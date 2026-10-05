# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime gate for Frontier trace logging."""

from contextlib import contextmanager
import json
import os
import time
from typing import Any, Mapping, Optional, TextIO

import torch

_SKIP_WARMUP = os.environ.get("VLLM_FRONTIER_TRACE_SKIP_WARMUP", "0") == "1"
_TRACE_ACTIVE = not _SKIP_WARMUP
_PP_BOUNDARY_LOG_ENV_VAR = "VLLM_FRONTIER_PP_BOUNDARY_LOG_PATH"
_DP_PLACEMENT_LOG_DIR_ENV_VAR = "VLLM_FRONTIER_DP_PLACEMENT_LOG_DIR"
_PP_BOUNDARY_REQUIRED_FIELDS = (
    "model_name",
    "timestamp",
    "batch_id",
    "batch_size",
    "tensor_parallel_degree",
    "num_prefill_tokens",
    "num_decode_tokens",
    "request_ids",
    "pp_rank",
    "tp_rank",
    "pp_world_size",
    "is_first_rank",
    "is_last_rank",
    "activation_bytes_per_rank",
    "recv_start_ts",
    "recv_end_ts",
    "preprocess_start_ts",
    "preprocess_end_ts",
    "forward_start_ts",
    "forward_end_ts",
    "send_start_ts",
    "send_end_ts",
)


def activate() -> None:
    global _TRACE_ACTIVE
    _TRACE_ACTIVE = True


def deactivate() -> None:
    global _TRACE_ACTIVE
    _TRACE_ACTIVE = False


def is_active() -> bool:
    return _TRACE_ACTIVE


def should_skip_warmup() -> bool:
    return _SKIP_WARMUP


def get_pp_boundary_log_path() -> str:
    return os.environ.get(_PP_BOUNDARY_LOG_ENV_VAR, "")


def is_pp_boundary_logging_enabled() -> bool:
    return is_active() and bool(get_pp_boundary_log_path())


def log_pp_boundary_record(record: Mapping[str, Any]) -> bool:
    if not is_pp_boundary_logging_enabled():
        return False

    missing_fields = [
        field for field in _PP_BOUNDARY_REQUIRED_FIELDS if field not in record
    ]
    if missing_fields:
        raise ValueError(
            "PP boundary record is missing required fields: "
            f"{missing_fields}."
        )

    log_path = get_pp_boundary_log_path()
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    try:
        with open(log_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(dict(record)) + "\n")
    except OSError as exc:
        raise RuntimeError(
            f"Failed to open Frontier PP boundary log file: {log_path}"
        ) from exc
    return True


# Per-step host timing (CPU probe). The engine core and the output rank's model
# runner (TP rank 0; with one worker it runs in the engine core process) take
# time.perf_counter stamps, which read CLOCK_MONOTONIC and so compare across the
# processes of one host. The runner also records two CUDA event pairs per step,
# around the model forward and around compute_logits plus sampling, and returns
# its stamps and the two device times in ModelRunnerOutput.cpu_probe_step. The
# engine core keeps every step in memory, so no file write lands inside a timed
# step, and writes them once, from EngineCore.shutdown, which the engine core's
# signal handler reaches. Each engine core writes its own file: with data
# parallelism, DP rank d writes <stem>_dp<d><suffix> of the configured path.
#
# With pipeline parallelism the engine core runs a batch queue, and its record
# of a batch holds that batch's schedule stamps, the time its output returned
# and the end of update_from_output (engine_loop "batch_queue"). Every stage's
# worker (TP rank 0) also keeps a stage record per forward: its PP receive and
# send stamps, the runner's stamps with forward_start, the token count the
# forward ran and its CUDA graph mode, and the forward's device time, read from
# CUDA events when the record is written, so the probe adds no device sync to a
# stage. The worker writes them from Worker.shutdown to
# <stem>[_dp<d>]_pp<p><suffix>.
_CPU_PROBE_LOG_PATH = os.environ.get("VLLM_FRONTIER_CPU_PROBE_LOG_PATH", "")
CPU_PROBE_ENABLED = bool(_CPU_PROBE_LOG_PATH)
_CPU_PROBE_ENGINE_FIELDS = ("step_start", "schedule_end", "execute_end",
                            "update_end")
_CPU_PROBE_RUNNER_FIELDS = ("execute_start", "preprocess_end", "forward_end",
                            "sample_end", "bookkeep_end", "execute_return",
                            "forward_device_ms", "sample_device_ms")
_CPU_PROBE_STAGE_RUNNER_FIELDS = ("execute_start", "preprocess_end",
                                  "forward_start", "forward_end",
                                  "execute_return")
_cpu_probe_steps: list[tuple[str, tuple[float, ...],
                             Optional[tuple[float, ...]],
                             Mapping[str, int]]] = []
_cpu_probe_stage_steps: list[dict[str, Any]] = []


def new_cpu_probe_events() -> tuple[torch.cuda.Event, ...]:
    """Forward start, forward end, sample start and sample end of one step."""
    return tuple(torch.cuda.Event(enable_timing=True) for _ in range(4))


def cpu_probe_runner_step(
        stamps: tuple[float, ...],
        events: tuple[torch.cuda.Event, ...]) -> tuple[float, ...]:
    """The runner's stamps of one step followed by its forward and sampling
    device times (_CPU_PROBE_RUNNER_FIELDS).

    Without async scheduling the step's bookkeeping has already waited for the
    sampled token ids, which the stream produces after the last event, so the
    wait here returns at once.
    """
    events[3].synchronize()
    return stamps + (events[0].elapsed_time(events[1]),
                     events[2].elapsed_time(events[3]))


def record_cpu_probe_step(engine_loop: str, engine_stamps: tuple[float, ...],
                          runner_step: Optional[tuple[float, ...]],
                          num_scheduled_tokens: Mapping[str, int]) -> None:
    """Close one engine step; a step without a forward has no runner step."""
    _cpu_probe_steps.append(
        (engine_loop, engine_stamps, runner_step, num_scheduled_tokens))


def record_cpu_probe_stage_step(
        pipeline_stage_id: int, num_scheduled_tokens: Mapping[str, int],
        recv: tuple[Optional[float], Optional[float]],
        runner_stamps: tuple[float, ...],
        forward_events: tuple[torch.cuda.Event, torch.cuda.Event],
        num_input_tokens: int, cudagraph_runtime_mode: str,
        send: tuple[Optional[float], Optional[float]]) -> None:
    """Keep one PP stage's forward: receive, runner stamps
    (_CPU_PROBE_STAGE_RUNNER_FIELDS), send, the forward's event pair, and the
    token count it ran (padded for a CUDA graph and across DP ranks) with its
    CUDA graph mode."""
    record: dict[str, Any] = {
        "pipeline_stage_id": pipeline_stage_id,
        "recv_start": recv[0],
        "recv_end": recv[1],
    }
    record.update(zip(_CPU_PROBE_STAGE_RUNNER_FIELDS, runner_stamps))
    record.update(send_start=send[0], send_end=send[1],
                  forward_events=forward_events,
                  num_input_tokens=num_input_tokens,
                  cudagraph_runtime_mode=cudagraph_runtime_mode,
                  num_scheduled_tokens=dict(num_scheduled_tokens))
    _cpu_probe_stage_steps.append(record)


def _cpu_probe_log_path(data_parallel_rank: int, data_parallel_size: int,
                        pipeline_stage_id: Optional[int] = None) -> str:
    stem, suffix = os.path.splitext(_CPU_PROBE_LOG_PATH)
    if data_parallel_size > 1:
        stem += f"_dp{data_parallel_rank}"
    if pipeline_stage_id is not None:
        stem += f"_pp{pipeline_stage_id}"
    return stem + suffix


def _write_records(path: str, records) -> None:
    log_dir = os.path.dirname(path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def write_cpu_probe_log(data_parallel_rank: int,
                        data_parallel_size: int) -> None:
    if not CPU_PROBE_ENABLED:
        return

    def records():
        for step, (engine_loop, engine_stamps, runner_step,
                   num_scheduled_tokens) in enumerate(_cpu_probe_steps):
            record: dict[str, Any] = {"step": step, "engine_loop": engine_loop}
            record.update(zip(_CPU_PROBE_ENGINE_FIELDS, engine_stamps))
            if runner_step is None:
                record.update(dict.fromkeys(_CPU_PROBE_RUNNER_FIELDS))
            else:
                record.update(zip(_CPU_PROBE_RUNNER_FIELDS, runner_step))
            record["num_scheduled_tokens"] = dict(num_scheduled_tokens)
            yield record

    _write_records(
        _cpu_probe_log_path(data_parallel_rank, data_parallel_size),
        records())


def write_cpu_probe_stage_log(data_parallel_rank: int,
                              data_parallel_size: int) -> None:
    if not _cpu_probe_stage_steps:
        return

    def records():
        for step, record in enumerate(_cpu_probe_stage_steps):
            start, end = record.pop("forward_events")
            end.synchronize()
            yield {"step": step, **record,
                   "forward_device_ms": start.elapsed_time(end)}

    _write_records(
        _cpu_probe_log_path(data_parallel_rank, data_parallel_size,
                            _cpu_probe_stage_steps[0]["pipeline_stage_id"]),
        records())
    _cpu_probe_stage_steps.clear()


@contextmanager
def disable_for_warmup():
    if not _SKIP_WARMUP:
        yield
        return
    global _TRACE_ACTIVE
    _TRACE_ACTIVE = False
    try:
        yield
    finally:
        _TRACE_ACTIVE = True


# Data-parallel placement records. Every process that takes part in a placement
# decision -- each engine core, the coordinator, each API server -- writes its
# own file, so records never interleave and the reader can tell the roles
# apart. Each record reaches the file as it is written: engine cores and the
# coordinator are forked and stopped by a signal, so a forked process never
# runs exit hooks and a buffer held in memory would be lost with it.
_dp_placement_file: Optional[TextIO] = None
_dp_placement_file_pid: Optional[int] = None
_dp_placement_seq = 0


def get_dp_placement_log_dir() -> str:
    return os.environ.get(_DP_PLACEMENT_LOG_DIR_ENV_VAR, "")


def is_dp_placement_logging_enabled() -> bool:
    return is_active() and bool(get_dp_placement_log_dir())


def log_dp_placement_record(kind: str, **fields: Any) -> None:
    """Write one placement record of the given kind.

    `seq` orders the records one process wrote and is the tie-break when two
    of them carry the same correlation id. Timestamps are for reading; the
    join between processes is always a correlation id.
    """
    if not is_dp_placement_logging_enabled():
        return

    global _dp_placement_file, _dp_placement_file_pid, _dp_placement_seq
    pid = os.getpid()
    if _dp_placement_file_pid != pid:
        # A forked process inherits the parent's handle; it opens its own.
        log_dir = get_dp_placement_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"dp_placement_{pid}.jsonl")
        try:
            _dp_placement_file = open(log_path,
                                      "a",
                                      encoding="utf-8",
                                      buffering=1)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to open Frontier DP placement log file: {log_path}"
            ) from exc
        _dp_placement_file_pid = pid
        _dp_placement_seq = 0

    record: dict[str, Any] = {
        "kind": kind,
        "pid": pid,
        "seq": _dp_placement_seq,
        "monotonic": time.monotonic(),
    }
    record.update(fields)
    _dp_placement_file.write(json.dumps(record) + "\n")
    _dp_placement_seq += 1

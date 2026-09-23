# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime gate for Frontier trace logging."""

from contextlib import contextmanager
import json
import os
import time
from typing import Any, Mapping, Optional, TextIO

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

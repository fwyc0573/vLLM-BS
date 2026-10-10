# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

from vllm.v1 import frontier_trace


class _Event:
    """Stands in for one CUDA event of a forward's pair."""

    def __init__(self, elapsed_ms: float = 0.0):
        self._elapsed_ms = elapsed_ms

    def synchronize(self) -> None:
        pass

    def elapsed_time(self, end: "_Event") -> float:
        return end._elapsed_ms


def _read(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _scheduler_output(new_reqs=(), cached_reqs=()):
    """A SchedulerOutput's request fields: new_reqs holds (req_id,
    prompt_len, num_computed_tokens, num_scheduled_tokens), cached_reqs
    (req_id, num_computed_tokens, num_scheduled_tokens)."""
    return SimpleNamespace(
        num_scheduled_tokens={
            **{req[0]: req[3]
               for req in new_reqs},
            **{req[0]: req[2]
               for req in cached_reqs}
        },
        scheduled_new_reqs=[
            SimpleNamespace(req_id=req_id,
                            prompt_token_ids=[0] * prompt_len,
                            num_computed_tokens=computed)
            for req_id, prompt_len, computed, _ in new_reqs
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[req[0] for req in cached_reqs],
            num_computed_tokens=[req[1] for req in cached_reqs]),
    )


def test_dp_allreduce_stamps_are_taken_once(monkeypatch):
    monkeypatch.setattr(frontier_trace, "_cpu_probe_dp_allreduce",
                        (None, None))
    frontier_trace.record_cpu_probe_dp_allreduce(1.0, 1.5)
    assert frontier_trace.take_cpu_probe_dp_allreduce() == (1.0, 1.5)
    assert frontier_trace.take_cpu_probe_dp_allreduce() == (None, None)


def test_kv_load_stamps_are_taken_once(monkeypatch):
    monkeypatch.setattr(frontier_trace, "_cpu_probe_kv_load", (None, None))
    frontier_trace.record_cpu_probe_kv_load(1.0, 1.4)
    assert frontier_trace.take_cpu_probe_kv_load() == (1.0, 1.4)
    assert frontier_trace.take_cpu_probe_kv_load() == (None, None)


def test_single_stage_engine_writes_engine_and_stage_records(
        tmp_path, monkeypatch):
    monkeypatch.setattr(frontier_trace, "_CPU_PROBE_LOG_PATH",
                        str(tmp_path / "cpu_probe.jsonl"))
    monkeypatch.setattr(frontier_trace, "CPU_PROBE_ENABLED", True)
    monkeypatch.setattr(frontier_trace, "_cpu_probe_steps", [])
    monkeypatch.setattr(frontier_trace, "_cpu_probe_stage_steps", [])

    frontier_trace.record_cpu_probe_step(
        "step", (1.0, 1.1, 1.9, 2.0),
        _scheduler_output(new_reqs=[("req-0", 17, 0, 16)]))
    frontier_trace.record_cpu_probe_step("step", (2.1, 2.2, 2.2, 2.3),
                                         _scheduler_output())
    runner_stamps = (1.2, 1.3, 1.31, 1.32, 1.6, 1.7, 1.8, 1.85)
    frontier_trace.record_cpu_probe_stage_step(
        0, {"req-0": 16}, (None, None), runner_stamps, (None, None),
        (None, None), (_Event(), _Event(250.0)), 16, "NONE", (None, None))
    frontier_trace.write_cpu_probe_log(0, 1, None)
    frontier_trace.write_cpu_probe_stage_log(0, 1)

    engine = _read(tmp_path / "cpu_probe.jsonl")
    assert engine == [
        {"step": 0, "engine_loop": "step", "step_start": 1.0,
         "schedule_end": 1.1, "execute_end": 1.9, "update_end": 2.0,
         "num_scheduled_tokens": {"req-0": 16},
         "num_computed_tokens": {"req-0": 0},
         "num_prompt_tokens": {"req-0": 17}, "kv_role": None},
        {"step": 1, "engine_loop": "step", "step_start": 2.1,
         "schedule_end": 2.2, "execute_end": 2.2, "update_end": 2.3,
         "num_scheduled_tokens": {}, "num_computed_tokens": {},
         "num_prompt_tokens": {}, "kv_role": None},
    ]
    [stage] = _read(tmp_path / "cpu_probe_pp0.jsonl")
    assert stage == {
        "step": 0, "pipeline_stage_id": 0, "recv_start": None,
        "recv_end": None, "execute_start": 1.2, "preprocess_end": 1.3,
        "forward_start": 1.31, "forward_context_entered": 1.32,
        "forward_end": 1.6, "sample_end": 1.7, "bookkeep_end": 1.8,
        "execute_return": 1.85, "dp_allreduce_start": None,
        "dp_allreduce_end": None, "kv_load_start": None,
        "kv_load_end": None, "send_start": None, "send_end": None,
        "num_input_tokens": 16, "cudagraph_runtime_mode": "NONE",
        "num_scheduled_tokens": {"req-0": 16}, "forward_device_ms": 250.0,
    }


def test_dp_stage_record_names_its_engine_and_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(frontier_trace, "_CPU_PROBE_LOG_PATH",
                        str(tmp_path / "cpu_probe.jsonl"))
    monkeypatch.setattr(frontier_trace, "_cpu_probe_stage_steps", [])

    runner_stamps = (1.2, 1.3, 1.31, 1.4, 1.6, None, None, 1.62)
    frontier_trace.record_cpu_probe_stage_step(
        0, {"req-0": 1}, (None, None), runner_stamps, (1.32, 1.39),
        (None, None), (_Event(), _Event(3.0)), 2, "NONE", (1.62, 1.63))
    frontier_trace.write_cpu_probe_stage_log(1, 2)

    [stage] = _read(tmp_path / "cpu_probe_dp1_pp0.jsonl")
    assert (stage["dp_allreduce_start"], stage["dp_allreduce_end"]) == (1.32,
                                                                        1.39)
    assert (stage["sample_end"], stage["bookkeep_end"]) == (None, None)
    assert (stage["send_start"], stage["send_end"]) == (1.62, 1.63)


def test_engine_records_carry_each_request_phase_and_kv_role(
        tmp_path, monkeypatch):
    """A request's prompt length comes from its first schedule; its computed
    tokens are the scheduler output's, before the step. A PD decode instance's
    first step of a request starts after the connector's prompt tokens."""
    monkeypatch.setattr(frontier_trace, "_CPU_PROBE_LOG_PATH",
                        str(tmp_path / "cpu_probe.jsonl"))
    monkeypatch.setattr(frontier_trace, "CPU_PROBE_ENABLED", True)
    monkeypatch.setattr(frontier_trace, "_cpu_probe_steps", [])

    frontier_trace.record_cpu_probe_step(
        "step", (1.0, 1.1, 1.9, 2.0),
        _scheduler_output(new_reqs=[("req-0", 65, 64, 1)]))
    frontier_trace.record_cpu_probe_step(
        "step", (2.1, 2.2, 2.8, 2.9),
        _scheduler_output(new_reqs=[("req-1", 9, 8, 1)],
                          cached_reqs=[("req-0", 65, 1)]))
    frontier_trace.write_cpu_probe_log(0, 1, "kv_consumer")

    first, second = _read(tmp_path / "cpu_probe.jsonl")
    assert first["num_computed_tokens"] == {"req-0": 64}
    assert first["num_prompt_tokens"] == {"req-0": 65}
    assert second["num_scheduled_tokens"] == {"req-1": 1, "req-0": 1}
    assert second["num_computed_tokens"] == {"req-1": 8, "req-0": 65}
    assert second["num_prompt_tokens"] == {"req-1": 9, "req-0": 65}
    assert first["kv_role"] == second["kv_role"] == "kv_consumer"

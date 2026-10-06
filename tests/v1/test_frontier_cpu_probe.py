# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

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


def test_dp_allreduce_stamps_are_taken_once(monkeypatch):
    monkeypatch.setattr(frontier_trace, "_cpu_probe_dp_allreduce",
                        (None, None))
    frontier_trace.record_cpu_probe_dp_allreduce(1.0, 1.5)
    assert frontier_trace.take_cpu_probe_dp_allreduce() == (1.0, 1.5)
    assert frontier_trace.take_cpu_probe_dp_allreduce() == (None, None)


def test_single_stage_engine_writes_engine_and_stage_records(
        tmp_path, monkeypatch):
    monkeypatch.setattr(frontier_trace, "_CPU_PROBE_LOG_PATH",
                        str(tmp_path / "cpu_probe.jsonl"))
    monkeypatch.setattr(frontier_trace, "CPU_PROBE_ENABLED", True)
    monkeypatch.setattr(frontier_trace, "_cpu_probe_steps", [])
    monkeypatch.setattr(frontier_trace, "_cpu_probe_stage_steps", [])

    frontier_trace.record_cpu_probe_step("step", (1.0, 1.1, 1.9, 2.0),
                                         {"req-0": 16})
    frontier_trace.record_cpu_probe_step("step", (2.1, 2.2, 2.2, 2.3), {})
    runner_stamps = (1.2, 1.3, 1.31, 1.32, 1.6, 1.7, 1.8, 1.85)
    frontier_trace.record_cpu_probe_stage_step(
        0, {"req-0": 16}, (None, None), runner_stamps, (None, None),
        (_Event(), _Event(250.0)), 16, "NONE", (None, None))
    frontier_trace.write_cpu_probe_log(0, 1)
    frontier_trace.write_cpu_probe_stage_log(0, 1)

    engine = _read(tmp_path / "cpu_probe.jsonl")
    assert engine == [
        {"step": 0, "engine_loop": "step", "step_start": 1.0,
         "schedule_end": 1.1, "execute_end": 1.9, "update_end": 2.0,
         "num_scheduled_tokens": {"req-0": 16}},
        {"step": 1, "engine_loop": "step", "step_start": 2.1,
         "schedule_end": 2.2, "execute_end": 2.2, "update_end": 2.3,
         "num_scheduled_tokens": {}},
    ]
    [stage] = _read(tmp_path / "cpu_probe_pp0.jsonl")
    assert stage == {
        "step": 0, "pipeline_stage_id": 0, "recv_start": None,
        "recv_end": None, "execute_start": 1.2, "preprocess_end": 1.3,
        "forward_start": 1.31, "forward_context_entered": 1.32,
        "forward_end": 1.6, "sample_end": 1.7, "bookkeep_end": 1.8,
        "execute_return": 1.85, "dp_allreduce_start": None,
        "dp_allreduce_end": None, "send_start": None, "send_end": None,
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
        (_Event(), _Event(3.0)), 2, "NONE", (1.62, 1.63))
    frontier_trace.write_cpu_probe_stage_log(1, 2)

    [stage] = _read(tmp_path / "cpu_probe_dp1_pp0.jsonl")
    assert (stage["dp_allreduce_start"], stage["dp_allreduce_end"]) == (1.32,
                                                                        1.39)
    assert (stage["sample_end"], stage["bookkeep_end"]) == (None, None)
    assert (stage["send_start"], stage["send_end"]) == (1.62, 1.63)

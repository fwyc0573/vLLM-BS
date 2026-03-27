# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.v1 import frontier_trace


def _pp_boundary_record() -> dict:
    return {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "timestamp": 1000.0,
        "batch_id": 7,
        "batch_size": 2,
        "tensor_parallel_degree": 1,
        "num_prefill_tokens": 0,
        "num_decode_tokens": 2,
        "request_ids": ["cmpl-0-0", "cmpl-1-0"],
        "pp_rank": 1,
        "pp_world_size": 8,
        "is_first_rank": False,
        "is_last_rank": False,
        "activation_bytes_per_rank": 16384,
        "recv_start_ts": 10.000,
        "recv_end_ts": 10.003,
        "preprocess_start_ts": 10.003,
        "preprocess_end_ts": 10.007,
        "forward_start_ts": 10.012,
        "forward_end_ts": 10.019,
        "send_start_ts": 10.020,
        "send_end_ts": 10.024,
    }


def test_log_pp_boundary_record_skips_when_trace_is_inactive(
    tmp_path,
    monkeypatch,
) -> None:
    log_path = tmp_path / "pp_boundary.jsonl"
    monkeypatch.setenv("VLLM_FRONTIER_PP_BOUNDARY_LOG_PATH", str(log_path))
    frontier_trace.deactivate()

    assert frontier_trace.log_pp_boundary_record(_pp_boundary_record()) is False
    assert not log_path.exists()

    frontier_trace.activate()


def test_log_pp_boundary_record_writes_jsonl_when_enabled(
    tmp_path,
    monkeypatch,
) -> None:
    log_path = tmp_path / "pp_boundary.jsonl"
    monkeypatch.setenv("VLLM_FRONTIER_PP_BOUNDARY_LOG_PATH", str(log_path))
    frontier_trace.activate()

    assert frontier_trace.log_pp_boundary_record(_pp_boundary_record()) is True

    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    assert records[0]["model_name"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert records[0]["pp_rank"] == 1
    assert records[0]["activation_bytes_per_rank"] == 16384


def test_log_pp_boundary_record_rejects_missing_required_fields(
    tmp_path,
    monkeypatch,
) -> None:
    log_path = tmp_path / "pp_boundary.jsonl"
    monkeypatch.setenv("VLLM_FRONTIER_PP_BOUNDARY_LOG_PATH", str(log_path))
    frontier_trace.activate()

    with pytest.raises(ValueError, match="missing required fields"):
        frontier_trace.log_pp_boundary_record({"pp_rank": 1})

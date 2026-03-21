import json

import pytest

from vllm.v1.utils import FrontierCudaEventOpLogger


class _FakeProfilerStat:

    def __init__(self, key: str, cuda_time_total: float, count: int) -> None:
        self.key = key
        self.cuda_time_total = cuda_time_total
        self.count = count


class _FakeProfilerStatDeviceTime:

    def __init__(self, key: str, device_time_total: float, count: int) -> None:
        self.key = key
        self.device_time_total = device_time_total
        self.count = count


class _FakeProfiler:

    def __init__(self) -> None:
        self._entered = False
        self._exited = False
        self._stats = [
            _FakeProfilerStat("frontier_attn_prefill", 2500.0, 2),
            _FakeProfilerStat("frontier_attn_decode", 1200.0, 1),
            _FakeProfilerStat("some_unrelated_scope", 9999.0, 3),
        ]

    def __enter__(self):
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._exited = True
        return False

    def key_averages(self):
        if not self._entered:
            raise RuntimeError("profiler enter was not called")
        if not self._exited:
            raise RuntimeError("profiler exit was not called")
        return self._stats

    def export_chrome_trace(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as trace_file:
            json.dump({"traceEvents": []}, trace_file)


class _FakeCudaEvent:

    def __init__(self, enable_timing: bool = True) -> None:
        if not enable_timing:
            raise RuntimeError("enable_timing must be True")

    def record(self) -> None:
        return

    def elapsed_time(self, other: "_FakeCudaEvent") -> float:
        del other
        return 1.75


class _FakeTPGroup:

    def __init__(self) -> None:
        self.world_size = 2
        self.unique_name = "tp:0"
        self.rank_in_group = 1
        self.all_reduce_inputs = []

    def all_reduce(self, input_):
        self.all_reduce_inputs.append(input_)
        return input_


def _read_single_record(path) -> dict:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    return json.loads(lines[0])


def _read_records(path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


def test_record_function_mode_logs_timing_mode_and_cuda_time(tmp_path,
                                                             monkeypatch):
    log_path = tmp_path / "frontier_record_function.jsonl"
    fake_profiler = _FakeProfiler()

    monkeypatch.setattr("torch.cuda.synchronize", lambda: None)
    monkeypatch.setattr("torch.profiler.profile", lambda *args, **kwargs: fake_profiler)
    monkeypatch.setattr(
        "vllm.v1.utils._frontier_collect_record_function_scopes",
        lambda *args, **kwargs: [
            {
                "op_name": "attn_prefill",
                "cuda_time_ms": 2.5,
                "count": 1,
                "scope_seq": 0,
            }
        ],
    )

    logger = FrontierCudaEventOpLogger(
        str(log_path),
        scopes=["attn_prefill"],
        timing_mode="record_function",
    )

    logger.start_batch(
        batch_id=0,
        batch_size=1,
        batch_num_tokens=4,
        batch_num_prefill_tokens=4,
        batch_num_decode_tokens=0,
    )
    with logger.scope("attn_prefill"):
        pass
    logger.finish_batch()

    record = _read_single_record(log_path)
    assert record["op_name"] == "attn_prefill"
    assert record["timing_mode"] == "record_function"
    assert record["aggregation_mode"] == "per_scope"
    assert record["scope_seq"] == 0
    assert record["count"] == 1
    assert record["cuda_time_ms"] == pytest.approx(2.5, rel=1e-6)


def test_record_function_mode_accepts_device_time_total(tmp_path, monkeypatch):
    log_path = tmp_path / "frontier_record_function_device_time.jsonl"
    fake_profiler = _FakeProfiler()
    fake_profiler._stats = [
        _FakeProfilerStatDeviceTime("frontier_attn_prefill", 3300.0, 3),
    ]

    monkeypatch.setattr("torch.cuda.synchronize", lambda: None)
    monkeypatch.setattr("torch.profiler.profile", lambda *args, **kwargs: fake_profiler)
    monkeypatch.setattr(
        "vllm.v1.utils._frontier_collect_record_function_scopes",
        lambda *args, **kwargs: [
            {
                "op_name": "attn_prefill",
                "cuda_time_ms": 3.3,
                "count": 1,
                "scope_seq": 0,
            }
        ],
    )

    logger = FrontierCudaEventOpLogger(
        str(log_path),
        scopes=["attn_prefill"],
        timing_mode="record_function",
    )

    logger.start_batch(
        batch_id=2,
        batch_size=1,
        batch_num_tokens=4,
        batch_num_prefill_tokens=4,
        batch_num_decode_tokens=0,
    )
    with logger.scope("attn_prefill"):
        pass
    logger.finish_batch()

    record = _read_single_record(log_path)
    assert record["op_name"] == "attn_prefill"
    assert record["timing_mode"] == "record_function"
    assert record["aggregation_mode"] == "per_scope"
    assert record["scope_seq"] == 0
    assert record["count"] == 1
    assert record["cuda_time_ms"] == pytest.approx(3.3, rel=1e-6)


def test_cuda_event_mode_logs_timing_mode_and_cuda_time(tmp_path, monkeypatch):
    log_path = tmp_path / "frontier_cuda_event.jsonl"

    monkeypatch.setattr("torch.cuda.synchronize", lambda: None)
    monkeypatch.setattr("torch.cuda.Event", _FakeCudaEvent)

    logger = FrontierCudaEventOpLogger(
        str(log_path),
        scopes=["attn_prefill"],
        timing_mode="cuda_event",
    )

    logger.start_batch(
        batch_id=1,
        batch_size=1,
        batch_num_tokens=4,
        batch_num_prefill_tokens=4,
        batch_num_decode_tokens=0,
    )
    with logger.scope("attn_prefill"):
        pass
    logger.finish_batch()

    record = _read_single_record(log_path)
    assert record["op_name"] == "attn_prefill"
    assert record["timing_mode"] == "cuda_event"
    assert record["count"] == 1
    assert record["cuda_time_ms"] == pytest.approx(1.75, rel=1e-6)


def test_cuda_event_mode_per_scope_emits_one_record_per_scope_occurrence(
    tmp_path,
    monkeypatch,
):
    log_path = tmp_path / "frontier_cuda_event_per_scope.jsonl"

    monkeypatch.setattr("torch.cuda.synchronize", lambda: None)
    monkeypatch.setattr("torch.cuda.Event", _FakeCudaEvent)

    logger = FrontierCudaEventOpLogger(
        str(log_path),
        scopes=["attn_prefill"],
        timing_mode="cuda_event",
        aggregation_mode="per_scope",
    )

    logger.start_batch(
        batch_id=4,
        batch_size=1,
        batch_num_tokens=4,
        batch_num_prefill_tokens=4,
        batch_num_decode_tokens=0,
    )
    with logger.scope("attn_prefill"):
        pass
    with logger.scope("attn_prefill"):
        pass
    logger.finish_batch()

    records = _read_records(log_path)
    assert len(records) == 2
    assert [record["scope_seq"] for record in records] == [0, 1]
    for record in records:
        assert record["op_name"] == "attn_prefill"
        assert record["timing_mode"] == "cuda_event"
        assert record["aggregation_mode"] == "per_scope"
        assert record["count"] == 1
        assert record["cuda_time_ms"] == pytest.approx(1.75, rel=1e-6)


def test_cuda_event_mode_batch_sum_legacy_keeps_one_aggregated_record(
    tmp_path,
    monkeypatch,
):
    log_path = tmp_path / "frontier_cuda_event_batch_sum_legacy.jsonl"

    monkeypatch.setattr("torch.cuda.synchronize", lambda: None)
    monkeypatch.setattr("torch.cuda.Event", _FakeCudaEvent)

    logger = FrontierCudaEventOpLogger(
        str(log_path),
        scopes=["attn_prefill"],
        timing_mode="cuda_event",
        aggregation_mode="batch_sum",
    )

    logger.start_batch(
        batch_id=5,
        batch_size=1,
        batch_num_tokens=4,
        batch_num_prefill_tokens=4,
        batch_num_decode_tokens=0,
    )
    with logger.scope("attn_prefill"):
        pass
    with logger.scope("attn_prefill"):
        pass
    logger.finish_batch()

    record = _read_single_record(log_path)
    assert record["op_name"] == "attn_prefill"
    assert record["timing_mode"] == "cuda_event"
    assert record["aggregation_mode"] == "batch_sum_legacy"
    assert record["count"] == 2
    assert record["cuda_time_ms"] == pytest.approx(3.5, rel=1e-6)


def test_invalid_timing_mode_raises(tmp_path):
    with pytest.raises(ValueError, match="VLLM_FRONTIER_OP_TIMING_MODE"):
        FrontierCudaEventOpLogger(
            str(tmp_path / "invalid_mode.jsonl"),
            scopes=["attn_prefill"],
            timing_mode="invalid_mode",
        )


def test_tensor_model_parallel_all_reduce_records_meta_inside_active_scope(
    tmp_path,
    monkeypatch,
):
    from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
    from vllm.v1 import frontier_trace

    log_path = tmp_path / "frontier_collective_meta.jsonl"
    fake_tp_group = _FakeTPGroup()

    monkeypatch.setattr("torch.cuda.synchronize", lambda: None)
    monkeypatch.setattr("torch.cuda.Event", _FakeCudaEvent)
    monkeypatch.setattr(
        "vllm.distributed.communication_op.get_tp_group",
        lambda: fake_tp_group,
    )

    logger = FrontierCudaEventOpLogger(
        str(log_path),
        scopes=["expert_parallel_allreduce"],
        meta_enabled=True,
        timing_mode="cuda_event",
        aggregation_mode="per_scope",
    )

    logger.start_batch(
        batch_id=6,
        batch_size=1,
        batch_num_tokens=1,
        batch_num_prefill_tokens=0,
        batch_num_decode_tokens=1,
    )
    trace_was_active = frontier_trace.is_active()
    frontier_trace.activate()
    try:
        with logger.activate():
            sentinel = object()
            result = tensor_model_parallel_all_reduce(
                sentinel,
                record_scope_name="expert_parallel_allreduce",
            )
    finally:
        if not trace_was_active:
            frontier_trace.deactivate()
    logger.finish_batch()

    assert result is sentinel
    assert fake_tp_group.all_reduce_inputs == [sentinel]

    record = _read_single_record(log_path)
    assert record["op_name"] == "expert_parallel_allreduce"
    assert record["timing_mode"] == "cuda_event"
    assert record["aggregation_mode"] == "per_scope"
    assert record["scope_seq"] == 0
    assert record["count"] == 1
    assert record["cuda_time_ms"] == pytest.approx(1.75, rel=1e-6)
    assert record["meta"] == {
        "collective_base_op_name": "expert_parallel_allreduce",
        "collective_domain": "TP",
        "collective_group_unique_name": "tp:0",
        "collective_rank_in_group": 1,
        "collective_world_size": 2,
    }



def test_cuda_event_mode_allows_empty_pure_decode_batch(tmp_path, monkeypatch):
    log_path = tmp_path / "frontier_cuda_event_empty_decode.jsonl"
    sync_calls = []

    monkeypatch.setattr("torch.cuda.synchronize", lambda: sync_calls.append(True))
    monkeypatch.setattr("torch.cuda.Event", _FakeCudaEvent)

    logger = FrontierCudaEventOpLogger(
        str(log_path),
        scopes=["attn_decode"],
        timing_mode="cuda_event",
        allow_empty_pure_decode_batch=True,
    )

    logger.start_batch(
        batch_id=3,
        batch_size=2,
        batch_num_tokens=2,
        batch_num_prefill_tokens=0,
        batch_num_decode_tokens=2,
    )
    logger.finish_batch()

    assert sync_calls == [True]
    assert log_path.read_text(encoding="utf-8") == ""

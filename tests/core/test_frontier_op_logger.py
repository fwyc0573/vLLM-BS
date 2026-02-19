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


class _FakeCudaEvent:

    def __init__(self, enable_timing: bool = True) -> None:
        if not enable_timing:
            raise RuntimeError("enable_timing must be True")

    def record(self) -> None:
        return

    def elapsed_time(self, other: "_FakeCudaEvent") -> float:
        del other
        return 1.75


def _read_single_record(path) -> dict:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    return json.loads(lines[0])


def test_record_function_mode_logs_timing_mode_and_cuda_time(tmp_path,
                                                             monkeypatch):
    log_path = tmp_path / "frontier_record_function.jsonl"
    fake_profiler = _FakeProfiler()

    monkeypatch.setattr("torch.cuda.synchronize", lambda: None)
    monkeypatch.setattr("torch.profiler.profile", lambda *args, **kwargs: fake_profiler)

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
    assert record["count"] == 2
    assert record["cuda_time_ms"] == pytest.approx(2.5, rel=1e-6)


def test_record_function_mode_accepts_device_time_total(tmp_path, monkeypatch):
    log_path = tmp_path / "frontier_record_function_device_time.jsonl"
    fake_profiler = _FakeProfiler()
    fake_profiler._stats = [
        _FakeProfilerStatDeviceTime("frontier_attn_prefill", 3300.0, 3),
    ]

    monkeypatch.setattr("torch.cuda.synchronize", lambda: None)
    monkeypatch.setattr("torch.profiler.profile", lambda *args, **kwargs: fake_profiler)

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
    assert record["count"] == 3
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


def test_invalid_timing_mode_raises(tmp_path):
    with pytest.raises(ValueError, match="VLLM_FRONTIER_OP_TIMING_MODE"):
        FrontierCudaEventOpLogger(
            str(tmp_path / "invalid_mode.jsonl"),
            scopes=["attn_prefill"],
            timing_mode="invalid_mode",
        )

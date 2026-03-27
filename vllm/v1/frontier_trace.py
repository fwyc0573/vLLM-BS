# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime gate for Frontier trace logging."""

from contextlib import contextmanager
import json
import os
from typing import Any, Mapping

_SKIP_WARMUP = os.environ.get("VLLM_FRONTIER_TRACE_SKIP_WARMUP", "0") == "1"
_TRACE_ACTIVE = not _SKIP_WARMUP
_PP_BOUNDARY_LOG_ENV_VAR = "VLLM_FRONTIER_PP_BOUNDARY_LOG_PATH"
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

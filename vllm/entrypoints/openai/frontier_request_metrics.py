# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from dataclasses import asdict, is_dataclass
from threading import Lock
from typing import Any, Mapping, Optional

from vllm.logger import init_logger
from vllm.outputs import RequestOutput

logger = init_logger(__name__)


class FrontierRequestMetricsJSONLLogger:
    """Append finished Frontier-style request metrics to a JSONL file."""

    ENV_VAR = "VLLM_FRONTIER_REQUEST_METRICS_LOG_PATH"

    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        self._lock = Lock()
        self._logged_request_keys: set[tuple[str, Any, Any]] = set()

        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self._file = open(log_path, "a", encoding="utf-8", buffering=1)
        logger.info("Frontier request metrics JSONL logging enabled: %s",
                    log_path)

    @classmethod
    def from_env(cls) -> Optional["FrontierRequestMetricsJSONLLogger"]:
        log_path = os.getenv(cls.ENV_VAR, "")
        if not log_path:
            return None
        return cls(log_path)

    def log(self, request_output: RequestOutput) -> None:
        if not request_output.finished or request_output.metrics is None:
            return

        request_id = request_output.request_id
        payload = self._serialize_metrics(request_output.metrics)
        payload.setdefault("request_id", request_id)
        request_key = (
            request_id,
            payload.get("arrival_time"),
            payload.get("completion_time"),
        )
        with self._lock:
            if request_key in self._logged_request_keys:
                return

            self._file.write(json.dumps(payload, sort_keys=True) + "\n")
            self._file.flush()
            self._logged_request_keys.add(request_key)

    @staticmethod
    def _serialize_metrics(metrics: Any) -> dict[str, Any]:
        if is_dataclass(metrics):
            payload = asdict(metrics)
        elif isinstance(metrics, Mapping):
            payload = dict(metrics)
        elif hasattr(metrics, "model_dump"):
            payload = metrics.model_dump()
        else:
            raise TypeError(
                "RequestOutput.metrics must be a dataclass, mapping, or "
                "pydantic model for Frontier JSONL export.")

        if not isinstance(payload, dict):
            raise TypeError(
                "Serialized RequestOutput.metrics payload must be a dict.")
        return payload

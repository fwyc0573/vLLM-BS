# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Optional


class FrontierKVTransferJSONLLogger:
    """Append request-level KV transfer contract events to a JSONL file."""

    ENV_VAR = "VLLM_FRONTIER_KV_TRANSFER_LOG_PATH"

    def __init__(self, log_path: str, *, role: str, rank: int,
                 local_rank: int) -> None:
        self.log_path = log_path
        self.role = role
        self.rank = int(rank)
        self.local_rank = int(local_rank)
        self._lock = threading.Lock()

        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self._fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)

    @classmethod
    def from_env(
        cls,
        *,
        role: str,
        rank: int,
        local_rank: int,
    ) -> Optional["FrontierKVTransferJSONLLogger"]:
        log_path = os.getenv(cls.ENV_VAR, "")
        if not log_path:
            return None
        return cls(log_path, role=role, rank=rank, local_rank=local_rank)

    def log_producer_layer_send_start(
        self,
        *,
        request_id: str,
        layer_name: str,
        remote_address: str,
    ) -> None:
        self._log_event(
            event="producer_layer_send_start",
            request_id=request_id,
            layer_name=layer_name,
            remote_address=remote_address,
        )

    def log_consumer_layer_inject_end(
        self,
        *,
        request_id: str,
        layer_name: str,
    ) -> None:
        self._log_event(
            event="consumer_layer_inject_end",
            request_id=request_id,
            layer_name=layer_name,
        )

    def _log_event(
        self,
        *,
        event: str,
        request_id: str,
        layer_name: str,
        **extra: Any,
    ) -> None:
        payload: dict[str, Any] = {
            "timestamp": time.time(),
            "event": event,
            "request_id": request_id,
            "layer_name": layer_name,
            "role": self.role,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "transport_backend": "p2p_nccl",
            "includes_queueing": False,
            "includes_inject": event == "consumer_layer_inject_end",
        }
        payload.update(extra)
        record = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        with self._lock:
            os.write(self._fd, record)

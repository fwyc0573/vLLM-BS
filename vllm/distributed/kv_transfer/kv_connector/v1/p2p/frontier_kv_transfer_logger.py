# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import time
from threading import Lock
from typing import Any, Optional

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

VLLM_FRONTIER_KV_TRANSFER_LOG_PATH = "VLLM_FRONTIER_KV_TRANSFER_LOG_PATH"


class FrontierKVTransferJSONLLogger:
    """Append Frontier KV transfer contract events to a JSONL file."""

    ENV_VAR = VLLM_FRONTIER_KV_TRANSFER_LOG_PATH

    def __init__(
        self,
        log_path: str,
        role: KVConnectorRole,
        rank: int,
        local_rank: int,
    ) -> None:
        if not log_path:
            raise ValueError("KV transfer log path must not be empty.")

        self.log_path = log_path
        self._role = role
        self._rank = rank
        self._local_rank = local_rank
        self._lock = Lock()

        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self._fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)

    @classmethod
    def from_env(
        cls,
        role: KVConnectorRole,
        rank: int,
        local_rank: int,
    ) -> Optional["FrontierKVTransferJSONLLogger"]:
        log_path = os.getenv(cls.ENV_VAR, "")
        if not log_path:
            return None
        return cls(
            log_path=log_path,
            role=role,
            rank=rank,
            local_rank=local_rank,
        )

    def log_producer_layer_send_start(
        self,
        request_id: str,
        layer_name: str,
        remote_address: str,
    ) -> None:
        self._write_event(
            event="producer_layer_send_start",
            request_id=request_id,
            layer_name=layer_name,
            includes_inject=False,
            remote_address=remote_address,
        )

    def log_consumer_layer_inject_end(
        self,
        request_id: str,
        layer_name: str,
    ) -> None:
        self._write_event(
            event="consumer_layer_inject_end",
            request_id=request_id,
            layer_name=layer_name,
            includes_inject=True,
        )

    def _write_event(
        self,
        event: str,
        request_id: str,
        layer_name: str,
        includes_inject: bool,
        **extra: Any,
    ) -> None:
        payload = {
            "timestamp": time.time(),
            "event": event,
            "request_id": request_id,
            "layer_name": layer_name,
            "rank": self._rank,
            "local_rank": self._local_rank,
            "role": self._role.name.lower(),
            "transport_backend": "p2p_nccl",
            "includes_queueing": False,
            "includes_inject": includes_inject,
        }
        payload.update(extra)
        with self._lock:
            os.write(
                self._fd,
                (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"),
            )

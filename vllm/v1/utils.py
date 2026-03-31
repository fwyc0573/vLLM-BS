# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import contextlib
import contextvars
from dataclasses import dataclass
import math
import json
import multiprocessing
import os
import time
import weakref
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager
from multiprocessing import connection
from multiprocessing.process import BaseProcess
from typing import (TYPE_CHECKING, Any, Callable, Generic, Optional, TypeVar,
                    Union, overload)

import torch
from torch.autograd.profiler import record_function

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.v1 import frontier_trace
from vllm.usage.usage_lib import (UsageContext, is_usage_stats_enabled,
                                  usage_message)
from vllm.utils import (get_open_port, get_open_zmq_ipc_path, get_tcp_uri,
                        kill_process_tree)

if TYPE_CHECKING:
    import numpy as np

    from vllm.v1.engine.coordinator import DPCoordinator
    from vllm.v1.engine.utils import (CoreEngineActorManager,
                                      CoreEngineProcManager)

logger = init_logger(__name__)

T = TypeVar("T")

DEFAULT_FRONTIER_CUDA_EVENT_OP_SCOPES = [
    "input_layernorm",
    "attn_pre_proj",
    "attn_rope",
    "attn_kv_cache_save",
    "attn_prefill",
    "attn_decode",
    "attn_post_proj",
    "post_attention_layernorm",
    "moe_gating",
    "moe_shuffling",
    "moe_grouped_gemm",
    "expert_parallel_alltoall_dispatch",
    "expert_parallel_alltoall_combine",
    "expert_parallel_allreduce",
    "moe_tensor_parallel_allreduce",
    "tensor_parallel_allreduce",
    "kv_p2p_send",
    "kv_p2p_recv",
    "add",
]

_FRONTIER_CUDA_EVENT_OP_LOGGER: contextvars.ContextVar[
    Optional["FrontierCudaEventOpLogger"]
] = contextvars.ContextVar("frontier_cuda_event_op_logger", default=None)
_FRONTIER_MOE_ROUTING_LOGGER: contextvars.ContextVar[
    Optional["FrontierMoeRoutingLogger"]
] = contextvars.ContextVar("frontier_moe_routing_logger", default=None)
_FRONTIER_MOE_ROUTING_CONTEXT: contextvars.ContextVar[
    Optional["FrontierMoeRoutingContext"]
] = contextvars.ContextVar("frontier_moe_routing_context", default=None)
_FRONTIER_RUNTIME_POSITIONS_META: contextvars.ContextVar[
    Optional[dict[str, Any]]
] = contextvars.ContextVar("frontier_runtime_positions_meta", default=None)


def _frontier_contextvars_enabled() -> bool:
    return not torch.compiler.is_compiling()


def _frontier_find_child_events(trace_events: list[dict[str, Any]],
                                parent_event: dict[str, Any]) -> list[dict[str, Any]]:
    if "dur" not in parent_event or "ts" not in parent_event:
        return []

    children: list[dict[str, Any]] = []
    parent_start = parent_event["ts"]
    parent_end = parent_start + parent_event["dur"]
    for event in trace_events:
        if event is parent_event:
            continue
        if "dur" not in event or "ts" not in event:
            continue
        event_start = event["ts"]
        event_end = event_start + event["dur"]
        if event_start > parent_start and event_end < parent_end:
            children.append(event)
    return children


def _frontier_find_correlated_trace_event(
    trace_events: list[dict[str, Any]],
    trace_event: dict[str, Any],
) -> Optional[dict[str, Any]]:
    correlation = trace_event.get("args", {}).get("correlation")
    if correlation is None:
        return None
    for event in trace_events:
        if event is trace_event:
            continue
        if event.get("args", {}).get("correlation") == correlation:
            return event
    return None


def _frontier_build_record_function_trace_path(log_path: str, batch_id: int) -> str:
    log_dir = os.path.dirname(log_path)
    trace_dir = os.path.join(log_dir or ".", "frontier_profiler_traces")
    os.makedirs(trace_dir, exist_ok=True)
    return os.path.join(
        trace_dir,
        f"frontier_batch_{batch_id}_{os.getpid()}_{time.time_ns()}.json",
    )


def _frontier_aggregate_record_function_trace(
    trace_path: str,
    scopes: set[str],
    allow_zero_cuda_ops: Optional[set[str]] = None,
) -> dict[str, dict[str, float]]:
    scope_rows = _frontier_collect_record_function_scopes(
        trace_path,
        scopes,
        allow_zero_cuda_ops=allow_zero_cuda_ops,
    )
    aggregated: dict[str, dict[str, float]] = {}
    for row in scope_rows:
        op_name = str(row["op_name"])
        stats = aggregated.setdefault(op_name, {
            "cuda_time_ms": 0.0,
            "count": 0.0,
        })
        stats["cuda_time_ms"] += float(row["cuda_time_ms"])
        stats["count"] += float(row["count"])
    return aggregated


def _frontier_collect_record_function_scopes(
    trace_path: str,
    scopes: set[str],
    allow_zero_cuda_ops: Optional[set[str]] = None,
) -> list[dict[str, float | int | str]]:
    with open(trace_path, encoding="utf-8") as trace_file:
        trace_events = json.load(trace_file).get("traceEvents", [])

    scope_rows: list[dict[str, float | int | str]] = []
    scope_seq_by_op: dict[str, int] = {}
    allow_zero_cuda_ops = set(allow_zero_cuda_ops or set())
    for event in trace_events:
        if event.get("cat") != "user_annotation":
            continue
        raw_name = event.get("name", "")
        if not isinstance(raw_name, str) or not raw_name.startswith("frontier_"):
            continue
        op_name = raw_name[len("frontier_"):]
        if op_name not in scopes:
            continue

        cuda_time_us = 0.0
        for child in _frontier_find_child_events(trace_events, event):
            if child.get("cat") not in ("cuda_runtime", "cuda_driver"):
                continue
            correlated_event = _frontier_find_correlated_trace_event(trace_events, child)
            if correlated_event is None:
                continue
            duration_us = correlated_event.get("dur")
            if duration_us is None:
                continue
            cuda_time_us += float(duration_us)

        cuda_time_ms = cuda_time_us / 1000.0
        if cuda_time_ms <= 0.0 and op_name not in allow_zero_cuda_ops:
            raise RuntimeError(
                f"Non-positive CUDA time for op {op_name}: {cuda_time_us}us")
        scope_seq = scope_seq_by_op.get(op_name, 0)
        scope_seq_by_op[op_name] = scope_seq + 1
        scope_rows.append({
            "op_name": op_name,
            "cuda_time_ms": cuda_time_ms,
            "count": 1,
            "scope_seq": scope_seq,
        })

    return scope_rows


class FrontierCudaEventOpLogger:
    """Collect per-op timing for Frontier comparison."""

    def __init__(self,
                 log_path: str,
                 scopes: Optional[Sequence[str]] = None,
                 meta_enabled: bool = False,
                 scope_mode: str = "default",
                 timing_mode: str = "record_function",
                 aggregation_mode: str = "per_scope",
                 allow_empty_pure_decode_batch: bool = False) -> None:
        if not log_path:
            raise ValueError("Frontier CUDA event log path is required.")
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self._log_path = log_path
        try:
            self._log_file = open(log_path, "a", encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(
                f"Failed to open Frontier CUDA event log file: {log_path}"
            ) from exc
        self._scopes = set(scopes or DEFAULT_FRONTIER_CUDA_EVENT_OP_SCOPES)
        if not self._scopes:
            raise ValueError("Frontier CUDA event scope list is empty.")
        timing_mode = timing_mode.strip().lower()
        if timing_mode not in {"cuda_event", "record_function"}:
            raise ValueError(
                "VLLM_FRONTIER_OP_TIMING_MODE must be one of "
                "['cuda_event', 'record_function']."
            )
        self._timing_mode = timing_mode
        aggregation_mode = aggregation_mode.strip().lower()
        if aggregation_mode not in {"per_scope", "batch_sum"}:
            raise ValueError(
                "VLLM_FRONTIER_OP_AGG_MODE must be one of "
                "['per_scope', 'batch_sum']."
            )
        self._aggregation_mode = aggregation_mode
        self._pending_events: list[tuple[str, int, torch.cuda.Event,
                                         torch.cuda.Event]] = []
        self._pending_meta: dict[object, dict[str, Any]] = {}
        self._profiler: Optional[torch.profiler.profile] = None
        self._meta_enabled = meta_enabled
        scope_mode = scope_mode.strip().lower()
        if scope_mode not in {"default", "kernel_only"}:
            raise ValueError(
                "VLLM_FRONTIER_CUDA_EVENT_SCOPE_MODE must be one of "
                "['default', 'kernel_only']."
            )
        self._scope_mode = scope_mode
        self._allow_empty_pure_decode_batch = allow_empty_pure_decode_batch
        self._batch_active = False
        self._batch_meta: dict[str, Any] = {}
        self._scope_seq_by_op: dict[str, int] = {}
        self._active_scope_keys: list[tuple[str, int]] = []

    @contextlib.contextmanager
    def activate(self) -> Iterator[None]:
        token = _FRONTIER_CUDA_EVENT_OP_LOGGER.set(self)
        try:
            yield
        finally:
            _FRONTIER_CUDA_EVENT_OP_LOGGER.reset(token)

    def should_record(self, op_name: str) -> bool:
        return op_name in self._scopes

    def _requires_kernel_only_post_sync(self, op_name: str) -> bool:
        return (
            op_name.endswith("allreduce")
            or "alltoall" in op_name
        )

    def start_batch(
        self,
        batch_id: int,
        batch_size: int,
        batch_num_tokens: int,
        batch_num_prefill_tokens: int,
        batch_num_decode_tokens: int,
        batch_request_num_tokens: Optional[list[int]] = None,
        pp_rank: Optional[int] = None,
    ) -> None:
        if self._batch_active:
            raise RuntimeError("Frontier CUDA event batch already active.")
        self._batch_active = True
        self._batch_meta = {
            "batch_id": batch_id,
            "batch_size": batch_size,
            "batch_num_tokens": batch_num_tokens,
            "batch_num_prefill_tokens": batch_num_prefill_tokens,
            "batch_num_decode_tokens": batch_num_decode_tokens,
            "timing_mode": self._timing_mode,
            "scope_mode": self._scope_mode,
            "aggregation_mode": (
                "per_scope"
                if self._aggregation_mode == "per_scope"
                else "batch_sum_legacy"
            ),
        }
        if batch_request_num_tokens is not None:
            self._batch_meta["batch_request_num_tokens"] = batch_request_num_tokens
        if pp_rank is not None:
            self._batch_meta["pp_rank"] = int(pp_rank)
        self._pending_events = []
        self._pending_meta = {}
        self._scope_seq_by_op = {}
        self._active_scope_keys = []
        if self._timing_mode == "record_function":
            self._profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ])
            self._profiler.__enter__()

    @contextlib.contextmanager
    def scope(self, op_name: str) -> Iterator[None]:
        if not self._batch_active:
            raise RuntimeError(
                "Frontier CUDA event logger used before start_batch().")
        if op_name not in self._scopes:
            yield
            return
        scope_seq = self._scope_seq_by_op.get(op_name, 0)
        self._scope_seq_by_op[op_name] = scope_seq + 1
        scope_key = (op_name, scope_seq)
        if self._timing_mode == "record_function":
            self._active_scope_keys.append(scope_key)
            try:
                with record_function(f"frontier_{op_name}"):
                    yield
            finally:
                popped_key = self._active_scope_keys.pop()
                if popped_key != scope_key:
                    raise RuntimeError(
                        "Frontier CUDA event logger scope stack mismatch."
                    )
            return

        if self._scope_mode == "kernel_only":
            # Synchronize before timing to remove queued work from the scope.
            torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        self._active_scope_keys.append(scope_key)
        try:
            yield
        finally:
            if (
                self._scope_mode == "kernel_only"
                and self._requires_kernel_only_post_sync(op_name)
            ):
                torch.cuda.synchronize()
            end_event.record()
            self._pending_events.append((op_name, scope_seq, start_event, end_event))
            popped_key = self._active_scope_keys.pop()
            if popped_key != scope_key:
                raise RuntimeError(
                    "Frontier CUDA event logger scope stack mismatch."
                )

    def record_meta(self, op_name: str, meta: dict[str, Any]) -> None:
        if not self._meta_enabled:
            return
        if not self._batch_active:
            raise RuntimeError(
                "Frontier CUDA event logger meta recorded before start_batch()."
            )
        if op_name not in self._scopes:
            return
        meta_key: object = op_name
        if self._aggregation_mode == "per_scope":
            if not self._active_scope_keys:
                raise RuntimeError(
                    f"Frontier CUDA event logger meta recorded outside active scope for {op_name}."
                )
            active_op_name, scope_seq = self._active_scope_keys[-1]
            if active_op_name != op_name:
                raise RuntimeError(
                    "Frontier CUDA event logger meta op does not match active scope: "
                    f"{op_name} vs {active_op_name}"
                )
            meta_key = (op_name, scope_seq)
        existing = self._pending_meta.get(meta_key)
        if existing is None:
            self._pending_meta[meta_key] = meta
            return
        if existing != meta:
            raise RuntimeError(
                f"Inconsistent meta for op {op_name}: {existing} vs {meta}"
            )

    def finish_batch(self) -> None:
        if not self._batch_active:
            raise RuntimeError(
                "Frontier CUDA event logger finish_batch without start_batch.")
        records: list[dict[str, Any]] = []
        allow_empty_batch = (
            self._allow_empty_pure_decode_batch
            and self._batch_meta.get("batch_num_prefill_tokens", 0) == 0
            and self._batch_meta.get("batch_num_decode_tokens", 0) > 0)
        if self._timing_mode == "record_function":
            if self._profiler is None:
                raise RuntimeError(
                    "Frontier record_function profiler is not initialized.")
            self._profiler.__exit__(None, None, None)
            torch.cuda.synchronize()
            trace_path = _frontier_build_record_function_trace_path(
                self._log_path, int(self._batch_meta.get("batch_id", -1)))
            self._profiler.export_chrome_trace(trace_path)
            allow_zero_cuda_ops: set[str] = set()
            if (self._batch_meta.get("batch_num_prefill_tokens", 0) == 0
                    and self._batch_meta.get("batch_num_decode_tokens", 0) > 0):
                allow_zero_cuda_ops.add("attn_kv_cache_save")
            if self._aggregation_mode == "per_scope":
                records = _frontier_collect_record_function_scopes(
                    trace_path,
                    self._scopes,
                    allow_zero_cuda_ops=allow_zero_cuda_ops,
                )
            else:
                aggregated = _frontier_aggregate_record_function_trace(
                    trace_path,
                    self._scopes,
                    allow_zero_cuda_ops=allow_zero_cuda_ops,
                )
                records = [
                    {
                        "op_name": op_name,
                        "cuda_time_ms": stats["cuda_time_ms"],
                        "count": int(stats["count"]),
                    }
                    for op_name, stats in aggregated.items()
                ]
            self._profiler = None
        else:
            if not self._pending_events:
                if allow_empty_batch:
                    torch.cuda.synchronize()
                    self._batch_active = False
                    self._pending_events = []
                    self._pending_meta = {}
                    self._scope_seq_by_op = {}
                    self._active_scope_keys = []
                    return
                raise RuntimeError(
                    "Frontier CUDA event logger captured no operations.")
            torch.cuda.synchronize()
            if self._aggregation_mode == "per_scope":
                for op_name, scope_seq, start_event, end_event in self._pending_events:
                    duration_ms = start_event.elapsed_time(end_event)
                    if duration_ms <= 0:
                        raise RuntimeError(
                            f"Non-positive CUDA time for op {op_name}: {duration_ms}")
                    records.append({
                        "op_name": op_name,
                        "cuda_time_ms": float(duration_ms),
                        "count": 1,
                        "scope_seq": scope_seq,
                    })
            else:
                aggregated: dict[str, dict[str, float]] = {}
                for op_name, _scope_seq, start_event, end_event in self._pending_events:
                    duration_ms = start_event.elapsed_time(end_event)
                    if duration_ms <= 0:
                        raise RuntimeError(
                            f"Non-positive CUDA time for op {op_name}: {duration_ms}")
                    stats = aggregated.setdefault(op_name, {
                        "cuda_time_ms": 0.0,
                        "count": 0.0
                    })
                    stats["cuda_time_ms"] += float(duration_ms)
                    stats["count"] += 1.0
                records = [
                    {
                        "op_name": op_name,
                        "cuda_time_ms": stats["cuda_time_ms"],
                        "count": int(stats["count"]),
                    }
                    for op_name, stats in aggregated.items()
                ]

        if not records:
            if allow_empty_batch:
                torch.cuda.synchronize()
                self._batch_active = False
                self._pending_events = []
                self._pending_meta = {}
                self._scope_seq_by_op = {}
                self._active_scope_keys = []
                return
            raise RuntimeError(
                "Frontier CUDA event logger captured no operations.")
        timestamp = time.time()
        for pending_record in records:
            op_name = str(pending_record["op_name"])
            scope_seq = pending_record.get("scope_seq")
            meta_key: object = op_name
            if self._aggregation_mode == "per_scope":
                if scope_seq is None:
                    raise RuntimeError(
                        f"Missing scope_seq for per-scope record of op {op_name}."
                    )
                meta_key = (op_name, int(scope_seq))
            meta = self._pending_meta.get(meta_key)
            if self._meta_enabled and meta is None:
                raise RuntimeError(
                    f"Missing runtime meta for op {op_name} in batch.")
            record = {
                **self._batch_meta,
                "op_name": op_name,
                "cuda_time_ms": float(pending_record["cuda_time_ms"]),
                "count": int(pending_record["count"]),
                "timestamp": timestamp,
            }
            if scope_seq is not None:
                record["scope_seq"] = int(scope_seq)
            if meta is not None:
                record["meta"] = meta
            self._log_file.write(json.dumps(record) + "\n")
        self._log_file.flush()
        self._batch_active = False
        self._pending_events = []
        self._pending_meta = {}
        self._scope_seq_by_op = {}
        self._active_scope_keys = []


class FrontierMoeRoutingLogger:
    """Collect per-expert token distribution for Frontier MoE routing analysis."""

    def __init__(self, log_path: str) -> None:
        if not log_path:
            raise ValueError("Frontier MoE routing log path is required.")
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        try:
            self._log_file = open(log_path, "a", encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(
                f"Failed to open Frontier MoE routing log file: {log_path}"
            ) from exc
        self._batch_active = False
        self._batch_meta: dict[str, int] = {}

    @contextlib.contextmanager
    def activate(self) -> Iterator[None]:
        token = _FRONTIER_MOE_ROUTING_LOGGER.set(self)
        try:
            yield
        finally:
            _FRONTIER_MOE_ROUTING_LOGGER.reset(token)

    def start_batch(
        self,
        batch_id: int,
        batch_size: int,
        batch_num_tokens: int,
        batch_num_prefill_tokens: int,
        batch_num_decode_tokens: int,
    ) -> None:
        if self._batch_active:
            raise RuntimeError("Frontier MoE routing batch already active.")
        self._batch_active = True
        self._batch_meta = {
            "batch_id": batch_id,
            "batch_size": batch_size,
            "batch_num_tokens": batch_num_tokens,
            "batch_num_prefill_tokens": batch_num_prefill_tokens,
            "batch_num_decode_tokens": batch_num_decode_tokens,
        }

    def finish_batch(self) -> None:
        if not self._batch_active:
            raise RuntimeError(
                "Frontier MoE routing logger finish_batch without start_batch."
            )
        self._batch_active = False
        self._batch_meta = {}
        self._log_file.flush()

    def log_routing(
        self,
        layer_name: str,
        topk_ids: torch.Tensor,
        num_tokens: int,
        router_topk: int,
        global_num_experts: int,
        local_num_experts: int,
        ep_rank: int,
        ep_size: int,
        expert_map: Optional[torch.Tensor],
    ) -> None:
        if not self._batch_active:
            raise RuntimeError(
                "Frontier MoE routing logger used before start_batch()."
            )
        if num_tokens <= 0:
            raise RuntimeError(
                f"Invalid num_tokens for MoE routing: {num_tokens}"
            )
        if router_topk <= 0:
            raise RuntimeError(
                f"Invalid router_topk for MoE routing: {router_topk}"
            )
        if global_num_experts <= 0:
            raise RuntimeError(
                f"Invalid global_num_experts for MoE routing: {global_num_experts}"
            )

        topk_ids_cpu = topk_ids.detach().to("cpu")
        flat_ids = topk_ids_cpu.reshape(-1).to(torch.int64)
        if flat_ids.numel() == 0:
            raise RuntimeError("MoE routing topk_ids is empty.")
        min_id = int(flat_ids.min().item())
        max_id = int(flat_ids.max().item())
        if min_id < 0 or max_id >= global_num_experts:
            raise RuntimeError(
                f"MoE routing expert id out of range: min={min_id}, max={max_id}, "
                f"global_num_experts={global_num_experts}"
            )

        expected_total_routed = num_tokens * router_topk
        num_experts_per_device = local_num_experts or global_num_experts
        if num_experts_per_device <= 0:
            raise RuntimeError(
                f"Invalid num_experts_per_device: {num_experts_per_device}"
            )

        if expert_map is None:
            counts_tensor = torch.bincount(
                flat_ids, minlength=global_num_experts
            )
            total_routed_tokens = int(counts_tensor.sum().item())
            if total_routed_tokens != expected_total_routed:
                raise RuntimeError(
                    f"MoE routing token mismatch: total_routed_tokens={total_routed_tokens}, "
                    f"expected={expected_total_routed}"
                )
            counts = counts_tensor.tolist()
        else:
            expert_map_cpu = expert_map.detach().to("cpu").to(torch.int64)
            if expert_map_cpu.numel() < global_num_experts:
                raise RuntimeError(
                    "MoE routing expert_map length does not cover all experts."
                )
            local_ids = expert_map_cpu[flat_ids]
            local_ids = local_ids[local_ids >= 0]
            expected_total_routed = int(local_ids.numel())
            counts_tensor = torch.bincount(
                local_ids, minlength=num_experts_per_device
            )
            total_routed_tokens = int(counts_tensor.sum().item())
            if total_routed_tokens != expected_total_routed:
                raise RuntimeError(
                    f"MoE routing local token mismatch: total_routed_tokens={total_routed_tokens}, "
                    f"expected_local={expected_total_routed}"
                )
            counts = counts_tensor.tolist()

        if len(counts) != num_experts_per_device:
            raise RuntimeError(
                f"MoE routing counts length mismatch: "
                f"{len(counts)} vs {num_experts_per_device}"
            )
        if total_routed_tokens < 0:
            raise RuntimeError(
                f"MoE routing total_routed_tokens invalid: {total_routed_tokens}"
            )

        if total_routed_tokens == 0:
            mean = 0.0
            variance = 0.0
            std = 0.0
            load_imbalance_cv = 0.0
            min_load_ratio = 0.0
            max_load_ratio = 0.0
            expert_utilization = 0.0
            load_entropy = 0.0
            gini = 0.0
        else:
            mean = total_routed_tokens / num_experts_per_device
            variance = sum((c - mean) ** 2 for c in counts) / num_experts_per_device
            std = math.sqrt(variance)
            load_imbalance_cv = std / mean if mean > 0 else 0.0
            min_load_ratio = min(counts) / mean if mean > 0 else 0.0
            max_load_ratio = max(counts) / mean if mean > 0 else 0.0
            expert_utilization = sum(1 for c in counts if c > 0) / num_experts_per_device
            probs = [c / total_routed_tokens for c in counts if c > 0]
            load_entropy = -sum(p * math.log2(p) for p in probs) if probs else 0.0
            sorted_counts = sorted(counts)
            gini = (
                (2 * sum((i + 1) * x for i, x in enumerate(sorted_counts)))
                / (num_experts_per_device * total_routed_tokens)
                - (num_experts_per_device + 1) / num_experts_per_device
            )

        record = {
            **self._batch_meta,
            "layer_name": layer_name,
            "num_tokens": num_tokens,
            "router_topk": router_topk,
            "global_num_experts": global_num_experts,
            "num_experts_per_device": num_experts_per_device,
            "ep_rank": ep_rank,
            "ep_size": ep_size,
            "total_routed_tokens": total_routed_tokens,
            "expected_total_routed_tokens": expected_total_routed,
            "tokens_per_expert_avg": mean,
            "tokens_to_experts_ratio": mean,
            "expert_utilization": expert_utilization,
            "min_load_ratio": min_load_ratio,
            "load_imbalance_cv": load_imbalance_cv,
            "max_load_ratio": max_load_ratio,
            "load_entropy": load_entropy,
            "load_gini_coefficient": gini,
            "load_distribution": "runtime",
            "per_expert_tokens": {
                str(i): int(c) for i, c in enumerate(counts) if c > 0
            },
            "timestamp": time.time(),
        }
        self._log_file.write(json.dumps(record) + "\n")
        self._log_file.flush()


@dataclass(frozen=True)
class FrontierMoeRoutingContext:
    layer_name: str
    num_tokens: int
    router_topk: int
    global_num_experts: int
    local_num_experts: int
    ep_rank: int
    ep_size: int
    expert_map: Optional[torch.Tensor]


@contextlib.contextmanager
def frontier_moe_routing_context(
    *,
    layer_name: str,
    num_tokens: int,
    router_topk: int,
    global_num_experts: int,
    local_num_experts: int,
    ep_rank: int,
    ep_size: int,
    expert_map: Optional[torch.Tensor],
) -> Iterator[None]:
    if not _frontier_contextvars_enabled():
        yield
        return

    logger = _FRONTIER_MOE_ROUTING_LOGGER.get()
    if logger is None:
        yield
        return

    if not layer_name:
        raise RuntimeError("Frontier MoE routing layer_name is required.")
    if num_tokens <= 0:
        raise RuntimeError(
            f"Invalid num_tokens for MoE routing context: {num_tokens}"
        )
    if router_topk <= 0:
        raise RuntimeError(
            f"Invalid router_topk for MoE routing context: {router_topk}"
        )
    if global_num_experts <= 0:
        raise RuntimeError(
            f"Invalid global_num_experts for MoE routing context: {global_num_experts}"
        )
    if local_num_experts <= 0:
        raise RuntimeError(
            f"Invalid local_num_experts for MoE routing context: {local_num_experts}"
        )
    if ep_size <= 0:
        raise RuntimeError(
            f"Invalid ep_size for MoE routing context: {ep_size}"
        )
    if ep_rank < 0 or ep_rank >= ep_size:
        raise RuntimeError(
            f"Invalid ep_rank for MoE routing context: {ep_rank}"
        )

    context = FrontierMoeRoutingContext(
        layer_name=layer_name,
        num_tokens=int(num_tokens),
        router_topk=router_topk,
        global_num_experts=global_num_experts,
        local_num_experts=local_num_experts,
        ep_rank=ep_rank,
        ep_size=ep_size,
        expert_map=expert_map,
    )
    token = _FRONTIER_MOE_ROUTING_CONTEXT.set(context)
    try:
        yield
    finally:
        _FRONTIER_MOE_ROUTING_CONTEXT.reset(token)


def log_frontier_moe_routing_from_context(
    topk_ids: torch.Tensor,
) -> None:
    if not _frontier_contextvars_enabled():
        return

    logger = _FRONTIER_MOE_ROUTING_LOGGER.get()
    if logger is None:
        return
    context = _FRONTIER_MOE_ROUTING_CONTEXT.get()
    if context is None:
        raise RuntimeError(
            "Frontier MoE routing context missing for active logger."
        )
    logger.log_routing(
        layer_name=context.layer_name,
        topk_ids=topk_ids,
        num_tokens=context.num_tokens,
        router_topk=context.router_topk,
        global_num_experts=context.global_num_experts,
        local_num_experts=context.local_num_experts,
        ep_rank=context.ep_rank,
        ep_size=context.ep_size,
        expert_map=context.expert_map,
    )

class ConstantList(Generic[T], Sequence):

    def __init__(self, x: list[T]) -> None:
        self._x = x

    def append(self, item):
        raise TypeError("Cannot append to a constant list")

    def extend(self, item):
        raise TypeError("Cannot extend a constant list")

    def insert(self, item):
        raise TypeError("Cannot insert into a constant list")

    def pop(self, item):
        raise TypeError("Cannot pop from a constant list")

    def remove(self, item):
        raise TypeError("Cannot remove from a constant list")

    def clear(self):
        raise TypeError("Cannot clear a constant list")

    def index(self,
              item: T,
              start: int = 0,
              stop: Optional[int] = None) -> int:
        return self._x.index(item, start,
                             stop if stop is not None else len(self._x))

    @overload
    def __getitem__(self, item: int) -> T:
        ...

    @overload
    def __getitem__(self, s: slice, /) -> list[T]:
        ...

    def __getitem__(self, item: Union[int, slice]) -> Union[T, list[T]]:
        return self._x[item]

    @overload
    def __setitem__(self, item: int, value: T):
        ...

    @overload
    def __setitem__(self, s: slice, value: T, /):
        ...

    def __setitem__(self, item: Union[int, slice], value: Union[T, list[T]]):
        raise TypeError("Cannot set item in a constant list")

    def __delitem__(self, item):
        raise TypeError("Cannot delete item from a constant list")

    def __iter__(self):
        return iter(self._x)

    def __contains__(self, item):
        return item in self._x

    def __len__(self):
        return len(self._x)

    def __repr__(self):
        return f"ConstantList({self._x})"


class CpuGpuBuffer:
    """Buffer to easily copy tensors between CPU and GPU."""

    def __init__(
        self,
        *size: Union[int, torch.SymInt],
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool,
        with_numpy: bool = True,
    ) -> None:
        self.cpu = torch.zeros(*size,
                               dtype=dtype,
                               device="cpu",
                               pin_memory=pin_memory)
        self.gpu = self.cpu.to(device)
        self.np: np.ndarray
        # To keep type hints simple (avoiding generics and subclasses), we
        # only conditionally create the numpy array attribute. This can cause
        # AttributeError if `self.np` is accessed when `with_numpy=False`.
        if with_numpy:
            if dtype == torch.bfloat16:
                raise ValueError(
                    "Bfloat16 torch tensors cannot be directly cast to a "
                    "numpy array, so call CpuGpuBuffer with with_numpy=False")
            self.np = self.cpu.numpy()

    def copy_to_gpu(self, n: Optional[int] = None) -> torch.Tensor:
        if n is None:
            return self.gpu.copy_(self.cpu, non_blocking=True)
        return self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)

    def copy_to_cpu(self, n: Optional[int] = None) -> torch.Tensor:
        """NOTE: Because this method is non-blocking, explicit synchronization
        is needed to ensure the data is copied to CPU."""
        if n is None:
            return self.cpu.copy_(self.gpu, non_blocking=True)
        return self.cpu[:n].copy_(self.gpu[:n], non_blocking=True)


def get_engine_client_zmq_addr(local_only: bool,
                               host: str,
                               port: int = 0) -> str:
    """Assign a new ZMQ socket address.

    If local_only is True, participants are colocated and so a unique IPC
    address will be returned.

    Otherwise, the provided host and port will be used to construct a TCP
    address (port == 0 means assign an available port)."""

    return get_open_zmq_ipc_path() if local_only else (get_tcp_uri(
        host, port or get_open_port()))


class APIServerProcessManager:
    """Manages a group of API server processes.

    Handles creation, monitoring, and termination of API server worker
    processes. Also monitors extra processes to check if they are healthy.
    """

    def __init__(
        self,
        target_server_fn: Callable,
        listen_address: str,
        sock: Any,
        args: argparse.Namespace,
        num_servers: int,
        input_addresses: list[str],
        output_addresses: list[str],
        stats_update_address: Optional[str] = None,
    ):
        """Initialize and start API server worker processes.

        Args:
            target_server_fn: Function to call for each API server process
            listen_address: Address to listen for client connections
            sock: Socket for client connections
            args: Command line arguments
            num_servers: Number of API server processes to start
            input_addresses: Input addresses for each API server
            output_addresses: Output addresses for each API server
            stats_update_address: Optional stats update address
        """
        self.listen_address = listen_address
        self.sock = sock
        self.args = args

        # Start API servers
        spawn_context = multiprocessing.get_context("spawn")
        self.processes: list[BaseProcess] = []

        for i, in_addr, out_addr in zip(range(num_servers), input_addresses,
                                        output_addresses):
            client_config = {
                "input_address": in_addr,
                "output_address": out_addr,
                "client_count": num_servers,
                "client_index": i
            }
            if stats_update_address is not None:
                client_config["stats_update_address"] = stats_update_address

            proc = spawn_context.Process(target=target_server_fn,
                                         name=f"ApiServer_{i}",
                                         args=(listen_address, sock, args,
                                               client_config))
            self.processes.append(proc)
            proc.start()

        logger.info("Started %d API server processes", len(self.processes))

        # Shutdown only the API server processes on garbage collection
        # The extra processes are managed by their owners
        self._finalizer = weakref.finalize(self, shutdown, self.processes)

    def close(self) -> None:
        self._finalizer()


def wait_for_completion_or_failure(
        api_server_manager: APIServerProcessManager,
        engine_manager: Optional[Union["CoreEngineProcManager",
                                       "CoreEngineActorManager"]] = None,
        coordinator: Optional["DPCoordinator"] = None) -> None:
    """Wait for all processes to complete or detect if any fail.

    Raises an exception if any process exits with a non-zero status.

    Args:
        api_server_manager: The manager for API servers.
        engine_manager: The manager for engine processes.
            If CoreEngineProcManager, it manages local engines;
            if CoreEngineActorManager, it manages all engines.
        coordinator: The coordinator for data parallel.
    """

    from vllm.v1.engine.utils import (CoreEngineActorManager,
                                      CoreEngineProcManager)

    try:
        logger.info("Waiting for API servers to complete ...")
        # Create a mapping of sentinels to their corresponding processes
        # for efficient lookup
        sentinel_to_proc: dict[Any, BaseProcess] = {
            proc.sentinel: proc
            for proc in api_server_manager.processes
        }

        if coordinator:
            sentinel_to_proc[coordinator.proc.sentinel] = coordinator.proc

        actor_run_refs = []
        if isinstance(engine_manager, CoreEngineProcManager):
            for proc in engine_manager.processes:
                sentinel_to_proc[proc.sentinel] = proc
        elif isinstance(engine_manager, CoreEngineActorManager):
            actor_run_refs = engine_manager.get_run_refs()

        # Check if any process terminates
        while sentinel_to_proc or actor_run_refs:
            # Wait for any process to terminate
            ready_sentinels: list[Any] = connection.wait(sentinel_to_proc,
                                                         timeout=5)

            # Process any terminated processes
            for sentinel in ready_sentinels:
                proc = sentinel_to_proc.pop(sentinel)

                # Check if process exited with error
                if proc.exitcode != 0:
                    raise RuntimeError(
                        f"Process {proc.name} (PID: {proc.pid}) "
                        f"died with exit code {proc.exitcode}")

            if actor_run_refs:
                import ray
                _, actor_run_refs = ray.wait(actor_run_refs, timeout=5)

    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down API servers...")
    except Exception as e:
        logger.exception("Exception occurred while running API servers: %s",
                         str(e))
        raise
    finally:
        logger.info("Terminating remaining processes ...")
        api_server_manager.close()
        if coordinator:
            coordinator.close()
        if engine_manager:
            engine_manager.close()


# Note(rob): shutdown function cannot be a bound method,
# else the gc cannot collect the object.
def shutdown(procs: list[BaseProcess]):
    # Shutdown the process.
    for proc in procs:
        if proc.is_alive():
            proc.terminate()

    # Allow 5 seconds for remaining procs to terminate.
    deadline = time.monotonic() + 5
    for proc in procs:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if proc.is_alive():
            proc.join(remaining)

    for proc in procs:
        if proc.is_alive() and (pid := proc.pid) is not None:
            kill_process_tree(pid)


def copy_slice(from_tensor: torch.Tensor, to_tensor: torch.Tensor,
               length: int) -> torch.Tensor:
    """
    Copy the first length elements of a tensor into another tensor in a
    non-blocking manner.

    Used to copy pinned CPU tensor data to pre-allocated GPU tensors.

    Returns the sliced target tensor.
    """
    return to_tensor[:length].copy_(from_tensor[:length], non_blocking=True)


def report_usage_stats(
        vllm_config,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT) -> None:
    """Report usage statistics if enabled."""

    if not is_usage_stats_enabled():
        return

    from vllm.model_executor.model_loader import get_architecture_class_name

    usage_message.report_usage(
        get_architecture_class_name(vllm_config.model_config),
        usage_context,
        extra_kvs={
            # Common configuration
            "dtype":
            str(vllm_config.model_config.dtype),
            "tensor_parallel_size":
            vllm_config.parallel_config.tensor_parallel_size,
            "block_size":
            vllm_config.cache_config.block_size,
            "gpu_memory_utilization":
            vllm_config.cache_config.gpu_memory_utilization,
            "kv_cache_memory_bytes":
            vllm_config.cache_config.kv_cache_memory_bytes,
            # Quantization
            "quantization":
            vllm_config.model_config.quantization,
            "kv_cache_dtype":
            str(vllm_config.cache_config.cache_dtype),

            # Feature flags
            "enable_lora":
            bool(vllm_config.lora_config),
            "enable_prefix_caching":
            vllm_config.cache_config.enable_prefix_caching,
            "enforce_eager":
            vllm_config.model_config.enforce_eager,
            "disable_custom_all_reduce":
            vllm_config.parallel_config.disable_custom_all_reduce,
        })


def record_function_or_nullcontext(name: str) -> AbstractContextManager:
    profiler_ctx: AbstractContextManager
    if envs.VLLM_CUSTOM_SCOPES_FOR_PROFILING:
        profiler_ctx = record_function(name)
    else:
        profiler_ctx = contextlib.nullcontext()

    if not _frontier_contextvars_enabled():
        return profiler_ctx

    op_logger = _FRONTIER_CUDA_EVENT_OP_LOGGER.get()
    if (op_logger is None or not op_logger.should_record(name)
            or not frontier_trace.is_active()):
        return profiler_ctx

    @contextlib.contextmanager
    def _combined_context() -> Iterator[None]:
        with profiler_ctx:
            with op_logger.scope(name):
                yield

    return _combined_context()


def record_frontier_op_meta(op_name: str, meta: dict[str, Any]) -> None:
    if not _frontier_contextvars_enabled() or not frontier_trace.is_active():
        return
    op_logger = _FRONTIER_CUDA_EVENT_OP_LOGGER.get()
    if op_logger is None or not op_logger.should_record(op_name):
        return
    op_logger.record_meta(op_name, meta)


def should_record_frontier_op_meta(op_name: str) -> bool:
    if not _frontier_contextvars_enabled() or not frontier_trace.is_active():
        return False
    op_logger = _FRONTIER_CUDA_EVENT_OP_LOGGER.get()
    return op_logger is not None and op_logger.should_record(op_name)


def set_frontier_positions_meta(meta: dict[str, Any]) -> None:
    if not _frontier_contextvars_enabled() or not frontier_trace.is_active():
        return
    _FRONTIER_RUNTIME_POSITIONS_META.set(meta)


def get_frontier_positions_meta() -> Optional[dict[str, Any]]:
    if not _frontier_contextvars_enabled():
        return None
    return _FRONTIER_RUNTIME_POSITIONS_META.get()


def log_frontier_moe_routing(
    *,
    layer_name: str,
    topk_ids: torch.Tensor,
    num_tokens: int,
    router_topk: int,
    global_num_experts: int,
    local_num_experts: int,
    ep_rank: int,
    ep_size: int,
    expert_map: Optional[torch.Tensor],
) -> None:
    if not _frontier_contextvars_enabled() or not frontier_trace.is_active():
        return
    logger = _FRONTIER_MOE_ROUTING_LOGGER.get()
    if logger is None:
        return
    logger.log_routing(
        layer_name=layer_name,
        topk_ids=topk_ids,
        num_tokens=num_tokens,
        router_topk=router_topk,
        global_num_experts=global_num_experts,
        local_num_experts=local_num_experts,
        ep_rank=ep_rank,
        ep_size=ep_size,
        expert_map=expert_map,
    )

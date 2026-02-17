# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from vllm import LLM, SamplingParams
from vllm.request_generator import (FixedLengthConfig, RequestGeneratorConfig,
                                    VLLMRequestGenerator)
from vllm.v1.engine.output_processor import RequestState


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: {value!r}. "
        "Expected one of: true/false, 1/0, yes/no, on/off.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure per-request TTFT for offline vLLM V1 runs.")
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        help="Model path or HF model id.")
    parser.add_argument("--num-requests",
                        type=int,
                        default=256,
                        help="Number of requests to issue.")
    parser.add_argument("--prefill-tokens",
                        type=int,
                        default=2048,
                        help="Prefill tokens per request.")
    parser.add_argument("--decode-tokens",
                        type=int,
                        default=2,
                        help="Decode tokens per request.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for request generation.")
    parser.add_argument("--warmup-iters",
                        type=int,
                        default=0,
                        help="Warmup iterations before measurement.")
    parser.add_argument("--gpu-memory-utilization",
                        type=float,
                        default=0.9,
                        help="GPU memory utilization.")
    parser.add_argument("--tensor-parallel-size",
                        type=int,
                        default=8,
                        help="Tensor parallel size.")
    parser.add_argument("--pipeline-parallel-size",
                        type=int,
                        default=1,
                        help="Pipeline parallel size.")
    parser.add_argument("--data-parallel-size",
                        type=int,
                        default=2,
                        help="Data parallel size (align NUM_REPLICAS).")
    parser.add_argument("--enable-expert-parallel",
                        action="store_true",
                        help="Enable expert parallel.")
    parser.add_argument("--vocab-size",
                        type=int,
                        default=128815,
                        help="Vocabulary size for token-id prompt generation.")
    parser.add_argument("--load-format",
                        type=str,
                        default="dummy",
                        help="Model load format (dummy recommended).")
    parser.add_argument("--max-num-seqs",
                        type=int,
                        default=1023,
                        help="Maximum number of sequences in a batch.")
    parser.add_argument("--max-num-batched-tokens",
                        type=int,
                        default=16384,
                        help="Maximum number of batched tokens.")
    parser.add_argument("--max-model-len-buffer",
                        type=int,
                        default=16,
                        help="Additional buffer tokens added to prefill+decode length.")
    parser.add_argument("--enable-chunked-prefill",
                        type=_parse_bool,
                        default=False,
                        help="Explicit chunked prefill toggle.")
    parser.add_argument("--enforce-eager",
                        type=_parse_bool,
                        default=False,
                        help="Force eager execution (required by certain instrumentation).")
    parser.add_argument("--results-csv",
                        type=Path,
                        required=True,
                        help="Output CSV path for TTFT results.")
    parser.add_argument("--results-json",
                        type=Path,
                        default=None,
                        help="Optional JSON output path.")
    parser.add_argument("--request-metrics-csv",
                        type=Path,
                        default=None,
                        help="Optional CSV path for request-level metrics (ttft/tpot/e2e).")
    parser.add_argument("--request-metrics-json",
                        type=Path,
                        default=None,
                        help="Optional JSON path for request-level metrics (ttft/tpot/e2e).")
    parser.add_argument("--enable-profiler",
                        type=_parse_bool,
                        default=False,
                        help="Enable torch profiler around generate() only.")
    return parser.parse_args()


def _build_prompts(num_requests: int, prefill_tokens: int, decode_tokens: int,
                   seed: int, vocab_size: int) -> list[Any]:
    safe_max_token_id = max(vocab_size - 1000, 1001)
    cfg = RequestGeneratorConfig(
        num_requests=num_requests,
        length_config=FixedLengthConfig(prefill_tokens=prefill_tokens,
                                        decode_tokens=decode_tokens),
        seed=seed,
        use_token_ids=True,
        vocab_size=vocab_size,
        min_token_id=1000,
        max_token_id=safe_max_token_id,
    )
    generator = VLLMRequestGenerator(cfg, tokenizer=None)
    requests = generator.generate()
    return [req.prompt for req in requests]


def _to_sortable_request_id(request_id: str) -> tuple[int, str]:
    try:
        return (0, f"{int(request_id):016d}")
    except ValueError:
        return (1, request_id)


def _safe_interval_seconds(end_ts: float, start_ts: float) -> float:
    if end_ts <= 0.0 or start_ts <= 0.0:
        return 0.0
    return max(0.0, end_ts - start_ts)


def _build_request_metrics_payload(
    *,
    request_id: str,
    prompt_len: int,
    arrival_time_s: float,
    first_token_latency_s: float,
    scheduled_ts_s: float,
    first_token_ts_s: float,
    last_token_ts_s: float,
    num_decode_tokens: int,
    finished: bool,
) -> dict[str, Any]:
    ttft_ms = max(0.0, float(first_token_latency_s) * 1000.0)
    decode_time_s = _safe_interval_seconds(last_token_ts_s, first_token_ts_s)
    request_e2e_ms = ttft_ms + (decode_time_s * 1000.0)
    model_exec_time_ms = (
        _safe_interval_seconds(last_token_ts_s, scheduled_ts_s) * 1000.0
    )
    tpot_ms = 0.0
    if num_decode_tokens > 1:
        tpot_ms = (decode_time_s * 1000.0) / float(num_decode_tokens - 1)

    completion_time_s = 0.0
    if finished:
        completion_time_s = float(arrival_time_s) + (request_e2e_ms / 1000.0)

    return {
        "request_id": request_id,
        "request_e2e_time": request_e2e_ms,
        "ttft": ttft_ms,
        "tpot": tpot_ms,
        "request_model_execution_time": model_exec_time_ms,
        "request_num_prefill_tokens": int(prompt_len),
        "request_num_decode_tokens": int(num_decode_tokens),
        "arrival_time": float(arrival_time_s),
        "completion_time": completion_time_s,
    }


def _ensure_request_output_metrics_patch() -> None:
    # vLLM V1 does not always attach request-level metrics to RequestOutput.
    # Patch this once so benchmark scripts can consume per-request TTFT/TPOT.
    if getattr(RequestState, "_frontier_metrics_patch_applied", False):
        return

    original_new_request_output = RequestState._new_request_output

    def _patched_new_request_output(
        self: RequestState,
        request_id: str,
        outputs: Any,
        finished: bool,
        kv_transfer_params: Any = None,
    ) -> Any:
        request_output = original_new_request_output(
            self,
            request_id,
            outputs,
            finished,
            kv_transfer_params,
        )
        if getattr(request_output, "metrics", None) is None and self.stats is not None:
            metrics_payload = _build_request_metrics_payload(
                request_id=request_id,
                prompt_len=len(self.prompt_token_ids),
                arrival_time_s=float(self.stats.arrival_time),
                first_token_latency_s=float(self.stats.first_token_latency),
                scheduled_ts_s=float(self.stats.scheduled_ts),
                first_token_ts_s=float(self.stats.first_token_ts),
                last_token_ts_s=float(self.stats.last_token_ts),
                num_decode_tokens=int(self.stats.num_generation_tokens),
                finished=bool(finished),
            )
            request_output.metrics = SimpleNamespace(**metrics_payload)
        return request_output

    RequestState._new_request_output = _patched_new_request_output  # type: ignore[assignment]
    RequestState._frontier_metrics_patch_applied = True  # type: ignore[attr-defined]


def _require_metric_field(metrics: Any, field: str) -> float:
    if metrics is None:
        raise RuntimeError("RequestOutput.metrics is None; TTFT cannot be computed.")
    if not hasattr(metrics, field):
        raise RuntimeError(
            f"RequestOutput.metrics missing field '{field}'. "
            "Ensure disable_log_stats=False and V1 metric export is enabled.")
    value = getattr(metrics, field)
    return float(value)


def main() -> None:
    args = _parse_args()
    _ensure_request_output_metrics_patch()

    if args.enable_profiler and not os.getenv("VLLM_TORCH_PROFILER_DIR"):
        raise RuntimeError(
            "--enable-profiler=True requires VLLM_TORCH_PROFILER_DIR.")

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.95,
        max_tokens=args.decode_tokens,
        ignore_eos=True,
        detokenize=False,
    )

    required_max_model_len = (args.prefill_tokens + args.decode_tokens
                              + args.max_model_len_buffer)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        skip_tokenizer_init=True,
        pipeline_parallel_size=args.pipeline_parallel_size,
        data_parallel_size=args.data_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=required_max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        load_format=args.load_format,
        enable_chunked_prefill=args.enable_chunked_prefill,
        disable_log_stats=False,
    )

    try:
        max_model_len = llm.llm_engine.model_config.max_model_len  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise RuntimeError(
            "Failed to read max_model_len from the vLLM engine.") from exc
    if max_model_len < required_max_model_len:
        raise RuntimeError(
            "max_model_len override did not take effect. "
            f"required={required_max_model_len}, engine={max_model_len}.")

    warmup_prompts = _build_prompts(args.num_requests, args.prefill_tokens,
                                    args.decode_tokens, args.seed + 1,
                                    args.vocab_size)
    prompts = _build_prompts(args.num_requests, args.prefill_tokens,
                             args.decode_tokens, args.seed, args.vocab_size)

    for _ in range(args.warmup_iters):
        llm.generate(warmup_prompts, sampling_params)

    profiler_started = False
    if args.enable_profiler:
        llm.start_profile()
        profiler_started = True

    try:
        outputs = llm.generate(prompts, sampling_params)
        torch.cuda.synchronize()
    finally:
        if profiler_started:
            llm.stop_profile()

    records: list[dict[str, Any]] = []
    for output in outputs:
        metrics = output.metrics
        ttft_ms = _require_metric_field(metrics, "ttft")
        tpot_ms = _require_metric_field(metrics, "tpot")
        request_e2e_ms = _require_metric_field(metrics, "request_e2e_time")
        arrival_time = _require_metric_field(metrics, "arrival_time")
        completion_time = _require_metric_field(metrics, "completion_time")
        num_prefill_tokens = int(_require_metric_field(metrics, "request_num_prefill_tokens"))
        num_decode_tokens = int(_require_metric_field(metrics, "request_num_decode_tokens"))
        prefill_complete_time = arrival_time + (ttft_ms / 1000.0)
        records.append({
            "request_id": str(output.request_id),
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "request_e2e_ms": request_e2e_ms,
            "arrival_time_s": arrival_time,
            "completion_time_s": completion_time,
            "prefill_complete_timestamp_s": prefill_complete_time,
            "request_num_prefill_tokens": num_prefill_tokens,
            "request_num_decode_tokens": num_decode_tokens,
        })

    records.sort(key=lambda item: _to_sortable_request_id(str(item["request_id"])))
    ttft_values = np.array([float(item["ttft_ms"]) for item in records],
                           dtype=np.float64)

    if ttft_values.size == 0:
        raise RuntimeError("No TTFT records were collected.")

    p90 = float(np.percentile(ttft_values, 90))
    p95 = float(np.percentile(ttft_values, 95))
    p99 = float(np.percentile(ttft_values, 99))

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.results_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["request_id", "ttft_ms"])
        for item in records:
            writer.writerow([item["request_id"], f"{float(item['ttft_ms']):.6f}"])
        writer.writerow(["p90_ttft_ms", f"{p90:.6f}"])
        writer.writerow(["p95_ttft_ms", f"{p95:.6f}"])
        writer.writerow(["p99_ttft_ms", f"{p99:.6f}"])

    if args.results_json is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "per_request": records,
            "summary": {
                "p90_ttft_ms": p90,
                "p95_ttft_ms": p95,
                "p99_ttft_ms": p99,
            },
        }
        args.results_json.write_text(json.dumps(payload, indent=2),
                                     encoding="utf-8")

    if args.request_metrics_csv is not None:
        args.request_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.request_metrics_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "request_id",
                "ttft_ms",
                "tpot_ms",
                "request_e2e_ms",
                "arrival_time_s",
                "completion_time_s",
                "prefill_complete_timestamp_s",
                "request_num_prefill_tokens",
                "request_num_decode_tokens",
            ])
            for item in records:
                writer.writerow([
                    item["request_id"],
                    f"{float(item['ttft_ms']):.6f}",
                    f"{float(item['tpot_ms']):.6f}",
                    f"{float(item['request_e2e_ms']):.6f}",
                    f"{float(item['arrival_time_s']):.9f}",
                    f"{float(item['completion_time_s']):.9f}",
                    f"{float(item['prefill_complete_timestamp_s']):.9f}",
                    int(item["request_num_prefill_tokens"]),
                    int(item["request_num_decode_tokens"]),
                ])

    if args.request_metrics_json is not None:
        args.request_metrics_json.parent.mkdir(parents=True, exist_ok=True)
        request_payload = {
            "per_request": records,
            "summary": {
                "p90_ttft_ms": p90,
                "p95_ttft_ms": p95,
                "p99_ttft_ms": p99,
            },
        }
        args.request_metrics_json.write_text(
            json.dumps(request_payload, indent=2),
            encoding="utf-8",
        )

    print(f"Wrote TTFT CSV to: {args.results_csv}")
    if args.results_json is not None:
        print(f"Wrote TTFT JSON to: {args.results_json}")
    if args.request_metrics_csv is not None:
        print(f"Wrote request metrics CSV to: {args.request_metrics_csv}")
    if args.request_metrics_json is not None:
        print(f"Wrote request metrics JSON to: {args.request_metrics_json}")
    print(f"TTFT percentiles (ms): p90={p90:.6f}, p95={p95:.6f}, p99={p99:.6f}")


if __name__ == "__main__":
    main()

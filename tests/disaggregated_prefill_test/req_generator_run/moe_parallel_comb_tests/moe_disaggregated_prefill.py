#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Custom disaggregated prefill script for MoE parallel combination tests.
This script extends the original disaggregated_prefill.py to support
tensor parallel, pipeline parallel, and expert parallel configurations.
"""

import argparse
import os
import time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MoE Disaggregated prefill/decode with parallel configurations.")

    parser.add_argument("--role",
                        type=str,
                        required=True,
                        choices=["prefill", "decode"],
                        help="Run role: prefill (producer) or decode (consumer).")
    parser.add_argument("--num-requests",
                        type=int,
                        default=1,
                        help="Number of requests to run.")
    parser.add_argument("--prefill-tokens",
                        type=int,
                        default=1024,
                        help="Prefill tokens per request (FixedLengthConfig).")
    parser.add_argument("--decode-tokens",
                        type=int,
                        default=4096,
                        help="Decode tokens per request (FixedLengthConfig).")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--warmup-iters",
                        type=int,
                        default=0,
                        help="Number of warmup iterations.")
    parser.add_argument(
        "--model",
        type=str,
        default="mmnga/Mixtral-Fusion-4x7B-Instruct-v0.1",
        help="Model name or path.")
    parser.add_argument("--gpu-memory-utilization",
                        type=float,
                        default=0.8,
                        help="GPU memory utilization for each engine.")
    parser.add_argument("--kv-port",
                        type=int,
                        required=True,
                        help="Base port for KV transfer (shared by both roles).")
    parser.add_argument(
        "--sync-file",
        type=str,
        required=True,
        help=("Marker file path used to synchronize decode with prefill. "
              "Prefill creates it after finishing; decode waits for it."))
    parser.add_argument("--prefill-timeout",
                        type=float,
                        default=1500.0,
                        help="Timeout waiting for prefill completion (seconds).")

    # Parallel configuration arguments
    parser.add_argument("--tensor-parallel-size",
                        type=int,
                        default=1,
                        help="Tensor parallel size.")
    parser.add_argument("--pipeline-parallel-size",
                        type=int,
                        default=1,
                        help="Pipeline parallel size.")
    parser.add_argument("--data-parallel-size",
                        type=int,
                        default=1,
                        help="Data parallel size.")
    parser.add_argument("--enable-expert-parallel",
                        action="store_true",
                        help="Enable expert parallel for MoE models.")

    # Profiling (aligned with tests/disaggregated_prefill_test patterns).
    parser.add_argument("--profile",
                        action="store_true",
                        help="Enable profiling and write traces to disk.")
    parser.add_argument(
        "--profile-max-decode-tokens",
        type=int,
        default=8196,
        help=(
            "When --profile is enabled, cap decode max_tokens to avoid "
            "excessively large profiler traces that can stall stop_profile(). "
            "Set to a value >= --decode-tokens to profile the full decode."))
    return parser.parse_args()


def _generate_workload_prompts(args: argparse.Namespace,
                              seed_offset: int = 0):
    """Generate deterministic workload prompts using vLLM's request generator.

    This matches the workload construction approach used in
    `prefill_with_generator.py` / `decode_with_generator.py`.
    
    Note: This function dynamically determines the model's vocabulary size
    to ensure generated token IDs are within valid range for any model.
    """
    from transformers import AutoTokenizer
    from vllm.request_generator import (FixedLengthConfig, RequestGeneratorConfig,
                                        VLLMRequestGenerator)

    # Get the model's actual vocabulary size to avoid out-of-vocabulary errors
    # Different models have different vocab sizes (e.g., Llama-3: 128256, Mixtral: 32000)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    vocab_size = len(tokenizer)
    
    # Use safe token ID range: [1000, vocab_size - 1000] to avoid special tokens
    # at both ends of the vocabulary
    safe_max_token_id = max(vocab_size - 1000, 2000)  # Ensure at least some range
    
    # Create request generator config with model-specific vocabulary size
    request_generator_config = RequestGeneratorConfig(
        num_requests=args.num_requests,
        length_config=FixedLengthConfig(
            prefill_tokens=args.prefill_tokens,
            decode_tokens=args.decode_tokens,
        ),
        seed=args.seed + seed_offset,
        vocab_size=vocab_size,
        min_token_id=1000,
        max_token_id=safe_max_token_id,
    )
    
    # Generate requests
    request_generator = VLLMRequestGenerator(request_generator_config)
    requests = request_generator.generate()
    
    # Extract prompts from requests
    prompts = [request.prompt for request in requests]
    return prompts


def run_prefill(args: argparse.Namespace):
    """Run prefill (producer) role."""
    import torch
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.v1.utils import record_function_or_nullcontext

    print(f"[PREFILL] Starting prefill with TP={args.tensor_parallel_size}, "
          f"PP={args.pipeline_parallel_size}, DP={args.data_parallel_size}, "
          f"EP={args.enable_expert_parallel}")

    # Generate workload prompts
    warmup_prompts = _generate_workload_prompts(args, seed_offset=1)
    prompts = _generate_workload_prompts(args, seed_offset=0)
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    # Using P2pNcclConnector to transmit KV caches between vLLM instances.
    # This instance is the prefill node (kv_producer, rank 0).
    ktc = KVTransferConfig(
        kv_connector="P2pNcclConnector",
        kv_role="kv_producer",
        kv_rank=0,
        kv_parallel_size=2,
        kv_port=args.kv_port,
    )

    # Create LLM with parallel configuration
    llm = LLM(
        model=args.model,
        kv_transfer_config=ktc,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        data_parallel_size=args.data_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
        enforce_eager=False,
    )

    # Wait for decode to be ready before warmup (to avoid NCCL hang)
    # The decode process creates a "decode_ready" file after LLM initialization
    decode_ready_file = args.sync_file.replace("prefill_done", "decode_ready")
    print(f"[PREFILL] Waiting for decode to be ready (sync file: {decode_ready_file})")
    start_time = time.time()
    while not os.path.exists(decode_ready_file):
        if time.time() - start_time > args.prefill_timeout:
            raise TimeoutError(f"Decode did not become ready within {args.prefill_timeout} seconds")
        time.sleep(0.1)
    print("[PREFILL] Decode is ready, starting warmup...")

    for _ in range(args.warmup_iters):
        llm.generate(warmup_prompts, sampling_params)
    if args.profile:
        llm.start_profile()
    with record_function_or_nullcontext("e2e_llm_generate_prefill"):
        wall_start = time.perf_counter()
        llm.generate(prompts, sampling_params)
        torch.cuda.synchronize()  # Ensure all GPU operations complete
        wall_end = time.perf_counter()
    if args.profile:
        llm.stop_profile()
    prefill_wall_time = wall_end - wall_start
    print(f"[PREFILL] Total generation wall-clock time: {prefill_wall_time:.6f} seconds",
          flush=True)
    print("Prefill node is finished.", flush=True)
    if os.path.exists(args.sync_file):
        raise RuntimeError(
            f"Sync file already exists: {args.sync_file}. Please remove it "
            "before running.")
    with open(args.sync_file, "w") as f:
        f.write("prefill_done\n")

    # Keep the prefill node running to maintain KV cache connection
    print("[PREFILL] Keeping prefill node alive for KV cache transfer...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[PREFILL] Prefill node shutting down.")


def run_decode(args: argparse.Namespace):
    """Run decode (consumer) role."""
    import torch
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.v1.utils import record_function_or_nullcontext

    print(f"[DECODE] Starting decode with TP={args.tensor_parallel_size}, "
          f"PP={args.pipeline_parallel_size}, DP={args.data_parallel_size}, "
          f"EP={args.enable_expert_parallel}")

    # Generate workload prompts (same as prefill)
    warmup_prompts = _generate_workload_prompts(args, seed_offset=1)
    prompts = _generate_workload_prompts(args, seed_offset=0)
    
    # For decode, we use max_tokens from profile_max_decode_tokens if profiling
    max_tokens = args.profile_max_decode_tokens if args.profile else args.decode_tokens
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=max_tokens)

    # Using P2pNcclConnector to receive KV caches from prefill instance.
    # This instance is the decode node (kv_consumer, rank 1).
    ktc = KVTransferConfig(
        kv_connector="P2pNcclConnector",
        kv_role="kv_consumer",
        kv_rank=1,
        kv_parallel_size=2,
        kv_port=args.kv_port,
    )

    # Create LLM with parallel configuration
    llm = LLM(
        model=args.model,
        kv_transfer_config=ktc,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        data_parallel_size=args.data_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
        enforce_eager=False,
    )

    # Signal that decode is ready (LLM and P2pNcclEngine initialized)
    # This allows prefill to start warmup without NCCL hang
    decode_ready_file = args.sync_file.replace("prefill_done", "decode_ready")
    print(f"[DECODE] LLM initialized, signaling ready (sync file: {decode_ready_file})")
    with open(decode_ready_file, "w") as f:
        f.write("decode_ready\n")

    # Wait for prefill to complete
    print(f"[DECODE] Waiting for prefill completion (sync file: {args.sync_file})")
    start_time = time.time()
    while not os.path.exists(args.sync_file):
        if time.time() - start_time > args.prefill_timeout:
            raise TimeoutError(f"Prefill did not complete within {args.prefill_timeout} seconds")
        time.sleep(0.1)
    print("[DECODE] Prefill completed, starting decode...")

    for _ in range(args.warmup_iters):
        llm.generate(warmup_prompts, sampling_params)
    if args.profile:
        llm.start_profile()
    with record_function_or_nullcontext("e2e_llm_generate_decode"):
        wall_start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        torch.cuda.synchronize()  # Ensure all GPU operations complete
        wall_end = time.perf_counter()
    if args.profile:
        llm.stop_profile()
    decode_wall_time = wall_end - wall_start
    print(f"[DECODE] Total generation wall-clock time: {decode_wall_time:.6f} seconds",
          flush=True)
    
    # Print some sample outputs
    print("[DECODE] Sample outputs:")
    for i, output in enumerate(outputs[:2]):  # Show first 2 outputs
        print(f"  Request {i}: {output.outputs[0].text[:100]}...")
    
    print("Decode node is finished.", flush=True)


def main():
    args = _parse_args()
    
    print(f"Starting MoE disaggregated prefill test:")
    print(f"  Role: {args.role}")
    print(f"  Model: {args.model}")
    print(f"  Tensor Parallel Size: {args.tensor_parallel_size}")
    print(f"  Pipeline Parallel Size: {args.pipeline_parallel_size}")
    print(f"  Data Parallel Size: {args.data_parallel_size}")
    print(f"  Expert Parallel: {args.enable_expert_parallel}")
    print(f"  Requests: {args.num_requests}")
    print(f"  Prefill tokens: {args.prefill_tokens}")
    print(f"  Decode tokens: {args.decode_tokens}")
    print(f"  KV Port: {args.kv_port}")
    print(f"  Sync file: {args.sync_file}")
    
    if args.role == "prefill":
        run_prefill(args)
    elif args.role == "decode":
        run_decode(args)
    else:
        raise ValueError(f"Unknown role: {args.role}")


if __name__ == "__main__":
    main()
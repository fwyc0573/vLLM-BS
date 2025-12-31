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
    """Run prefill (producer) role.
    
    IMPORTANT: P2pNcclConnector uses NCCL for point-to-point communication,
    which requires both sender (prefill) and receiver (decode) to participate
    in the communication simultaneously. Therefore, prefill and decode must
    call generate() at the same time to avoid NCCL blocking/timeout.
    
    Synchronization flow:
    1. Prefill waits for decode_ready (decode LLM initialized)
    2. Prefill signals prefill_ready (prefill LLM initialized)
    3. Decode sees prefill_ready and starts warmup generate
    4. Prefill starts warmup generate (both execute simultaneously)
    5. After warmup, prefill signals prefill_warmup_done
    6. Decode sees prefill_warmup_done and starts actual generate
    7. Prefill starts actual generate (both execute simultaneously)
    8. Prefill signals prefill_done when finished
    """
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
    # This instance is the prefill node (kv_producer).
    # NOTE: kv_rank is set to 1 because in DP mode, the decode process
    # typically initializes first and gets the lower port range (kv_rank=0).
    # The prefill process initializes later and gets the higher port range.
    # This ensures the port allocation matches the actual initialization order.
    ktc = KVTransferConfig(
        kv_connector="P2pNcclConnector",
        kv_role="kv_producer",
        kv_rank=1,  # Prefill uses higher port range
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

    # Define sync file paths
    decode_ready_file = args.sync_file.replace("prefill_done", "decode_ready")
    prefill_ready_file = args.sync_file.replace("prefill_done", "prefill_ready")
    prefill_warmup_done_file = args.sync_file.replace("prefill_done", "prefill_warmup_done")

    # Wait for decode to be ready (LLM initialized)
    print(f"[PREFILL] Waiting for decode to be ready (sync file: {decode_ready_file})")
    start_time = time.time()
    while not os.path.exists(decode_ready_file):
        if time.time() - start_time > args.prefill_timeout:
            raise TimeoutError(f"Decode did not become ready within {args.prefill_timeout} seconds")
        time.sleep(0.1)
    print("[PREFILL] Decode is ready.")

    # Signal that prefill is ready
    print(f"[PREFILL] Signaling prefill ready (sync file: {prefill_ready_file})")
    with open(prefill_ready_file, "w") as f:
        f.write("prefill_ready\n")

    # Small delay to ensure decode sees the signal and starts warmup
    time.sleep(0.5)

    # Start warmup - decode should be executing warmup generate simultaneously
    print("[PREFILL] Starting warmup...")
    for _ in range(args.warmup_iters):
        llm.generate(warmup_prompts, sampling_params)
    print("[PREFILL] Warmup completed.")

    # Signal warmup done
    print(f"[PREFILL] Signaling warmup done (sync file: {prefill_warmup_done_file})")
    with open(prefill_warmup_done_file, "w") as f:
        f.write("prefill_warmup_done\n")

    # Small delay to ensure decode sees the signal and starts actual generate
    time.sleep(0.5)

    # Start actual prefill - decode should be executing generate simultaneously
    if args.profile:
        llm.start_profile()
    print("[PREFILL] Starting actual prefill generation...")
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
    
    # Signal prefill done
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
    """Run decode (consumer) role.
    
    IMPORTANT: P2pNcclConnector uses NCCL for point-to-point communication,
    which requires both sender (prefill) and receiver (decode) to participate
    in the communication simultaneously. Therefore, prefill and decode must
    call generate() at the same time to avoid NCCL blocking/timeout.
    
    Synchronization flow:
    1. Decode initializes LLM and signals decode_ready
    2. Decode waits for prefill_ready (prefill LLM initialized)
    3. Decode starts warmup generate (prefill does the same simultaneously)
    4. Decode waits for prefill_warmup_done
    5. Decode starts actual generate (prefill does the same simultaneously)
    6. Decode waits for prefill_done to confirm completion
    """
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
    # This instance is the decode node (kv_consumer).
    # NOTE: kv_rank is set to 0 because in DP mode, the decode process
    # typically initializes first and gets the lower port range.
    # This ensures the port allocation matches the actual initialization order.
    ktc = KVTransferConfig(
        kv_connector="P2pNcclConnector",
        kv_role="kv_consumer",
        kv_rank=0,  # Decode uses lower port range
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

    # Define sync file paths
    decode_ready_file = args.sync_file.replace("prefill_done", "decode_ready")
    prefill_ready_file = args.sync_file.replace("prefill_done", "prefill_ready")
    prefill_warmup_done_file = args.sync_file.replace("prefill_done", "prefill_warmup_done")

    # Signal that decode is ready (LLM and P2pNcclEngine initialized)
    print(f"[DECODE] LLM initialized, signaling ready (sync file: {decode_ready_file})")
    with open(decode_ready_file, "w") as f:
        f.write("decode_ready\n")

    # Wait for prefill to be ready
    print(f"[DECODE] Waiting for prefill to be ready (sync file: {prefill_ready_file})")
    start_time = time.time()
    while not os.path.exists(prefill_ready_file):
        if time.time() - start_time > args.prefill_timeout:
            raise TimeoutError(f"Prefill did not become ready within {args.prefill_timeout} seconds")
        time.sleep(0.1)
    print("[DECODE] Prefill is ready.")

    # Small delay to ensure prefill starts warmup
    time.sleep(0.5)

    # Start warmup - prefill should be executing warmup generate simultaneously
    # This is CRITICAL: decode must call generate() to receive KV cache from prefill
    print("[DECODE] Starting warmup (receiving KV cache from prefill)...")
    for _ in range(args.warmup_iters):
        llm.generate(warmup_prompts, sampling_params)
    print("[DECODE] Warmup completed.")

    # Wait for prefill warmup to complete
    print(f"[DECODE] Waiting for prefill warmup to complete (sync file: {prefill_warmup_done_file})")
    start_time = time.time()
    while not os.path.exists(prefill_warmup_done_file):
        if time.time() - start_time > args.prefill_timeout:
            raise TimeoutError(f"Prefill warmup did not complete within {args.prefill_timeout} seconds")
        time.sleep(0.1)
    print("[DECODE] Prefill warmup completed.")

    # Small delay to ensure prefill starts actual generate
    time.sleep(0.5)

    # Start actual decode - prefill should be executing generate simultaneously
    if args.profile:
        llm.start_profile()
    print("[DECODE] Starting actual decode generation (receiving KV cache from prefill)...")
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
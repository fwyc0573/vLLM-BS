# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file demonstrates the example usage of disaggregated prefilling
We will launch 2 vllm instances (GPU 0 for prefill and GPU 1 for decode),
and then transfer the KV cache between them.
"""

import argparse
import os
import time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Disaggregated prefill/decode with KV transfer.")

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
        default="unsloth/Llama-3.2-1B-Instruct",
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
    """
    from vllm.request_generator import (FixedLengthConfig, RequestGeneratorConfig,
                                        VLLMRequestGenerator)

    config = RequestGeneratorConfig(
        num_requests=args.num_requests,
        length_config=FixedLengthConfig(prefill_tokens=args.prefill_tokens,
                                        decode_tokens=args.decode_tokens),
        seed=int(args.seed) + int(seed_offset),
    )
    generator = VLLMRequestGenerator(config)
    requests = generator.generate()
    prompts = [r.prompt for r in requests]
    return prompts


def _validate_runtime_env(args: argparse.Namespace) -> None:
    # Prefer controlling environment variables in shell scripts. Fail fast.
    if os.environ.get("VLLM_USE_V1") != "1":
        raise RuntimeError(
            "VLLM_USE_V1 must be set to 1. Please export VLLM_USE_V1=1 "
            "before running this example.")
    if args.profile and not os.environ.get("VLLM_TORCH_PROFILER_DIR"):
        raise RuntimeError(
            "Profiling is enabled but VLLM_TORCH_PROFILER_DIR is not set. "
            "Please export VLLM_TORCH_PROFILER_DIR before running.")


def _wait_for_file(path: str, timeout_s: float) -> None:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if os.path.exists(path):
            return
        time.sleep(0.1)
    raise TimeoutError(f"Timeout waiting for sync file: {path}")


def run_prefill(args: argparse.Namespace):
    _validate_runtime_env(args)
    import torch
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.v1.utils import record_function_or_nullcontext

    # The prefill node receives two requests, while the decode node receives
    # three requests. So the decode node will only receive the KV Cache for
    # requests 1 and 3. The decode node will use the KV Cache of requests 1
    # and 3 and do prefilling on request 2.
    warmup_prompts = _generate_workload_prompts(args, seed_offset=1)
    prompts = _generate_workload_prompts(args, seed_offset=0)
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    # Using P2pNcclConnector to transmit KV caches between vLLM instances.
    # This instance is the prefill node (kv_producer, rank 0).
    # The number of parallel instances for KV cache transfer is set to 2,
    # as required for P2pNcclConnector.
    ktc = KVTransferConfig(
        kv_connector="P2pNcclConnector",
        kv_role="kv_producer",
        kv_rank=0,
        kv_parallel_size=2,
        kv_port=args.kv_port,
    )

    # Set GPU memory utilization to 0.8 for an A6000 GPU with 40GB
    # memory. You may need to adjust the value to fit your GPU.
    llm = LLM(
        model=args.model,
        kv_transfer_config=ktc,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
    )

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

    # To keep the prefill node running in case the decode node is not done;
    # otherwise, the script might exit prematurely, causing incomplete decoding.
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Script stopped by user.")


def run_decode(args: argparse.Namespace):
    _validate_runtime_env(args)
    import torch
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.v1.utils import record_function_or_nullcontext

    warmup_prompts = _generate_workload_prompts(args, seed_offset=1)
    prompts = _generate_workload_prompts(args, seed_offset=0)
    max_decode_tokens = int(args.decode_tokens)
    if args.profile:
        cap = int(args.profile_max_decode_tokens)
        if cap <= 0:
            raise ValueError("--profile-max-decode-tokens must be > 0")
        if cap < max_decode_tokens:
            print(
                "NOTE: --profile is enabled; capping decode max_tokens from "
                f"{max_decode_tokens} to {cap} to keep profiler traces "
                "manageable. Override with --profile-max-decode-tokens.")
        max_decode_tokens = min(max_decode_tokens, cap)

    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.95,
        max_tokens=max_decode_tokens,
        ignore_eos=True,
        stop=None,
        stop_token_ids=None,
    )

    # Using P2pNcclConnector to transmit KV caches between vLLM instances.
    # This instance is the decode node (kv_consumer, rank 1).
    # The number of parallel instances for KV cache transfer is set to 2,
    # as required for P2pNcclConnector.
    ktc = KVTransferConfig(
        kv_connector="P2pNcclConnector",
        kv_role="kv_consumer",
        kv_rank=1,
        kv_parallel_size=2,
        kv_port=args.kv_port,
    )

    # Set GPU memory utilization to 0.8 for an A6000 GPU with 40GB
    # memory. You may need to adjust the value to fit your GPU.
    llm = LLM(
        model=args.model,
        kv_transfer_config=ktc,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
        max_num_seqs=64,
    )

    print("Waiting for prefill node to finish...")
    _wait_for_file(args.sync_file, timeout_s=args.prefill_timeout)

    for _ in range(args.warmup_iters):
        llm.generate(warmup_prompts, sampling_params)

    # At this point when the prefill_done is set, the kv-cache should have been
    # transferred to this decode node, so we can start decoding.
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
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


def main():
    args = _parse_args()
    if args.role == "prefill":
        run_prefill(args)
    elif args.role == "decode":
        run_decode(args)
    else:
        raise ValueError(f"Unknown role: {args.role}")


if __name__ == "__main__":
    main()

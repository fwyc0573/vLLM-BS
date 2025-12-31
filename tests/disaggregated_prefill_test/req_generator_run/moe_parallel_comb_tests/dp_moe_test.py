#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Data Parallel MoE Test Script

This script tests vLLM's Data Parallel (DP) functionality with MoE models.
It properly initializes DP by spawning multiple processes with correct
environment variables.

Usage:
    python dp_moe_test.py --dp-size 2 --model mmnga/Mixtral-Fusion-4x7B-Instruct-v0.1
    python dp_moe_test.py --dp-size 2 --enable-expert-parallel --model mmnga/Mixtral-Fusion-4x7B-Instruct-v0.1
"""

import argparse
import os
import sys
import time
from multiprocessing import Process

from vllm.utils import get_open_port


def parse_args():
    parser = argparse.ArgumentParser(description="Data Parallel MoE Test")
    parser.add_argument(
        "--model",
        type=str,
        default="mmnga/Mixtral-Fusion-4x7B-Instruct-v0.1",
        help="Model name or path",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=2,
        help="Data parallel size"
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size"
    )
    parser.add_argument(
        "--enable-expert-parallel",
        action="store_true",
        help="Enable expert parallel for MoE models"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=4,
        help="Number of requests to process"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Maximum tokens to generate per request"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for each process"
    )
    return parser.parse_args()


def run_dp_rank(
    model: str,
    dp_size: int,
    local_dp_rank: int,
    global_dp_rank: int,
    dp_master_ip: str,
    dp_master_port: int,
    tp_size: int,
    enable_expert_parallel: bool,
    num_requests: int,
    max_tokens: int,
    gpu_memory_utilization: float,
):
    """Run inference for a single DP rank."""
    # Set DP environment variables
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # Import vLLM after setting environment variables
    from vllm import LLM, SamplingParams

    print(f"[DP Rank {global_dp_rank}] Starting with:")
    print(f"  - Model: {model}")
    print(f"  - DP Size: {dp_size}")
    print(f"  - TP Size: {tp_size}")
    print(f"  - Expert Parallel: {enable_expert_parallel}")
    print(f"  - DP Master: {dp_master_ip}:{dp_master_port}")

    # Generate prompts - each DP rank processes different prompts
    all_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "Machine learning is",
        "Deep learning differs from",
        "Natural language processing enables",
        "Computer vision is used for",
    ]

    # Distribute prompts across DP ranks
    prompts_per_rank = len(all_prompts) // dp_size
    start_idx = global_dp_rank * prompts_per_rank
    end_idx = start_idx + prompts_per_rank
    prompts = all_prompts[start_idx:end_idx]

    if len(prompts) == 0:
        prompts = ["Placeholder prompt"]

    print(f"[DP Rank {global_dp_rank}] Processing {len(prompts)} prompts")

    # Create sampling params
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_tokens
    )

    # Create LLM instance
    # Note: CUDA_VISIBLE_DEVICES is set automatically by vLLM for DP
    llm = LLM(
        model=model,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=enable_expert_parallel,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=False,
    )

    print(f"[DP Rank {global_dp_rank}] LLM initialized, starting generation...")

    # Generate outputs
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.perf_counter()

    # Print results
    print(f"\n[DP Rank {global_dp_rank}] Generation completed in {end_time - start_time:.2f}s")
    for i, output in enumerate(outputs[:3]):  # Print first 3 outputs
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"[DP Rank {global_dp_rank}] Prompt: {prompt!r}")
        print(f"[DP Rank {global_dp_rank}] Generated: {generated_text!r}")
        print()

    print(f"[DP Rank {global_dp_rank}] Test completed successfully!")

    # Give engines time to clean up
    time.sleep(1)


def main():
    args = parse_args()

    dp_size = args.dp_size
    tp_size = args.tp_size

    # Get DP master address
    dp_master_ip = "127.0.0.1"
    dp_master_port = get_open_port()

    print("=" * 60)
    print(f"Data Parallel MoE Test")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"DP Size: {dp_size}")
    print(f"TP Size: {tp_size}")
    print(f"Expert Parallel: {args.enable_expert_parallel}")
    print(f"DP Master: {dp_master_ip}:{dp_master_port}")
    print(f"Num Requests: {args.num_requests}")
    print(f"Max Tokens: {args.max_tokens}")
    print("=" * 60)

    # Spawn processes for each DP rank
    procs = []
    for local_dp_rank in range(dp_size):
        global_dp_rank = local_dp_rank  # Single node, so local == global

        proc = Process(
            target=run_dp_rank,
            args=(
                args.model,
                dp_size,
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port,
                tp_size,
                args.enable_expert_parallel,
                args.num_requests,
                args.max_tokens,
                args.gpu_memory_utilization,
            ),
        )
        proc.start()
        procs.append(proc)
        print(f"Started DP rank {global_dp_rank} (PID: {proc.pid})")

    # Wait for all processes to complete
    exit_code = 0
    for i, proc in enumerate(procs):
        proc.join(timeout=args.timeout)
        if proc.exitcode is None:
            print(f"ERROR: DP rank {i} (PID: {proc.pid}) timed out after {args.timeout}s")
            proc.kill()
            exit_code = 1
        elif proc.exitcode != 0:
            print(f"ERROR: DP rank {i} (PID: {proc.pid}) exited with code {proc.exitcode}")
            exit_code = proc.exitcode
        else:
            print(f"DP rank {i} (PID: {proc.pid}) completed successfully")

    print("=" * 60)
    if exit_code == 0:
        print("All DP ranks completed successfully!")
    else:
        print(f"Test failed with exit code {exit_code}")
    print("=" * 60)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

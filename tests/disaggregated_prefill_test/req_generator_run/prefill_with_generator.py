# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Prefill example using vLLM Request Generator.

This script generates workload requests using the request generator
and runs the prefill phase of disaggregated inference.
"""

import argparse
import json
import os
import sys
import shutil

# Add vLLM to path if needed
VLLM_PATH = "/research/d1/gds/ytyang/yichengfeng/frontier/sota-infer-engine/vllm"
if VLLM_PATH not in sys.path:
    sys.path.insert(0, VLLM_PATH)

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.v1.utils import record_function_or_nullcontext
from vllm.request_generator import (
    VLLMRequestGenerator,
    RequestGeneratorConfig,
    FixedLengthConfig,
    UniformLengthConfig,
    KVTransferSync,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Prefill with Request Generator")
    
    # Request generation config
    parser.add_argument("--num-requests", type=int, default=4,
                        help="Number of requests to generate")
    parser.add_argument("--prefill-tokens", type=int, default=1024,
                        help="Input length for each request (fixed mode)")
    parser.add_argument("--decode-tokens", type=int, default=4096,
                        help="Output length for each request")
    parser.add_argument("--length-mode", type=str, default="fixed",
                        choices=["fixed", "uniform"],
                        help="Length distribution mode")
    parser.add_argument("--min-tokens", type=int, default=512,
                        help="Min total tokens (uniform mode)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max total tokens (uniform mode)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Model config
    parser.add_argument("--model", type=str, 
                        default="unsloth/Llama-3.2-1B-Instruct",
                        help="Model to use")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7,
                        help="GPU memory utilization")
    
    # Output config
    parser.add_argument("--output-file", type=str, default="output.txt",
                        help="File to save prompts for decode phase")
    parser.add_argument("--metadata-file", type=str, default="metadata.json",
                        help="File to save request metadata")
    
    # KV Transfer sync
    parser.add_argument("--kv-storage-path", type=str, default="local_storage",
                        help="Path for KV cache storage (used for sync)")
    parser.add_argument("--enable-kv-sync", action="store_true",
                        help="Enable KV transfer synchronization marker")
    
    # Profiling
    parser.add_argument("--profile", action="store_true",
                        help="Enable profiling")
    parser.add_argument("--warmup-iters", type=int, default=3,
                        help="Number of warmup iterations")
    
    return parser.parse_args()


def create_request_generator(args) -> VLLMRequestGenerator:
    """Create request generator based on command line arguments."""
    
    if args.length_mode == "fixed":
        length_config = FixedLengthConfig(
            prefill_tokens=args.prefill_tokens,
            decode_tokens=args.decode_tokens,
        )
    elif args.length_mode == "uniform":
        length_config = UniformLengthConfig(
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            prefill_to_decode_ratio=args.prefill_tokens / max(1, args.decode_tokens),
        )
    else:
        raise ValueError(f"Unknown length mode: {args.length_mode}")
    
    config = RequestGeneratorConfig(
        num_requests=args.num_requests,
        length_config=length_config,
        seed=args.seed,
    )
    
    return VLLMRequestGenerator(config)


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Prefill Phase - Request Generator Example")
    print("=" * 60)
    
    # Create request generator
    print("\n[1] Creating request generator...")
    generator = create_request_generator(args)
    
    # Print workload summary
    summary = generator.get_workload_summary()
    print(f"\nWorkload Summary:")
    print(f"  - Requests: {summary['num_requests']}")
    print(f"  - Prefill tokens: min={summary['prefill_tokens']['min']}, "
          f"max={summary['prefill_tokens']['max']}, "
          f"mean={summary['prefill_tokens']['mean']:.1f}")
    print(f"  - Decode tokens: min={summary['decode_tokens']['min']}, "
          f"max={summary['decode_tokens']['max']}, "
          f"mean={summary['decode_tokens']['mean']:.1f}")
    
    # Generate requests
    print("\n[2] Generating vLLM requests...")
    requests = generator.generate()
    
    # Prepare prompts for vLLM
    # For prefill, we only generate 1 token to just do the KV cache computation
    prompts = [r.prompt for r in requests]
    
    # Prefill only generates 1 token
    prefill_sampling_params = SamplingParams(
        temperature=0,
        top_p=0.95,
        max_tokens=1,
    )
    
    # Initialize vLLM
    print(f"\n[3] Initializing vLLM with model: {args.model}")
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_transfer_config=KVTransferConfig(
            kv_connector="SharedStorageConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "shared_storage_path": args.kv_storage_path
            },
        ),
    )
    
    # Warmup
    if args.warmup_iters > 0:
        print(f"\n[4] Running {args.warmup_iters} warmup iterations...")
        for i in range(args.warmup_iters):
            _ = llm.generate(prompts, prefill_sampling_params)

        # Prevent warmup KV cache contamination for the actual measurement run.
        # Remove all KV cache artifacts produced during warmup.
        if os.path.isdir(args.kv_storage_path):
            shutil.rmtree(args.kv_storage_path)
        os.makedirs(args.kv_storage_path, exist_ok=True)
    
    # Run prefill with profiling
    print("\n[5] Running prefill phase...")
    if args.profile:
        llm.start_profile()
    
    with record_function_or_nullcontext("e2e_llm_generate_prefill"):
        outputs = llm.generate(prompts, prefill_sampling_params)
    
    if args.profile:
        llm.stop_profile()
    
    # Process outputs and save for decode phase
    print("\n[6] Processing outputs...")
    new_prompts = []
    metadata = []
    
    for i, (output, req) in enumerate(zip(outputs, requests)):
        prompt_ids = req.prompt["prompt_token_ids"]
        generated_ids = output.outputs[0].token_ids
        
        # NOTE: Save original prompt WITHOUT generated tokens
        # This is required for SharedStorageConnector's hash matching logic:
        # - _found_match_for_request uses align_to_block_size(len-1, block_size)
        # - make_meta uses align_to_block_size(len, block_size)
        # If we include generated tokens (prompt_len becomes 1025), the hash
        # computed by make_meta will differ from what was saved during prefill.
        # By keeping the original prompt length (1024), both computations align.
        new_prompts.append(prompt_ids)
        
        metadata.append({
            "request_id": req.request_id,
            "prefill_tokens": req.prefill_tokens,
            "decode_tokens": req.decode_tokens,
            "generated_tokens": len(generated_ids),
        })
        
        print(f"  Request {i}: prefill={req.prefill_tokens}, "
              f"generated={len(generated_ids)}, "
              f"saved_prompt_len={len(prompt_ids)}")
    
    # Save prompts for decode phase (as JSON for token IDs)
    print(f"\n[7] Saving outputs to {args.output_file}...")
    with open(args.output_file, "w") as f:
        json.dump(new_prompts, f)
    
    # Save metadata
    print(f"    Saving metadata to {args.metadata_file}...")
    with open(args.metadata_file, "w") as f:
        json.dump({
            "requests": metadata,
            "config": {
                "num_requests": args.num_requests,
                "length_mode": args.length_mode,
                "prefill_tokens": args.prefill_tokens,
                "decode_tokens": args.decode_tokens,
                "seed": args.seed,
            }
        }, f, indent=2)
    
    # Mark KV transfer as complete (if sync enabled)
    if args.enable_kv_sync:
        print(f"\n[8] Marking KV transfer complete...")
        kv_sync = KVTransferSync(storage_path=args.kv_storage_path)
        kv_sync.mark_transfer_complete(metadata={
            "num_requests": len(new_prompts),
            "total_prefill_tokens": sum(len(p) for p in new_prompts),
        })
        print(f"    Marker file: {kv_sync.marker_file}")
    
    print("\n" + "=" * 60)
    print("Prefill phase complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

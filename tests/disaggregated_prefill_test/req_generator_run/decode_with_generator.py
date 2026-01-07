# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Decode example using vLLM Request Generator.

This script loads prompts from the prefill phase and runs the decode phase
of disaggregated inference with controlled output lengths.
"""

import argparse
import json
import os
import sys

# Add vLLM to path if needed - dynamically resolve path
VLLM_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if VLLM_PATH not in sys.path:
    sys.path.insert(0, VLLM_PATH)

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.v1.utils import record_function_or_nullcontext
from vllm.request_generator import KVTransferSync


def parse_args():
    parser = argparse.ArgumentParser(description="Decode with Request Generator")
    
    # Input config
    parser.add_argument("--input-file", type=str, default="output.txt",
                        help="File with prompts from prefill phase")
    parser.add_argument("--metadata-file", type=str, default="metadata.json",
                        help="File with request metadata")
    
    # Override decode length (optional)
    parser.add_argument("--decode-tokens", type=int, default=None,
                        help="Override decode tokens (uses metadata if not set)")
    
    # Model config
    parser.add_argument("--model", type=str, 
                        default="unsloth/Llama-3.2-1B-Instruct",
                        help="Model to use")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7,
                        help="GPU memory utilization")
    parser.add_argument("--max-num-batched-tokens", type=int, default=512,
                        help="Max tokens per batch")
    parser.add_argument("--max-num-seqs", type=int, default=64,
                        help="Max sequences per batch")
    
    # Profiling
    parser.add_argument("--profile", action="store_true",
                        help="Enable profiling")
    parser.add_argument("--warmup-iters", type=int, default=3,
                        help="Number of warmup iterations")
    
    # KV Transfer sync
    parser.add_argument("--kv-storage-path", type=str, default="local_storage",
                        help="Path for KV cache storage (used for sync)")
    parser.add_argument("--enable-kv-sync", action="store_true",
                        help="Enable KV transfer synchronization (wait for prefill)")
    parser.add_argument("--kv-sync-timeout", type=float, default=60.0,
                        help="Timeout for waiting on KV transfer (seconds)")
    
    return parser.parse_args()


def load_prompts_and_metadata(args):
    """Load prompts and metadata from prefill phase."""
    
    # Load token ID prompts
    print(f"Loading prompts from {args.input_file}...")
    with open(args.input_file, "r") as f:
        prompt_ids_list = json.load(f)
    
    # Load metadata
    metadata = None
    if os.path.exists(args.metadata_file):
        print(f"Loading metadata from {args.metadata_file}...")
        with open(args.metadata_file, "r") as f:
            metadata = json.load(f)
    
    return prompt_ids_list, metadata


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Decode Phase - Request Generator Example")
    print("=" * 60)
    
    # Wait for KV transfer to complete (if sync enabled)
    if args.enable_kv_sync:
        print("\n[0] Waiting for KV transfer from prefill phase...")
        kv_sync = KVTransferSync(storage_path=args.kv_storage_path)
        try:
            kv_sync.wait_for_transfer_complete(
                timeout=args.kv_sync_timeout,
                verbose=True,
            )
            # Print transfer metadata if available
            transfer_meta = kv_sync.get_transfer_metadata()
            if transfer_meta:
                print(f"    Transfer metadata: {transfer_meta.get('num_requests', '?')} requests")
        except TimeoutError as e:
            print(f"ERROR: {e}")
            print("Hint: Make sure prefill phase ran with --enable-kv-sync")
            sys.exit(1)
        except RuntimeError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    
    # Load prompts and metadata
    print("\n[1] Loading data from prefill phase...")
    prompt_ids_list, metadata = load_prompts_and_metadata(args)
    
    num_requests = len(prompt_ids_list)
    print(f"  Loaded {num_requests} prompts")
    
    # Determine decode tokens for each request
    if args.decode_tokens is not None:
        # Use override for all requests
        decode_tokens_list = [args.decode_tokens] * num_requests
        print(f"  Using override decode_tokens={args.decode_tokens} for all requests")
    elif metadata is not None and "requests" in metadata:
        # Use per-request decode tokens from metadata
        decode_tokens_list = [req["decode_tokens"] for req in metadata["requests"]]
        print(f"  Using per-request decode tokens from metadata")
    else:
        # Default fallback
        decode_tokens_list = [64] * num_requests
        print(f"  Using default decode_tokens=64 for all requests")
    
    # Print summary
    print(f"\nDecode Summary:")
    print(f"  - Requests: {num_requests}")
    print(f"  - Decode tokens: min={min(decode_tokens_list)}, "
          f"max={max(decode_tokens_list)}, "
          f"mean={sum(decode_tokens_list)/len(decode_tokens_list):.1f}")
    for i, (ids, dec_len) in enumerate(zip(prompt_ids_list, decode_tokens_list)):
        print(f"    Request {i}: input_len={len(ids)}, decode_len={dec_len}")
    
    # Prepare prompts (TokensPrompt format)
    prompts = [{"prompt_token_ids": ids} for ids in prompt_ids_list]
    
    # Create per-request sampling params
    sampling_params_list = [
        SamplingParams(
            temperature=0,
            top_p=0.95,
            max_tokens=dec_len,
            ignore_eos=True,
            stop=None,
            stop_token_ids=None,
        )
        for dec_len in decode_tokens_list
    ]
    
    # Initialize vLLM
    print(f"\n[2] Initializing vLLM with model: {args.model}")
    llm = LLM(
        model=args.model,
        enforce_eager=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
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
        print(f"\n[3] Running {args.warmup_iters} warmup iterations...")
        for i in range(args.warmup_iters):
            _ = llm.generate(prompts, sampling_params_list)
    
    # Run decode with profiling
    print("\n[4] Running decode phase...")
    if args.profile:
        llm.start_profile()
    
    with record_function_or_nullcontext("e2e_llm_generate_decode"):
        outputs = llm.generate(prompts, sampling_params_list)
    
    if args.profile:
        llm.stop_profile()
    
    # Process outputs
    print("\n[5] Results:")
    print("-" * 60)
    
    total_generated = 0
    for i, (output, expected_len) in enumerate(zip(outputs, decode_tokens_list)):
        generated_ids = output.outputs[0].token_ids
        generated_len = len(generated_ids)
        total_generated += generated_len
        
        # Check if we got expected length
        status = "✓" if generated_len == expected_len else f"✗ (expected {expected_len})"
        
        print(f"Request {i}: generated {generated_len} tokens {status}")
        
        # Show first few tokens of generated text (if detokenizable)
        if output.outputs[0].text:
            text_preview = output.outputs[0].text[:50]
            if len(output.outputs[0].text) > 50:
                text_preview += "..."
            print(f"  Preview: {text_preview!r}")
    
    print("-" * 60)
    print(f"Total generated tokens: {total_generated}")
    print(f"Expected total: {sum(decode_tokens_list)}")
    
    print("\n" + "=" * 60)
    print("Decode phase complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

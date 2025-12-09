#!/usr/bin/env python3
"""
Phase 2: Batch/Sequence Variation Tests

Tests how profiling overhead varies with:
- Batch sizes: 1, 4, 8, 16
- Sequence lengths: 512, 1024, 2048

Uses 1B model (Llama-3.2-1B-Instruct) for faster iteration.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Script directory
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parents[2]  # vllm root

# Test configurations for Phase 2
BATCH_SIZES = [1, 4, 8, 16]
SEQ_LENGTHS = [512, 1024, 2048]

# Use TinyLlama for Phase 2 (faster iteration, publicly accessible)
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_NAME = "TinyLlama-1.1B"
EXPECTED_SCOPES = 22 * 10 + 6  # TinyLlama has 22 layers * ~10 scopes/layer + global scopes ≈ 226

def get_python_exe():
    """Get the Python executable path."""
    return sys.executable

def run_single_test(batch_size: int, seq_len: int, profiling_on: bool, 
                    run_index: int, results_dir: Path) -> dict:
    """Run a single test configuration as subprocess."""
    
    test_id = f"P2-B{batch_size}S{seq_len}" + ("A" if profiling_on else "B")
    
    print(f"\n{'='*70}")
    print(f"Test: {test_id} (Run {run_index})")
    print(f"  Model: {MODEL_NAME} ({MODEL_ID})")
    print(f"  Batch: {batch_size}, Seq: {seq_len}")
    print(f"  Profiling: {'ON' if profiling_on else 'OFF'}")
    print(f"  Expected scopes: ~{EXPECTED_SCOPES if profiling_on else 0}")
    print(f"{'='*70}")
    
    # Build the test script to run
    test_code = f'''
import os
import sys
sys.path.insert(0, "{PROJECT_ROOT}")

# Set environment before any vllm imports
os.environ["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "{'1' if profiling_on else '0'}"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # Run in single-process mode for profiling

import json
import time
import torch
from vllm import LLM, SamplingParams

def run_test():
    # Initialize model
    llm = LLM(
        model="{MODEL_ID}",
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        max_model_len=2048,
    )
    
    # Create prompts
    prompts = ["Hello, my name is" * ({seq_len} // 20 + 1)][:1] * {batch_size}
    prompts = [p[:{seq_len}] for p in prompts]
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
    )
    
    # Warmup
    _ = llm.generate(prompts[:1], sampling_params)
    torch.cuda.synchronize()
    
    # Profile the actual run
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        start_time = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
    
    e2e_latency = (end_time - start_time) * 1000
    
    # Extract timing info
    events = prof.key_averages()
    total_self_cpu = sum(e.self_cpu_time_total for e in events) / 1000
    # Note: Use self_device_time_total for CUDA time
    total_self_cuda = sum(e.self_device_time_total for e in events) / 1000
    
    # Count custom scopes (actual scope names from record_function_or_nullcontext)
    custom_scope_names = ["mlp_up_proj", "mlp_act", "mlp_down_proj", "attn_pre_proj", 
                          "attn_rope", "attn", "attn_post_proj", "input_layernorm", 
                          "post_attention_layernorm", "attn_kv_cache_save", "attn_prefill",
                          "attn_decode", "Preprocess", "Forward", "Postprocess", 
                          "Sample", "Bookkeep", "Draft", "EPLB"]
    scope_count = sum(1 for e in events if any(s in e.key for s in custom_scope_names))
    
    result = {{
        "test_id": "{test_id}",
        "model": "{MODEL_NAME}",
        "model_hf_id": "{MODEL_ID}",
        "batch_size": {batch_size},
        "seq_len": {seq_len},
        "profiling_enabled": {profiling_on},
        "run_index": {run_index},
        "self_cpu_time_ms": round(total_self_cpu, 3),
        "self_cuda_time_ms": round(total_self_cuda, 3),
        "e2e_latency_ms": round(e2e_latency, 3),
        "total_scope_calls": scope_count,
        "timestamp": "{datetime.now().isoformat()}",
        "success": True,
        "error_message": None
    }}
    
    print(json.dumps(result))
    return result

if __name__ == "__main__":
    run_test()
'''
    
    # Run as subprocess
    python_exe = get_python_exe()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["HF_HOME"] = "/local/ytyang/yichengfeng/.hf_saved_menu"
    
    print(f"  Running subprocess...")
    print(f"  Python exe: {python_exe}")
    print(f"  PYTHONPATH: {PROJECT_ROOT}")
    
    start = time.time()
    try:
        proc = subprocess.run(
            [python_exe, "-c", test_code],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,  # 10 minute timeout
        )
        elapsed = time.time() - start
        print(f"  Subprocess completed in {elapsed:.1f}s")
        
        if proc.returncode != 0:
            print(f"  ERROR: Process failed with return code {proc.returncode}")
            print(f"  STDERR: {proc.stderr[:2000] if proc.stderr else 'None'}")
            return {
                "test_id": test_id,
                "model": MODEL_NAME,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "profiling_enabled": profiling_on,
                "run_index": run_index,
                "success": False,
                "error_message": proc.stderr[:1000] if proc.stderr else "Unknown error",
                "timestamp": datetime.now().isoformat(),
            }
        
        # Parse result from stdout (last line should be JSON)
        output_lines = proc.stdout.strip().split('\n')
        for line in reversed(output_lines):
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    result = json.loads(line)
                    # Save individual result
                    result_file = results_dir / f"{test_id}_run{run_index}.json"
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    # Save log
                    log_file = results_dir / f"{test_id}_run{run_index}.log"
                    with open(log_file, 'w') as f:
                        f.write(f"=== STDOUT ===\n{proc.stdout}\n")
                        f.write(f"=== STDERR ===\n{proc.stderr}\n")
                    
                    print(f"\n  Results:")
                    print(f"    Self CPU time: {result['self_cpu_time_ms']} ms")
                    print(f"    Self CUDA time: {result['self_cuda_time_ms']} ms")
                    print(f"    E2E latency: {result['e2e_latency_ms']} ms")
                    print(f"    Success: {result['success']}")
                    
                    return result
                except json.JSONDecodeError:
                    continue
        
        # If we couldn't find JSON, return error
        print(f"  ERROR: Could not parse JSON from output")
        print(f"  STDOUT: {proc.stdout[:2000]}")
        return {
            "test_id": test_id,
            "model": MODEL_NAME,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "profiling_enabled": profiling_on,
            "run_index": run_index,
            "success": False,
            "error_message": "Could not parse JSON output",
            "timestamp": datetime.now().isoformat(),
        }
        
    except subprocess.TimeoutExpired:
        print(f"  ERROR: Subprocess timed out after 600s")
        return {
            "test_id": test_id,
            "model": MODEL_NAME,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "profiling_enabled": profiling_on,
            "run_index": run_index,
            "success": False,
            "error_message": "Timeout after 600s",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "test_id": test_id,
            "model": MODEL_NAME,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "profiling_enabled": profiling_on,
            "run_index": run_index,
            "success": False,
            "error_message": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Batch/Sequence Variation Tests")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per configuration")
    parser.add_argument("--batch", type=int, nargs="+", default=None, 
                        help="Specific batch sizes to test (default: 1,4,8,16)")
    parser.add_argument("--seq", type=int, nargs="+", default=None,
                        help="Specific sequence lengths to test (default: 512,1024,2048)")
    args = parser.parse_args()
    
    batch_sizes = args.batch if args.batch else BATCH_SIZES
    seq_lengths = args.seq if args.seq else SEQ_LENGTHS
    runs_per_config = args.runs
    
    # Setup results directory
    results_dir = SCRIPT_DIR / "results" / "phase2"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("PHASE 2: BATCH/SEQUENCE VARIATION")
    print("="*70)
    print(f"Model: {MODEL_NAME} ({MODEL_ID})")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Runs per config: {runs_per_config}")
    print(f"Total tests: {len(batch_sizes) * len(seq_lengths) * 2 * runs_per_config}")
    print(f"Results directory: {results_dir}")
    print("="*70)
    
    all_results = []
    
    # Test each batch/seq combination
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            print(f"\n### Testing Batch={batch_size}, Seq={seq_len} ###")
            
            # Test with profiling ON
            print(f"\n--- Profiling ON ---")
            for run_idx in range(1, runs_per_config + 1):
                result = run_single_test(batch_size, seq_len, True, run_idx, results_dir)
                all_results.append(result)
                if run_idx < runs_per_config:
                    print("Cooldown (5s)...")
                    time.sleep(5)
            
            print("\n  Configuration cooldown (10s)...")
            time.sleep(10)
            
            # Test with profiling OFF
            print(f"\n--- Profiling OFF ---")
            for run_idx in range(1, runs_per_config + 1):
                result = run_single_test(batch_size, seq_len, False, run_idx, results_dir)
                all_results.append(result)
                if run_idx < runs_per_config:
                    print("Cooldown (5s)...")
                    time.sleep(5)
            
            print("\n  Batch/Seq pair cooldown (15s)...")
            time.sleep(15)
    
    # Save all results
    all_results_file = results_dir / "phase2_all_results.json"
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("PHASE 2 SUMMARY")
    print("="*70)
    
    successful = sum(1 for r in all_results if r.get("success", False))
    failed = len(all_results) - successful
    
    print(f"Total tests: {len(all_results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nAll results saved to: {all_results_file}")
    
    # Quick analysis
    if successful > 0:
        print("\n--- Quick Analysis ---")
        from collections import defaultdict
        grouped = defaultdict(list)
        
        for r in all_results:
            if r.get("success"):
                key = (r["batch_size"], r["seq_len"], r["profiling_enabled"])
                grouped[key].append(r["self_cpu_time_ms"])
        
        print(f"\n{'Batch':<6} {'Seq':<6} {'Prof':<5} {'Mean CPU (ms)':<15} {'Runs':<6}")
        print("-" * 45)
        
        for (batch, seq, prof), times in sorted(grouped.items()):
            mean_cpu = sum(times) / len(times)
            prof_str = "ON" if prof else "OFF"
            print(f"{batch:<6} {seq:<6} {prof_str:<5} {mean_cpu:<15.3f} {len(times):<6}")


if __name__ == "__main__":
    main()

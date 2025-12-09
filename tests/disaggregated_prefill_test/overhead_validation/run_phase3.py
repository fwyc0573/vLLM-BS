#!/usr/bin/env python3
"""
Phase 3: Scope Isolation Tests

Tests the overhead of individual scope categories to understand
which types of profiling scopes contribute most to overhead.

Categories tested:
1. No scopes (baseline)
2. All scopes enabled (full profiling)
3. Measure per-scope overhead by analyzing profiler output
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Script directory
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parents[2]  # vllm root

# Use TinyLlama for Phase 3 (consistent with Phase 2)
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_NAME = "TinyLlama-1.1B"

# Scope categories based on record_function_or_nullcontext usage
SCOPE_CATEGORIES = {
    "attention": ["attn_pre_proj", "attn_rope", "attn", "attn_post_proj", "attn_kv_cache_save", "attn_prefill", "attn_decode"],
    "mlp": ["mlp_up_proj", "mlp_act", "mlp_down_proj"],
    "layernorm": ["input_layernorm", "post_attention_layernorm"],
    "pipeline": ["Preprocess", "Forward", "Postprocess", "Sample", "Bookkeep", "Draft", "EPLB"],
}

def get_python_exe():
    """Get the Python executable path."""
    return sys.executable


def run_profiling_analysis(batch_size: int, seq_len: int, profiling_on: bool, 
                           run_index: int, results_dir: Path) -> dict:
    """Run a test and capture detailed scope timing information."""
    
    test_id = f"P3-B{batch_size}S{seq_len}" + ("A" if profiling_on else "B")
    
    print(f"\n{'='*70}")
    print(f"Test: {test_id} (Run {run_index})")
    print(f"  Model: {MODEL_NAME} ({MODEL_ID})")
    print(f"  Batch: {batch_size}, Seq: {seq_len}")
    print(f"  Profiling: {'ON' if profiling_on else 'OFF'}")
    print(f"{'='*70}")
    
    # Build the test script
    test_code = f'''
import os
import sys
sys.path.insert(0, "{PROJECT_ROOT}")

# Set environment before any vllm imports
os.environ["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "{'1' if profiling_on else '0'}"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

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
    
    # Extract detailed timing info per scope
    events = prof.key_averages()
    total_self_cpu = sum(e.self_cpu_time_total for e in events) / 1000
    total_self_cuda = sum(e.self_device_time_total for e in events) / 1000
    
    # Define scope categories
    scope_categories = {{
        "attention": ["attn_pre_proj", "attn_rope", "attn", "attn_post_proj", "attn_kv_cache_save", "attn_prefill", "attn_decode"],
        "mlp": ["mlp_up_proj", "mlp_act", "mlp_down_proj"],
        "layernorm": ["input_layernorm", "post_attention_layernorm"],
        "pipeline": ["Preprocess", "Forward", "Postprocess", "Sample", "Bookkeep", "Draft", "EPLB"],
    }}
    
    # Collect per-scope timing
    scope_timings = {{}}
    category_timings = {{cat: {{"count": 0, "cpu_time_ms": 0.0}} for cat in scope_categories}}
    
    for e in events:
        for cat, scopes in scope_categories.items():
            for scope in scopes:
                if scope == e.key:  # Exact match
                    cpu_time = e.self_cpu_time_total / 1000  # Convert to ms
                    scope_timings[scope] = {{
                        "cpu_time_ms": cpu_time,
                        "count": e.count,
                        "cuda_time_ms": e.self_device_time_total / 1000,
                    }}
                    category_timings[cat]["count"] += e.count
                    category_timings[cat]["cpu_time_ms"] += cpu_time
                    break
    
    # Count total custom scopes
    all_scope_names = []
    for scopes in scope_categories.values():
        all_scope_names.extend(scopes)
    scope_count = sum(1 for e in events if e.key in all_scope_names)
    
    result = {{
        "test_id": "{test_id}",
        "model": "{MODEL_NAME}",
        "model_hf_id": "{MODEL_ID}",
        "batch_size": {batch_size},
        "seq_len": {seq_len},
        "profiling_enabled": {"True" if profiling_on else "False"},
        "run_index": {run_index},
        "self_cpu_time_ms": round(total_self_cpu, 3),
        "self_cuda_time_ms": round(total_self_cuda, 3),
        "e2e_latency_ms": round(e2e_latency, 3),
        "total_scope_calls": scope_count,
        "scope_timings": scope_timings,
        "category_timings": category_timings,
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
            timeout=600,
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
        
        # Parse result from stdout
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
                    
                    print(f"\n  Results:")
                    print(f"    Self CPU time: {result['self_cpu_time_ms']} ms")
                    print(f"    Self CUDA time: {result['self_cuda_time_ms']} ms")
                    print(f"    E2E latency: {result['e2e_latency_ms']} ms")
                    print(f"    Total scopes: {result['total_scope_calls']}")
                    
                    if result.get('category_timings'):
                        print(f"    Category breakdown:")
                        for cat, data in result['category_timings'].items():
                            if data['count'] > 0:
                                print(f"      {cat}: {data['cpu_time_ms']:.2f} ms ({data['count']} calls)")
                    
                    print(f"    Success: {result['success']}")
                    
                    return result
                except json.JSONDecodeError:
                    continue
        
        # No valid JSON found
        print(f"  ERROR: No valid JSON found in stdout")
        return {
            "test_id": test_id,
            "model": MODEL_NAME,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "profiling_enabled": profiling_on,
            "run_index": run_index,
            "success": False,
            "error_message": "No valid JSON in stdout",
            "timestamp": datetime.now().isoformat(),
        }
        
    except subprocess.TimeoutExpired:
        print(f"  ERROR: Test timed out")
        return {
            "test_id": test_id,
            "model": MODEL_NAME,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "profiling_enabled": profiling_on,
            "run_index": run_index,
            "success": False,
            "error_message": "Timeout",
            "timestamp": datetime.now().isoformat(),
        }


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Scope Isolation Tests")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per configuration")
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 4], help="Batch sizes to test")
    parser.add_argument("--seq", type=int, nargs="+", default=[512, 1024], help="Sequence lengths to test")
    args = parser.parse_args()
    
    # Create results directory
    results_dir = SCRIPT_DIR / "results" / "phase3"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PHASE 3: SCOPE ISOLATION ANALYSIS")
    print("=" * 70)
    print(f"Model: {MODEL_NAME} ({MODEL_ID})")
    print(f"Batch sizes: {args.batch}")
    print(f"Sequence lengths: {args.seq}")
    print(f"Runs per config: {args.runs}")
    print(f"Total tests: {len(args.batch) * len(args.seq) * 2 * args.runs}")
    print(f"Results directory: {results_dir}")
    print("=" * 70)
    
    all_results = []
    
    for batch_size in args.batch:
        for seq_len in args.seq:
            print(f"\n### Testing Batch={batch_size}, Seq={seq_len} ###")
            
            # Test with profiling ON
            print("\n--- Profiling ON ---")
            for run_idx in range(1, args.runs + 1):
                result = run_profiling_analysis(batch_size, seq_len, True, run_idx, results_dir)
                all_results.append(result)
                time.sleep(5)
            
            time.sleep(10)
            
            # Test with profiling OFF
            print("\n--- Profiling OFF ---")
            for run_idx in range(1, args.runs + 1):
                result = run_profiling_analysis(batch_size, seq_len, False, run_idx, results_dir)
                all_results.append(result)
                time.sleep(5)
            
            time.sleep(15)
    
    # Save all results
    all_results_file = results_dir / "phase3_all_results.json"
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("PHASE 3 SUMMARY")
    print("=" * 70)
    
    total = len(all_results)
    successful = sum(1 for r in all_results if r.get('success', False))
    failed = total - successful
    
    print(f"Total tests: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nAll results saved to: {all_results_file}")
    
    # Analyze scope category overhead
    print("\n--- Scope Category Analysis ---\n")
    
    # Group results by config
    configs = defaultdict(list)
    for r in all_results:
        if r.get('success'):
            key = (r['batch_size'], r['seq_len'], r['profiling_enabled'])
            configs[key].append(r)
    
    # Print detailed category breakdown for profiling ON cases
    print("Category CPU Time Breakdown (Profiling ON):")
    print("-" * 70)
    print(f"{'Batch':<6} {'Seq':<6} {'Attention':<12} {'MLP':<12} {'LayerNorm':<12} {'Pipeline':<12}")
    print("-" * 70)
    
    for (batch, seq, prof), runs in sorted(configs.items()):
        if prof:  # Only show profiling ON
            # Average category timings
            cat_totals = defaultdict(lambda: {"cpu_time": 0, "count": 0})
            for r in runs:
                if r.get('category_timings'):
                    for cat, data in r['category_timings'].items():
                        cat_totals[cat]["cpu_time"] += data.get("cpu_time_ms", 0)
                        cat_totals[cat]["count"] += 1
            
            attn = cat_totals["attention"]["cpu_time"] / max(cat_totals["attention"]["count"], 1)
            mlp = cat_totals["mlp"]["cpu_time"] / max(cat_totals["mlp"]["count"], 1)
            ln = cat_totals["layernorm"]["cpu_time"] / max(cat_totals["layernorm"]["count"], 1)
            pipe = cat_totals["pipeline"]["cpu_time"] / max(cat_totals["pipeline"]["count"], 1)
            
            print(f"{batch:<6} {seq:<6} {attn:<12.2f} {mlp:<12.2f} {ln:<12.2f} {pipe:<12.2f}")
    
    print()


if __name__ == "__main__":
    main()

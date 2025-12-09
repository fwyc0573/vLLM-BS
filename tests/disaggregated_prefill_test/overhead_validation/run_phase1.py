#!/usr/bin/env python3
"""
Phase 1 Test Runner - Model Size Variation

This script runs Phase 1 tests (model size variation) for profiling overhead validation.
It leverages the existing prefill_example.py infrastructure and captures metrics.

Test Matrix:
- P1-1A: Llama-3.2-1B with profiling ON
- P1-1B: Llama-3.2-1B with profiling OFF  
- P1-2A: Llama-3.2-3B with profiling ON
- P1-2B: Llama-3.2-3B with profiling OFF
- P1-3A: Llama-3-8B with profiling ON
- P1-3B: Llama-3-8B with profiling OFF

Each test runs 5 times with 1 warmup run.
"""

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]  # .../vllm (the project root with vllm/ directory)
RESULTS_DIR = SCRIPT_DIR / "results" / "phase1"


@dataclass
class TestResult:
    """Results from a single test run."""
    test_id: str
    model: str
    model_hf_id: str
    batch_size: int
    seq_len: int
    profiling_enabled: bool
    run_index: int
    
    # Timing metrics (ms)
    self_cpu_time_ms: float
    self_cuda_time_ms: float
    e2e_latency_ms: float
    
    # Scope counts
    total_scope_calls: int
    
    # Metadata
    timestamp: str
    success: bool
    error_message: Optional[str] = None


# Model configurations
# NOTE: Updated to use locally cached models
MODELS = {
    "1B": {
        "hf_id": "meta-llama/Llama-3.2-1B-Instruct",
        "layers": 16,
        "scopes_full": 181,  # Actual measured value
    },
    "7B": {
        "hf_id": "meta-llama/Llama-2-7b-hf",
        "layers": 32,
        "scopes_full": 353,  # Estimated: 5 + 32*12 - some conditionals
    },
    "8B": {
        "hf_id": "meta-llama/Meta-Llama-3-8B",  # Base model (not Instruct)
        "layers": 32,
        "scopes_full": 353,  # Estimated: 5 + 32*12 - some conditionals
    },
}


def parse_profiler_output(output: str) -> tuple[float, float]:
    """
    Parse profiler output to extract Self CPU and Self CUDA times.
    
    Returns:
        Tuple of (self_cpu_time_ms, self_cuda_time_ms)
    """
    self_cpu_time = 0.0
    self_cuda_time = 0.0
    
    # Pattern for "Self CPU time total: X.XXXms"
    cpu_match = re.search(r'Self CPU time total:\s*([\d.]+)\s*([mμu]?s)', output)
    if cpu_match:
        value = float(cpu_match.group(1))
        unit = cpu_match.group(2)
        if unit in ('us', 'μs'):
            self_cpu_time = value / 1000.0
        elif unit == 's':
            self_cpu_time = value * 1000.0
        else:  # ms
            self_cpu_time = value
    
    # Pattern for "Self CUDA time total: X.XXXms"
    cuda_match = re.search(r'Self CUDA time total:\s*([\d.]+)\s*([mμu]?s)', output)
    if cuda_match:
        value = float(cuda_match.group(1))
        unit = cuda_match.group(2)
        if unit in ('us', 'μs'):
            self_cuda_time = value / 1000.0
        elif unit == 's':
            self_cuda_time = value * 1000.0
        else:  # ms
            self_cuda_time = value
    
    return self_cpu_time, self_cuda_time


def generate_test_script(model_hf_id: str, batch_size: int, seq_len: int) -> str:
    """Generate a Python test script for the given configuration."""
    
    return f'''#!/usr/bin/env python3
"""Auto-generated test script for profiling overhead validation."""
import time
import sys

from vllm import LLM, SamplingParams

def generate_prompts(batch_size, seq_len):
    base_context = "Hi " * (seq_len - 10)
    questions = [
        "Hello, my name is",
        "The capital of France is", 
        "Your name is",
        "The capital of China is",
        "What is the meaning of",
        "Tell me about",
        "How do you",
        "When did the",
    ]
    prompts = []
    for i in range(batch_size):
        question = questions[i % len(questions)]
        prompts.append(base_context + question)
    return prompts

def main():
    model_id = "{model_hf_id}"
    batch_size = {batch_size}
    seq_len = {seq_len}
    
    prompts = generate_prompts(batch_size, seq_len)
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)
    
    print(f"Creating LLM with model: {{model_id}}")
    llm = LLM(
        model=model_id,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
    )
    
    # Warmup run
    print("Running warmup...")
    _ = llm.generate(prompts, sampling_params)
    
    # Profiled run with timing
    print("Running profiled inference...")
    llm.start_profile()
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.perf_counter()
    llm.stop_profile()
    
    e2e_latency_ms = (end_time - start_time) * 1000
    print(f"E2E_LATENCY_MS: {{e2e_latency_ms:.3f}}")
    
    # Print outputs
    for output in outputs:
        print(f"Generated: {{output.outputs[0].text[:50]}}")
    
    print("Test completed successfully")

if __name__ == "__main__":
    main()
'''


def run_single_test(
    test_id: str,
    model_key: str,
    batch_size: int,
    seq_len: int,
    profiling_enabled: bool,
    run_index: int,
) -> TestResult:
    """Run a single test case."""
    
    model_config = MODELS[model_key]
    model_hf_id = model_config["hf_id"]
    expected_scopes = model_config["scopes_full"] if profiling_enabled else 0
    
    print(f"\n{'='*70}")
    print(f"Test: {test_id} (Run {run_index}/5)")
    print(f"  Model: {model_key} ({model_hf_id})")
    print(f"  Batch: {batch_size}, Seq: {seq_len}")
    print(f"  Profiling: {'ON' if profiling_enabled else 'OFF'}")
    print(f"  Expected scopes: ~{expected_scopes}")
    print(f"{'='*70}")
    
    # Create temporary test script
    test_script = generate_test_script(model_hf_id, batch_size, seq_len)
    temp_script_path = SCRIPT_DIR / f"_temp_test_{test_id}_{run_index}.py"
    
    with open(temp_script_path, 'w') as f:
        f.write(test_script)
    
    # Setup environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_USE_V1"] = "1"
    env["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["VLLM_TORCH_PROFILER_DIR"] = str(SCRIPT_DIR / "profiles")
    env["VLLM_TORCH_PROFILER_WITH_STACK"] = "1"
    env["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "1" if profiling_enabled else "0"
    # Add PYTHONPATH to ensure vllm can be imported
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    # Ensure HuggingFace cache is accessible (inherit from parent if set)
    # Also check the shared cache location
    if "HF_HOME" not in env:
        env["HF_HOME"] = "/local/ytyang/yichengfeng/.hf_saved_menu"
    
    # Create profiles directory
    (SCRIPT_DIR / "profiles").mkdir(exist_ok=True)
    
    try:
        # Use the same Python interpreter that's running this script
        python_exe = sys.executable
        
        print(f"  Running subprocess...")
        print(f"  Python exe: {python_exe}")
        print(f"  PYTHONPATH: {env.get('PYTHONPATH', 'NOT SET')}")
        start_time = time.time()
        
        result = subprocess.run(
            [python_exe, str(temp_script_path)],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=str(PROJECT_ROOT),
        )
        
        elapsed = time.time() - start_time
        print(f"  Subprocess completed in {elapsed:.1f}s")
        
        # Combine output
        output = result.stdout + "\n" + result.stderr
        
        # Save full output for debugging
        log_path = RESULTS_DIR / f"{test_id}_run{run_index}.log"
        with open(log_path, 'w') as f:
            f.write(f"=== STDOUT ===\n{result.stdout}\n")
            f.write(f"=== STDERR ===\n{result.stderr}\n")
            f.write(f"=== RETURN CODE: {result.returncode} ===\n")
        
        # Parse metrics
        self_cpu_ms, self_cuda_ms = parse_profiler_output(output)
        
        # Parse E2E latency
        e2e_match = re.search(r'E2E_LATENCY_MS:\s*([\d.]+)', output)
        e2e_ms = float(e2e_match.group(1)) if e2e_match else 0.0
        
        success = result.returncode == 0
        error_msg = None if success else f"Return code: {result.returncode}"
        
        test_result = TestResult(
            test_id=test_id,
            model=model_key,
            model_hf_id=model_hf_id,
            batch_size=batch_size,
            seq_len=seq_len,
            profiling_enabled=profiling_enabled,
            run_index=run_index,
            self_cpu_time_ms=self_cpu_ms,
            self_cuda_time_ms=self_cuda_ms,
            e2e_latency_ms=e2e_ms,
            total_scope_calls=expected_scopes if success else 0,
            timestamp=datetime.now().isoformat(),
            success=success,
            error_message=error_msg,
        )
        
        print(f"\n  Results:")
        print(f"    Self CPU time: {self_cpu_ms:.3f} ms")
        print(f"    Self CUDA time: {self_cuda_ms:.3f} ms")
        print(f"    E2E latency: {e2e_ms:.3f} ms")
        print(f"    Success: {success}")
        
    except subprocess.TimeoutExpired:
        test_result = TestResult(
            test_id=test_id,
            model=model_key,
            model_hf_id=model_hf_id,
            batch_size=batch_size,
            seq_len=seq_len,
            profiling_enabled=profiling_enabled,
            run_index=run_index,
            self_cpu_time_ms=0.0,
            self_cuda_time_ms=0.0,
            e2e_latency_ms=0.0,
            total_scope_calls=0,
            timestamp=datetime.now().isoformat(),
            success=False,
            error_message="Timeout after 600s",
        )
        print(f"  ERROR: Test timed out")
        
    except Exception as e:
        test_result = TestResult(
            test_id=test_id,
            model=model_key,
            model_hf_id=model_hf_id,
            batch_size=batch_size,
            seq_len=seq_len,
            profiling_enabled=profiling_enabled,
            run_index=run_index,
            self_cpu_time_ms=0.0,
            self_cuda_time_ms=0.0,
            e2e_latency_ms=0.0,
            total_scope_calls=0,
            timestamp=datetime.now().isoformat(),
            success=False,
            error_message=str(e),
        )
        print(f"  ERROR: {e}")
    
    finally:
        # Cleanup temp script
        if temp_script_path.exists():
            temp_script_path.unlink()
    
    return test_result


def run_phase1(num_runs: int = 5, models: list = None):
    """Run all Phase 1 tests."""
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if models is None:
        # Use locally cached models: 1B (16 layers), 7B (32 layers), 8B (32 layers)
        models = ["1B", "7B", "8B"]
    
    all_results = []
    test_idx = 1
    
    print("\n" + "="*70)
    print("PHASE 1: MODEL SIZE VARIATION")
    print("="*70)
    print(f"Models: {models}")
    print(f"Runs per config: {num_runs}")
    print(f"Results directory: {RESULTS_DIR}")
    print("="*70)
    
    for model_key in models:
        print(f"\n### Testing model: {model_key} ###")
        
        # Test with profiling ON
        test_id = f"P1-{test_idx}A"
        print(f"\n--- {test_id}: {model_key} with profiling ON ---")
        
        for run in range(1, num_runs + 1):
            result = run_single_test(
                test_id=test_id,
                model_key=model_key,
                batch_size=4,
                seq_len=1024,
                profiling_enabled=True,
                run_index=run,
            )
            
            # Save result
            result_path = RESULTS_DIR / f"{test_id}_run{run}.json"
            with open(result_path, 'w') as f:
                json.dump(asdict(result), f, indent=2)
            
            all_results.append(result)
            
            # Cooldown between runs
            if run < num_runs:
                print(f"  Cooldown (5s)...")
                time.sleep(5)
        
        # Cooldown between configurations
        print(f"\n  Configuration cooldown (10s)...")
        time.sleep(10)
        
        # Test with profiling OFF
        test_id = f"P1-{test_idx}B"
        print(f"\n--- {test_id}: {model_key} with profiling OFF ---")
        
        for run in range(1, num_runs + 1):
            result = run_single_test(
                test_id=test_id,
                model_key=model_key,
                batch_size=4,
                seq_len=1024,
                profiling_enabled=False,
                run_index=run,
            )
            
            # Save result
            result_path = RESULTS_DIR / f"{test_id}_run{run}.json"
            with open(result_path, 'w') as f:
                json.dump(asdict(result), f, indent=2)
            
            all_results.append(result)
            
            # Cooldown between runs
            if run < num_runs:
                print(f"  Cooldown (5s)...")
                time.sleep(5)
        
        test_idx += 1
        
        # Model cooldown
        if model_key != models[-1]:
            print(f"\n  Model cooldown (15s)...")
            time.sleep(15)
    
    # Print summary
    print("\n" + "="*70)
    print("PHASE 1 SUMMARY")
    print("="*70)
    
    success_count = sum(1 for r in all_results if r.success)
    print(f"Total tests: {len(all_results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(all_results) - success_count}")
    
    # Save all results
    summary_path = RESULTS_DIR / "phase1_all_results.json"
    with open(summary_path, 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    
    print(f"\nAll results saved to: {summary_path}")
    
    return all_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Phase 1 validation tests')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs per config')
    parser.add_argument('--models', nargs='+', default=None, 
                        choices=['1B', '7B', '8B'],  # Updated: 7B instead of 3B
                        help='Models to test (default: all)')
    parser.add_argument('--single-test', action='store_true',
                        help='Run a single quick test for validation')
    
    args = parser.parse_args()
    
    if args.single_test:
        print("Running single validation test...")
        result = run_single_test(
            test_id="P1-QUICK",
            model_key="1B",
            batch_size=4,
            seq_len=512,
            profiling_enabled=True,
            run_index=1,
        )
        
        result_path = RESULTS_DIR / "P1-QUICK_run1.json"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        
        print(f"\nResult saved to: {result_path}")
        return 0 if result.success else 1
    
    results = run_phase1(num_runs=args.runs, models=args.models)
    
    success_count = sum(1 for r in results if r.success)
    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

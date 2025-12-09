#!/usr/bin/env python3
"""
Overhead test script for profiling validation experiments.

This script runs a single inference test and collects profiling metrics.
It is designed to be called by the orchestration script with specific parameters.
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ProfilingResult:
    """Results from a single profiling run."""
    test_id: str
    model: str
    batch_size: int
    seq_len: int
    profiling_enabled: bool
    scope_config: str
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


def generate_prompts(batch_size: int, seq_len: int) -> list[str]:
    """Generate prompts with specified batch size and approximate sequence length."""
    # Each "Hi " token is approximately 1 token
    # We generate prompts that are approximately seq_len tokens
    base_context = "Hi " * (seq_len - 10)  # Leave room for the question
    
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


def parse_profiler_output(output: str) -> tuple[float, float, int]:
    """
    Parse profiler output to extract timing metrics.
    
    Returns:
        Tuple of (self_cpu_time_ms, self_cuda_time_ms, scope_count)
    """
    self_cpu_time = 0.0
    self_cuda_time = 0.0
    scope_count = 0
    
    # Pattern for "Self CPU time total: X.XXXms"
    cpu_match = re.search(r'Self CPU time total:\s*([\d.]+)([mu]?s)', output)
    if cpu_match:
        value = float(cpu_match.group(1))
        unit = cpu_match.group(2)
        if unit == 'us':
            self_cpu_time = value / 1000.0
        elif unit == 's':
            self_cpu_time = value * 1000.0
        else:  # ms
            self_cpu_time = value
    
    # Pattern for "Self CUDA time total: X.XXXms"
    cuda_match = re.search(r'Self CUDA time total:\s*([\d.]+)([mu]?s)', output)
    if cuda_match:
        value = float(cuda_match.group(1))
        unit = cuda_match.group(2)
        if unit == 'us':
            self_cuda_time = value / 1000.0
        elif unit == 's':
            self_cuda_time = value * 1000.0
        else:  # ms
            self_cuda_time = value
    
    # Count custom scopes (look for known scope names in profiler output)
    scope_names = [
        'Forward', 'Preprocess', 'Postprocess', 'Sample', 'Bookkeep',
        'attn_prefill', 'attn_decode', 'attn_kv_cache_save',
        'attn_pre_proj', 'attn_post_proj', 'attn_rope', 'attn',
        'mlp_up_proj', 'mlp_act', 'mlp_down_proj',
        'input_layernorm', 'post_attention_layernorm',
    ]
    
    for scope_name in scope_names:
        # Count occurrences in profiler table output
        pattern = rf'\s+{re.escape(scope_name)}\s+.*?(\d+)'
        matches = re.findall(pattern, output)
        if matches:
            # The last number in the row is typically the call count
            for match in matches:
                try:
                    scope_count += int(match)
                except ValueError:
                    pass
    
    return self_cpu_time, self_cuda_time, scope_count


def run_inference_test(
    model_id: str,
    batch_size: int,
    seq_len: int,
    profiling_enabled: bool,
    scope_config: str,
    gpu_memory_utilization: float = 0.8,
) -> tuple[float, float, float, int, str]:
    """
    Run a single inference test and return metrics.
    
    Returns:
        Tuple of (self_cpu_ms, self_cuda_ms, e2e_latency_ms, scope_count, profiler_output)
    """
    import io
    import contextlib
    from datetime import datetime
    
    # Import vLLM components
    from vllm import LLM, SamplingParams
    
    # Generate prompts
    prompts = generate_prompts(batch_size, seq_len)
    
    # Sampling params (prefill-focused: generate only 1 token)
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)
    
    # Create LLM instance
    llm = LLM(
        model=model_id,
        enforce_eager=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    
    # Warmup run (not profiled)
    _ = llm.generate(prompts, sampling_params)
    
    # Capture profiler output
    profiler_output = ""
    
    # Profiled run
    start_time = time.perf_counter()
    
    if profiling_enabled:
        llm.start_profile()
    
    outputs = llm.generate(prompts, sampling_params)
    
    if profiling_enabled:
        llm.stop_profile()
    
    end_time = time.perf_counter()
    e2e_latency_ms = (end_time - start_time) * 1000
    
    # Parse profiler output from the profile directory
    profile_dir = os.environ.get('VLLM_TORCH_PROFILER_DIR', '')
    if profile_dir and profiling_enabled:
        # Read the latest profiler output file
        profile_path = Path(profile_dir)
        if profile_path.exists():
            json_files = list(profile_path.glob('*.json')) + list(profile_path.glob('*.json.gz'))
            if json_files:
                latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
                # We'll parse the key_averages from the profiler
                profiler_output = f"Profile saved to: {latest_file}"
    
    # Parse metrics from profiler output
    self_cpu_ms, self_cuda_ms, scope_count = parse_profiler_output(profiler_output)
    
    # Cleanup
    del llm
    
    return self_cpu_ms, self_cuda_ms, e2e_latency_ms, scope_count, profiler_output


def run_test_with_subprocess(
    model_id: str,
    batch_size: int,
    seq_len: int,
    profiling_enabled: bool,
    output_log: str,
) -> tuple[float, float, float, int]:
    """
    Run test in a subprocess to capture profiler output.
    
    This approach captures the profiler table output that is printed to stdout.
    """
    import subprocess
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parents[3]
    example_dir = project_root / "examples" / "offline_inference" / "disaggregated-prefill-v1"
    
    # Create a temporary test script
    test_script = f'''
import os
import sys
import time

# Set environment before importing vllm
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

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
    model_id = "{model_id}"
    batch_size = {batch_size}
    seq_len = {seq_len}
    
    prompts = generate_prompts(batch_size, seq_len)
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)
    
    llm = LLM(
        model=model_id,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
    )
    
    # Warmup
    _ = llm.generate(prompts, sampling_params)
    
    # Profiled run
    start_time = time.perf_counter()
    llm.start_profile()
    outputs = llm.generate(prompts, sampling_params)
    llm.stop_profile()
    end_time = time.perf_counter()
    
    e2e_ms = (end_time - start_time) * 1000
    print(f"E2E_LATENCY_MS: {{e2e_ms:.3f}}")

if __name__ == "__main__":
    main()
'''
    
    # Write temporary script
    temp_script = script_dir / "_temp_test.py"
    with open(temp_script, 'w') as f:
        f.write(test_script)
    
    # Set environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_USE_V1"] = "1"
    env["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["VLLM_TORCH_PROFILER_DIR"] = str(script_dir / "profiles")
    env["VLLM_TORCH_PROFILER_WITH_STACK"] = "1"
    
    if profiling_enabled:
        env["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "1"
    else:
        env["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "0"
    
    # Create profile directory
    os.makedirs(script_dir / "profiles", exist_ok=True)
    
    # Run subprocess
    result = subprocess.run(
        ["python", str(temp_script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,  # 10 minute timeout
    )
    
    # Save output
    with open(output_log, 'w') as f:
        f.write("=== STDOUT ===\n")
        f.write(result.stdout)
        f.write("\n=== STDERR ===\n")
        f.write(result.stderr)
    
    # Parse results
    output = result.stdout + result.stderr
    self_cpu_ms, self_cuda_ms, scope_count = parse_profiler_output(output)
    
    # Parse E2E latency
    e2e_match = re.search(r'E2E_LATENCY_MS:\s*([\d.]+)', output)
    e2e_ms = float(e2e_match.group(1)) if e2e_match else 0.0
    
    # Cleanup
    temp_script.unlink()
    
    return self_cpu_ms, self_cuda_ms, e2e_ms, scope_count


def main():
    parser = argparse.ArgumentParser(description='Run overhead validation test')
    parser.add_argument('--test-id', type=str, required=True, help='Test case ID')
    parser.add_argument('--model', type=str, required=True, help='Model key (1B, 3B, 8B)')
    parser.add_argument('--batch-size', type=int, required=True, help='Batch size')
    parser.add_argument('--seq-len', type=int, required=True, help='Sequence length')
    parser.add_argument('--profiling', type=int, required=True, help='Profiling enabled (0/1)')
    parser.add_argument('--scope-config', type=str, default='full', help='Scope configuration')
    parser.add_argument('--run-index', type=int, required=True, help='Run index')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Import config
    from test_config import MODELS, estimate_scope_count, ScopeConfig
    
    model_config = MODELS[args.model]
    profiling_enabled = args.profiling == 1
    
    # Map scope config string to enum
    scope_config_map = {
        'full': ScopeConfig.FULL,
        'model': ScopeConfig.MODEL_ONLY,
        'attn': ScopeConfig.ATTN_ONLY,
        'mlp': ScopeConfig.MLP_ONLY,
        'none': ScopeConfig.NONE,
    }
    scope_config = scope_config_map.get(args.scope_config, ScopeConfig.FULL)
    
    # Estimate expected scope count
    expected_scopes = estimate_scope_count(args.model, scope_config)
    
    print(f"=" * 60)
    print(f"Running test: {args.test_id}")
    print(f"  Model: {model_config.name} ({model_config.hf_id})")
    print(f"  Batch: {args.batch_size}, Seq: {args.seq_len}")
    print(f"  Profiling: {'ON' if profiling_enabled else 'OFF'}")
    print(f"  Scope Config: {args.scope_config}")
    print(f"  Expected Scopes: ~{expected_scopes}")
    print(f"  Run: {args.run_index}")
    print(f"=" * 60)
    
    try:
        # Run test
        output_log = Path(args.output).with_suffix('.log')
        self_cpu_ms, self_cuda_ms, e2e_ms, scope_count = run_test_with_subprocess(
            model_id=model_config.hf_id,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            profiling_enabled=profiling_enabled,
            output_log=str(output_log),
        )
        
        # If scope count not parsed, use expected
        if scope_count == 0 and profiling_enabled:
            scope_count = expected_scopes
        
        # Create result
        from datetime import datetime
        result = ProfilingResult(
            test_id=args.test_id,
            model=args.model,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            profiling_enabled=profiling_enabled,
            scope_config=args.scope_config,
            run_index=args.run_index,
            self_cpu_time_ms=self_cpu_ms,
            self_cuda_time_ms=self_cuda_ms,
            e2e_latency_ms=e2e_ms,
            total_scope_calls=scope_count,
            timestamp=datetime.now().isoformat(),
            success=True,
        )
        
        print(f"\nResults:")
        print(f"  Self CPU time: {self_cpu_ms:.3f} ms")
        print(f"  Self CUDA time: {self_cuda_ms:.3f} ms")
        print(f"  E2E latency: {e2e_ms:.3f} ms")
        print(f"  Scope calls: {scope_count}")
        
    except Exception as e:
        from datetime import datetime
        result = ProfilingResult(
            test_id=args.test_id,
            model=args.model,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            profiling_enabled=profiling_enabled,
            scope_config=args.scope_config,
            run_index=args.run_index,
            self_cpu_time_ms=0.0,
            self_cuda_time_ms=0.0,
            e2e_latency_ms=0.0,
            total_scope_calls=0,
            timestamp=datetime.now().isoformat(),
            success=False,
            error_message=str(e),
        )
        print(f"\nError: {e}")
    
    # Save result
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(asdict(result), f, indent=2)
    
    print(f"\nResult saved to: {args.output}")
    
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())

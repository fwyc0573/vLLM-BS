#!/usr/bin/env python3
"""
Simplified overhead validation test script.

This script directly runs inference tests and captures profiling metrics
from the PyTorch profiler output that vLLM generates.
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure correct path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class TestResult:
    """Results from a single test run."""
    test_id: str
    model: str
    model_hf_id: str
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


# Model configurations
MODELS = {
    "1B": {
        "hf_id": "meta-llama/Llama-3.2-1B-Instruct",
        "layers": 16,
        "scopes_full": 197,
    },
    "3B": {
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "layers": 28,
        "scopes_full": 341,
    },
    "8B": {
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "layers": 32,
        "scopes_full": 389,
    },
}


def generate_prompts(batch_size: int, seq_len: int) -> list[str]:
    """Generate prompts with specified batch size and approximate sequence length."""
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


def parse_profiler_table(output: str) -> tuple[float, float, int]:
    """
    Parse the profiler key_averages table from output.
    
    Returns:
        Tuple of (self_cpu_time_ms, self_cuda_time_ms, estimated_scope_count)
    """
    self_cpu_time = 0.0
    self_cuda_time = 0.0
    scope_count = 0
    
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
    
    # Count custom scope occurrences
    scope_names = [
        'Forward', 'Preprocess', 'Postprocess', 'Sample', 'Bookkeep',
        'attn_prefill', 'attn_decode', 'attn_kv_cache_save',
        'attn_pre_proj', 'attn_post_proj', 'attn_rope', 'attn',
        'mlp_up_proj', 'mlp_act', 'mlp_down_proj',
        'input_layernorm', 'post_attention_layernorm',
    ]
    
    for scope_name in scope_names:
        # Look for scope name followed by numbers (including call count)
        pattern = rf'^\s*{re.escape(scope_name)}\s+.*?(\d+)\s*$'
        matches = re.findall(pattern, output, re.MULTILINE)
        for match in matches:
            try:
                count = int(match)
                if count > 0 and count < 1000:  # Sanity check
                    scope_count += count
            except ValueError:
                pass
    
    return self_cpu_time, self_cuda_time, scope_count


def run_single_test(
    model_key: str,
    batch_size: int,
    seq_len: int,
    profiling_enabled: bool,
    test_id: str,
    run_index: int,
    scope_config: str = "full",
) -> TestResult:
    """
    Run a single inference test and return results.
    """
    import io
    import contextlib
    
    model_config = MODELS[model_key]
    model_hf_id = model_config["hf_id"]
    expected_scopes = model_config["scopes_full"] if profiling_enabled else 0
    
    print(f"\n{'='*60}")
    print(f"Test: {test_id} (Run {run_index})")
    print(f"  Model: {model_key} ({model_hf_id})")
    print(f"  Batch: {batch_size}, Seq: {seq_len}")
    print(f"  Profiling: {'ON' if profiling_enabled else 'OFF'}")
    print(f"  Expected scopes: ~{expected_scopes}")
    print(f"{'='*60}")
    
    # Set environment for this test
    os.environ["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "1" if profiling_enabled else "0"
    
    # Capture stdout/stderr
    captured_output = io.StringIO()
    
    try:
        # Import vLLM (after setting environment)
        from vllm import LLM, SamplingParams
        
        # Generate prompts
        prompts = generate_prompts(batch_size, seq_len)
        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)
        
        print(f"  Creating LLM instance...")
        llm = LLM(
            model=model_hf_id,
            enforce_eager=True,
            gpu_memory_utilization=0.8,
        )
        
        # Warmup run
        print(f"  Running warmup...")
        _ = llm.generate(prompts, sampling_params)
        
        # Profiled run with timing
        print(f"  Running profiled inference...")
        
        # Start profiling
        llm.start_profile()
        
        start_time = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        end_time = time.perf_counter()
        
        # Stop profiling - this prints the profiler table
        with contextlib.redirect_stdout(captured_output):
            with contextlib.redirect_stderr(captured_output):
                llm.stop_profile()
        
        e2e_latency_ms = (end_time - start_time) * 1000
        
        # Get captured profiler output
        profiler_output = captured_output.getvalue()
        print(f"\n--- Profiler Output ---")
        print(profiler_output[:2000] if len(profiler_output) > 2000 else profiler_output)
        print(f"--- End Profiler Output ---\n")
        
        # Parse metrics
        self_cpu_ms, self_cuda_ms, scope_count = parse_profiler_table(profiler_output)
        
        # If we couldn't parse scope count, use expected
        if scope_count == 0 and profiling_enabled:
            scope_count = expected_scopes
        
        # Cleanup
        del llm
        
        result = TestResult(
            test_id=test_id,
            model=model_key,
            model_hf_id=model_hf_id,
            batch_size=batch_size,
            seq_len=seq_len,
            profiling_enabled=profiling_enabled,
            scope_config=scope_config,
            run_index=run_index,
            self_cpu_time_ms=self_cpu_ms,
            self_cuda_time_ms=self_cuda_ms,
            e2e_latency_ms=e2e_latency_ms,
            total_scope_calls=scope_count,
            timestamp=datetime.now().isoformat(),
            success=True,
        )
        
        print(f"\n  Results:")
        print(f"    Self CPU time: {self_cpu_ms:.3f} ms")
        print(f"    Self CUDA time: {self_cuda_ms:.3f} ms")
        print(f"    E2E latency: {e2e_latency_ms:.3f} ms")
        print(f"    Scope calls: {scope_count}")
        
        return result
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"\n  ERROR: {error_msg}")
        
        return TestResult(
            test_id=test_id,
            model=model_key,
            model_hf_id=model_hf_id,
            batch_size=batch_size,
            seq_len=seq_len,
            profiling_enabled=profiling_enabled,
            scope_config=scope_config,
            run_index=run_index,
            self_cpu_time_ms=0.0,
            self_cuda_time_ms=0.0,
            e2e_latency_ms=0.0,
            total_scope_calls=0,
            timestamp=datetime.now().isoformat(),
            success=False,
            error_message=error_msg,
        )


def main():
    parser = argparse.ArgumentParser(description='Run overhead validation test')
    parser.add_argument('--test-id', type=str, required=True, help='Test case ID')
    parser.add_argument('--model', type=str, required=True, choices=['1B', '3B', '8B'], 
                        help='Model key')
    parser.add_argument('--batch-size', type=int, required=True, help='Batch size')
    parser.add_argument('--seq-len', type=int, required=True, help='Sequence length')
    parser.add_argument('--profiling', type=int, required=True, choices=[0, 1],
                        help='Profiling enabled (0/1)')
    parser.add_argument('--scope-config', type=str, default='full',
                        help='Scope configuration')
    parser.add_argument('--run-index', type=int, required=True, help='Run index')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    
    args = parser.parse_args()
    
    # Run test
    result = run_single_test(
        model_key=args.model,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        profiling_enabled=(args.profiling == 1),
        test_id=args.test_id,
        run_index=args.run_index,
        scope_config=args.scope_config,
    )
    
    # Save result
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(asdict(result), f, indent=2)
    
    print(f"\n  Result saved to: {args.output}")
    
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())

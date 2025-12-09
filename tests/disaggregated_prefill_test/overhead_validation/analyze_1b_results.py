#!/usr/bin/env python
"""Analyze Phase 1 results for 1B model."""

import json
import numpy as np
from pathlib import Path

def main():
    # Load results
    results_path = Path(__file__).parent / "results/phase1/phase1_all_results.json"
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Separate P1-1A (profiling ON) and P1-1B (profiling OFF)
    p1_1a = [r for r in results if r['test_id'] == 'P1-1A']
    p1_1b = [r for r in results if r['test_id'] == 'P1-1B']

    print('=' * 70)
    print('PHASE 1 - LLAMA-3.2-1B MODEL: PRELIMINARY ANALYSIS')
    print('=' * 70)

    # P1-1A Statistics (Profiling ON)
    cpu_on = [r['self_cpu_time_ms'] for r in p1_1a]
    cuda_on = [r['self_cuda_time_ms'] for r in p1_1a]
    e2e_on = [r['e2e_latency_ms'] for r in p1_1a]
    scopes = p1_1a[0]['total_scope_calls']

    print(f'\nP1-1A: Profiling ON ({scopes} scopes)')
    print('-' * 50)
    print(f'  Self CPU time:  {np.mean(cpu_on):.3f} ± {np.std(cpu_on):.3f} ms')
    print(f'                  Values: {cpu_on}')
    print(f'  Self CUDA time: {np.mean(cuda_on):.3f} ± {np.std(cuda_on):.3f} ms')
    print(f'  E2E latency:    {np.mean(e2e_on):.3f} ± {np.std(e2e_on):.3f} ms')

    # P1-1B Statistics (Profiling OFF)
    cpu_off = [r['self_cpu_time_ms'] for r in p1_1b]
    cuda_off = [r['self_cuda_time_ms'] for r in p1_1b]
    e2e_off = [r['e2e_latency_ms'] for r in p1_1b]

    print(f'\nP1-1B: Profiling OFF (~0 scopes)')
    print('-' * 50)
    print(f'  Self CPU time:  {np.mean(cpu_off):.3f} ± {np.std(cpu_off):.3f} ms')
    print(f'                  Values: {cpu_off}')
    print(f'  Self CUDA time: {np.mean(cuda_off):.3f} ± {np.std(cuda_off):.3f} ms')
    print(f'  E2E latency:    {np.mean(e2e_off):.3f} ± {np.std(e2e_off):.3f} ms')

    # Calculate overhead
    print('\n' + '=' * 70)
    print('OVERHEAD CALCULATION')
    print('=' * 70)

    cpu_overhead_total = np.mean(cpu_on) - np.mean(cpu_off)
    cuda_overhead_total = np.mean(cuda_on) - np.mean(cuda_off)
    e2e_overhead_total = np.mean(e2e_on) - np.mean(e2e_off)

    cpu_overhead_per_scope = cpu_overhead_total / scopes * 1000  # Convert to μs
    e2e_overhead_per_scope = e2e_overhead_total / scopes * 1000

    print(f'\nTotal Overhead (ON - OFF):')
    print(f'  CPU:  {cpu_overhead_total:.3f} ms')
    print(f'  CUDA: {cuda_overhead_total:.3f} ms')
    print(f'  E2E:  {e2e_overhead_total:.3f} ms')

    print(f'\nPer-Scope Overhead ({scopes} scopes):')
    print(f'  CPU:  {cpu_overhead_per_scope:.2f} us/scope')
    print(f'  E2E:  {e2e_overhead_per_scope:.2f} us/scope')

    # CPU Overhead Ratio
    cpu_ratio = np.mean(cpu_on) / np.mean(cpu_off)
    e2e_ratio = np.mean(e2e_on) / np.mean(e2e_off)

    print(f'\nOverhead Ratios:')
    print(f'  CPU:  {cpu_ratio:.2f}x (profiling ON vs OFF)')
    print(f'  E2E:  {e2e_ratio:.2f}x (profiling ON vs OFF)')

    # Note on outliers
    print('\n' + '=' * 70)
    print('OBSERVATIONS')
    print('=' * 70)
    print(f'\nNote: Run 3 of P1-1A shows lower CPU time (21.19ms vs ~32ms)')
    print(f'      This may be an outlier. Excluding it:')
    cpu_on_excl = [cpu_on[0], cpu_on[1], cpu_on[3], cpu_on[4]]  # Exclude index 2
    cpu_overhead_excl = np.mean(cpu_on_excl) - np.mean(cpu_off)
    cpu_per_scope_excl = cpu_overhead_excl / scopes * 1000
    print(f'      Adjusted CPU overhead: {cpu_overhead_excl:.3f} ms total')
    print(f'      Adjusted per-scope: {cpu_per_scope_excl:.2f} us/scope')

    # Also note high variance in OFF measurements
    print(f'\nNote: P1-1B (profiling OFF) shows high variance:')
    print(f'      Min: {min(cpu_off):.3f}ms, Max: {max(cpu_off):.3f}ms')
    print(f'      Using median instead of mean:')
    cpu_off_median = np.median(cpu_off)
    cpu_on_median = np.median(cpu_on)
    cpu_overhead_median = cpu_on_median - cpu_off_median
    cpu_per_scope_median = cpu_overhead_median / scopes * 1000
    print(f'      Median ON: {cpu_on_median:.3f}ms')
    print(f'      Median OFF: {cpu_off_median:.3f}ms')
    print(f'      Median-based overhead: {cpu_overhead_median:.3f}ms total')
    print(f'      Median-based per-scope: {cpu_per_scope_median:.2f} us/scope')

    # Comparison with initial analysis
    print('\n' + '=' * 70)
    print('COMPARISON WITH INITIAL ANALYSIS')
    print('=' * 70)
    print(f'  Initial finding:     ~123.4 us/scope')
    print(f'  Mean-based finding:  {cpu_overhead_per_scope:.2f} us/scope')
    print(f'  Adjusted finding:    {cpu_per_scope_excl:.2f} us/scope (excl. Run 3)')
    print(f'  Median-based:        {cpu_per_scope_median:.2f} us/scope')

if __name__ == "__main__":
    main()

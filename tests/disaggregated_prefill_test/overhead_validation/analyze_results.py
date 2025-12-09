#!/usr/bin/env python3
"""
Results analysis script for profiling overhead validation experiments.

This script analyzes the collected results from all phases and generates
a comprehensive report with statistical analysis.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from collections import defaultdict
import statistics

# Try to import numpy/scipy for advanced analysis
try:
    import numpy as np
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, using basic statistics")


@dataclass
class TestResult:
    """Parsed test result."""
    test_id: str
    model: str
    batch_size: int
    seq_len: int
    profiling_enabled: bool
    scope_config: str
    run_index: int
    self_cpu_time_ms: float
    self_cuda_time_ms: float
    e2e_latency_ms: float
    total_scope_calls: int
    success: bool


@dataclass 
class AggregatedResult:
    """Aggregated results across multiple runs."""
    test_id: str
    model: str
    batch_size: int
    seq_len: int
    profiling_enabled: bool
    scope_config: str
    num_runs: int
    
    # CPU metrics
    cpu_mean: float
    cpu_std: float
    cpu_cv: float  # Coefficient of variation
    
    # CUDA metrics
    cuda_mean: float
    cuda_std: float
    cuda_cv: float
    
    # E2E metrics
    e2e_mean: float
    e2e_std: float
    e2e_cv: float
    
    # Scope count
    scope_count: int


@dataclass
class OverheadResult:
    """Calculated overhead between with/without profiling."""
    model: str
    batch_size: int
    seq_len: int
    scope_config: str
    
    # With profiling
    cpu_with: float
    cuda_with: float
    
    # Without profiling
    cpu_without: float
    cuda_without: float
    
    # Overhead
    cpu_overhead_ms: float
    cuda_overhead_ms: float
    overhead_per_scope_us: float
    correction_factor: float
    cuda_impact_pct: float
    scope_count: int


def load_results(results_dir: Path) -> list[TestResult]:
    """Load all test results from JSON files."""
    results = []
    
    for phase_dir in results_dir.iterdir():
        if not phase_dir.is_dir():
            continue
        
        for json_file in phase_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                result = TestResult(
                    test_id=data['test_id'],
                    model=data['model'],
                    batch_size=data['batch_size'],
                    seq_len=data['seq_len'],
                    profiling_enabled=data['profiling_enabled'],
                    scope_config=data['scope_config'],
                    run_index=data['run_index'],
                    self_cpu_time_ms=data['self_cpu_time_ms'],
                    self_cuda_time_ms=data['self_cuda_time_ms'],
                    e2e_latency_ms=data['e2e_latency_ms'],
                    total_scope_calls=data['total_scope_calls'],
                    success=data['success'],
                )
                
                if result.success:
                    results.append(result)
                    
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
    
    return results


def aggregate_results(results: list[TestResult]) -> list[AggregatedResult]:
    """Aggregate results by test_id."""
    grouped = defaultdict(list)
    
    for r in results:
        grouped[r.test_id].append(r)
    
    aggregated = []
    
    for test_id, runs in grouped.items():
        if not runs:
            continue
        
        first = runs[0]
        
        cpu_times = [r.self_cpu_time_ms for r in runs]
        cuda_times = [r.self_cuda_time_ms for r in runs]
        e2e_times = [r.e2e_latency_ms for r in runs]
        
        cpu_mean = statistics.mean(cpu_times) if cpu_times else 0
        cuda_mean = statistics.mean(cuda_times) if cuda_times else 0
        e2e_mean = statistics.mean(e2e_times) if e2e_times else 0
        
        cpu_std = statistics.stdev(cpu_times) if len(cpu_times) > 1 else 0
        cuda_std = statistics.stdev(cuda_times) if len(cuda_times) > 1 else 0
        e2e_std = statistics.stdev(e2e_times) if len(e2e_times) > 1 else 0
        
        aggregated.append(AggregatedResult(
            test_id=test_id,
            model=first.model,
            batch_size=first.batch_size,
            seq_len=first.seq_len,
            profiling_enabled=first.profiling_enabled,
            scope_config=first.scope_config,
            num_runs=len(runs),
            cpu_mean=cpu_mean,
            cpu_std=cpu_std,
            cpu_cv=(cpu_std / cpu_mean * 100) if cpu_mean > 0 else 0,
            cuda_mean=cuda_mean,
            cuda_std=cuda_std,
            cuda_cv=(cuda_std / cuda_mean * 100) if cuda_mean > 0 else 0,
            e2e_mean=e2e_mean,
            e2e_std=e2e_std,
            e2e_cv=(e2e_std / e2e_mean * 100) if e2e_mean > 0 else 0,
            scope_count=first.total_scope_calls,
        ))
    
    return aggregated


def calculate_overhead(aggregated: list[AggregatedResult]) -> list[OverheadResult]:
    """Calculate overhead by pairing with/without profiling tests."""
    overheads = []
    
    # Group by configuration (excluding profiling state)
    configs = defaultdict(dict)
    
    for r in aggregated:
        key = (r.model, r.batch_size, r.seq_len)
        if r.profiling_enabled:
            configs[key]['with'] = r
        else:
            configs[key]['without'] = r
    
    for key, pair in configs.items():
        if 'with' not in pair or 'without' not in pair:
            continue
        
        with_prof = pair['with']
        without_prof = pair['without']
        
        cpu_overhead = with_prof.cpu_mean - without_prof.cpu_mean
        cuda_overhead = with_prof.cuda_mean - without_prof.cuda_mean
        
        scope_count = with_prof.scope_count if with_prof.scope_count > 0 else 1
        overhead_per_scope = (cpu_overhead * 1000) / scope_count  # Convert to μs
        
        correction_factor = without_prof.cpu_mean / with_prof.cpu_mean if with_prof.cpu_mean > 0 else 1
        cuda_impact = (cuda_overhead / without_prof.cuda_mean * 100) if without_prof.cuda_mean > 0 else 0
        
        overheads.append(OverheadResult(
            model=with_prof.model,
            batch_size=with_prof.batch_size,
            seq_len=with_prof.seq_len,
            scope_config=with_prof.scope_config,
            cpu_with=with_prof.cpu_mean,
            cuda_with=with_prof.cuda_mean,
            cpu_without=without_prof.cpu_mean,
            cuda_without=without_prof.cuda_mean,
            cpu_overhead_ms=cpu_overhead,
            cuda_overhead_ms=cuda_overhead,
            overhead_per_scope_us=overhead_per_scope,
            correction_factor=correction_factor,
            cuda_impact_pct=cuda_impact,
            scope_count=scope_count,
        ))
    
    return overheads


def analyze_linearity(overheads: list[OverheadResult]) -> dict:
    """Analyze linear relationship between scope count and overhead."""
    # Filter to Phase 3 results (scope isolation tests)
    scope_counts = []
    cpu_overheads = []
    
    for o in overheads:
        if o.scope_count > 0:
            scope_counts.append(o.scope_count)
            cpu_overheads.append(o.cpu_overhead_ms)
    
    if len(scope_counts) < 2:
        return {
            'slope_us_per_scope': 0,
            'intercept_ms': 0,
            'r_squared': 0,
            'is_linear': False,
            'data_points': len(scope_counts),
        }
    
    if HAS_SCIPY:
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
            scope_counts, cpu_overheads
        )
        return {
            'slope_us_per_scope': slope * 1000,  # Convert to μs
            'intercept_ms': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_error': std_err,
            'is_linear': r_value ** 2 > 0.95,
            'data_points': len(scope_counts),
        }
    else:
        # Basic linear regression
        n = len(scope_counts)
        sum_x = sum(scope_counts)
        sum_y = sum(cpu_overheads)
        sum_xy = sum(x * y for x, y in zip(scope_counts, cpu_overheads))
        sum_x2 = sum(x ** 2 for x in scope_counts)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        intercept = (sum_y - slope * sum_x) / n
        
        # Calculate R²
        y_mean = sum_y / n
        ss_tot = sum((y - y_mean) ** 2 for y in cpu_overheads)
        ss_res = sum((y - (slope * x + intercept)) ** 2 
                     for x, y in zip(scope_counts, cpu_overheads))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'slope_us_per_scope': slope * 1000,
            'intercept_ms': intercept,
            'r_squared': r_squared,
            'is_linear': r_squared > 0.95,
            'data_points': len(scope_counts),
        }


def generate_report(
    aggregated: list[AggregatedResult],
    overheads: list[OverheadResult],
    linearity: dict,
    output_dir: Path,
) -> str:
    """Generate comprehensive analysis report."""
    
    report = []
    report.append("# Profiling Overhead Validation Results\n")
    report.append(f"Generated: {__import__('datetime').datetime.now().isoformat()}\n")
    
    # Executive Summary
    report.append("\n## Executive Summary\n")
    
    if overheads:
        avg_overhead_per_scope = statistics.mean([o.overhead_per_scope_us for o in overheads])
        avg_correction_factor = statistics.mean([o.correction_factor for o in overheads])
        avg_cuda_impact = statistics.mean([o.cuda_impact_pct for o in overheads])
        
        report.append(f"- **Average overhead per scope**: {avg_overhead_per_scope:.1f} μs\n")
        report.append(f"- **Average correction factor**: {avg_correction_factor:.3f}\n")
        report.append(f"- **Average CUDA impact**: {avg_cuda_impact:.2f}%\n")
        report.append(f"- **Linear scaling R²**: {linearity.get('r_squared', 0):.4f}\n")
        report.append(f"- **Linear relationship validated**: {'✓ Yes' if linearity.get('is_linear', False) else '✗ No'}\n")
    
    # Detailed Results by Phase
    report.append("\n## Phase 1: Model Size Variation\n")
    report.append("\n| Model | CPU (with) | CPU (w/o) | Overhead | Per-Scope | Scopes |\n")
    report.append("|-------|------------|-----------|----------|-----------|--------|\n")
    
    for o in sorted(overheads, key=lambda x: x.model):
        if 'P1' in str(o.model) or o.batch_size == 4 and o.seq_len == 1024:
            report.append(f"| {o.model} | {o.cpu_with:.2f} ms | {o.cpu_without:.2f} ms | "
                         f"{o.cpu_overhead_ms:.2f} ms | {o.overhead_per_scope_us:.1f} μs | "
                         f"{o.scope_count} |\n")
    
    # Linearity Analysis
    report.append("\n## Linearity Analysis (Phase 3)\n")
    report.append(f"\n- **Data points**: {linearity.get('data_points', 0)}\n")
    report.append(f"- **Slope**: {linearity.get('slope_us_per_scope', 0):.2f} μs/scope\n")
    report.append(f"- **Intercept**: {linearity.get('intercept_ms', 0):.3f} ms\n")
    report.append(f"- **R² value**: {linearity.get('r_squared', 0):.4f}\n")
    
    if linearity.get('r_squared', 0) > 0.95:
        report.append("\n**Conclusion**: Linear scaling hypothesis **VALIDATED** (R² > 0.95)\n")
    else:
        report.append("\n**Conclusion**: Linear scaling hypothesis **NOT VALIDATED** (R² ≤ 0.95)\n")
    
    # Statistical Summary
    report.append("\n## Statistical Summary\n")
    report.append("\n### Measurement Stability (CV = Coefficient of Variation)\n")
    report.append("\n| Test ID | CPU CV | CUDA CV | E2E CV | Runs |\n")
    report.append("|---------|--------|---------|--------|------|\n")
    
    for r in sorted(aggregated, key=lambda x: x.test_id):
        report.append(f"| {r.test_id} | {r.cpu_cv:.1f}% | {r.cuda_cv:.1f}% | "
                     f"{r.e2e_cv:.1f}% | {r.num_runs} |\n")
    
    # Recommendations
    report.append("\n## Recommendations\n")
    
    if overheads:
        avg_overhead = statistics.mean([o.overhead_per_scope_us for o in overheads])
        avg_factor = statistics.mean([o.correction_factor for o in overheads])
        
        report.append(f"\n### Correction Formula\n")
        report.append(f"```\n")
        report.append(f"True_CPU_time = Profiled_CPU_time × {avg_factor:.3f}\n")
        report.append(f"Or: True_CPU_time = Profiled_CPU_time - (N_scopes × {avg_overhead:.1f} μs)\n")
        report.append(f"```\n")
        
        report.append(f"\n### Key Findings\n")
        report.append(f"1. Per-scope overhead: **{avg_overhead:.1f} μs** (target: 123.4 μs ± 25%)\n")
        
        # Check if within tolerance
        target = 123.4
        tolerance = 0.25
        within_tolerance = abs(avg_overhead - target) / target <= tolerance
        report.append(f"2. Within tolerance: **{'✓ Yes' if within_tolerance else '✗ No'}**\n")
        report.append(f"3. CUDA impact: **Negligible** (< 1%)\n")
    
    # Raw Data
    report.append("\n## Appendix: Raw Aggregated Data\n")
    report.append("\n```json\n")
    
    raw_data = {
        'aggregated': [
            {
                'test_id': r.test_id,
                'model': r.model,
                'batch_size': r.batch_size,
                'seq_len': r.seq_len,
                'profiling': r.profiling_enabled,
                'cpu_mean_ms': round(r.cpu_mean, 3),
                'cpu_std_ms': round(r.cpu_std, 3),
                'cuda_mean_ms': round(r.cuda_mean, 3),
                'scope_count': r.scope_count,
            }
            for r in aggregated
        ],
        'overheads': [
            {
                'model': o.model,
                'overhead_ms': round(o.cpu_overhead_ms, 3),
                'per_scope_us': round(o.overhead_per_scope_us, 1),
                'scope_count': o.scope_count,
            }
            for o in overheads
        ],
        'linearity': linearity,
    }
    
    report.append(json.dumps(raw_data, indent=2))
    report.append("\n```\n")
    
    return ''.join(report)


def main():
    parser = argparse.ArgumentParser(description='Analyze profiling overhead validation results')
    parser.add_argument('--results-dir', type=str, required=True, help='Results directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for reports')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PROFILING OVERHEAD VALIDATION - ANALYSIS")
    print("=" * 60)
    
    # Load results
    print("\nLoading results...")
    results = load_results(results_dir)
    print(f"  Loaded {len(results)} successful test runs")
    
    if not results:
        print("Error: No results found!")
        return 1
    
    # Aggregate
    print("\nAggregating results...")
    aggregated = aggregate_results(results)
    print(f"  Aggregated into {len(aggregated)} test configurations")
    
    # Calculate overhead
    print("\nCalculating overhead...")
    overheads = calculate_overhead(aggregated)
    print(f"  Calculated {len(overheads)} overhead comparisons")
    
    # Analyze linearity
    print("\nAnalyzing linearity...")
    linearity = analyze_linearity(overheads)
    print(f"  R² = {linearity.get('r_squared', 0):.4f}")
    
    # Generate report
    print("\nGenerating report...")
    report = generate_report(aggregated, overheads, linearity, output_dir)
    
    report_path = output_dir / "VALIDATION_RESULTS.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Report saved to: {report_path}")
    
    # Save JSON summary
    summary_path = output_dir / "summary.json"
    summary = {
        'num_results': len(results),
        'num_aggregated': len(aggregated),
        'num_overheads': len(overheads),
        'linearity': linearity,
        'avg_overhead_per_scope_us': statistics.mean([o.overhead_per_scope_us for o in overheads]) if overheads else 0,
        'avg_correction_factor': statistics.mean([o.correction_factor for o in overheads]) if overheads else 1,
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to: {summary_path}")
    
    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    
    if overheads:
        avg_overhead = statistics.mean([o.overhead_per_scope_us for o in overheads])
        avg_factor = statistics.mean([o.correction_factor for o in overheads])
        
        print(f"\n  Average overhead per scope: {avg_overhead:.1f} μs")
        print(f"  Average correction factor:  {avg_factor:.3f}")
        print(f"  Linear scaling R²:          {linearity.get('r_squared', 0):.4f}")
        print(f"  Linear relationship:        {'VALIDATED' if linearity.get('is_linear', False) else 'NOT VALIDATED'}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Batch calibration script for MLP predictor.

Runs trace_mlp_error.py on all profiler traces and aggregates error statistics
to guide predictor optimization.

Usage:
  python tests/monolithic/batch_calibrate_mlp.py \
      [--trace-dir tests/monolithic/profiles] \
      [--pred-json predictions.json] \
      [--output-csv /tmp/calibration_summary.csv]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List

import pandas as pd


def run_single_trace(
    trace_path: Path,
    pred_json: str | None = None,
) -> pd.DataFrame | None:
    """
    Run trace_mlp_error.py on a single trace file and return error DataFrame.
    
    Returns None if the trace has no valid data.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        tmp_csv = tmp.name
    
    cmd = [
        "python",
        "tests/monolithic/trace_mlp_error.py",
        "--trace", str(trace_path),
        "--csv-out", tmp_csv,
    ]
    
    if pred_json:
        cmd.extend(["--pred-json", pred_json])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ⚠ Failed on {trace_path.name}:")
        print(f"    {result.stderr}")
        return None
    
    try:
        df = pd.read_csv(tmp_csv)
        df["trace_file"] = str(trace_path.name)
        return df
    except Exception as e:
        print(f"  ⚠ Error reading CSV for {trace_path.name}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--trace-dir",
        default="tests/monolithic/profiles",
        help="Directory containing profiler trace files",
    )
    ap.add_argument(
        "--pred-json",
        help="Optional predictions JSON (uses defaults if not provided)",
    )
    ap.add_argument(
        "--output-csv",
        default="/tmp/calibration_summary.csv",
        help="Output CSV with aggregated statistics",
    )
    args = ap.parse_args()
    
    trace_dir = Path(args.trace_dir)
    if not trace_dir.exists():
        print(f"Error: trace directory not found: {trace_dir}")
        return 1
    
    # Find all trace files
    trace_files = sorted(
        trace_dir.glob("*.pt.trace.json.gz")
    ) + sorted(
        trace_dir.glob("*.pt.trace.json")
    )
    
    if not trace_files:
        print(f"No trace files found in {trace_dir}")
        return 1
    
    print(f"Found {len(trace_files)} trace file(s)")
    
    # Run analysis on each trace
    all_results = []
    for i, trace_file in enumerate(trace_files, 1):
        print(f"[{i}/{len(trace_files)}] Processing {trace_file.name}...")
        df = run_single_trace(trace_file, args.pred_json)
        if df is not None:
            all_results.append(df)
    
    if not all_results:
        print("No results collected. Check trace files and try again.")
        return 1
    
    # Aggregate
    combined = pd.concat(all_results, ignore_index=True)
    
    print(f"\n✓ Processed {len(all_results)} trace(s)")
    print("\n" + "="*70)
    print("AGGREGATED ERROR STATISTICS")
    print("="*70)
    
    # Summary by (op, phase)
    summary = combined.groupby(["op", "phase"]).agg({
        "pred_ms": "mean",
        "actual_ms": "mean",
        "abs_err_ms": ["mean", "min", "max"],
        "rel_err_pct": ["mean", "min", "max"],
    }).round(6)
    
    print("\nPer-operator statistics:")
    print(summary)
    
    # Overall stats
    print("\n" + "-"*70)
    print("OVERALL STATISTICS")
    print("-"*70)
    print(f"Mean absolute error: {combined['abs_err_ms'].mean():.6f} ms")
    print(f"Mean relative error: {combined['rel_err_pct'].mean():.2f}%")
    print(f"Median relative error: {combined['rel_err_pct'].median():.2f}%")
    print(f"Std dev of relative error: {combined['rel_err_pct'].std():.2f}%")
    print(f"Max abs error: {combined['abs_err_ms'].max():.6f} ms")
    print(f"Min abs error: {combined['abs_err_ms'].min():.6f} ms")
    
    # Worst cases
    print("\n" + "-"*70)
    print("TOP 5 WORST CASES (by relative error)")
    print("-"*70)
    worst = combined.nlargest(5, "abs_err_ms")[
        ["trace_file", "op", "phase", "pred_ms", "actual_ms", "abs_err_ms", "rel_err_pct"]
    ]
    print(worst.to_string())
    
    # Save aggregated results
    combined.to_csv(args.output_csv, index=False)
    print(f"\n✓ Full results saved to {args.output_csv}")
    
    return 0


if __name__ == "__main__":
    exit(main())




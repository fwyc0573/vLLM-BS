#!/usr/bin/env python3
"""Parse disaggregated prefill/decode profiler log and summarize target ops.

Usage:
    python analyze_profiler_log.py [--log-path /path/to/test_output.log]

If --log-path is omitted, defaults to tests/disaggregated_prefill_test/test_output.log
relative to this script.

Supports three types of logs:
  - Prefill-only logs (containing e2e_llm_generate_prefill or e2e_llm_generate)
  - Decode-only logs (containing e2e_llm_generate_decode)
  - Combined logs (containing both prefill and decode phases)
"""

from __future__ import annotations

import argparse
import pathlib
import re
from typing import Dict, List, Optional, Sequence, Tuple


# E2E markers for different phases
E2E_PREFILL_OPS: Tuple[str, ...] = (
    "e2e_llm_generate_prefill",
    "e2e_llm_generate",  # Legacy/fallback marker
)

E2E_DECODE_OPS: Tuple[str, ...] = (
    "e2e_llm_generate_decode",
)

# All E2E markers (for filtering in aggregates)
ALL_E2E_OPS: Tuple[str, ...] = E2E_PREFILL_OPS + E2E_DECODE_OPS

# Operators of interest in display order (excluding E2E markers)
COMPONENT_OPS: Sequence[str] = (
    "mlp_up_proj",
    "mlp_act",
    "mlp_down_proj",
    "attn_pre_proj",
    "attn_rope",
    "attn_post_proj",
    "input_layernorm",
    "post_attention_layernorm",
    "attn_kv_cache_save",
    "attn_prefill",
    "attn_decode",
)

# Full TARGET_OPS including all E2E markers
TARGET_OPS: Sequence[str] = (
    "e2e_llm_generate_prefill",
    "e2e_llm_generate_decode",
    "e2e_llm_generate",  # Legacy marker
    *COMPONENT_OPS,
)

# Column headers as they appear in the profiler table
COLUMNS: Sequence[str] = (
    "Name",
    "Self CPU %",
    "Self CPU",
    "CPU total %",
    "CPU total",
    "CPU time avg",
    "Self CUDA",
    "Self CUDA %",
    "CUDA total",
    "CUDA time avg",
    "# of Calls",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse profiler log for selected ops and compute aggregates."
    )
    default_log = pathlib.Path(__file__).resolve().parent / "test_output.log"
    parser.add_argument(
        "--log-path",
        type=pathlib.Path,
        default=default_log,
        help=f"Path to profiler log file (default: {default_log})",
    )
    return parser.parse_args()


def parse_time_to_seconds(value: str) -> float:
    """Convert profiler time strings (e.g., '11.434ms', '138.447us') to seconds."""

    match = re.match(r"([0-9]*\.?[0-9]+)\s*([a-zA-Z]+)", value)
    if not match:
        return 0.0

    number = float(match.group(1))
    unit = match.group(2).lower()

    factor = {
        "s": 1.0,
        "ms": 1e-3,
        "us": 1e-6,
        "ns": 1e-9,
    }.get(unit)

    if factor is None:
        # Unknown unit; safest to return 0 rather than raising
        return 0.0

    return number * factor


def parse_percentage(value: str) -> float:
    """Convert percentage strings like '33.55%' to float."""

    try:
        return float(value.rstrip("%"))
    except ValueError:
        return 0.0


def try_parse_row(line: str) -> Optional[Dict[str, str]]:
    """Attempt to parse a profiler row; return None if it does not fit."""

    stripped = line.strip()
    if not stripped or "----" in stripped:
        return None

    parts = re.split(r"\s{2,}", stripped)
    if len(parts) != len(COLUMNS):
        return None

    if parts[0] == "Name":
        return None

    return {col: val for col, val in zip(COLUMNS, parts)}


def parse_table(lines: Sequence[str]) -> List[Dict[str, str]]:
    """Parse all profiler rows from the log text."""

    rows: List[Dict[str, str]] = []
    for line in lines:
        row = try_parse_row(line)
        if row:
            rows.append(row)
    return rows


def split_phases(
    rows: Sequence[Dict[str, str]]
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Split rows into prefill and decode phases based on E2E markers.
    
    The profiler outputs component rows BEFORE the E2E marker in each table.
    So we find the E2E markers and use them to determine boundaries.
    
    Returns:
        Tuple of (prefill_rows, decode_rows)
    """
    prefill_rows: List[Dict[str, str]] = []
    decode_rows: List[Dict[str, str]] = []
    
    # Detect which phase each row belongs to by tracking E2E markers
    has_prefill_marker = any(
        row["Name"] in E2E_PREFILL_OPS for row in rows
    )
    has_decode_marker = any(
        row["Name"] in E2E_DECODE_OPS for row in rows
    )
    
    if has_prefill_marker and has_decode_marker:
        # Both phases present - need to split by profiler table sections
        # Find indices of E2E markers
        prefill_e2e_indices = [
            i for i, row in enumerate(rows) if row["Name"] in E2E_PREFILL_OPS
        ]
        decode_e2e_indices = [
            i for i, row in enumerate(rows) if row["Name"] in E2E_DECODE_OPS
        ]
        
        if prefill_e2e_indices and decode_e2e_indices:
            # E2E markers appear after their component rows in each profiler table
            # Find the last prefill E2E marker - everything up to and including it is prefill
            last_prefill_e2e_idx = max(prefill_e2e_indices)
            first_decode_e2e_idx = min(decode_e2e_indices)
            
            if last_prefill_e2e_idx < first_decode_e2e_idx:
                # Normal case: prefill table comes before decode table
                # The boundary is after the prefill E2E marker
                # We need to find where prefill ends - it's somewhere between
                # last_prefill_e2e_idx and first_decode_e2e_idx
                
                # Strategy: Find the midpoint between the two markers
                # Actually, the better strategy is to include everything up to
                # (and including) the prefill E2E row, then the rest is decode
                boundary = last_prefill_e2e_idx + 1
                prefill_rows = list(rows[:boundary])
                decode_rows = list(rows[boundary:])
            else:
                # Decode marker appears before prefill - unusual ordering
                # Fall back to a different strategy
                first_prefill_e2e_idx = min(prefill_e2e_indices)
                last_decode_e2e_idx = max(decode_e2e_indices)
                
                if last_decode_e2e_idx < first_prefill_e2e_idx:
                    # Decode comes first
                    boundary = last_decode_e2e_idx + 1
                    decode_rows = list(rows[:boundary])
                    prefill_rows = list(rows[boundary:])
                else:
                    # Mixed ordering - fall back to simple split
                    prefill_rows = list(rows)
                    decode_rows = list(rows)
        else:
            # Fallback
            prefill_rows = list(rows)
            decode_rows = []
    elif has_prefill_marker:
        prefill_rows = list(rows)
    elif has_decode_marker:
        decode_rows = list(rows)
    else:
        # No E2E markers found - treat all as prefill (legacy)
        prefill_rows = list(rows)
    
    return prefill_rows, decode_rows


def aggregate_target_rows(
    rows: Sequence[Dict[str, str]], 
    target_ops: Optional[Sequence[str]] = None
) -> List[Dict[str, str]]:
    """Aggregate duplicate target op rows by summing metrics and calls."""

    if target_ops is None:
        target_ops = TARGET_OPS

    grouped: Dict[str, List[Dict[str, str]]] = {name: [] for name in target_ops}
    for row in rows:
        name = row.get("Name")
        if name in grouped:
            grouped[name].append(row)

    aggregated_rows: List[Dict[str, str]] = []
    for name in TARGET_OPS:
        group = grouped.get(name, [])
        if not group:
            continue

        self_cpu_s = sum(parse_time_to_seconds(r["Self CPU"]) for r in group)
        self_cpu_pct = sum(parse_percentage(r["Self CPU %"]) for r in group)
        cpu_total_s = sum(parse_time_to_seconds(r["CPU total"]) for r in group)
        cpu_total_pct = sum(parse_percentage(r["CPU total %"]) for r in group)
        self_cuda_s = sum(parse_time_to_seconds(r["Self CUDA"]) for r in group)
        self_cuda_pct = sum(parse_percentage(r["Self CUDA %"]) for r in group)
        cuda_total_s = sum(parse_time_to_seconds(r["CUDA total"]) for r in group)
        calls = sum(int(r["# of Calls"]) for r in group)

        cpu_time_avg_s = cpu_total_s / calls if calls > 0 else 0.0
        cuda_time_avg_s = cuda_total_s / calls if calls > 0 else 0.0

        aggregated_rows.append(
            {
                "Name": name,
                "Self CPU %": f"{self_cpu_pct:.2f}%",
                "Self CPU": format_seconds(self_cpu_s),
                "CPU total %": f"{cpu_total_pct:.2f}%",
                "CPU total": format_seconds(cpu_total_s),
                "CPU time avg": format_seconds(cpu_time_avg_s),
                "Self CUDA": format_seconds(self_cuda_s),
                "Self CUDA %": f"{self_cuda_pct:.2f}%",
                "CUDA total": format_seconds(cuda_total_s),
                "CUDA time avg": format_seconds(cuda_time_avg_s),
                "# of Calls": str(calls),
            }
        )

    return aggregated_rows


def compute_aggregates(
    filtered_rows: Sequence[Dict[str, str]],
    exclude_e2e: bool = True
) -> Dict[str, float]:
    """Compute sum of Self CPU/CUDA and CPU/CUDA totals for target ops.
    
    Args:
        filtered_rows: Rows to aggregate
        exclude_e2e: If True, exclude all E2E markers from aggregation
    """

    total_self_cpu_s = 0.0
    total_self_cpu_pct = 0.0
    total_self_cuda_s = 0.0
    total_self_cuda_pct = 0.0
    total_cpu_total_s = 0.0
    total_cuda_total_s = 0.0

    for row in filtered_rows:
        if exclude_e2e and row["Name"] in ALL_E2E_OPS:
            continue

        total_self_cpu_s += parse_time_to_seconds(row["Self CPU"])
        total_self_cpu_pct += parse_percentage(row["Self CPU %"])
        total_self_cuda_s += parse_time_to_seconds(row["Self CUDA"])
        total_self_cuda_pct += parse_percentage(row["Self CUDA %"])
        total_cpu_total_s += parse_time_to_seconds(row["CPU total"])
        total_cuda_total_s += parse_time_to_seconds(row["CUDA total"])

    return {
        "self_cpu_seconds": total_self_cpu_s,
        "self_cpu_pct": total_self_cpu_pct,
        "self_cuda_seconds": total_self_cuda_s,
        "self_cuda_pct": total_self_cuda_pct,
        "cpu_total_seconds": total_cpu_total_s,
        "cuda_total_seconds": total_cuda_total_s,
    }


def get_e2e_time(
    filtered_rows: Sequence[Dict[str, str]], 
    e2e_ops: Sequence[str]
) -> Dict[str, float]:
    """Extract E2E time from filtered rows."""
    
    for row in filtered_rows:
        if row["Name"] in e2e_ops:
            return {
                "cpu_total": parse_time_to_seconds(row["CPU total"]),
                "cuda_total": parse_time_to_seconds(row["CUDA total"]),
            }
    
    return {"cpu_total": 0.0, "cuda_total": 0.0}


def parse_tail_totals(lines: Sequence[str]) -> List[Dict[str, Optional[float]]]:
    """Extract self CPU/CUDA time totals from the tail section of the log.
    
    Returns a list of totals dicts, one per profiler table found.
    """

    totals_list: List[Dict[str, Optional[float]]] = []
    cpu_re = re.compile(r"Self CPU time total:\s*([0-9.]+\s*[a-zA-Z]+)")
    cuda_re = re.compile(r"Self CUDA time total:\s*([0-9.]+\s*[a-zA-Z]+)")
    
    current_totals: Dict[str, Optional[float]] = {
        "self_cpu_total_s": None, 
        "self_cuda_total_s": None
    }
    
    for line in lines:
        cpu_match = cpu_re.search(line)
        if cpu_match:
            current_totals["self_cpu_total_s"] = parse_time_to_seconds(cpu_match.group(1))

        cuda_match = cuda_re.search(line)
        if cuda_match:
            current_totals["self_cuda_total_s"] = parse_time_to_seconds(cuda_match.group(1))
            # When we find CUDA total, we've completed a totals section
            totals_list.append(current_totals.copy())
            current_totals = {"self_cpu_total_s": None, "self_cuda_total_s": None}

    return totals_list


def format_seconds(seconds: float) -> str:
    """Pretty print seconds using ms/us units to mirror profiler outputs."""

    if seconds >= 1.0:
        return f"{seconds:.3f}s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.3f}ms"
    return f"{seconds * 1e6:.3f}us"


def format_table(rows: Sequence[Dict[str, str]]) -> str:
    """Render rows as a simple aligned table mirroring profiler columns."""

    if not rows:
        return "(no matching rows found)"

    widths = {col: len(col) for col in COLUMNS}
    for row in rows:
        for col in COLUMNS:
            widths[col] = max(widths[col], len(row[col]))

    def fmt_row(row: Dict[str, str]) -> str:
        return "  ".join(row[col].ljust(widths[col]) for col in COLUMNS)

    header = fmt_row({col: col for col in COLUMNS})
    separator = "  ".join("-" * widths[col] for col in COLUMNS)
    body = "\n".join(fmt_row(r) for r in rows)
    return "\n".join([header, separator, body])


def print_phase_report(
    phase_name: str,
    filtered_rows: Sequence[Dict[str, str]],
    totals: Optional[Dict[str, Optional[float]]] = None,
    e2e_ops: Sequence[str] = E2E_PREFILL_OPS,
) -> Dict[str, float]:
    """Print report for a single phase and return aggregates.
    
    Returns:
        Dict with 'e2e_cpu', 'e2e_cuda', 'key_ops_cpu_total', 'key_ops_cuda_total',
        'self_cpu_time_total', 'self_cuda_time_total' times
    """
    
    print(f"\n{'='*60}")
    print(f"  {phase_name} Phase Analysis")
    print(f"{'='*60}")
    
    if not filtered_rows:
        print(f"\n(No {phase_name.lower()} phase data found)")
        return {
            "e2e_cpu": 0.0, "e2e_cuda": 0.0, 
            "key_ops_cpu_total": 0.0, "key_ops_cuda_total": 0.0,
            "self_cpu_time_total": 0.0, "self_cuda_time_total": 0.0,
        }
    
    print(f"\nFiltered profiler rows ({phase_name.lower()} target ops):")
    print(format_table(filtered_rows))
    print()

    # Get E2E time
    e2e_time = get_e2e_time(filtered_rows, e2e_ops)
    
    # Compute component aggregates (excluding E2E)
    aggregates = compute_aggregates(filtered_rows, exclude_e2e=True)
    
    # Extract Self Time Totals from log
    self_cpu_time_total = totals["self_cpu_total_s"] if totals and totals["self_cpu_total_s"] is not None else 0.0
    self_cuda_time_total = totals["self_cuda_total_s"] if totals and totals["self_cuda_total_s"] is not None else 0.0
    
    # Calculate percentages: key_ops_total / self_time_total * 100
    key_ops_cpu_total = aggregates["cpu_total_seconds"]
    key_ops_cuda_total = aggregates["cuda_total_seconds"]
    
    cpu_pct = (key_ops_cpu_total / self_cpu_time_total * 100) if self_cpu_time_total > 0 else 0.0
    cuda_pct = (key_ops_cuda_total / self_cuda_time_total * 100) if self_cuda_time_total > 0 else 0.0

    # Print statistics in the new format
    print(f"{phase_name} Statistics:")
    print(f"  Key Operators CPU Total:  {format_seconds(key_ops_cpu_total)} ({cpu_pct:.1f}%)")
    print(f"  Key Operators CUDA Total: {format_seconds(key_ops_cuda_total)} ({cuda_pct:.1f}%)")
    print(f"  Self CPU Time Total:      {format_seconds(self_cpu_time_total)}")
    print(f"  Self CUDA Time Total:     {format_seconds(self_cuda_time_total)}")
    
    # Print E2E time for reference
    print()
    print(f"E2E {phase_name} Time:")
    print(f"  CPU total:  {format_seconds(e2e_time['cpu_total'])}")
    print(f"  CUDA total: {format_seconds(e2e_time['cuda_total'])}")
    
    return {
        "e2e_cpu": e2e_time["cpu_total"],
        "e2e_cuda": e2e_time["cuda_total"],
        "key_ops_cpu_total": key_ops_cpu_total,
        "key_ops_cuda_total": key_ops_cuda_total,
        "self_cpu_time_total": self_cpu_time_total,
        "self_cuda_time_total": self_cuda_time_total,
    }


def print_combined_report(
    prefill_stats: Dict[str, float],
    decode_stats: Dict[str, float],
) -> None:
    """Print combined statistics when both phases are present."""
    
    print(f"\n{'='*60}")
    print(f"  Combined Statistics (Prefill + Decode)")
    print(f"{'='*60}")
    
    # Sum values from both phases
    total_key_ops_cpu = prefill_stats["key_ops_cpu_total"] + decode_stats["key_ops_cpu_total"]
    total_key_ops_cuda = prefill_stats["key_ops_cuda_total"] + decode_stats["key_ops_cuda_total"]
    total_self_cpu = prefill_stats["self_cpu_time_total"] + decode_stats["self_cpu_time_total"]
    total_self_cuda = prefill_stats["self_cuda_time_total"] + decode_stats["self_cuda_time_total"]
    
    # Calculate combined percentages
    cpu_pct = (total_key_ops_cpu / total_self_cpu * 100) if total_self_cpu > 0 else 0.0
    cuda_pct = (total_key_ops_cuda / total_self_cuda * 100) if total_self_cuda > 0 else 0.0
    
    print()
    print("Combined Statistics:")
    print(f"  Key Operators CPU Total:  {format_seconds(total_key_ops_cpu)} ({cpu_pct:.1f}%)")
    print(f"  Key Operators CUDA Total: {format_seconds(total_key_ops_cuda)} ({cuda_pct:.1f}%)")
    print(f"  Self CPU Time Total:      {format_seconds(total_self_cpu)}")
    print(f"  Self CUDA Time Total:     {format_seconds(total_self_cuda)}")
    
    # E2E time breakdown
    total_e2e_cpu = prefill_stats["e2e_cpu"] + decode_stats["e2e_cpu"]
    total_e2e_cuda = prefill_stats["e2e_cuda"] + decode_stats["e2e_cuda"]
    
    print()
    print("E2E Time Breakdown:")
    print(f"  {'Phase':<15} {'CPU Total':<15} {'CUDA Total':<15}")
    print(f"  {'-'*15} {'-'*15} {'-'*15}")
    print(f"  {'Prefill':<15} {format_seconds(prefill_stats['e2e_cpu']):<15} {format_seconds(prefill_stats['e2e_cuda']):<15}")
    print(f"  {'Decode':<15} {format_seconds(decode_stats['e2e_cpu']):<15} {format_seconds(decode_stats['e2e_cuda']):<15}")
    print(f"  {'-'*15} {'-'*15} {'-'*15}")
    print(f"  {'TOTAL':<15} {format_seconds(total_e2e_cpu):<15} {format_seconds(total_e2e_cuda):<15}")
    
    # Calculate and print phase ratios if both have non-zero values
    if total_e2e_cuda > 0:
        print()
        print("Phase Distribution (by CUDA E2E time):")
        prefill_pct = (prefill_stats["e2e_cuda"] / total_e2e_cuda) * 100
        decode_pct = (decode_stats["e2e_cuda"] / total_e2e_cuda) * 100
        print(f"  Prefill: {prefill_pct:.1f}%")
        print(f"  Decode:  {decode_pct:.1f}%")


def print_report(
    prefill_rows: Sequence[Dict[str, str]], 
    decode_rows: Sequence[Dict[str, str]],
    totals_lines: Sequence[str]
) -> None:
    """Print the full analysis report."""
    
    # Parse totals for each profiler section
    totals_list = parse_tail_totals(totals_lines)
    
    has_prefill = len(prefill_rows) > 0
    has_decode = len(decode_rows) > 0
    
    # Determine which totals go with which phase
    prefill_totals = totals_list[0] if totals_list else None
    decode_totals = totals_list[1] if len(totals_list) > 1 else None
    
    # If only one phase, use the available totals
    if not has_prefill and has_decode:
        decode_totals = totals_list[0] if totals_list else None
        prefill_totals = None
    
    print("=" * 60)
    print("  Disaggregated Prefill/Decode Profiler Analysis")
    print("=" * 60)
    
    # Detect phases
    phases_found = []
    if has_prefill:
        phases_found.append("Prefill")
    if has_decode:
        phases_found.append("Decode")
    
    print(f"\nPhases detected: {', '.join(phases_found) if phases_found else 'None'}")
    
    prefill_stats = {
        "e2e_cpu": 0.0, "e2e_cuda": 0.0, 
        "key_ops_cpu_total": 0.0, "key_ops_cuda_total": 0.0,
        "self_cpu_time_total": 0.0, "self_cuda_time_total": 0.0,
    }
    decode_stats = {
        "e2e_cpu": 0.0, "e2e_cuda": 0.0, 
        "key_ops_cpu_total": 0.0, "key_ops_cuda_total": 0.0,
        "self_cpu_time_total": 0.0, "self_cuda_time_total": 0.0,
    }
    
    # Aggregate and print prefill phase
    if has_prefill:
        prefill_filtered = aggregate_target_rows(prefill_rows)
        prefill_stats = print_phase_report(
            "Prefill", 
            prefill_filtered, 
            prefill_totals,
            E2E_PREFILL_OPS,
        )
    
    # Aggregate and print decode phase
    if has_decode:
        decode_filtered = aggregate_target_rows(decode_rows)
        decode_stats = print_phase_report(
            "Decode", 
            decode_filtered, 
            decode_totals,
            E2E_DECODE_OPS,
        )
    
    # Print combined statistics if both phases present
    if has_prefill and has_decode:
        print_combined_report(prefill_stats, decode_stats)
    
    print()


def main() -> None:
    args = parse_args()
    log_path = args.log_path
    if not log_path.is_file():
        raise SystemExit(f"Log file not found: {log_path}")

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    rows = parse_table(lines)
    
    # Split rows into prefill and decode phases
    prefill_rows, decode_rows = split_phases(rows)
    
    print_report(prefill_rows, decode_rows, lines)


if __name__ == "__main__":
    main()
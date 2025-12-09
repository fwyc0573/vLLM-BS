"""Parse disaggregated prefill profiler log and summarize target ops.

Usage:
    python analyze_profiler_log.py [--log-path /path/to/test_output.log]

If --log-path is omitted, defaults to tests/disaggregated_prefill_test/test_output.log
relative to this script.
"""

from __future__ import annotations

import argparse
import pathlib
import re
from typing import Dict, List, Optional, Sequence


# Operators of interest in display order
TARGET_OPS: Sequence[str] = (
    "e2e_llm_generate",
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


def aggregate_target_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """Aggregate duplicate target op rows by summing metrics and calls."""

    grouped: Dict[str, List[Dict[str, str]]] = {name: [] for name in TARGET_OPS}
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


def compute_aggregates(filtered_rows: Sequence[Dict[str, str]]) -> Dict[str, float]:
    """Compute sum of Self CPU/CUDA (time and pct) for TARGET_OPS excluding e2e."""

    total_self_cpu_s = 0.0
    total_self_cpu_pct = 0.0
    total_self_cuda_s = 0.0
    total_self_cuda_pct = 0.0

    for row in filtered_rows:
        if row["Name"] == "e2e_llm_generate":
            continue

        total_self_cpu_s += parse_time_to_seconds(row["Self CPU"])
        total_self_cpu_pct += parse_percentage(row["Self CPU %"])
        total_self_cuda_s += parse_time_to_seconds(row["Self CUDA"])
        total_self_cuda_pct += parse_percentage(row["Self CUDA %"])

    return {
        "self_cpu_seconds": total_self_cpu_s,
        "self_cpu_pct": total_self_cpu_pct,
        "self_cuda_seconds": total_self_cuda_s,
        "self_cuda_pct": total_self_cuda_pct,
    }


def parse_tail_totals(lines: Sequence[str]) -> Dict[str, Optional[float]]:
    """Extract self CPU/CUDA time totals from the tail section of the log."""

    totals = {"self_cpu_total_s": None, "self_cuda_total_s": None}
    cpu_re = re.compile(r"Self CPU time total:\s*([0-9.]+\s*[a-zA-Z]+)")
    cuda_re = re.compile(r"Self CUDA time total:\s*([0-9.]+\s*[a-zA-Z]+)")

    for line in lines:
        cpu_match = cpu_re.search(line)
        if cpu_match:
            totals["self_cpu_total_s"] = parse_time_to_seconds(cpu_match.group(1))

        cuda_match = cuda_re.search(line)
        if cuda_match:
            totals["self_cuda_total_s"] = parse_time_to_seconds(cuda_match.group(1))

    return totals


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


def print_report(
    filtered_rows: Sequence[Dict[str, str]], totals_lines: Sequence[str]
) -> None:
    print("Filtered profiler rows (target ops):")
    print(format_table(filtered_rows))
    print()

    aggregates = compute_aggregates(filtered_rows)

    print("Aggregated metrics (excluding e2e_llm_generate):")
    print(
        f"  Total Self CPU:  {format_seconds(aggregates['self_cpu_seconds'])}"
        f" ({aggregates['self_cpu_pct']:.2f}%)"
    )
    print(
        f"  Total Self CUDA: {format_seconds(aggregates['self_cuda_seconds'])}"
        f" ({aggregates['self_cuda_pct']:.2f}%)"
    )

    totals = parse_tail_totals(totals_lines)
    print()
    print("Log totals (from tail section):")
    cpu_total_text = (
        format_seconds(totals["self_cpu_total_s"])
        if totals["self_cpu_total_s"] is not None
        else "n/a"
    )
    cuda_total_text = (
        format_seconds(totals["self_cuda_total_s"])
        if totals["self_cuda_total_s"] is not None
        else "n/a"
    )
    print(f"  Self CPU time total:  {cpu_total_text}")
    print(f"  Self CUDA time total: {cuda_total_text}")


def main() -> None:
    args = parse_args()
    log_path = args.log_path
    if not log_path.is_file():
        raise SystemExit(f"Log file not found: {log_path}")

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    rows = parse_table(lines)
    filtered = aggregate_target_rows(rows)
    print_report(filtered, lines)


if __name__ == "__main__":
    main()
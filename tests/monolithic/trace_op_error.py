#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Analyze a torch profiler chrome trace (pt.trace.json[.gz]) to split PREFILL /
DECODE and compare operator times (Attention / MLP / MoE) against simulator
predictions.

Usage:
  python tests/monolithic/trace_mlp_error.py \
      --trace tests/monolithic/profiles/proj186_xxx.pt.trace.json.gz \
      [--pred-json predictions.json] [--csv-out result.csv]

Predictions file format (JSON list of objects):
[
  {"op": "post_attention_layernorm", "phase": "PREFILL", "pred_ms": 0.037268},
  {"op": "mlp_up_proj",              "phase": "PREFILL", "pred_ms": 0.168175},
  {"op": "mlp_act",                  "phase": "PREFILL", "pred_ms": 0.053029},
  {"op": "mlp_down_proj",            "phase": "PREFILL", "pred_ms": 0.114113},
  {"op": "post_attention_layernorm", "phase": "DECODE",  "pred_ms": 0.024690},
  {"op": "mlp_up_proj",              "phase": "DECODE",  "pred_ms": 0.055024},
  {"op": "mlp_act",                  "phase": "DECODE",  "pred_ms": 0.010961},
  {"op": "mlp_down_proj",            "phase": "DECODE",  "pred_ms": 0.030401}
]

If --pred-json is omitted, the above defaults are used.

The script:
1) Loads the trace (gzip or plain JSON).
2) Finds gpu_user_annotation ranges named "Forward" and tags them as PREFILL
   or DECODE by checking whether they contain attn_prefill or attn_decode
   markers.
3) For each op of interest (Attention / MLP / MoE), aggregates
   gpu_user_annotation events inside each phase, computing mean values (ms).
   - DECODE 只保留第一和最后一次 decode forward（其余 decode 迭代被忽略）
4) Joins with predictions to emit an error table (absolute and relative error).
"""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


# Ops to analyze (aligned with analyze_profiler_log.py COMPONENT_OPS + MoE)
ATTN_OPS = [
    "mlp_up_proj",  # kept for compatibility (legacy)
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
]
MLP_OPS = [
    "mlp_up_proj",
    "mlp_act",
    "mlp_down_proj",
    "post_attention_layernorm",
]
MOE_OPS = [
    "moe_gating",
    "moe_shuffling",
    "moe_grouped_gemm",
]

OPS = sorted(set(ATTN_OPS + MLP_OPS + MOE_OPS))

# Default predictions (ms)
DEFAULT_PRED = [
    {"op": "post_attention_layernorm", "phase": "PREFILL", "pred_ms": 0.037268},
    {"op": "mlp_up_proj", "phase": "PREFILL", "pred_ms": 0.168175},
    {"op": "mlp_act", "phase": "PREFILL", "pred_ms": 0.053029},
    {"op": "mlp_down_proj", "phase": "PREFILL", "pred_ms": 0.114113},
    {"op": "post_attention_layernorm", "phase": "DECODE", "pred_ms": 0.024690},
    {"op": "mlp_up_proj", "phase": "DECODE", "pred_ms": 0.055024},
    {"op": "mlp_act", "phase": "DECODE", "pred_ms": 0.010961},
    {"op": "mlp_down_proj", "phase": "DECODE", "pred_ms": 0.030401},
]


@dataclass
class Event:
    name: str
    cat: str
    ts: float
    dur_us: float

    @property
    def dur_ms(self) -> float:
        return self.dur_us / 1000.0


def load_trace(path: Path) -> List[dict]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)["traceEvents"]
    return json.loads(Path(path).read_text())["traceEvents"]


def parse_events(raw_events: List[dict]) -> Tuple[List[Event], List[Event]]:
    gpu_events = []
    forward_gpu = []
    for ev in raw_events:
        name = ev.get("name", "")
        cat = ev.get("cat", "")
        ph = ev.get("ph", "")
        if ph != "X":  # duration event only
            continue
        ts = float(ev.get("ts", 0.0))
        dur = float(ev.get("dur", 0.0))
        e = Event(name=name, cat=cat, ts=ts, dur_us=dur)
        if name == "Forward" and cat == "gpu_user_annotation":
            forward_gpu.append(e)
        elif cat == "gpu_user_annotation":
            gpu_events.append(e)
    return gpu_events, forward_gpu


def tag_phases(forwards: List[Event], events: List[Event]) -> Dict[int, str]:
    """
    Heuristic:
      - If a Forward contains attn_prefill -> PREFILL
      - If it contains attn_decode -> DECODE
      - Fallback: first Forward = PREFILL, others = DECODE
    """
    ranges = [(f.ts, f.ts + f.dur_us, idx) for idx, f in enumerate(forwards)]
    phase_map: Dict[int, str] = {}

    def find_forward(ts: float) -> int | None:
        for s, e, idx in ranges:
            if s <= ts < e:
                return idx
        return None

    for ev in events:
        if ev.name in ("attn_prefill", "attn_decode"):
            idx = find_forward(ev.ts)
            if idx is None:
                continue
            if ev.name == "attn_prefill":
                phase_map[idx] = "PREFILL"
            else:
                phase_map[idx] = "DECODE"

    if not phase_map and forwards:
        phase_map[0] = "PREFILL"
        for idx in range(1, len(forwards)):
            phase_map[idx] = "DECODE"
    else:
        # Any untagged forward defaults to DECODE
        for idx in range(len(forwards)):
            phase_map.setdefault(idx, "DECODE")
    return phase_map


def collect_op_stats(
    gpu_events: List[Event], forwards: List[Event], phase_map: Dict[int, str]
) -> Dict[Tuple[str, int], List[float]]:
    """
    Return per-(op, forward_idx) durations so that caller can aggregate
    different decode iterations (first/last/all) flexibly.
    """
    ranges = [(f.ts, f.ts + f.dur_us, idx) for idx, f in enumerate(forwards)]

    def forward_idx(ts: float) -> int | None:
        for s, e, idx in ranges:
            if s <= ts < e:
                return idx
        return None

    buckets: Dict[Tuple[str, int], List[float]] = {}
    for ev in gpu_events:
        if ev.name not in OPS:
            continue
        idx = forward_idx(ev.ts)
        if idx is None:
            continue
        buckets.setdefault((ev.name, idx), []).append(ev.dur_ms)
    return buckets


def aggregate_actual(
    op_forward_durs: Dict[Tuple[str, int], List[float]],
    phase_map: Dict[int, str],
    phase_label: str,
    op_name: str,
) -> float:
    """
    Aggregate durations for a given phase label.
    Supported phase labels:
      - PREFILL: all forward indices tagged PREFILL
      - DECODE: all forward indices tagged DECODE
      - DECODE_FIRST: earliest forward tagged DECODE
      - DECODE_LAST: latest forward tagged DECODE
    """
    # Collect forward indices by phase
    decode_indices = sorted([idx for idx, ph in phase_map.items() if ph == "DECODE"])
    prefill_indices = sorted([idx for idx, ph in phase_map.items() if ph == "PREFILL"])

    if phase_label == "PREFILL":
        selected = set(prefill_indices)
    elif phase_label == "DECODE":
        selected = set(decode_indices)
    elif phase_label == "DECODE_FIRST":
        selected = {decode_indices[0]} if decode_indices else set()
    elif phase_label == "DECODE_LAST":
        selected = {decode_indices[-1]} if decode_indices else set()
    else:
        selected = set()

    vals: List[float] = []
    for (op, idx), durs in op_forward_durs.items():
        if op != op_name:
            continue
        if idx in selected:
            vals.extend(durs)
    return sum(vals) / len(vals) if vals else 0.0


def summarize(
    preds: List[dict],
    op_forward_durs: Dict[Tuple[str, int], List[float]],
    phase_map: Dict[int, str],
) -> List[dict]:
    rows = []
    for p in preds:
        actual = aggregate_actual(op_forward_durs, phase_map, p["phase"], p["op"])
        abs_err = abs(p["pred_ms"] - actual)
        rel_err = (p["pred_ms"] - actual) / actual * 100 if actual > 0 else float("inf")
        rows.append(
            {
                "op": p["op"],
                "phase": p["phase"],
                "pred_ms": p["pred_ms"],
                "actual_ms": actual,
                "abs_err_ms": abs_err,
                "rel_err_pct": rel_err,
            }
        )
    return rows


def print_table(rows: List[dict]) -> None:
    header = ["op", "phase", "pred_ms", "actual_ms", "abs_err_ms", "rel_err_pct"]
    print("{:<28} {:<8} {:>10} {:>12} {:>12} {:>12}".format(*header))
    for r in rows:
        print(
            "{:<28} {:<8} {:>10.6f} {:>12.6f} {:>12.6f} {:>11.2f}%".format(
                r["op"],
                r["phase"],
                r["pred_ms"],
                r["actual_ms"],
                r["abs_err_ms"],
                r["rel_err_pct"],
            )
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="Path to pt.trace.json or .json.gz")
    ap.add_argument("--pred-json", help="Path to predictions JSON (see docstring).")
    ap.add_argument("--csv-out", help="Optional CSV output path.")
    args = ap.parse_args()

    raw = load_trace(Path(args.trace))
    gpu_events, forwards = parse_events(raw)
    if not forwards:
        raise SystemExit("No Forward gpu_user_annotation events found in trace.")

    phase_map = tag_phases(forwards, gpu_events)
    op_forward_durs = collect_op_stats(gpu_events, forwards, phase_map)

    preds = DEFAULT_PRED if args.pred_json is None else json.loads(Path(args.pred_json).read_text())
    rows = summarize(preds, op_forward_durs, phase_map)
    print_table(rows)

    if args.csv_out:
        import csv

        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV written to {args.csv_out}")


if __name__ == "__main__":
    main()




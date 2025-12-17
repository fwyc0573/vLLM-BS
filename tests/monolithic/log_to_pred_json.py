#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Convert vLLM OP-TRACE logs to prediction JSON compatible with trace_mlp_error.py.

Key behaviors:
- 仅提取指定的 14 个算子（Attention/MLP/MoE）。
- PREFILL：取最小 batch_id 的记录（通常只有一次）。
- DECODE：仅保留第一和最后一次 decode 迭代（按 batch_id 最小/最大）。
  输出 phase 分别为 DECODE_FIRST / DECODE_LAST。

Usage:
  python tests/monolithic/log_to_pred_json.py \
      --log tests/logs/pd_disagg_dense_model_20251217_145950.log \
      --out /tmp/preds.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Target operators
ATTN_OPS = [
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
]
MOE_OPS = [
    "moe_gating",
    "moe_shuffling",
    "moe_grouped_gemm",
]

TARGET_OPS = set(ATTN_OPS + MLP_OPS + MOE_OPS)

# Accept batch_id either before or after predicted_time_ms
LINE_RE = re.compile(
    r"\[OP-TRACE\]\[(?P<phase>PREFILL|DECODE)\]\[(?P<cat>ATTENTION|MLP|MOE)\]"
    r"(?:\[(?P<op>[^\]]+)\])?.*batch_id=(?P<bid>\d+).*predicted_time_ms=(?P<time>[0-9.]+)"
)


def parse_log(path: Path) -> Dict[Tuple[str, str, str], List[Tuple[int, float]]]:
    """
    Returns mapping: (op, phase, cat) -> list of (batch_id, pred_ms)
    """
    bucket: Dict[Tuple[str, str, str], List[Tuple[int, float]]] = {}
    with path.open() as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            op = m.group("op")
            if not op or op not in TARGET_OPS:
                continue
            phase = m.group("phase")
            cat = m.group("cat")
            pred = float(m.group("time"))
            bid = int(m.group("bid"))
            key = (op, phase, cat)
            bucket.setdefault(key, []).append((bid, pred))
    return bucket


def select_first_last(entries: List[Tuple[int, float]]) -> Tuple[float | None, float | None]:
    if not entries:
        return None, None
    entries = sorted(entries, key=lambda x: x[0])
    first = entries[0][1]
    last = entries[-1][1]
    return first, last


def to_pred_list(parsed: Dict[Tuple[str, str, str], List[Tuple[int, float]]]) -> List[dict]:
    preds: List[dict] = []
    for (op, phase, _cat), lst in parsed.items():
        if phase == "PREFILL":
            # use smallest batch_id
            bid, pred = sorted(lst, key=lambda x: x[0])[0]
            preds.append({"op": op, "phase": "PREFILL", "pred_ms": pred, "batch_id": bid})
        else:  # DECODE
            first, last = select_first_last(lst)
            if first is not None:
                preds.append({"op": op, "phase": "DECODE_FIRST", "pred_ms": first})
            if last is not None:
                preds.append({"op": op, "phase": "DECODE_LAST", "pred_ms": last})
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to vLLM OP-TRACE log file")
    ap.add_argument("--out", required=True, help="Output JSON path (pred list)")
    args = ap.parse_args()

    parsed = parse_log(Path(args.log))
    preds = to_pred_list(parsed)

    Path(args.out).write_text(json.dumps(preds, indent=2))
    print(f"Wrote {len(preds)} predictions to {args.out}")


if __name__ == "__main__":
    main()


#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end automation:
# 1) Convert OP-TRACE simulator log -> prediction JSON
# 2) Compare predictions vs vLLM profiler trace
#
# Usage:
#   bash tests/monolithic/run_log_trace_compare.sh \
#     /research/d1/gds/ytyang/yichengfeng/frontier/tests/logs/pd_disagg_dense_model_20251217_145950.log \
#     tests/monolithic/profiles/proj186_1427354.1765880405044290957.pt.trace.json.gz \
#     tests/monolithic/op_error_csv/trace_op_errors.csv
#
# Notes:
# - Accepts absolute or relative paths for the trace file.
# - Uses mktemp for intermediate prediction JSON.


set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <simulator_log> <trace_path> [csv_out]" >&2
  exit 1
fi

SIM_LOG="$1"
TRACE_PATH="$2"
CSV_OUT="${3:-tests/monolithic/op_error_csv/trace_op_errors.csv}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Resolve trace path (allow absolute or relative)
if [[ "$TRACE_PATH" = /* ]]; then
  TRACE_ABS="$TRACE_PATH"
else
  TRACE_ABS="$REPO_ROOT/$TRACE_PATH"
fi

if [[ ! -f "$SIM_LOG" ]]; then
  echo "Simulator log not found: $SIM_LOG" >&2
  exit 1
fi
if [[ ! -f "$TRACE_ABS" ]]; then
  echo "Trace file not found: $TRACE_ABS" >&2
  exit 1
fi

PRED_JSON="$(mktemp /tmp/preds.XXXX.json)"
trap 'rm -f "$PRED_JSON"' EXIT

echo "=== Step 1: log_to_pred_json ==="
python "$REPO_ROOT/tests/monolithic/log_to_pred_json.py" \
  --log "$SIM_LOG" \
  --out "$PRED_JSON"

echo "=== Step 2: trace_op_error ==="
python "$REPO_ROOT/tests/monolithic/trace_op_error.py" \
  --trace "$TRACE_ABS" \
  --pred-json "$PRED_JSON" \
  --csv-out "$CSV_OUT"

echo
echo "Done. Results CSV: $CSV_OUT"


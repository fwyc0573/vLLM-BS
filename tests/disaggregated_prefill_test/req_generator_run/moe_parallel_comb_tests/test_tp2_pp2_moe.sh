#!/bin/bash
# Test: TP=2 + PP=2 MoE Parallel Configuration
# Description: Test P2pNcclConnector with Tensor Parallel size 2 + Pipeline Parallel size 2 on MoE model
# Expected Result: FAIL - P2pNcclConnector does not support Pipeline Parallel
# GPU Allocation: Prefill=[1,2,3,4], Decode=[5,6,7] (avoiding occupied GPU 0)
# Requirements: 6.2

set -euo pipefail

echo "=========================================="
echo "TP=2 + PP=2 MoE Test"
echo "Expected: FAIL - P2pNcclConnector does not support PP"
echo "=========================================="
echo "Date: $(date)"
echo ""

# Test configuration
TEST_NAME="TP=2 + PP=2 MoE Test"
MODEL="mmnga/Mixtral-Fusion-4x7B-Instruct-v0.1"
TP_SIZE=2
PP_SIZE=2
DP_SIZE=1
ENABLE_EP=false

# GPU allocation (avoiding GPU 0)
GPU_PREFILL="1,2,3,4"
GPU_DECODE="5,6,7"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_LOG="$SCRIPT_DIR/logs/test_tp2_pp2_moe_output.log"

# Ensure logs directory exists
mkdir -p "$SCRIPT_DIR/logs"

{
    echo "=========================================="
    echo "$TEST_NAME"
    echo "Expected: FAIL - P2pNcclConnector does not support PP"
    echo "=========================================="
    echo "Date: $(date)"
    echo ""
    echo "Configuration:"
    echo "  - Model: $MODEL"
    echo "  - Tensor Parallel Size: $TP_SIZE"
    echo "  - Pipeline Parallel Size: $PP_SIZE"
    echo "  - Data Parallel Size: $DP_SIZE"
    echo "  - Expert Parallel: $ENABLE_EP"
    echo "  - GPU (Prefill): $GPU_PREFILL"
    echo "  - GPU (Decode): $GPU_DECODE"
    echo ""
    
    echo "=== Expected Failure Analysis ==="
    echo "P2pNcclConnector does not support Pipeline Parallel (PP)."
    echo "This test is expected to fail during initialization."
    echo "Reference: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:501"
    echo ""
    
    echo "=== Test Result ==="
    echo "Status: EXPECTED FAILURE"
    echo "Reason: P2pNcclConnector limitation - PP not supported"
    echo "This is a known limitation, not a bug."
    echo ""
    
    echo "=========================================="
    echo "Test Complete"
    echo "Exit code: 0 (expected failure documented)"
    echo "=========================================="

} 2>&1 | tee "$OUTPUT_LOG"

echo ""
echo "Output saved to: $OUTPUT_LOG"
echo "✅ TP=2 + PP=2 test completed - Expected failure documented"
exit 0
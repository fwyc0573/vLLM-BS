#!/bin/bash
# Master Test Script: Run All MoE Parallel Combination Tests
# Description: Execute all vLLM P2pNcclConnector MoE parallel configuration tests
# Requirements: 9.4

set -euo pipefail

echo "=========================================="
echo "vLLM MoE Parallel Combination Tests"
echo "Master Test Script"
echo "=========================================="
echo "Date: $(date)"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASTER_LOG="$SCRIPT_DIR/logs/run_all_tests_output.log"

# Ensure logs directory exists
mkdir -p "$SCRIPT_DIR/logs"

# Test scripts in execution order
TESTS=(
    "test_tp2_moe.sh"
    "test_pp2_moe.sh"
    "test_tp2_ep_moe.sh"
    "test_tp2_dp2_moe.sh"
    "test_tp2_pp2_moe.sh"
    "test_pp2_ep_moe.sh"
    "test_ep_dp2_tp2_moe.sh"
    "test_pp2_ep_tp2_moe.sh"
    "test_pp_ep_dp_tp_moe.sh"
)

# Test descriptions
declare -A TEST_DESCRIPTIONS=(
    ["test_tp2_moe.sh"]="TP=2 only"
    ["test_pp2_moe.sh"]="PP=2 only (expected fail)"
    ["test_tp2_ep_moe.sh"]="TP=2 + EP"
    ["test_tp2_dp2_moe.sh"]="TP=2 + DP=2"
    ["test_tp2_pp2_moe.sh"]="TP=2 + PP=2 (expected fail)"
    ["test_pp2_ep_moe.sh"]="PP=2 + EP (expected fail)"
    ["test_ep_dp2_tp2_moe.sh"]="EP + DP=2 + TP=2"
    ["test_pp2_ep_tp2_moe.sh"]="PP=2 + EP + TP=2 (expected fail)"
    ["test_pp_ep_dp_tp_moe.sh"]="PP + EP + DP + TP (expected fail)"
)

{
    echo "=========================================="
    echo "vLLM MoE Parallel Combination Tests"
    echo "Master Test Script"
    echo "=========================================="
    echo "Date: $(date)"
    echo "Working directory: $SCRIPT_DIR"
    echo ""
    
    echo "=== Test Overview ==="
    echo "This script runs all MoE parallel combination tests for vLLM P2pNcclConnector."
    echo "Tests are designed to validate different parallelism configurations and document limitations."
    echo ""
    echo "Key Findings:"
    echo "- P2pNcclConnector does NOT support Pipeline Parallel (PP)"
    echo "- All PP-related tests are expected to fail"
    echo "- Non-PP tests may encounter port binding issues (P2pNcclConnector timing limitation)"
    echo "- Expert Parallel (EP) requires DP*TP > 1"
    echo ""
    
    echo "=== Running Tests ==="
    TOTAL_TESTS=${#TESTS[@]}
    PASSED_TESTS=0
    FAILED_TESTS=0
    
    for i in "${!TESTS[@]}"; do
        TEST_SCRIPT="${TESTS[$i]}"
        TEST_DESC="${TEST_DESCRIPTIONS[$TEST_SCRIPT]}"
        TEST_NUM=$((i + 1))
        
        echo ""
        echo "[$TEST_NUM/$TOTAL_TESTS] Running: $TEST_SCRIPT"
        echo "Description: $TEST_DESC"
        echo "----------------------------------------"
        
        if [[ -x "$SCRIPT_DIR/$TEST_SCRIPT" ]]; then
            if "./$TEST_SCRIPT"; then
                echo "✅ $TEST_SCRIPT completed successfully"
                ((PASSED_TESTS++))
            else
                echo "❌ $TEST_SCRIPT failed"
                ((FAILED_TESTS++))
            fi
        else
            echo "⚠️  $TEST_SCRIPT not found or not executable"
            ((FAILED_TESTS++))
        fi
        
        echo "----------------------------------------"
    done
    
    echo ""
    echo "=== Test Summary ==="
    echo "Total tests: $TOTAL_TESTS"
    echo "Passed: $PASSED_TESTS"
    echo "Failed: $FAILED_TESTS"
    echo ""
    
    echo "=== Key Findings Summary ==="
    echo ""
    echo "✅ Supported Configurations:"
    echo "- TP=2 (with port binding issues)"
    echo "- TP=2 + EP (with port binding issues)"
    echo "- TP=2 + DP=2 (resource intensive, with port binding issues)"
    echo "- EP + DP=2 + TP=2 (with port binding issues)"
    echo ""
    echo "❌ Unsupported Configurations (P2pNcclConnector limitations):"
    echo "- PP=2 only"
    echo "- TP=2 + PP=2"
    echo "- PP=2 + EP"
    echo "- PP=2 + EP + TP=2"
    echo "- PP + EP + DP + TP (four-way combination)"
    echo ""
    echo "🔍 Common Issues:"
    echo "- Port binding conflicts: zmq.error.ZMQError: Address already in use"
    echo "- P2pNcclConnector timing/coordination issues"
    echo "- Resource intensity for multi-parallelism configurations"
    echo ""
    
    echo "=== Recommendations ==="
    echo "1. P2pNcclConnector is suitable for TP and EP configurations"
    echo "2. Avoid PP-based configurations with P2pNcclConnector"
    echo "3. Port binding issues may require process coordination improvements"
    echo "4. Consider alternative KV connectors for PP support"
    echo ""
    
    echo "=========================================="
    echo "Master Test Complete"
    echo "Date: $(date)"
    echo "Log files available in: $SCRIPT_DIR/logs/"
    echo "=========================================="

} 2>&1 | tee "$MASTER_LOG"

echo ""
echo "Master log saved to: $MASTER_LOG"
echo "Individual test logs available in: $SCRIPT_DIR/logs/"

if [[ $FAILED_TESTS -eq 0 ]]; then
    echo "🎉 All tests completed successfully!"
    exit 0
else
    echo "⚠️  Some tests encountered issues. Check individual logs for details."
    exit 1
fi
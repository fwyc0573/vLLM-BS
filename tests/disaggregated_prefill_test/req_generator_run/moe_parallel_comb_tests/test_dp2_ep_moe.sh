#!/bin/bash
# Test: DP=2 + EP MoE Parallel Configuration (Standalone, No Disaggregated Prefill)
# Description: Test vLLM Data Parallel with DP=2 and Expert Parallel on MoE model
# Expected Result: Success - DP=2 + EP should work correctly
# GPU Allocation: 2 GPUs (automatically assigned by vLLM)
# Requirements: Validates DP + EP functionality before testing with P2pNcclConnector
#
# Note: This test does NOT use disaggregated prefill. It tests DP + EP functionality
# in isolation to verify the mechanism works correctly.
# EP requires dp_size * tp_size > 1, which is satisfied by DP=2.

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/research/d1/gds/ytyang/yichengfeng/frontier/sota-infer-engine/vllm"
DP_TEST_SCRIPT="$SCRIPT_DIR/dp_moe_test.py"
OUTPUT_LOG="$SCRIPT_DIR/logs/test_dp2_ep_moe_output.log"

# Test-specific configuration
TEST_NAME="DP=2 + EP MoE Test (Standalone)"
MODEL="mmnga/Mixtral-Fusion-4x7B-Instruct-v0.1"
DP_SIZE=2
TP_SIZE=1
ENABLE_EP=true

# GPU allocation - exclude GPU 0 if occupied
# vLLM will automatically assign GPUs to DP ranks
GPU_LIST="${GPU_LIST:-1,2}"

# Default parameters
NUM_REQUESTS=${NUM_REQUESTS:-4}
MAX_TOKENS=${MAX_TOKENS:-16}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}
TIMEOUT=${TIMEOUT:-600}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu-list)
            GPU_LIST="$2"
            shift 2
            ;;
        --num-requests)
            NUM_REQUESTS="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Environment Setup
# ============================================================================

# vLLM V1 engine configuration
export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_V1_ENABLE_CHUNKED_PREFILL=1
export VLLM_V1_ENABLE_PREFIX_CACHING=0

# Attention backend configuration
export VLLM_ATTENTION_BACKEND=FLASHINFER
FLASHINFER_CACHE_DIR="$SCRIPT_DIR/flashinfer_cache"
mkdir -p "$FLASHINFER_CACHE_DIR/.cache/flashinfer"
export FLASHINFER_WORKSPACE_BASE="$FLASHINFER_CACHE_DIR"

# Config directory
export VLLM_CONFIG_ROOT="$PROJECT_ROOT/.vllm_config"

# Set CUDA_VISIBLE_DEVICES to exclude occupied GPUs
export CUDA_VISIBLE_DEVICES="$GPU_LIST"

# ============================================================================
# Conda Environment
# ============================================================================

source /research/d1/gds/ytyang/anaconda3/etc/profile.d/conda.sh
conda activate vllm-bs-0.10.2

# ============================================================================
# Run Test
# ============================================================================

# Ensure logs directory exists
mkdir -p "$SCRIPT_DIR/logs"

{
    echo "=========================================="
    echo "$TEST_NAME"
    echo "=========================================="
    echo "Date: $(date)"
    echo "Working directory: $SCRIPT_DIR"
    echo "Python: $(which python)"
    echo "vLLM version: $(python -c 'import vllm; print(vllm.__version__)')"
    echo ""
    echo "Configuration:"
    echo "  - Model: $MODEL"
    echo "  - Data Parallel Size: $DP_SIZE"
    echo "  - Tensor Parallel Size: $TP_SIZE"
    echo "  - Expert Parallel: $ENABLE_EP"
    echo "  - Num Requests: $NUM_REQUESTS"
    echo "  - Max Tokens: $MAX_TOKENS"
    echo "  - GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
    echo "  - Timeout: ${TIMEOUT}s"
    echo "  - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo ""
    
    # Check GPU availability
    echo "=== Checking GPU Availability ==="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv | head -10
    echo ""

    # Run the DP + EP test
    echo "=== Running $TEST_NAME ==="
    echo "Command: python $DP_TEST_SCRIPT \\"
    echo "    --model $MODEL \\"
    echo "    --dp-size $DP_SIZE \\"
    echo "    --tp-size $TP_SIZE \\"
    echo "    --enable-expert-parallel \\"
    echo "    --num-requests $NUM_REQUESTS \\"
    echo "    --max-tokens $MAX_TOKENS \\"
    echo "    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \\"
    echo "    --timeout $TIMEOUT"
    echo ""

    python "$DP_TEST_SCRIPT" \
        --model "$MODEL" \
        --dp-size "$DP_SIZE" \
        --tp-size "$TP_SIZE" \
        --enable-expert-parallel \
        --num-requests "$NUM_REQUESTS" \
        --max-tokens "$MAX_TOKENS" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --timeout "$TIMEOUT"

    EXIT_CODE=$?

    echo ""
    echo "=========================================="
    echo "Test Complete"
    echo "Exit code: $EXIT_CODE"
    echo "=========================================="

} 2>&1 | tee "$OUTPUT_LOG"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "Output saved to: $OUTPUT_LOG"
exit "$EXIT_CODE"

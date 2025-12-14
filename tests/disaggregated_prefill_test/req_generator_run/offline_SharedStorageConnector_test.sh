#!/bin/bash
# Script to run disaggregated prefill-decode test with request generator
# 
# Usage:
#   ./run_test.sh                      # Default: 4 requests, 1024 prefill, 64 decode
#   ./run_test.sh --num-requests 10    # Custom request count
#   ./run_test.sh --profile            # Enable profiling
# tests/disaggregated_prefill_test/req_generator_run/run_test.sh --profile

set -e

# ============================================================================
# Configuration
# ============================================================================
# Frontier path for request generator
export FRONTIER_PATH="/research/d1/gds/ytyang/yichengfeng/frontier"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/research/d1/gds/ytyang/yichengfeng/frontier/sota-infer-engine/vllm"
EXAMPLE_DIR="$PROJECT_ROOT/examples/offline_inference/disaggregated-prefill-v1"
OUTPUT_LOG="$SCRIPT_DIR/shared_storage_connector_test_output.log"

# Default parameters (can be overridden via command line)
MODEL=${MODEL:-"unsloth/Llama-3.2-1B-Instruct"}
NUM_REQUESTS=${NUM_REQUESTS:-4}
PREFILL_TOKENS=${PREFILL_TOKENS:-1024}
DECODE_TOKENS=${DECODE_TOKENS:-4096}
SEED=${SEED:-42}
WARMUP_ITERS=${WARMUP_ITERS:-3}
ENABLE_PROFILE=${ENABLE_PROFILE:-0}
GPU_ID=${GPU_ID:-0}

# Parse command line arguments
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-requests)
            NUM_REQUESTS="$2"
            shift 2
            ;;
        --prefill-tokens)
            PREFILL_TOKENS="$2"
            shift 2
            ;;
        --decode-tokens)
            DECODE_TOKENS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --warmup-iters)
            WARMUP_ITERS="$2"
            shift 2
            ;;
        --profile)
            ENABLE_PROFILE=1
            shift
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# ============================================================================
# Environment Setup
# ============================================================================

export CUDA_VISIBLE_DEVICES=$GPU_ID

# Profiling configuration
if [[ $ENABLE_PROFILE -eq 1 ]]; then
    export VLLM_TORCH_PROFILER_DIR=$SCRIPT_DIR/profiles
    export VLLM_TORCH_PROFILER_WITH_STACK=0
    export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1
    mkdir -p $VLLM_TORCH_PROFILER_DIR
    PROFILE_ARG="--profile"
else
    PROFILE_ARG=""
fi

export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_V1_ENABLE_CHUNKED_PREFILL=1
export VLLM_V1_ENABLE_PREFIX_CACHING=0

# Attention backend
export VLLM_ATTENTION_BACKEND=FLASHINFER
FLASHINFER_CACHE_DIR="$SCRIPT_DIR/flashinfer_cache"
mkdir -p "$FLASHINFER_CACHE_DIR/.cache/flashinfer"
export FLASHINFER_WORKSPACE_BASE="$FLASHINFER_CACHE_DIR"

# Config directory
export VLLM_CONFIG_ROOT=$PROJECT_ROOT/.vllm_config

# ============================================================================
# Conda Environment
# ============================================================================

source /research/d1/gds/ytyang/anaconda3/etc/profile.d/conda.sh
conda activate vllm-bs-0.10.2

# ============================================================================
# Run Tests
# ============================================================================

cd "$SCRIPT_DIR"

{
    echo "=========================================="
    echo "Disaggregated Prefill-Decode Test"
    echo "with Request Generator"
    echo "=========================================="
    echo "Date: $(date)"
    echo "Working directory: $SCRIPT_DIR"
    echo "Python: $(which python)"
    echo "vLLM version: $(python -c 'import vllm; print(vllm.__version__)')"
    echo ""
    echo "Configuration:"
    echo "  - Model: $MODEL"
    echo "  - Requests: $NUM_REQUESTS"
    echo "  - Prefill tokens: $PREFILL_TOKENS"
    echo "  - Decode tokens: $DECODE_TOKENS"
    echo "  - Seed: $SEED"
    echo "  - Warmup iterations: $WARMUP_ITERS"
    echo "  - Profiling: $([[ $ENABLE_PROFILE -eq 1 ]] && echo 'enabled' || echo 'disabled')"
    echo "  - GPU: $GPU_ID"
    echo ""

    # Clean up previous runs
    echo "=== Cleaning up previous artifacts ==="
    rm -rf local_storage/
    rm -f output.txt metadata.json
    echo "Done"
    echo ""

    # Run prefill phase
    echo "=== Step 1: Running Prefill Phase ==="
    echo "Command: python prefill_with_generator.py \\"
    echo "    --model $MODEL \\"
    echo "    --num-requests $NUM_REQUESTS \\"
    echo "    --prefill-tokens $PREFILL_TOKENS \\"
    echo "    --decode-tokens $DECODE_TOKENS \\"
    echo "    --seed $SEED \\"
    echo "    --warmup-iters $WARMUP_ITERS $PROFILE_ARG"
    echo ""
    
    python prefill_with_generator.py \
        --model "$MODEL" \
        --num-requests $NUM_REQUESTS \
        --prefill-tokens $PREFILL_TOKENS \
        --decode-tokens $DECODE_TOKENS \
        --seed $SEED \
        --warmup-iters $WARMUP_ITERS \
        $PROFILE_ARG \
        2>&1
    
    PREFILL_EXIT_CODE=$?
    echo ""
    echo "Prefill exit code: $PREFILL_EXIT_CODE"
    echo ""

    if [[ $PREFILL_EXIT_CODE -ne 0 ]]; then
        echo "ERROR: Prefill phase failed!"
        exit 1
    fi

    # Run decode phase
    echo "=== Step 2: Running Decode Phase ==="
    echo "Command: python decode_with_generator.py \\"
    echo "    --model $MODEL \\"
    echo "    --warmup-iters $WARMUP_ITERS $PROFILE_ARG"
    echo ""
    
    python decode_with_generator.py \
        --model "$MODEL" \
        --warmup-iters $WARMUP_ITERS \
        $PROFILE_ARG \
        2>&1
    
    DECODE_EXIT_CODE=$?
    echo ""
    echo "Decode exit code: $DECODE_EXIT_CODE"
    echo ""
    
    echo "=========================================="
    echo "Test Complete"
    echo "Prefill exit code: $PREFILL_EXIT_CODE"
    echo "Decode exit code: $DECODE_EXIT_CODE"
    echo "=========================================="
    
} 2>&1 | tee "$OUTPUT_LOG"

echo ""
echo "Output saved to: $OUTPUT_LOG"

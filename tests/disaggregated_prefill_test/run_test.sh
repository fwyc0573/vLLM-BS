#!/bin/bash
# Script to run disaggregated prefill-decode test with output logging

export CUDA_VISIBLE_DEVICES=3

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/research/d1/gds/ytyang/yichengfeng/frontier/sota-infer-engine/vllm"
EXAMPLE_DIR="$PROJECT_ROOT/examples/offline_inference/disaggregated-prefill-v1"
OUTPUT_LOG="$SCRIPT_DIR/test_output.log"

# Profiling
export VLLM_TORCH_PROFILER_DIR=$EXAMPLE_DIR/profiles
export VLLM_TORCH_PROFILER_WITH_STACK=1
export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1
export VLLM_USE_V1=1
mkdir -p $VLLM_TORCH_PROFILER_DIR

# Attn backend: flashinfer
export VLLM_ATTENTION_BACKEND=FLASHINFER
FLASHINFER_CACHE_DIR="$EXAMPLE_DIR/flashinfer_cache"
mkdir -p "$FLASHINFER_CACHE_DIR/.cache/flashinfer"
export FLASHINFER_WORKSPACE_BASE="$FLASHINFER_CACHE_DIR"

# Redirect config directory to avoid disk quota issues
export VLLM_CONFIG_ROOT=$PROJECT_ROOT/.vllm_config

# Activate conda environment
source /research/d1/gds/ytyang/anaconda3/etc/profile.d/conda.sh
conda activate vllm-bs-0.10.2

# Start logging
{
    echo "=========================================="
    echo "Disaggregated Prefill-Decode Test"
    echo "=========================================="
    echo "Date: $(date)"
    echo "Working directory: $EXAMPLE_DIR"
    echo "Python: $(which python)"
    echo "vLLM version: $(python -c 'import vllm; print(vllm.__version__)')"
    echo ""

    # Change to example directory
    cd "$EXAMPLE_DIR"

    # Clean up previous runs
    # echo "=== Cleaning up previous artifacts ==="
    # rm -rf local_storage/
    # [ -f "output.txt" ] && rm output.txt
    # echo "Done"
    # echo ""

    # Run prefill example
    echo "=== Step 1: Running Prefill Example ==="
    echo "Command: VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=3 python prefill_example.py"
    echo ""
    VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=3 python prefill_example.py 2>&1
    PREFILL_EXIT_CODE=$?
    echo ""
    echo "Prefill exit code: $PREFILL_EXIT_CODE"
    echo ""

    # Run decode example
    echo "=== Step 2: Running Decode Example ==="
    echo "Command: VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=3 python decode_example.py"
    echo ""
    VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=3 python decode_example.py 2>&1
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


# below are ops that we profile and train in frontier:

# e2e_llm_generate

# mlp_up_proj
# mlp_act
# mlp_down_proj

# attn_pre_proj
# attn_rope
# attn_post_proj
# input_layernorm
# post_attention_layernorm

# attn_kv_cache_save
# attn_prefill
# attn_decode
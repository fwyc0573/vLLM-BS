#!/bin/bash
# Monolithic offline profiling launcher for simple_profiling.py
# Requirements:
#   - Single GPU (default: GPU 0)
#   - Conda env: vllm-bs-0.10.2
# tests/monolithic/offline_monolithic_profiling.sh --profile --model unsloth/Llama-3.2-1B-Instruct --gpu 7
# tests/monolithic/offline_monolithic_profiling.sh --model unsloth/Llama-3.2-1B-Instruct --gpu 7
# tests/monolithic/offline_monolithic_profiling.sh --profile --model microsoft/Phi-tiny-MoE-instruct --gpu 5
# tests/monolithic/offline_monolithic_profiling.sh  --gpu 6
set -euo pipefail

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT="/research/d1/gds/ytyang/yichengfeng/frontier/sota-infer-engine/vllm"
EXAMPLE_DIR="$PROJECT_ROOT/examples/offline_inference"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_LOG="$SCRIPT_DIR/offline_monolithic_profiling.log"

# -----------------------------------------------------------------------------
# Defaults (overridable via env or CLI)
# -----------------------------------------------------------------------------
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 # 允许突破原来model的max_model_len限制


GPU_ID=${GPU_ID:-1}
NUM_REQUESTS=${NUM_REQUESTS:-32}
PREFILL_TOKENS=${PREFILL_TOKENS:-512}
DECODE_TOKENS=${DECODE_TOKENS:-32}
SEED=${SEED:-42}
WARMUP_ITERS=${WARMUP_ITERS:-3}
ENABLE_PROFILE=${ENABLE_PROFILE:-0}
PROFILE_MAX_DECODE_TOKENS=${PROFILE_MAX_DECODE_TOKENS:-2048}
# microsoft/Phi-tiny-MoE-instruct
# unsloth/Llama-3.2-1B-Instruct
MODEL=${MODEL:-"mmnga/Mixtral-Fusion-4x7B-Instruct-v0.1"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
LOAD_FORMAT=${LOAD_FORMAT:-"auto"}  # 添加 load_format 支持

# -----------------------------------------------------------------------------
# CLI parsing (mirrors reference script flags where applicable)
# -----------------------------------------------------------------------------
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-requests) NUM_REQUESTS="$2"; shift 2 ;;
        --prefill-tokens) PREFILL_TOKENS="$2"; shift 2 ;;
        --decode-tokens) DECODE_TOKENS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --warmup-iters) WARMUP_ITERS="$2"; shift 2 ;;
        --profile) ENABLE_PROFILE=1; shift ;;
        --profile-max-decode-tokens) PROFILE_MAX_DECODE_TOKENS="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --gpu) GPU_ID="$2"; shift 2 ;;
        --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --load-format) LOAD_FORMAT="$2"; shift 2 ;;  # 添加 load-format 参数解析
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Environment setup (monolithic v1, no chunked prefill, no prefix caching)
# -----------------------------------------------------------------------------
export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_V1_ENABLE_CHUNKED_PREFILL=1
export VLLM_V1_ENABLE_PREFIX_CACHING=0

# Set custom temp directory to avoid /tmp space issues
CUSTOM_TMP_DIR="/research/d1/gds/ytyang/tmp_cuda"
mkdir -p "$CUSTOM_TMP_DIR"
export TMPDIR="$CUSTOM_TMP_DIR"
export TMP="$CUSTOM_TMP_DIR"
export TEMP="$CUSTOM_TMP_DIR"
export CUDA_CACHE_PATH="$CUSTOM_TMP_DIR/cuda_cache"
mkdir -p "$CUDA_CACHE_PATH"

# Profiling config (matches reference parameters when enabled)
if [[ $ENABLE_PROFILE -eq 1 ]]; then
    export VLLM_TORCH_PROFILER_DIR="$SCRIPT_DIR/profiles"
    export VLLM_TORCH_PROFILER_WITH_STACK=0
    export VLLM_CUSTOM_SCOPES_FOR_PROFILING=0
    # # ban torch.compile for profiling
    # export VLLM_TORCH_COMPILE_LEVEL=1
    mkdir -p "$VLLM_TORCH_PROFILER_DIR"
    PROFILE_ARG="--profile --profile-max-decode-tokens $PROFILE_MAX_DECODE_TOKENS"
else
    PROFILE_ARG=""
fi

# Attention backend and config root (aligned with reference)
export VLLM_ATTENTION_BACKEND=FLASHINFER
FLASHINFER_CACHE_DIR="$SCRIPT_DIR/flashinfer_cache"
mkdir -p "$FLASHINFER_CACHE_DIR/.cache/flashinfer"
export FLASHINFER_WORKSPACE_BASE="$FLASHINFER_CACHE_DIR"
export VLLM_CONFIG_ROOT="$PROJECT_ROOT/.vllm_config"

# -----------------------------------------------------------------------------
# Conda environment
# -----------------------------------------------------------------------------
source /research/d1/gds/ytyang/anaconda3/etc/profile.d/conda.sh
conda activate vllm-bs-0.10.2

# -----------------------------------------------------------------------------
# Runtime checks (fail fast, no fallbacks)
# -----------------------------------------------------------------------------
echo "=== Checking GPU $GPU_ID availability ==="
if nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits | grep -qE '[0-9]'; then
    echo "ERROR: GPU $GPU_ID is busy. Choose an idle GPU via GPU_ID env or --gpu."
    exit 1
fi

if [[ $ENABLE_PROFILE -eq 1 && -z "${VLLM_TORCH_PROFILER_DIR:-}" ]]; then
    echo "ERROR: Profiling enabled but VLLM_TORCH_PROFILER_DIR not set."
    exit 1
fi

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
cd "$SCRIPT_DIR"

{
    echo "=========================================="
    echo "Monolithic Offline Profiling"
    echo "=========================================="
    echo "Date: $(date)"
    echo "Working directory: $SCRIPT_DIR"
    echo "Python: $(which python)"
    echo "vLLM version: $(python -c 'import vllm; print(vllm.__version__)')"
    echo ""
    echo "Configuration:"
    echo "  - Model: $MODEL"
    echo "  - Load Format: $LOAD_FORMAT"  # 显示 load_format
    echo "  - Requests: $NUM_REQUESTS"
    echo "  - Prefill tokens: $PREFILL_TOKENS"
    echo "  - Decode tokens: $DECODE_TOKENS"
    echo "  - Seed: $SEED"
    echo "  - Warmup iterations: $WARMUP_ITERS"
    echo "  - Profiling: $([[ $ENABLE_PROFILE -eq 1 ]] && echo 'enabled' || echo 'disabled')"
    if [[ $ENABLE_PROFILE -eq 1 ]]; then
        echo "  - Profile max decode tokens: $PROFILE_MAX_DECODE_TOKENS"
        echo "  - Profiler dir: $VLLM_TORCH_PROFILER_DIR"
    fi
    echo "  - GPU: $GPU_ID"
    echo "  - GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
    echo "  - Custom Temp Dir: $TMPDIR"
    echo ""

    echo "=== Running monolithic profiling ==="
    echo "Command: CUDA_VISIBLE_DEVICES=$GPU_ID python $EXAMPLE_DIR/simple_profiling_fixed.py \\"
    echo "    --num-requests $NUM_REQUESTS \\"
    echo "    --prefill-tokens $PREFILL_TOKENS \\"
    echo "    --decode-tokens $DECODE_TOKENS \\"
    echo "    --seed $SEED \\"
    echo "    --warmup-iters $WARMUP_ITERS \\"
    echo "    --model \"$MODEL\" \\"
    echo "    --load-format $LOAD_FORMAT \\"
    echo "    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \\"
    echo "    $PROFILE_ARG"
    echo ""

    set -x
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    python "$EXAMPLE_DIR/simple_profiling_fixed.py" \
        --num-requests "$NUM_REQUESTS" \
        --prefill-tokens "$PREFILL_TOKENS" \
        --decode-tokens "$DECODE_TOKENS" \
        --seed "$SEED" \
        --warmup-iters "$WARMUP_ITERS" \
        --model "$MODEL" \
        --load-format "$LOAD_FORMAT" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        $PROFILE_ARG \
        $EXTRA_ARGS
    EXIT_CODE=$?
    set +x

    echo ""
    echo "Exit code: $EXIT_CODE"
    echo "=========================================="
    
    # Clean up temporary files
    echo "Cleaning up temporary files in $TMPDIR..."
    find "$TMPDIR" -name "tmpxft_*" -type f -delete 2>/dev/null || true
    echo "Cleanup completed."
} 2>&1 | tee "$OUTPUT_LOG"

exit "${EXIT_CODE:-0}"
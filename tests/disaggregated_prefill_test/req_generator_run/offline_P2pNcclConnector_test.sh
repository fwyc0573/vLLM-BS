#!/bin/bash
# Script to run disaggregated prefill-decode test with P2pNcclConnector
#
# This script tests the P2pNcclConnector backend for KV cache transfer
# between prefill and decode nodes using the Request Generator.
#
# IMPORTANT: This script uses separate torch compile cache directories for
# prefill and decode processes to avoid race conditions when compiling MoE models.
# Two types of cache isolation are implemented:
#   1. VLLM_CACHE_ROOT - Controls vLLM's torch_compile_cache location
#   2. TORCHINDUCTOR_CACHE_DIR - Controls PyTorch's aot_autograd and fxgraph cache
# The cache directories are automatically created in the script directory.
#
# Requirements:
#   - Two available GPUs (default: GPU 6 and GPU 7)
#   - Conda environment: vllm-bs-0.10.2
#
# Usage:
#   ./offline_P2pNcclConnector_test.sh                # Default settings
#   ./offline_P2pNcclConnector_test.sh --profile      # Enable profiling
#   ./offline_P2pNcclConnector_test.sh --num-requests 8 --prefill-tokens 512
#   ./offline_P2pNcclConnector_test.sh --profile --profile-max-decode-tokens 256
#   tests/disaggregated_prefill_test/req_generator_run/offline_P2pNcclConnector_test.sh --profile --gpu-prefill 1 --gpu-decode 2
#   tests/disaggregated_prefill_test/req_generator_run/offline_P2pNcclConnector_test.sh --gpu-prefill 1 --gpu-decode 2 --model "mmnga/Mixtral-Fusion-4x7B-Instruct-v0.1"
# Environment Variables:
#   GPU_PREFILL - GPU for prefill node (default: 6)
#   GPU_DECODE  - GPU for decode node (default: 7)
#   NUM_REQUESTS - Number of requests (default: 4)
#   PREFILL_TOKENS - Prefill tokens per request (default: 1024)
#   DECODE_TOKENS - Decode tokens per request (default: 64)

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

# Frontier path for request generator
export FRONTIER_PATH="/local/ycfeng/frontier"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/local/ycfeng/frontier/sota-infer-engine/vllm"
EXAMPLE_DIR="$PROJECT_ROOT/examples/offline_inference"
OUTPUT_LOG="$SCRIPT_DIR/p2p_nccl_connector_test_output.log"

# Default parameters (can be overridden via command line or environment variables)
NUM_REQUESTS=${NUM_REQUESTS:-8}
PREFILL_TOKENS=${PREFILL_TOKENS:-512}
DECODE_TOKENS=${DECODE_TOKENS:-2}
SEED=${SEED:-42}
WARMUP_ITERS=${WARMUP_ITERS:-3}
ENABLE_PROFILE=${ENABLE_PROFILE:-0}
PROFILE_MAX_DECODE_TOKENS=${PROFILE_MAX_DECODE_TOKENS:-256}

# GPU configuration - default to GPU 6 and 7 per user requirement
GPU_PREFILL=${GPU_PREFILL:-6}
GPU_DECODE=${GPU_DECODE:-7}

# Model configuration - align with SharedStorageConnector baseline defaults.
MODEL=${MODEL:-"unsloth/Llama-3.2-1B-Instruct"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.7}

# Timeout configuration
PREFILL_TIMEOUT=${PREFILL_TIMEOUT:-5000}
DECODE_TIMEOUT=${DECODE_TIMEOUT:-5000}

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
        --profile-max-decode-tokens)
            PROFILE_MAX_DECODE_TOKENS="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --gpu-prefill)
            GPU_PREFILL="$2"
            shift 2
            ;;
        --gpu-decode)
            GPU_DECODE="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --prefill-timeout)
            PREFILL_TIMEOUT="$2"
            shift 2
            ;;
        --decode-timeout)
            DECODE_TIMEOUT="$2"
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

# vLLM V1 engine configuration
export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_V1_ENABLE_CHUNKED_PREFILL=1
export VLLM_V1_ENABLE_PREFIX_CACHING=0

# Profiling configuration
# NOTE: Disabling heavy profiling options per user requirement
if [[ $ENABLE_PROFILE -eq 1 ]]; then
    export VLLM_TORCH_PROFILER_DIR="$SCRIPT_DIR/profiles"
    export VLLM_TORCH_PROFILER_WITH_STACK=0
    export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1
    mkdir -p "$VLLM_TORCH_PROFILER_DIR"
    PROFILE_ARG="--profile"
else
    PROFILE_ARG=""
fi

# Attention backend configuration
export VLLM_ATTENTION_BACKEND=FLASHINFER
FLASHINFER_CACHE_DIR="$SCRIPT_DIR/flashinfer_cache"
mkdir -p "$FLASHINFER_CACHE_DIR/.cache/flashinfer"
export FLASHINFER_WORKSPACE_BASE="$FLASHINFER_CACHE_DIR"

# Config directory
export VLLM_CONFIG_ROOT="$PROJECT_ROOT/.vllm_config"

# vLLM cache directories - separate for each process to avoid torch.compile race conditions
# This sets VLLM_CACHE_ROOT which controls the torch_compile_cache location
PREFILL_CACHE_DIR="$SCRIPT_DIR/vllm_cache_prefill"
DECODE_CACHE_DIR="$SCRIPT_DIR/vllm_cache_decode"
mkdir -p "$PREFILL_CACHE_DIR" "$DECODE_CACHE_DIR"

# PyTorch Inductor cache directories - separate for each process to avoid aot_autograd race conditions
# This is CRITICAL for MoE models where compilation takes longer and race conditions are more likely
# TORCHINDUCTOR_CACHE_DIR controls: aotautograd cache, fxgraph cache, and triton cache
PREFILL_INDUCTOR_CACHE_DIR="$SCRIPT_DIR/inductor_cache_prefill"
DECODE_INDUCTOR_CACHE_DIR="$SCRIPT_DIR/inductor_cache_decode"
mkdir -p "$PREFILL_INDUCTOR_CACHE_DIR" "$DECODE_INDUCTOR_CACHE_DIR"

# ============================================================================
# Conda Environment
# ============================================================================

source /local/ycfeng/anaconda3/etc/profile.d/conda.sh
conda activate vllm-bs-0.10.2

# ============================================================================
# Run Test
# ============================================================================

cd "$SCRIPT_DIR"

{
    echo "=========================================="
    echo "Disaggregated Prefill-Decode Test"
    echo "with P2pNcclConnector and Request Generator"
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
    if [[ $ENABLE_PROFILE -eq 1 ]]; then
        echo "  - Profile max decode tokens: $PROFILE_MAX_DECODE_TOKENS"
    fi
    echo "  - Seed: $SEED"
    echo "  - Warmup iterations: $WARMUP_ITERS"
    echo "  - Profiling: $([[ $ENABLE_PROFILE -eq 1 ]] && echo 'enabled' || echo 'disabled')"
    echo "  - GPU (Prefill): $GPU_PREFILL"
    echo "  - GPU (Decode): $GPU_DECODE"
    echo "  - GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
    echo "  - Prefill Timeout: ${PREFILL_TIMEOUT}s"
    echo "  - Decode Timeout: ${DECODE_TIMEOUT}s"
    echo "  - Prefill Cache Dir: $PREFILL_CACHE_DIR"
    echo "  - Decode Cache Dir: $DECODE_CACHE_DIR"
    echo "  - Prefill Inductor Cache Dir: $PREFILL_INDUCTOR_CACHE_DIR"
    echo "  - Decode Inductor Cache Dir: $DECODE_INDUCTOR_CACHE_DIR"
    echo ""
    
    # Check GPU availability
    echo "=== Checking GPU Availability ==="
    echo "Checking GPU $GPU_PREFILL and GPU $GPU_DECODE..."
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv | head -10
    echo ""

    if [[ "$GPU_PREFILL" -eq "$GPU_DECODE" ]]; then
        echo "ERROR: GPU_PREFILL ($GPU_PREFILL) and GPU_DECODE ($GPU_DECODE) must be different."
        exit 1
    fi

    echo "=== Checking GPU Occupancy ==="
    check_gpu_idle() {
        local gpu_id="$1"
        local procs
        procs="$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits | sed '/^$/d')"
        if [[ -n "$procs" ]]; then
            echo "ERROR: GPU $gpu_id is not idle. Found compute processes:"
            echo "$procs"
            echo ""
            echo "Please choose idle GPUs, e.g.:"
            echo "  GPU_PREFILL=0 GPU_DECODE=2 ./offline_P2pNcclConnector_test.sh"
            exit 1
        fi
    }
    check_gpu_idle "$GPU_PREFILL"
    check_gpu_idle "$GPU_DECODE"
    echo "OK: GPUs are idle."
    echo ""
    
    # Run the disaggregated prefill-decode test
    echo "=== Running Disaggregated Prefill-Decode Test ==="
    echo "Command (prefill): CUDA_VISIBLE_DEVICES=$GPU_PREFILL python $EXAMPLE_DIR/disaggregated_prefill.py \\"
    echo "    --role prefill \\"
    echo "    --num-requests $NUM_REQUESTS \\"
    echo "    --prefill-tokens $PREFILL_TOKENS \\"
    echo "    --decode-tokens $DECODE_TOKENS \\"
    echo "    --seed $SEED \\"
    echo "    --warmup-iters $WARMUP_ITERS \\"
    echo "    --model $MODEL \\"
    echo "    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \\"
    echo "    --kv-port \$KV_BASE_PORT \\"
    echo "    --sync-file \$SYNC_FILE \\"
    echo "    --prefill-timeout $PREFILL_TIMEOUT \\"
    echo "    $PROFILE_ARG"
    if [[ $ENABLE_PROFILE -eq 1 ]]; then
        echo "    --profile-max-decode-tokens $PROFILE_MAX_DECODE_TOKENS"
    fi
    echo ""

    KV_BASE_PORT="${KV_BASE_PORT:-$(
python - <<'PY'
import socket
start_port = 14579
block_size = 2
max_attempts = 100
for base in range(start_port, start_port + max_attempts):
    sockets = []
    try:
        for offset in range(block_size):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", base + offset))
            sockets.append(s)
        print(base)
        break
    except OSError:
        pass
    finally:
        for s in sockets:
            s.close()
else:
    raise SystemExit("Could not find a free port block for KV transfer.")
PY
)}"

    SYNC_FILE="$SCRIPT_DIR/prefill_done_${GPU_PREFILL}_${GPU_DECODE}_$$.marker"
    rm -f "$SYNC_FILE"

    prefill_pid=""
    cleanup() {
        if [[ -n "$prefill_pid" ]]; then
            kill "$prefill_pid" 2>/dev/null || true
            wait "$prefill_pid" 2>/dev/null || true
        fi
        rm -f "$SYNC_FILE" || true
        
        # Optional: Clean up torch compile cache directories
        # Uncomment the following lines if you want to clean up cache after each run
        # echo "Cleaning up torch compile cache directories..."
        # rm -rf "$PREFILL_CACHE_DIR" "$DECODE_CACHE_DIR" || true
    }
    trap cleanup EXIT

    echo "KV_BASE_PORT: $KV_BASE_PORT"
    echo "SYNC_FILE: $SYNC_FILE"
    if [[ $ENABLE_PROFILE -eq 1 ]]; then
        echo "Profiler traces: $VLLM_TORCH_PROFILER_DIR"
    fi
    echo ""

    # Start prefill (producer) in background.
    if [[ $ENABLE_PROFILE -eq 1 ]]; then
        CUDA_VISIBLE_DEVICES="$GPU_PREFILL" \
        VLLM_CACHE_ROOT="$PREFILL_CACHE_DIR" \
        TORCHINDUCTOR_CACHE_DIR="$PREFILL_INDUCTOR_CACHE_DIR" \
        python "$EXAMPLE_DIR/disaggregated_prefill.py" \
            --role prefill \
            --num-requests "$NUM_REQUESTS" \
            --prefill-tokens "$PREFILL_TOKENS" \
            --decode-tokens "$DECODE_TOKENS" \
            --seed "$SEED" \
            --warmup-iters "$WARMUP_ITERS" \
            --model "$MODEL" \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --kv-port "$KV_BASE_PORT" \
            --sync-file "$SYNC_FILE" \
            --prefill-timeout "$PREFILL_TIMEOUT" \
            $PROFILE_ARG \
            --profile-max-decode-tokens "$PROFILE_MAX_DECODE_TOKENS" \
            $EXTRA_ARGS \
            2>&1 &
    else
        CUDA_VISIBLE_DEVICES="$GPU_PREFILL" \
        VLLM_CACHE_ROOT="$PREFILL_CACHE_DIR" \
        TORCHINDUCTOR_CACHE_DIR="$PREFILL_INDUCTOR_CACHE_DIR" \
        python "$EXAMPLE_DIR/disaggregated_prefill.py" \
            --role prefill \
            --num-requests "$NUM_REQUESTS" \
            --prefill-tokens "$PREFILL_TOKENS" \
            --decode-tokens "$DECODE_TOKENS" \
            --seed "$SEED" \
            --warmup-iters "$WARMUP_ITERS" \
            --model "$MODEL" \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --kv-port "$KV_BASE_PORT" \
            --sync-file "$SYNC_FILE" \
            --prefill-timeout "$PREFILL_TIMEOUT" \
            $EXTRA_ARGS \
            2>&1 &
    fi
    prefill_pid=$!

    # Start decode (consumer) in background so we can enforce a timeout without
    # external dependencies.
    set +e
    if [[ $ENABLE_PROFILE -eq 1 ]]; then
        CUDA_VISIBLE_DEVICES="$GPU_DECODE" \
        VLLM_CACHE_ROOT="$DECODE_CACHE_DIR" \
        TORCHINDUCTOR_CACHE_DIR="$DECODE_INDUCTOR_CACHE_DIR" \
        python "$EXAMPLE_DIR/disaggregated_prefill.py" \
            --role decode \
            --num-requests "$NUM_REQUESTS" \
            --prefill-tokens "$PREFILL_TOKENS" \
            --decode-tokens "$DECODE_TOKENS" \
            --seed "$SEED" \
            --warmup-iters "$WARMUP_ITERS" \
            --model "$MODEL" \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --kv-port "$KV_BASE_PORT" \
            --sync-file "$SYNC_FILE" \
            --prefill-timeout "$PREFILL_TIMEOUT" \
            $PROFILE_ARG \
            --profile-max-decode-tokens "$PROFILE_MAX_DECODE_TOKENS" \
            $EXTRA_ARGS \
            2>&1 &
    else
        CUDA_VISIBLE_DEVICES="$GPU_DECODE" \
        VLLM_CACHE_ROOT="$DECODE_CACHE_DIR" \
        TORCHINDUCTOR_CACHE_DIR="$DECODE_INDUCTOR_CACHE_DIR" \
        python "$EXAMPLE_DIR/disaggregated_prefill.py" \
            --role decode \
            --num-requests "$NUM_REQUESTS" \
            --prefill-tokens "$PREFILL_TOKENS" \
            --decode-tokens "$DECODE_TOKENS" \
            --seed "$SEED" \
            --warmup-iters "$WARMUP_ITERS" \
            --model "$MODEL" \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --kv-port "$KV_BASE_PORT" \
            --sync-file "$SYNC_FILE" \
            --prefill-timeout "$PREFILL_TIMEOUT" \
            $EXTRA_ARGS \
            2>&1 &
    fi
    decode_pid=$!

    start_ts=$(date +%s)
    if ! kill -0 "$prefill_pid" 2>/dev/null; then
        echo "ERROR: Prefill process exited unexpectedly right after start."
        kill "$decode_pid" 2>/dev/null || true
        wait "$decode_pid" 2>/dev/null || true
        EXIT_CODE=1
    fi
    while kill -0 "$decode_pid" 2>/dev/null; do
        if ! kill -0 "$prefill_pid" 2>/dev/null; then
            echo "ERROR: Prefill process exited unexpectedly while decode is running."
            kill "$decode_pid" 2>/dev/null || true
            wait "$decode_pid" 2>/dev/null || true
            EXIT_CODE=1
            break
        fi
        now_ts=$(date +%s)
        elapsed=$((now_ts - start_ts))
        if (( elapsed > DECODE_TIMEOUT )); then
            echo "ERROR: Decode timed out after ${DECODE_TIMEOUT}s."
            kill "$decode_pid" 2>/dev/null || true
            wait "$decode_pid" 2>/dev/null || true
            EXIT_CODE=1
            break
        fi
        sleep 1
    done

    if [[ -z "${EXIT_CODE:-}" ]]; then
        wait "$decode_pid"
        EXIT_CODE=$?
    fi
    set -e

    # Clean up prefill process after decode finishes.
    kill "$prefill_pid" 2>/dev/null || true
    wait "$prefill_pid" 2>/dev/null || true

    echo ""
    echo "Exit code: $EXIT_CODE"
    echo ""
    
    echo "=========================================="
    echo "Test Complete"
    echo "Exit code: $EXIT_CODE"
    if [[ $ENABLE_PROFILE -eq 1 ]]; then
        echo "Profiler traces: $VLLM_TORCH_PROFILER_DIR"
    fi
    echo "=========================================="

} 2>&1 | tee "$OUTPUT_LOG" || true

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "Output saved to: $OUTPUT_LOG"
exit "$EXIT_CODE"


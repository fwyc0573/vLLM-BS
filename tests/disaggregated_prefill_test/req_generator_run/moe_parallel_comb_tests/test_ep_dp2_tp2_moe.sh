#!/bin/bash
# Test: EP + DP=2 + TP=2 MoE Parallel Configuration
# Description: Test P2pNcclConnector with Expert Parallel + Data Parallel size 2 + Tensor Parallel size 2 on MoE model
# Expected Result: Success - All three parallelism types should work together (no PP involved)
# GPU Allocation: Prefill=[0,1,2,3], Decode=[4,5,6,7] (requires 8 independent GPUs)
# Requirements: 6.3
# 
# IMPORTANT: This test requires 8 independent GPUs. Using overlapping GPUs between
# prefill and decode will cause NCCL communication deadlock.

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

# Frontier path for request generator
export FRONTIER_PATH="/research/d1/gds/ytyang/yichengfeng/frontier"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/research/d1/gds/ytyang/yichengfeng/frontier/sota-infer-engine/vllm"
EXAMPLE_DIR="$PROJECT_ROOT/examples/offline_inference"
MOE_SCRIPT="$SCRIPT_DIR/moe_disaggregated_prefill.py"
OUTPUT_LOG="$SCRIPT_DIR/logs/test_ep_dp2_tp2_moe_output.log"

# Test-specific configuration
TEST_NAME="EP + DP=2 + TP=2 MoE Test"
# MODEL="mmnga/Mixtral-Fusion-4x7B-Instruct-v0.1"
MODEL="microsoft/Phi-tiny-MoE-instruct"
TP_SIZE=2
PP_SIZE=1
DP_SIZE=2
ENABLE_EP=true

# GPU allocation for EP + DP=2 + TP=2: Need 4 GPUs for prefill (TP=2*DP=2), 4 GPUs for decode
# Using all 8 GPUs without overlap to avoid NCCL communication deadlock
# CRITICAL: GPU overlap between prefill and decode causes NCCL deadlock
GPU_PREFILL="0,1,2,3"
GPU_DECODE="4,5,6,7"

# Default parameters (can be overridden via command line or environment variables)
# Reduced for faster testing
NUM_REQUESTS=${NUM_REQUESTS:-2}
PREFILL_TOKENS=${PREFILL_TOKENS:-128}
DECODE_TOKENS=${DECODE_TOKENS:-4}
SEED=${SEED:-42}
WARMUP_ITERS=${WARMUP_ITERS:-1}
ENABLE_PROFILE=${ENABLE_PROFILE:-0}
PROFILE_MAX_DECODE_TOKENS=${PROFILE_MAX_DECODE_TOKENS:-256}

# Model configuration
# NOTE: Reduced from 0.8 to 0.7 to accommodate scenarios where GPUs may be
# partially occupied by other processes. This provides more headroom for
# memory allocation while still allowing efficient model loading.
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
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_V1_ENABLE_CHUNKED_PREFILL=1
export VLLM_V1_ENABLE_PREFIX_CACHING=0

# Profiling configuration
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
PREFILL_CACHE_DIR="$SCRIPT_DIR/vllm_cache_prefill_ep_dp2_tp2"
DECODE_CACHE_DIR="$SCRIPT_DIR/vllm_cache_decode_ep_dp2_tp2"
mkdir -p "$PREFILL_CACHE_DIR" "$DECODE_CACHE_DIR"

# PyTorch Inductor cache directories - separate for each process to avoid aot_autograd race conditions
PREFILL_INDUCTOR_CACHE_DIR="$SCRIPT_DIR/inductor_cache_prefill_ep_dp2_tp2"
DECODE_INDUCTOR_CACHE_DIR="$SCRIPT_DIR/inductor_cache_decode_ep_dp2_tp2"
mkdir -p "$PREFILL_INDUCTOR_CACHE_DIR" "$DECODE_INDUCTOR_CACHE_DIR"

# ============================================================================
# Conda Environment
# ============================================================================

source /research/d1/gds/ytyang/anaconda3/etc/profile.d/conda.sh
conda activate vllm-bs-0.10.2

# ============================================================================
# Run Test
# ============================================================================

cd "$SCRIPT_DIR"

# Ensure logs directory exists
mkdir -p "$SCRIPT_DIR/logs"

{
    echo "=========================================="
    echo "$TEST_NAME"
    echo "with P2pNcclConnector and Request Generator"
    echo "=========================================="
    echo "Date: $(date)"
    echo "Working directory: $SCRIPT_DIR"
    echo "Python: $(which python)"
    echo "vLLM version: $(python -c 'import vllm; print(vllm.__version__)')"
    echo ""
    echo "Configuration:"
    echo "  - Model: $MODEL"
    echo "  - Tensor Parallel Size: $TP_SIZE"
    echo "  - Pipeline Parallel Size: $PP_SIZE"
    echo "  - Data Parallel Size: $DP_SIZE"
    echo "  - Expert Parallel: $ENABLE_EP"
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
    echo "Checking GPUs: Prefill=$GPU_PREFILL, Decode=$GPU_DECODE"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv | head -10
    echo ""

    # Validate GPU allocation - check for overlap
    IFS=',' read -ra PREFILL_GPUS <<< "$GPU_PREFILL"
    IFS=',' read -ra DECODE_GPUS <<< "$GPU_DECODE"
    
    echo "=== Validating GPU Allocation ==="
    echo "Prefill GPUs: ${PREFILL_GPUS[*]}"
    echo "Decode GPUs: ${DECODE_GPUS[*]}"
    
    # Check for GPU overlap (which would cause NCCL deadlock)
    for p_gpu in "${PREFILL_GPUS[@]}"; do
        for d_gpu in "${DECODE_GPUS[@]}"; do
            if [[ "$p_gpu" == "$d_gpu" ]]; then
                echo "ERROR: GPU $p_gpu is used by both prefill and decode!"
                echo "GPU overlap between prefill and decode causes NCCL communication deadlock."
                echo "Please ensure 8 independent GPUs are available."
                exit 1
            fi
        done
    done
    echo "GPU allocation validated - no overlap detected."
    echo ""

    echo "=== Checking GPU Occupancy ==="
    check_gpu_idle() {
        local gpu_id="$1"
        local procs
        procs="$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits | sed '/^$/d')"
        if [[ -n "$procs" ]]; then
            echo "WARNING: GPU $gpu_id is not idle. Found compute processes:"
            echo "$procs"
            echo ""
            return 1
        fi
        return 0
    }
    
    # Check all allocated GPUs
    GPU_CHECK_FAILED=0
    for gpu in "${PREFILL_GPUS[@]}"; do
        if ! check_gpu_idle "$gpu"; then
            GPU_CHECK_FAILED=1
        fi
    done
    for gpu in "${DECODE_GPUS[@]}"; do
        if ! check_gpu_idle "$gpu"; then
            GPU_CHECK_FAILED=1
        fi
    done
    
    if [[ $GPU_CHECK_FAILED -eq 1 ]]; then
        echo "WARNING: Some required GPUs are occupied."
        echo "This test requires 8 independent GPUs: Prefill=$GPU_PREFILL, Decode=$GPU_DECODE"
        echo "Proceeding anyway - test may fail if GPU memory is insufficient."
        echo ""
    else
        echo "GPU check complete - all GPUs are available."
    fi
    echo ""
    
    # Run the disaggregated prefill-decode test with EP + DP=2 + TP=2
    echo "=== Running $TEST_NAME ==="
    echo "Command (prefill): CUDA_VISIBLE_DEVICES=$GPU_PREFILL python $MOE_SCRIPT \\"
    echo "    --role prefill \\"
    echo "    --tensor-parallel-size $TP_SIZE \\"
    echo "    --data-parallel-size $DP_SIZE \\"
    echo "    --enable-expert-parallel \\"
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

    # Calculate port block size based on parallelism configuration
    # P2pNcclConnector uses: port = kv_port + kv_rank * world_size + rank
    # where world_size = DP_SIZE * TP_SIZE and kv_parallel_size = 2 (prefill + decode)
    # Formula: block_size = kv_parallel_size * world_size = 2 * (DP_SIZE * TP_SIZE)
    PORT_BLOCK_SIZE=$((2 * DP_SIZE * TP_SIZE))
    
    KV_BASE_PORT="${KV_BASE_PORT:-$(
python - "$PORT_BLOCK_SIZE" <<'PY'
import socket
import sys
start_port = 14579
block_size = int(sys.argv[1]) if len(sys.argv) > 1 else 8
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

    SYNC_FILE="$SCRIPT_DIR/prefill_done_ep_dp2_tp2_$.marker"
    rm -f "$SYNC_FILE"
    # Clean up all sync files (prefill_done, decode_ready, prefill_ready, prefill_warmup_done)
    rm -f "${SYNC_FILE/prefill_done/decode_ready}"
    rm -f "${SYNC_FILE/prefill_done/prefill_ready}"
    rm -f "${SYNC_FILE/prefill_done/prefill_warmup_done}"

    prefill_pid=""
    cleanup() {
        if [[ -n "$prefill_pid" ]]; then
            kill "$prefill_pid" 2>/dev/null || true
            wait "$prefill_pid" 2>/dev/null || true
        fi
        # Clean up all sync files
        rm -f "$SYNC_FILE" || true
        rm -f "${SYNC_FILE/prefill_done/decode_ready}" || true
        rm -f "${SYNC_FILE/prefill_done/prefill_ready}" || true
        rm -f "${SYNC_FILE/prefill_done/prefill_warmup_done}" || true
    }
    trap cleanup EXIT

    echo "KV_BASE_PORT: $KV_BASE_PORT"
    echo "PORT_BLOCK_SIZE: $PORT_BLOCK_SIZE"
    echo "SYNC_FILE: $SYNC_FILE"
    if [[ $ENABLE_PROFILE -eq 1 ]]; then
        echo "Profiler traces: $VLLM_TORCH_PROFILER_DIR"
    fi
    echo ""

    # Start prefill (producer) in background with EP + DP=2 + TP=2
    # CRITICAL: Use different VLLM_DP_MASTER_PORT for prefill and decode to avoid
    # DP group initialization conflicts. Without this, prefill and decode processes
    # may try to join the same DP group, causing deadlock.
    PREFILL_DP_MASTER_PORT=29500
    DECODE_DP_MASTER_PORT=29600
    
    if [[ $ENABLE_PROFILE -eq 1 ]]; then
        CUDA_VISIBLE_DEVICES="$GPU_PREFILL" \
        VLLM_CACHE_ROOT="$PREFILL_CACHE_DIR" \
        TORCHINDUCTOR_CACHE_DIR="$PREFILL_INDUCTOR_CACHE_DIR" \
        VLLM_DP_MASTER_PORT="$PREFILL_DP_MASTER_PORT" \
        python "$MOE_SCRIPT" \
            --role prefill \
            --tensor-parallel-size "$TP_SIZE" \
            --data-parallel-size "$DP_SIZE" \
            --enable-expert-parallel \
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
        VLLM_DP_MASTER_PORT="$PREFILL_DP_MASTER_PORT" \
        python "$MOE_SCRIPT" \
            --role prefill \
            --tensor-parallel-size "$TP_SIZE" \
            --data-parallel-size "$DP_SIZE" \
            --enable-expert-parallel \
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

    # Start decode (consumer) in background with EP + DP=2 + TP=2
    # NOTE: Decode uses the same KV_BASE_PORT as Prefill because P2pNcclConnector
    # uses kv_rank to calculate different port offsets for producer and consumer.
    # Port allocation: port = kv_port + kv_rank * world_size + rank
    # - Prefill (kv_rank=0): ports kv_port+0 to kv_port+(world_size-1)
    # - Decode (kv_rank=1): ports kv_port+world_size to kv_port+(2*world_size-1)
    set +e
    if [[ $ENABLE_PROFILE -eq 1 ]]; then
        CUDA_VISIBLE_DEVICES="$GPU_DECODE" \
        VLLM_CACHE_ROOT="$DECODE_CACHE_DIR" \
        TORCHINDUCTOR_CACHE_DIR="$DECODE_INDUCTOR_CACHE_DIR" \
        VLLM_DP_MASTER_PORT="$DECODE_DP_MASTER_PORT" \
        python "$MOE_SCRIPT" \
            --role decode \
            --tensor-parallel-size "$TP_SIZE" \
            --data-parallel-size "$DP_SIZE" \
            --enable-expert-parallel \
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
        VLLM_DP_MASTER_PORT="$DECODE_DP_MASTER_PORT" \
        python "$MOE_SCRIPT" \
            --role decode \
            --tensor-parallel-size "$TP_SIZE" \
            --data-parallel-size "$DP_SIZE" \
            --enable-expert-parallel \
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

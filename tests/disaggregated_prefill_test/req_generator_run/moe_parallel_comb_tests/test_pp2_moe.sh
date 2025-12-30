#!/bin/bash
# Test: PP=2 MoE Parallel Configuration
# Description: Test P2pNcclConnector with Pipeline Parallel size 2 on MoE model
# Expected Result: FAIL - P2pNcclConnector does not support Pipeline Parallel
# GPU Allocation: Prefill=[0,1], Decode=[2,3]
# Requirements: 3.1, 3.2, 3.5

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
OUTPUT_LOG="$SCRIPT_DIR/logs/test_pp2_moe_output.log"

# Test-specific configuration
TEST_NAME="PP=2 MoE Test"
MODEL="mmnga/Mixtral-Fusion-4x7B-Instruct-v0.1"
TP_SIZE=1
PP_SIZE=2
DP_SIZE=1
ENABLE_EP=false

# GPU allocation for PP=2: Prefill uses [0,1], Decode uses [2,3]
# Note: PP=2 means 2 pipeline stages, so we need 2 GPUs per instance
GPU_PREFILL="0,1"
GPU_DECODE="2,3"

# Default parameters (can be overridden via command line or environment variables)
NUM_REQUESTS=${NUM_REQUESTS:-8}
PREFILL_TOKENS=${PREFILL_TOKENS:-512}
DECODE_TOKENS=${DECODE_TOKENS:-2}
SEED=${SEED:-42}
WARMUP_ITERS=${WARMUP_ITERS:-3}
ENABLE_PROFILE=${ENABLE_PROFILE:-0}
PROFILE_MAX_DECODE_TOKENS=${PROFILE_MAX_DECODE_TOKENS:-256}

# Model configuration
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
export VLLM_ENABLE_V1_MULTIPROCESSING=0
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
PREFILL_CACHE_DIR="$SCRIPT_DIR/vllm_cache_prefill_pp2"
DECODE_CACHE_DIR="$SCRIPT_DIR/vllm_cache_decode_pp2"
mkdir -p "$PREFILL_CACHE_DIR" "$DECODE_CACHE_DIR"

# PyTorch Inductor cache directories - separate for each process to avoid aot_autograd race conditions
PREFILL_INDUCTOR_CACHE_DIR="$SCRIPT_DIR/inductor_cache_prefill_pp2"
DECODE_INDUCTOR_CACHE_DIR="$SCRIPT_DIR/inductor_cache_decode_pp2"
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
    echo "IMPORTANT: This test is EXPECTED TO FAIL"
    echo "Reason: P2pNcclConnector does not support Pipeline Parallel (PP)"
    echo "Source: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:501"
    echo ""
    
    # Check GPU availability
    echo "=== Checking GPU Availability ==="
    echo "Checking GPUs: Prefill=$GPU_PREFILL, Decode=$GPU_DECODE"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv | head -10
    echo ""

    # Validate GPU allocation
    IFS=',' read -ra PREFILL_GPUS <<< "$GPU_PREFILL"
    IFS=',' read -ra DECODE_GPUS <<< "$GPU_DECODE"
    
    # Check for GPU overlap
    for prefill_gpu in "${PREFILL_GPUS[@]}"; do
        for decode_gpu in "${DECODE_GPUS[@]}"; do
            if [[ "$prefill_gpu" -eq "$decode_gpu" ]]; then
                echo "ERROR: GPU $prefill_gpu is allocated to both prefill and decode."
                exit 1
            fi
        done
    done

    echo "=== Checking GPU Occupancy ==="
    check_gpu_idle() {
        local gpu_id="$1"
        local procs
        procs="$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits | sed '/^$/d')"
        if [[ -n "$procs" ]]; then
            echo "ERROR: GPU $gpu_id is not idle. Found compute processes:"
            echo "$procs"
            echo ""
            echo "Please choose idle GPUs."
            exit 1
        fi
    }
    
    # Check all allocated GPUs
    for gpu in "${PREFILL_GPUS[@]}"; do
        check_gpu_idle "$gpu"
    done
    for gpu in "${DECODE_GPUS[@]}"; do
        check_gpu_idle "$gpu"
    done
    echo "OK: All GPUs are idle."
    echo ""
    
    # Run the disaggregated prefill-decode test with PP=2
    echo "=== Running $TEST_NAME ==="
    echo "Command (prefill): CUDA_VISIBLE_DEVICES=$GPU_PREFILL python $MOE_SCRIPT \\"
    echo "    --role prefill \\"
    echo "    --pipeline-parallel-size $PP_SIZE \\"
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

    SYNC_FILE="$SCRIPT_DIR/prefill_done_pp2_$.marker"
    rm -f "$SYNC_FILE"

    prefill_pid=""
    cleanup() {
        if [[ -n "$prefill_pid" ]]; then
            kill "$prefill_pid" 2>/dev/null || true
            wait "$prefill_pid" 2>/dev/null || true
        fi
        rm -f "$SYNC_FILE" || true
    }
    trap cleanup EXIT

    echo "KV_BASE_PORT: $KV_BASE_PORT"
    echo "SYNC_FILE: $SYNC_FILE"
    if [[ $ENABLE_PROFILE -eq 1 ]]; then
        echo "Profiler traces: $VLLM_TORCH_PROFILER_DIR"
    fi
    echo ""

    # Start prefill (producer) in background with PP=2
    if [[ $ENABLE_PROFILE -eq 1 ]]; then
        CUDA_VISIBLE_DEVICES="$GPU_PREFILL" \
        VLLM_CACHE_ROOT="$PREFILL_CACHE_DIR" \
        TORCHINDUCTOR_CACHE_DIR="$PREFILL_INDUCTOR_CACHE_DIR" \
        python "$MOE_SCRIPT" \
            --role prefill \
            --pipeline-parallel-size "$PP_SIZE" \
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
        python "$MOE_SCRIPT" \
            --role prefill \
            --pipeline-parallel-size "$PP_SIZE" \
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

    # Start decode (consumer) in background with PP=2
    set +e
    if [[ $ENABLE_PROFILE -eq 1 ]]; then
        CUDA_VISIBLE_DEVICES="$GPU_DECODE" \
        VLLM_CACHE_ROOT="$DECODE_CACHE_DIR" \
        TORCHINDUCTOR_CACHE_DIR="$DECODE_INDUCTOR_CACHE_DIR" \
        python "$MOE_SCRIPT" \
            --role decode \
            --pipeline-parallel-size "$PP_SIZE" \
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
        python "$MOE_SCRIPT" \
            --role decode \
            --pipeline-parallel-size "$PP_SIZE" \
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
    
    # Analyze the failure
    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "=== EXPECTED FAILURE ANALYSIS ==="
        echo "This test was expected to fail because:"
        echo "1. P2pNcclConnector does not support Pipeline Parallel (PP)"
        echo "2. The limitation is documented in the source code"
        echo "3. Any PP > 1 configuration will fail with P2pNcclConnector"
        echo ""
        echo "This confirms the architectural limitation of P2pNcclConnector."
        echo ""
    fi
    
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
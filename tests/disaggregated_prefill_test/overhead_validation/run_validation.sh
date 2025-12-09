#!/bin/bash
# ============================================================================
# PROFILING OVERHEAD VALIDATION - Main Orchestration Script
# ============================================================================
#
# Usage: ./run_validation.sh [phase] [--dry-run]
#
# Examples:
#   ./run_validation.sh           # Run all phases
#   ./run_validation.sh 1         # Run only Phase 1
#   ./run_validation.sh 1 --dry-run  # Dry run Phase 1
#
# Environment Requirements:
#   - conda environment: vllm-bs-0.10.2
#   - CUDA_VISIBLE_DEVICES will be se
#   - Models should be pre-downloaded
# ============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
REPORTS_DIR="$SCRIPT_DIR/reports"

# Parse arguments
PHASE="${1:-all}"
DRY_RUN=0
if [[ "$2" == "--dry-run" ]] || [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=1
    if [[ "$1" == "--dry-run" ]]; then
        PHASE="all"
    fi
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# Environment Setup
# ============================================================================

setup_environment() {
    log_info "Setting up environment..."
    
    # Activate conda environment
    source /research/d1/gds/ytyang/anaconda3/etc/profile.d/conda.sh
    conda activate vllm-bs-0.10.2
    
    # Set environment variables
    export CUDA_VISIBLE_DEVICES=0
    export VLLM_USE_V1=1
    export VLLM_ATTENTION_BACKEND=FLASHINFER
    export VLLM_ENABLE_V1_MULTIPROCESSING=0
    export VLLM_CONFIG_ROOT="$PROJECT_ROOT/.vllm_config"
    
    # FlashInfer cache
    export FLASHINFER_WORKSPACE_BASE="$SCRIPT_DIR/flashinfer_cache"
    mkdir -p "$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer"
    
    # Profiler settings
    export VLLM_TORCH_PROFILER_DIR="$SCRIPT_DIR/profiles"
    export VLLM_TORCH_PROFILER_WITH_STACK=1
    mkdir -p "$VLLM_TORCH_PROFILER_DIR"
    
    log_success "Environment configured"
}

# ============================================================================
# Test Execution Functions
# ============================================================================

run_single_test() {
    local TEST_ID=$1
    local MODEL=$2
    local BATCH_SIZE=$3
    local SEQ_LEN=$4
    local PROFILING=$5
    local SCOPE_CONFIG=$6
    local RUN_INDEX=$7
    local PHASE=$8
    
    local OUTPUT_FILE="$RESULTS_DIR/phase$PHASE/${TEST_ID}_run${RUN_INDEX}.json"
    
    log_info "Running: $TEST_ID (Run $RUN_INDEX)"
    log_info "  Model=$MODEL, Batch=$BATCH_SIZE, Seq=$SEQ_LEN, Prof=$PROFILING"
    
    if [[ $DRY_RUN -eq 1 ]]; then
        log_warn "[DRY RUN] Would execute test"
        return 0
    fi
    
    # Set profiling environment
    if [[ "$PROFILING" == "1" ]]; then
        export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1
    else
        export VLLM_CUSTOM_SCOPES_FOR_PROFILING=0
    fi
    
    # Run the test
    python "$SCRIPT_DIR/overhead_test.py" \
        --test-id "$TEST_ID" \
        --model "$MODEL" \
        --batch-size "$BATCH_SIZE" \
        --seq-len "$SEQ_LEN" \
        --profiling "$PROFILING" \
        --scope-config "$SCOPE_CONFIG" \
        --run-index "$RUN_INDEX" \
        --output "$OUTPUT_FILE"
    
    local EXIT_CODE=$?
    
    if [[ $EXIT_CODE -eq 0 ]]; then
        log_success "Test $TEST_ID Run $RUN_INDEX completed"
    else
        log_error "Test $TEST_ID Run $RUN_INDEX failed"
    fi
    
    return $EXIT_CODE
}

run_test_with_repetitions() {
    local TEST_ID=$1
    local MODEL=$2
    local BATCH_SIZE=$3
    local SEQ_LEN=$4
    local PROFILING=$5
    local SCOPE_CONFIG=$6
    local NUM_RUNS=$7
    local PHASE=$8
    
    log_info "=========================================="
    log_info "Test: $TEST_ID ($NUM_RUNS runs)"
    log_info "=========================================="
    
    local SUCCESS_COUNT=0
    
    for ((i=1; i<=NUM_RUNS; i++)); do
        run_single_test "$TEST_ID" "$MODEL" "$BATCH_SIZE" "$SEQ_LEN" \
            "$PROFILING" "$SCOPE_CONFIG" "$i" "$PHASE"
        
        if [[ $? -eq 0 ]]; then
            ((SUCCESS_COUNT++))
        fi
        
        # Cooldown between runs
        if [[ $i -lt $NUM_RUNS ]] && [[ $DRY_RUN -eq 0 ]]; then
            log_info "Cooldown (5s)..."
            sleep 5
        fi
    done
    
    log_info "Test $TEST_ID: $SUCCESS_COUNT/$NUM_RUNS runs succeeded"
    return 0
}

# ============================================================================
# Phase Execution
# ============================================================================

run_phase1() {
    log_info "============================================"
    log_info "PHASE 1: MODEL SIZE VARIATION"
    log_info "============================================"
    
    mkdir -p "$RESULTS_DIR/phase1"
    
    local MODELS=("1B" "3B" "8B")
    local NUM_RUNS=5
    local TEST_IDX=1
    
    for MODEL in "${MODELS[@]}"; do
        # With profiling
        run_test_with_repetitions "P1-${TEST_IDX}A" "$MODEL" 4 1024 1 "full" $NUM_RUNS 1
        
        # GPU cooldown between configurations
        if [[ $DRY_RUN -eq 0 ]]; then
            log_info "Configuration cooldown (10s)..."
            sleep 10
        fi
        
        # Without profiling
        run_test_with_repetitions "P1-${TEST_IDX}B" "$MODEL" 4 1024 0 "none" $NUM_RUNS 1
        
        ((TEST_IDX++))
        
        if [[ $DRY_RUN -eq 0 ]]; then
            log_info "Model cooldown (15s)..."
            sleep 15
        fi
    done
    
    log_success "Phase 1 completed"
}

run_phase2() {
    log_info "============================================"
    log_info "PHASE 2: BATCH/SEQUENCE VARIATION"
    log_info "============================================"
    
    mkdir -p "$RESULTS_DIR/phase2"
    
    local NUM_RUNS=5
    local TEST_IDX=1
    
    # Test configurations: (batch, seq_len)
    local CONFIGS=(
        "1 512"
        "1 2048"
        "8 512"
        "8 1024"
    )
    
    for CONFIG in "${CONFIGS[@]}"; do
        read -r BATCH SEQ <<< "$CONFIG"
        
        # With profiling
        run_test_with_repetitions "P2-${TEST_IDX}A" "1B" "$BATCH" "$SEQ" 1 "full" $NUM_RUNS 2
        
        if [[ $DRY_RUN -eq 0 ]]; then
            sleep 10
        fi
        
        # Without profiling
        run_test_with_repetitions "P2-${TEST_IDX}B" "1B" "$BATCH" "$SEQ" 0 "none" $NUM_RUNS 2
        
        ((TEST_IDX++))
        
        if [[ $DRY_RUN -eq 0 ]]; then
            sleep 10
        fi
    done
    
    log_success "Phase 2 completed"
}

run_phase3() {
    log_info "============================================"
    log_info "PHASE 3: SCOPE COUNT ISOLATION"
    log_info "============================================"
    
    mkdir -p "$RESULTS_DIR/phase3"
    
    local NUM_RUNS=5
    
    # Scope configurations: (config_name, profiling_on)
    local CONFIGS=(
        "full 1"
        "model 1"
        "attn 1"
        "mlp 1"
        "none 0"
    )
    
    local SUFFIXES=("A" "B" "C" "D" "E")
    local IDX=0
    
    for CONFIG in "${CONFIGS[@]}"; do
        read -r SCOPE_CFG PROFILING <<< "$CONFIG"
        local SUFFIX="${SUFFIXES[$IDX]}"
        
        run_test_with_repetitions "P3-1${SUFFIX}" "1B" 4 1024 "$PROFILING" "$SCOPE_CFG" $NUM_RUNS 3
        
        ((IDX++))
        
        if [[ $DRY_RUN -eq 0 ]]; then
            sleep 10
        fi
    done
    
    log_success "Phase 3 completed"
}

# ============================================================================
# Analysis
# ============================================================================

run_analysis() {
    log_info "============================================"
    log_info "RUNNING ANALYSIS"
    log_info "============================================"
    
    if [[ $DRY_RUN -eq 1 ]]; then
        log_warn "[DRY RUN] Would run analysis"
        return 0
    fi
    
    python "$SCRIPT_DIR/analyze_results.py" \
        --results-dir "$RESULTS_DIR" \
        --output-dir "$REPORTS_DIR"
    
    log_success "Analysis completed"
}

# ============================================================================
# Main
# ============================================================================

main() {
    echo "=============================================="
    echo "PROFILING OVERHEAD VALIDATION"
    echo "=============================================="
    echo "Phase: $PHASE"
    echo "Dry Run: $([[ $DRY_RUN -eq 1 ]] && echo 'Yes' || echo 'No')"
    echo "Script Dir: $SCRIPT_DIR"
    echo "Results Dir: $RESULTS_DIR"
    echo "=============================================="
    
    # Setup
    setup_environment
    
    # Create directories
    mkdir -p "$RESULTS_DIR" "$REPORTS_DIR"
    
    # Run phases
    case $PHASE in
        "1")
            run_phase1
            ;;
        "2")
            run_phase2
            ;;
        "3")
            run_phase3
            ;;
        "all")
            run_phase1
            run_phase2
            run_phase3
            run_analysis
            ;;
        "analyze")
            run_analysis
            ;;
        *)
            log_error "Unknown phase: $PHASE"
            echo "Usage: $0 [1|2|3|all|analyze] [--dry-run]"
            exit 1
            ;;
    esac
    
    log_success "Validation complete!"
}

# Run main
main "$@"

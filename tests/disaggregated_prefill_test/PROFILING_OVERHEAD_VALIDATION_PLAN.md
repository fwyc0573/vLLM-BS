# Profiling Overhead Validation Plan

## Document Information

| Field | Value |
|-------|-------|
| **Document Version** | 1.0 |
| **Created** | 2025-12-08 |
| **Author** | Profiling Analysis Team |
| **Status** | Ready for Execution |

### Modification History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-08 | Team | Initial experimental design document |

---

## 1. Executive Summary

### 1.1 Objective

Validate and generalize the profiling overhead findings from the initial Llama-3.2-1B analysis:
- **Per-scope overhead**: ~123.4 μs per `record_function` call
- **CPU correction factor**: 0.340× (profiled → true)
- **CUDA impact**: Negligible (+0.03%)

### 1.2 Key Questions to Answer

1. Is the per-scope overhead (~123.4 μs) **consistent** across different model configurations?
2. Does the overhead scale **linearly** with the number of scopes?
3. Are model architecture characteristics (hidden size, layer count) **independent** of overhead?
4. What is the **variance** of the overhead measurements?

### 1.3 Success Criteria

| Criterion | Target | Validation Method |
|-----------|--------|-------------------|
| Per-scope overhead consistency | 123.4 μs ± 25% | Compare across all configurations |
| Linear scaling (R²) | > 0.95 | Regression analysis on scope count vs overhead |
| CUDA time impact | < 1% | Compare with/without profiling |
| Measurement stability | CV < 10% | 5 runs per configuration |

---

## 2. Experimental Design

### 2.1 Test Dimensions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EXPERIMENTAL DIMENSIONS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Dimension 1: MODEL SIZE                                                    │
│  ├── Llama-3.2-1B  (16 layers, 2048 hidden)   ← Baseline                   │
│  ├── Llama-3.2-3B  (28 layers, 3072 hidden)                                │
│  └── Llama-3-8B    (32 layers, 4096 hidden)                                │
│                                                                             │
│  Dimension 2: BATCH × SEQUENCE LENGTH                                       │
│  ├── Batch: 1, 4, 8                                                        │
│  └── Seq Length: 512, 1024, 2048 tokens                                    │
│                                                                             │
│  Dimension 3: SCOPE COUNT ISOLATION                                         │
│  ├── Full instrumentation (all scopes)                                     │
│  ├── Model-only scopes (no runner scopes)                                  │
│  ├── Attention-only scopes                                                 │
│  └── MLP-only scopes                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Model Configurations

| Model | Layers | Hidden | Heads | KV Heads | Intermediate | Scopes/Forward | VRAM (BF16) |
|-------|--------|--------|-------|----------|--------------|----------------|-------------|
| `meta-llama/Llama-3.2-1B-Instruct` | 16 | 2048 | 32 | 8 | 8192 | ~181 | ~2 GB |
| `meta-llama/Llama-3.2-3B-Instruct` | 28 | 3072 | 24 | 8 | 8192 | ~309 | ~6 GB |
| `meta-llama/Meta-Llama-3-8B-Instruct` | 32 | 4096 | 32 | 8 | 14336 | ~353 | ~16 GB |

**Scope Count Formula:**
```
Scopes_per_forward = 5 (runner) + L × 10 (per-layer) + L × 2 (attention backend)
                   = 5 + 12L

For L=16: 5 + 12×16 = 197 (actual ~181 due to some conditional scopes)
For L=28: 5 + 12×28 = 341 (estimated ~309)
For L=32: 5 + 12×32 = 389 (estimated ~353)
```

### 2.3 Batch × Sequence Length Matrix

| Config ID | Batch Size | Seq Length | Total Tokens | Expected Token Pattern |
|-----------|------------|------------|--------------|------------------------|
| BS1_S512 | 1 | 512 | 512 | Single short sequence |
| BS1_S1024 | 1 | 1024 | 1024 | Single medium sequence |
| BS1_S2048 | 1 | 2048 | 2048 | Single long sequence |
| BS4_S512 | 4 | 512 | 2048 | Multiple short sequences |
| BS4_S1024 | 4 | 1024 | 4096 | Multiple medium sequences |
| BS4_S2048 | 4 | 2048 | 8192 | Multiple long sequences |
| BS8_S512 | 8 | 512 | 4096 | Many short sequences |
| BS8_S1024 | 8 | 1024 | 8192 | Many medium sequences |

### 2.4 Scope Isolation Configurations

| Config | Model Scopes | Attention Backend Scopes | Runner Scopes | Expected Scope Count (L=16) |
|--------|--------------|--------------------------|---------------|----------------------------|
| `FULL` | ✓ All | ✓ All | ✓ All | ~181 |
| `MODEL_ONLY` | ✓ All | ✗ Disabled | ✗ Disabled | ~160 |
| `ATTN_ONLY` | ✓ Attention | ✗ Disabled | ✗ Disabled | ~80 |
| `MLP_ONLY` | ✓ MLP | ✗ Disabled | ✗ Disabled | ~48 |
| `NONE` | ✗ Disabled | ✗ Disabled | ✗ Disabled | 0 |

---

## 3. Test Matrix

### 3.1 Phase 1: Model Size Variation (Primary Experiment)

**Objective**: Validate per-scope overhead consistency across model sizes

| Test ID | Model | Batch | Seq Len | Profiling | Runs | Priority |
|---------|-------|-------|---------|-----------|------|----------|
| P1-1A | Llama-3.2-1B | 4 | 1024 | ON | 5 | HIGH |
| P1-1B | Llama-3.2-1B | 4 | 1024 | OFF | 5 | HIGH |
| P1-2A | Llama-3.2-3B | 4 | 1024 | ON | 5 | HIGH |
| P1-2B | Llama-3.2-3B | 4 | 1024 | OFF | 5 | HIGH |
| P1-3A | Llama-3-8B | 4 | 1024 | ON | 5 | HIGH |
| P1-3B | Llama-3-8B | 4 | 1024 | OFF | 5 | HIGH |

**Expected Output**: Per-scope overhead for each model size

### 3.2 Phase 2: Batch/Sequence Variation

**Objective**: Confirm overhead is independent of workload characteristics

| Test ID | Model | Batch | Seq Len | Profiling | Runs | Priority |
|---------|-------|-------|---------|-----------|------|----------|
| P2-1A | Llama-3.2-1B | 1 | 512 | ON | 5 | MEDIUM |
| P2-1B | Llama-3.2-1B | 1 | 512 | OFF | 5 | MEDIUM |
| P2-2A | Llama-3.2-1B | 1 | 2048 | ON | 5 | MEDIUM |
| P2-2B | Llama-3.2-1B | 1 | 2048 | OFF | 5 | MEDIUM |
| P2-3A | Llama-3.2-1B | 8 | 512 | ON | 5 | MEDIUM |
| P2-3B | Llama-3.2-1B | 8 | 512 | OFF | 5 | MEDIUM |
| P2-4A | Llama-3.2-1B | 8 | 1024 | ON | 5 | MEDIUM |
| P2-4B | Llama-3.2-1B | 8 | 1024 | OFF | 5 | MEDIUM |

**Expected Output**: Overhead values across batch/sequence configurations

### 3.3 Phase 3: Scope Count Isolation (Linearity Test)

**Objective**: Validate linear scaling of overhead with scope count

| Test ID | Model | Scope Config | Expected Scopes | Profiling | Runs | Priority |
|---------|-------|--------------|-----------------|-----------|------|----------|
| P3-1A | Llama-3.2-1B | FULL | ~181 | ON | 5 | HIGH |
| P3-1B | Llama-3.2-1B | MODEL_ONLY | ~160 | ON | 5 | HIGH |
| P3-1C | Llama-3.2-1B | ATTN_ONLY | ~80 | ON | 5 | HIGH |
| P3-1D | Llama-3.2-1B | MLP_ONLY | ~48 | ON | 5 | HIGH |
| P3-1E | Llama-3.2-1B | NONE | 0 | OFF | 5 | HIGH |

**Expected Output**: Linear regression R² > 0.95

---

## 4. Metrics Collection

### 4.1 Primary Metrics

| Metric | Source | Unit | Description |
|--------|--------|------|-------------|
| `self_cpu_time` | PyTorch Profiler | ms | Total self CPU time |
| `self_cuda_time` | PyTorch Profiler | ms | Total self CUDA time |
| `total_scope_calls` | Log parsing | count | Number of `record_function` invocations |
| `e2e_latency` | Wall clock | ms | End-to-end inference latency |

### 4.2 Derived Metrics

| Metric | Formula | Unit |
|--------|---------|------|
| `cpu_overhead` | `cpu_with - cpu_without` | ms |
| `overhead_per_scope` | `cpu_overhead / total_scope_calls` | μs |
| `correction_factor` | `cpu_without / cpu_with` | ratio |
| `cuda_impact` | `(cuda_with - cuda_without) / cuda_without × 100` | % |

### 4.3 Statistical Metrics

| Metric | Description |
|--------|-------------|
| Mean | Average across N runs |
| Std Dev | Standard deviation |
| CV (Coefficient of Variation) | `std_dev / mean × 100%` |
| 95% CI | Confidence interval |

---

## 5. Implementation

### 5.1 Directory Structure

```
tests/disaggregated_prefill_test/
├── PROFILING_OVERHEAD_VALIDATION_PLAN.md   # This document
├── PROFILING_OVERHEAD_ANALYSIS.md          # Initial analysis results
├── overhead_validation/
│   ├── run_validation.sh                   # Main orchestration script
│   ├── run_single_test.sh                  # Single test runner
│   ├── test_config.py                      # Test configuration
│   ├── analyze_results.py                  # Results analysis
│   ├── results/                            # Raw results directory
│   │   ├── phase1/
│   │   ├── phase2/
│   │   └── phase3/
│   └── reports/                            # Analysis reports
```

### 5.2 Test Configuration File

```python
# test_config.py
TEST_CONFIGS = {
    "models": {
        "1B": "meta-llama/Llama-3.2-1B-Instruct",
        "3B": "meta-llama/Llama-3.2-3B-Instruct",
        "8B": "meta-llama/Meta-Llama-3-8B-Instruct",
    },
    "batch_seq_configs": [
        {"batch": 1, "seq_len": 512},
        {"batch": 1, "seq_len": 1024},
        {"batch": 1, "seq_len": 2048},
        {"batch": 4, "seq_len": 512},
        {"batch": 4, "seq_len": 1024},
        {"batch": 4, "seq_len": 2048},
        {"batch": 8, "seq_len": 512},
        {"batch": 8, "seq_len": 1024},
    ],
    "scope_configs": ["FULL", "MODEL_ONLY", "ATTN_ONLY", "MLP_ONLY", "NONE"],
    "runs_per_config": 5,
    "warmup_runs": 1,
}
```

### 5.3 Single Test Runner Script

```bash
#!/bin/bash
# run_single_test.sh
# Usage: ./run_single_test.sh <model> <batch> <seq_len> <profiling_on> <output_file>

MODEL=$1
BATCH_SIZE=$2
SEQ_LEN=$3
PROFILING_ON=$4
OUTPUT_FILE=$5

export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_ENABLE_V1_MULTIPROCESSING=0

if [ "$PROFILING_ON" = "1" ]; then
    export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1
else
    export VLLM_CUSTOM_SCOPES_FOR_PROFILING=0
fi

python overhead_test.py \
    --model "$MODEL" \
    --batch-size "$BATCH_SIZE" \
    --seq-len "$SEQ_LEN" \
    --output "$OUTPUT_FILE"
```

---

## 6. Expected Outcomes

### 6.1 Hypothesis Testing

| Hypothesis | Expected Result | Validation |
|------------|-----------------|------------|
| H1: Per-scope overhead is constant | ~123.4 μs ± 25% across all models | Compare overhead/scope across Phase 1 |
| H2: Overhead scales linearly with scope count | R² > 0.95 | Linear regression on Phase 3 data |
| H3: Batch/seq length don't affect overhead | CV < 15% across Phase 2 | Compare overhead/scope across configs |
| H4: CUDA time is unaffected | Impact < 1% | Compare CUDA times with/without profiling |

### 6.2 Expected Results Table

| Configuration | Est. Scopes | Est. Overhead (ms) | Est. Overhead/Scope (μs) |
|---------------|-------------|-------------------|-------------------------|
| Llama-1B, Full | ~181 | ~22.3 | ~123 |
| Llama-3B, Full | ~309 | ~38.1 | ~123 |
| Llama-8B, Full | ~353 | ~43.5 | ~123 |
| Llama-1B, ATTN_ONLY | ~80 | ~9.9 | ~123 |
| Llama-1B, MLP_ONLY | ~48 | ~5.9 | ~123 |

### 6.3 Statistical Validation Criteria

| Metric | Threshold | Action if Failed |
|--------|-----------|------------------|
| CV (per config) | < 10% | Increase runs to N=10 |
| R² (linearity) | > 0.95 | Investigate non-linear factors |
| Per-scope variance | < 25% | Review outlier configurations |
| CUDA impact | < 1% | Confirm profiling isolation |

---

## 7. Execution Timeline

### 7.1 Estimated Runtime

| Phase | Tests | Runs/Test | Est. Time/Run | Total Time |
|-------|-------|-----------|---------------|------------|
| Phase 1 | 6 | 5 | ~2 min | ~60 min |
| Phase 2 | 8 | 5 | ~2 min | ~80 min |
| Phase 3 | 5 | 5 | ~2 min | ~50 min |
| **Total** | **19** | **5** | - | **~3.5 hours** |

### 7.2 Execution Order

```
1. Phase 1: Model Size Variation (Priority: HIGH)
   └── Establishes baseline per-scope overhead for each model
   
2. Phase 3: Scope Count Isolation (Priority: HIGH)
   └── Validates linear scaling hypothesis
   
3. Phase 2: Batch/Sequence Variation (Priority: MEDIUM)
   └── Confirms workload independence
   
4. Analysis & Report Generation
   └── Compile results, compute statistics, generate final report
```

---

## 8. Analysis Methodology

### 8.1 Per-Configuration Analysis

For each test configuration:

```python
def analyze_config(results: list[dict]) -> dict:
    """Analyze results from N runs of a single configuration."""
    cpu_times = [r['self_cpu_time'] for r in results]
    cuda_times = [r['self_cuda_time'] for r in results]
    
    return {
        'cpu_mean': np.mean(cpu_times),
        'cpu_std': np.std(cpu_times),
        'cpu_cv': np.std(cpu_times) / np.mean(cpu_times) * 100,
        'cuda_mean': np.mean(cuda_times),
        'cuda_std': np.std(cuda_times),
        'cuda_cv': np.std(cuda_times) / np.mean(cuda_times) * 100,
    }
```

### 8.2 Overhead Calculation

```python
def calculate_overhead(with_prof: dict, without_prof: dict, scope_count: int) -> dict:
    """Calculate profiling overhead metrics."""
    cpu_overhead = with_prof['cpu_mean'] - without_prof['cpu_mean']
    cuda_overhead = with_prof['cuda_mean'] - without_prof['cuda_mean']
    
    return {
        'cpu_overhead_ms': cpu_overhead,
        'cuda_overhead_ms': cuda_overhead,
        'overhead_per_scope_us': (cpu_overhead * 1000) / scope_count,
        'correction_factor': without_prof['cpu_mean'] / with_prof['cpu_mean'],
        'cuda_impact_pct': (cuda_overhead / without_prof['cuda_mean']) * 100,
    }
```

### 8.3 Linearity Analysis (Phase 3)

```python
from scipy import stats

def analyze_linearity(scope_counts: list[int], overheads: list[float]) -> dict:
    """Perform linear regression to validate overhead scaling."""
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        scope_counts, overheads
    )
    
    return {
        'slope_us_per_scope': slope * 1000,  # Convert to μs
        'intercept_ms': intercept,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'std_error': std_err,
        'is_linear': r_value ** 2 > 0.95,
    }
```

---

## 9. Deliverables

### 9.1 Final Report Contents

1. **Summary Statistics Table**
   - Per-scope overhead for each model
   - Correction factors
   - Confidence intervals

2. **Linearity Analysis**
   - Scatter plot: Scope count vs. Overhead
   - Linear regression parameters
   - R² value and significance

3. **Generalized Correction Formula**
   ```
   True_CPU_time = Profiled_CPU_time - (N_scopes × overhead_per_scope)
   ```

4. **Recommendations**
   - Validated correction factor for simulator
   - Guidelines for different model configurations

### 9.2 Output Files

| File | Description |
|------|-------------|
| `results/raw_data.json` | All raw measurement data |
| `results/summary_statistics.csv` | Aggregated statistics per config |
| `reports/linearity_analysis.png` | Regression plot |
| `reports/VALIDATION_RESULTS.md` | Final comprehensive report |

---

## 10. Risk Mitigation

### 10.1 Potential Issues

| Risk | Mitigation |
|------|------------|
| High variance in measurements | Increase N to 10, add more warmup |
| Model loading failures | Pre-download all model weights |
| GPU memory issues (8B model) | Reduce batch size or use FP8 |
| Non-linear overhead | Investigate scope interaction effects |

### 10.2 Contingency Plans

1. **If R² < 0.95**: Investigate quadratic or logarithmic models
2. **If variance > 25%**: Add GPU temperature monitoring, increase cooldown
3. **If 8B model OOM**: Fall back to 3B as largest test case

---

## Appendix A: Scope Instrumentation Locations

### A.1 Model Layer Scopes (`llama.py`)

| Scope | Location | Function |
|-------|----------|----------|
| `input_layernorm` | `LlamaDecoderLayer.forward()` | Pre-attention RMSNorm |
| `attn_pre_proj` | `LlamaAttention.forward()` | QKV projection |
| `attn_rope` | `LlamaAttention.forward()` | Rotary position embedding |
| `attn` | `LlamaAttention.forward()` | Attention wrapper call |
| `attn_post_proj` | `LlamaAttention.forward()` | Output projection |
| `post_attention_layernorm` | `LlamaDecoderLayer.forward()` | Post-attention RMSNorm |
| `mlp_up_proj` | `LlamaMLP.forward()` | Gate/up projection |
| `mlp_act` | `LlamaMLP.forward()` | SiLU activation |
| `mlp_down_proj` | `LlamaMLP.forward()` | Down projection |

### A.2 Attention Backend Scopes (`flashinfer.py`)

| Scope | Location | Function |
|-------|----------|----------|
| `attn_kv_cache_save` | `FlashInferImpl.forward()` | KV cache write |
| `attn_prefill` | `FlashInferImpl.forward()` | Prefill attention kernel |
| `attn_decode` | `FlashInferImpl.forward()` | Decode attention kernel |

### A.3 Model Runner Scopes (`gpu_model_runner.py`)

| Scope | Location | Function |
|-------|----------|----------|
| `Preprocess` | `execute_model()` | Input preparation |
| `Forward` | `execute_model()` | Model forward pass |
| `Postprocess` | `execute_model()` | Output processing |
| `Sample` | `execute_model()` | Token sampling |
| `Bookkeep` | `execute_model()` | State management |

---

## Appendix B: Environment Setup

### B.1 Required Environment

```bash
# Conda environment
conda activate vllm-bs-0.10.2

# Environment variables
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_TORCH_PROFILER_DIR=./profiles
export VLLM_TORCH_PROFILER_WITH_STACK=1
```

### B.2 Hardware Requirements

| Model | Min VRAM | Recommended |
|-------|----------|-------------|
| Llama-3.2-1B | 4 GB | 8 GB |
| Llama-3.2-3B | 8 GB | 16 GB |
| Llama-3-8B | 20 GB | 40 GB |

### B.3 Model Pre-download

```bash
# Pre-download models to avoid network latency during tests
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct
```


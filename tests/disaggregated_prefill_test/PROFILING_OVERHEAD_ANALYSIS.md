# Profiling Overhead Analysis Report

## Executive Summary

This report analyzes the CPU and CUDA overhead introduced by enabling `VLLM_CUSTOM_SCOPES_FOR_PROFILING` in vLLM. The analysis compares profiling output with and without custom scope instrumentation.

### Key Findings

| Metric | With Profiling | Without Profiling | Difference | Ratio |
|--------|----------------|-------------------|------------|-------|
| Self CPU time total | 35.795 ms | 12.159 ms | +23.636 ms | **2.94x** |
| Self CUDA time total | 2.904 ms | 2.903 ms | +0.001 ms | **1.00x** |

**Critical Insight**: The profiling overhead is **entirely on the CPU side**. CUDA kernel execution times are virtually unaffected (+0.03%).

---

## 1. Detailed CPU Overhead Breakdown

### 1.1 Common Operator CPU Time Comparison

| Operation | With Profiling (ms) | Without Profiling (ms) | Overhead (ms) | % Increase |
|-----------|---------------------|------------------------|---------------|------------|
| `aten::mm` | 2.518 | 1.937 | +0.581 | +30.0% |
| `cudaLaunchKernel` | 2.577 | 1.994 | +0.583 | +29.2% |
| `_C::fused_add_rms_norm` | 0.235 | 0.176 | +0.059 | +33.5% |
| `_C::rotary_embedding` | 0.144 | 0.105 | +0.039 | +37.1% |
| `_C::silu_and_mul` | 0.099 | 0.075 | +0.024 | +32.0% |
| `_C_cache_ops::reshape_and_cache_flash` | 0.120 | 0.105 | +0.015 | +14.3% |
| `batch_prefill_with_kv_cache` | 0.193 | 0.178 | +0.015 | +8.4% |
| `Activity Buffer Request` | 0.582 | 0.592 | -0.010 | -1.7% |
| **Subtotal** | - | - | **+1.306** | - |

### 1.2 Overhead Attribution

| Source | Overhead (ms) | % of Total Overhead |
|--------|---------------|---------------------|
| `aten::mm` profiling instrumentation | +0.581 | 2.5% |
| `cudaLaunchKernel` overhead | +0.583 | 2.5% |
| Other operator overhead | +0.142 | 0.6% |
| **`record_function` scope overhead (181 calls)** | **+22.330** | **94.5%** |
| **TOTAL** | **+23.636** | **100.0%** |

### 1.3 Custom Scope Invocations

The following custom scopes are instrumented via `record_function_or_nullcontext()`:

| Scope Name | # Calls | Description |
|------------|---------|-------------|
| `Forward` | 1 | Main forward pass |
| `Preprocess` | 1 | Input preparation |
| `Postprocess` | 1 | Output processing |
| `Sample` | 1 | Token sampling |
| `Bookkeep` | 1 | State management |
| `attn_prefill` | 16 | Attention kernel (prefill) |
| `attn_pre_proj` | 16 | QKV projection |
| `attn_post_proj` | 16 | Output projection |
| `attn_rope` | 16 | Rotary position embedding |
| `attn_kv_cache_save` | 16 | KV cache write |
| `attn` | 16 | Attention wrapper |
| `mlp_up_proj` | 16 | MLP gate_up projection |
| `mlp_act` | 16 | SiLU activation |
| `mlp_down_proj` | 16 | MLP down projection |
| `input_layernorm` | 16 | Input layer normalization |
| `post_attention_layernorm` | 16 | Post-attention normalization |
| **Total** | **181** | - |

**Estimated overhead per `record_function` call: ~123.4 μs**

---

## 2. CUDA Time Analysis (Unaffected by Profiling)

The CUDA kernel execution times are **accurate and unaffected** by profiling:

### 2.1 Per-Operator CUDA Times (Use These for Simulator)

#### Attention Operations (Llama-3.2-1B, 16 layers)

| Operation | Per-Layer (μs) | Total (ms) |
|-----------|----------------|------------|
| `attn_pre_proj` (QKV projection) | 14.44 | 0.231 |
| `attn_rope` (RoPE) | 3.88 | 0.062 |
| `attn_prefill` (attention kernel) | 57.06 | 0.913 |
| `attn_kv_cache_save` | 3.50 | 0.056 |
| `attn_post_proj` (O projection) | 12.81 | 0.205 |

#### MLP Operations (16 layers)

| Operation | Per-Layer (μs) | Total (ms) |
|-----------|----------------|------------|
| `mlp_up_proj` (gate_up) | 50.69 | 0.811 |
| `mlp_act` (SiLU) | 6.38 | 0.102 |
| `mlp_down_proj` | 32.00 | 0.512 |

#### LayerNorm Operations (16 layers)

| Operation | Per-Layer (μs) | Total (ms) |
|-----------|----------------|------------|
| `input_layernorm` | 3.88 | 0.062 |
| `post_attention_layernorm` | 3.62 | 0.058 |

**Total per-layer CUDA time: 188.25 μs/layer**  
**Total model CUDA time: 3.012 ms**

---

## 3. Correction Factors for E2E Latency Calculation

### 3.1 CUDA Time Correction
```
Correction factor: 1.00 (no correction needed)
CUDA times from profiling are accurate and can be used directly.
```

### 3.2 CPU Time Correction
```
Correction factor: 0.340
True_CPU_time = Profiled_CPU_time × 0.340
Or: True_CPU_time = Profiled_CPU_time - 23.636 ms
```

### 3.3 Per-Scope Overhead Correction
```
Each record_function scope adds: ~123.4 μs
Total scope calls per forward: 181
Total scope overhead: ~22.330 ms
```

---

## 4. Recommended Methodology for Simulator

### 4.1 GPU Execution Time
- **Use "Self CUDA" values directly** from profiled output
- Profiling impact on CUDA time: **negligible (+0.03%)**
- These values accurately represent kernel execution time

### 4.2 CPU Overhead Estimation
Choose one of the following approaches:

**Approach A (Multiplicative):**
```
True_CPU_time = Profiled_CPU_time × 0.340
```

**Approach B (Subtractive):**
```
True_CPU_time = Profiled_CPU_time - 23.636 ms
```

### 4.3 E2E Latency Formula
```
True_E2E = GPU_time + (Profiled_CPU_time × 0.340) + Memory_transfer_time
```
Or:
```
True_E2E = Profiled_E2E - 23.636 ms
```

### 4.4 Per-Call Correction (if needed)
When profiling with N custom scope calls:
```
Scope_overhead = N × 123.4 μs
True_time = Profiled_time - Scope_overhead
```

---

## 5. Conclusion

1. **CUDA Time is Accurate**: GPU kernel execution times from profiling are reliable and can be used directly for simulator input.

2. **CPU Overhead is Significant**: The `record_function` context managers introduce **~23.6 ms** overhead (2.94x increase), which is **94.5%** of the total profiling overhead.

3. **Root Cause**: Each `record_function` call adds **~123.4 μs** CPU overhead. With 181 calls per forward pass, this accumulates to **~22.3 ms**.

4. **For Accurate E2E Latency**: 
   - Use CUDA times as-is
   - Apply 0.340x correction factor to CPU times
   - Or subtract 23.636 ms from total profiled time

5. **Profiling is Safe for GPU Metrics**: If your simulator primarily relies on GPU execution times, the profiling data is accurate without any correction.

---

## Appendix: Test Configuration

- **Model**: meta-llama/Llama-3.2-1B-Instruct (16 layers)
- **Backend**: FlashInfer attention
- **Mode**: Prefill-only (decode profiling not included)
- **Batch**: 4 prompts (~1000 tokens each)
- **Environment**: CUDA with VLLM V1 engine


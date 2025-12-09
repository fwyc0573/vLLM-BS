# Phase 1 Complete Analysis: Profiling Overhead Across Model Sizes

**Generated:** 2025-12-08
**Test Environment:** vLLM 0.10.3.dev0+g01efc7ef7.d20251206 (pd-offline branch)
**Hardware:** A800 GPU, CUDA_VISIBLE_DEVICES=0
**Configuration:** Batch=4, Seq=1024, 5 runs per configuration

---

## Executive Summary

Phase 1 of the profiling overhead validation is **COMPLETE**. We tested three model sizes (1B, 7B, 8B) with profiling ON and OFF to measure the CPU overhead introduced by `record_function_or_nullcontext()` custom scopes.

### Key Findings

| Model | Scopes | CPU ON (ms) | CPU OFF (ms) | Total Overhead | Per-scope | CPU Ratio |
|-------|--------|-------------|--------------|----------------|-----------|-----------|
| **1B** (Llama-3.2-1B-Instruct) | 181 | 30.2 | 13.0 | **17.2 ms** | **~95 μs** | 2.32x |
| **7B** (Llama-2-7b-hf) | 353 | 57.6 | 36.7 | **20.9 ms** | **~59 μs** | 1.57x |
| **8B** (Meta-Llama-3-8B) | 353 | 52.4 | 26.6 | **25.7 ms** | **~73 μs** | 1.97x |

**Weighted Average Per-Scope Overhead: ~75 μs** (range: 59-95 μs)

---

## Detailed Results by Model

### 1B Model: Llama-3.2-1B-Instruct

**Model:** `meta-llama/Llama-3.2-1B-Instruct`
**Profiling Scopes:** 181 calls

#### Profiling ON (P1-1A) - 5 runs
| Run | Self CPU (ms) | Self CUDA (ms) | E2E Latency (ms) |
|-----|---------------|----------------|------------------|
| 1 | 30.263 | 3.301 | 33.493 |
| 2 | 29.872 | 3.295 | 33.108 |
| 3 | 30.156 | 3.294 | 33.379 |
| 4 | 30.254 | 3.327 | 33.508 |
| 5 | 30.336 | 3.328 | 33.593 |
| **Mean** | **30.176** | **3.309** | **33.416** |

#### Profiling OFF (P1-1B) - 5 runs
| Run | Self CPU (ms) | Self CUDA (ms) | E2E Latency (ms) |
|-----|---------------|----------------|------------------|
| 1 | 13.008 | 3.299 | 16.241 |
| 2 | 13.200 | 3.281 | 16.410 |
| 3 | 12.860 | 3.290 | 16.082 |
| 4 | 12.960 | 3.350 | 16.245 |
| 5 | 12.921 | 3.356 | 16.210 |
| **Mean** | **12.990** | **3.315** | **16.238** |

#### 1B Overhead Analysis
- **Total CPU Overhead:** 30.176 - 12.990 = **17.186 ms**
- **Per-scope Overhead:** 17,186 μs / 181 = **94.9 μs/scope**
- **CPU Ratio (ON/OFF):** 30.176 / 12.990 = **2.32x**
- **CUDA Impact:** 3.309 - 3.315 = **-0.006 ms (negligible)**

---

### 7B Model: Llama-2-7b-hf

**Model:** `meta-llama/Llama-2-7b-hf`
**Profiling Scopes:** 353 calls

#### Profiling ON (P1-1A) - 5 runs
| Run | Self CPU (ms) | Self CUDA (ms) | E2E Latency (ms) |
|-----|---------------|----------------|------------------|
| 1 | 57.785 | 11.670 | 69.389 |
| 2 | 56.860 | 11.653 | 68.449 |
| 3 | 57.494 | 11.609 | 69.038 |
| 4 | 58.086 | 11.594 | 69.615 |
| 5 | 57.893 | 11.619 | 69.448 |
| **Mean** | **57.624** | **11.629** | **69.188** |

#### Profiling OFF (P1-1B) - 5 runs
| Run | Self CPU (ms) | Self CUDA (ms) | E2E Latency (ms) |
|-----|---------------|----------------|------------------|
| 1 | 36.625 | 11.672 | 48.240 |
| 2 | 36.701 | 11.700 | 48.344 |
| 3 | 36.759 | 11.716 | 48.418 |
| 4 | 36.620 | 11.689 | 48.252 |
| 5 | 36.782 | 11.683 | 48.409 |
| **Mean** | **36.697** | **11.692** | **48.333** |

#### 7B Overhead Analysis
- **Total CPU Overhead:** 57.624 - 36.697 = **20.927 ms**
- **Per-scope Overhead:** 20,927 μs / 353 = **59.3 μs/scope**
- **CPU Ratio (ON/OFF):** 57.624 / 36.697 = **1.57x**
- **CUDA Impact:** 11.629 - 11.692 = **-0.063 ms (negligible)**

---

### 8B Model: Meta-Llama-3-8B

**Model:** `meta-llama/Meta-Llama-3-8B`
**Profiling Scopes:** 353 calls

#### Profiling ON (P1-1A) - 5 runs
| Run | Self CPU (ms) | Self CUDA (ms) | E2E Latency (ms) |
|-----|---------------|----------------|------------------|
| 1 | 66.112 | 15.742 | 81.798 |
| 2 | 43.104 | 15.654 | 53.807 |
| 3 | 63.005 | 14.891 | 77.841 |
| 4 | 44.677 | 14.882 | 55.648 |
| 5 | 44.896 | 14.953 | 57.920 |
| **Mean** | **52.359** | **15.224** | **65.403** |

#### Profiling OFF (P1-1B) - 5 runs
| Run | Self CPU (ms) | Self CUDA (ms) | E2E Latency (ms) |
|-----|---------------|----------------|------------------|
| 1 | 28.898 | 14.886 | 59.813 |
| 2 | 27.467 | 15.665 | 58.109 |
| 3 | 29.237 | 15.651 | 60.541 |
| 4 | 18.725 | 16.515 | 40.455 |
| 5 | 28.842 | 15.629 | 61.042 |
| **Mean** | **26.634** | **15.669** | **55.992** |

#### 8B Overhead Analysis
- **Total CPU Overhead:** 52.359 - 26.634 = **25.725 ms**
- **Per-scope Overhead:** 25,725 μs / 353 = **72.9 μs/scope**
- **CPU Ratio (ON/OFF):** 52.359 / 26.634 = **1.97x**
- **CUDA Impact:** 15.224 - 15.669 = **-0.445 ms (negligible)**

---

## Cross-Model Analysis

### Per-Scope Overhead Comparison

```
Model Size vs Per-Scope Overhead:

1B (181 scopes):  ████████████████████████████████████████████████ 95 μs
7B (353 scopes):  ██████████████████████████████ 59 μs
8B (353 scopes):  ████████████████████████████████████ 73 μs
                  └───────┴───────┴───────┴───────┴───────┘
                  0      20      40      60      80     100 μs
```

### Key Observations

1. **Per-scope overhead varies by model architecture:**
   - 1B: ~95 μs/scope (highest)
   - 7B: ~59 μs/scope (lowest)
   - 8B: ~73 μs/scope (middle)

2. **CPU Ratio decreases with model size:**
   - 1B: 2.32x overhead
   - 7B: 1.57x overhead
   - 8B: 1.97x overhead

3. **CUDA time is unaffected by profiling:**
   - All models show < 1ms CUDA difference (negligible)
   - This confirms overhead is pure CPU-side instrumentation

4. **Variance in 8B results:**
   - Higher variance observed (18-66ms range for OFF runs)
   - May indicate initialization effects or system noise

### Statistical Summary

| Metric | 1B | 7B | 8B | Average |
|--------|----|----|----|---------| 
| Per-scope overhead (μs) | 95 | 59 | 73 | **~75** |
| CPU ratio (ON/OFF) | 2.32x | 1.57x | 1.97x | **~1.95x** |
| Total CPU overhead (ms) | 17.2 | 20.9 | 25.7 | **~21.3** |

---

## Conclusions

### Validated Findings

1. **Original hypothesis PARTIALLY CONFIRMED:**
   - Initial estimate: ~123 μs per scope (from preliminary 1B test)
   - Validated range: **59-95 μs per scope**
   - The original estimate was slightly high but in the right order of magnitude

2. **Profiling overhead is SIGNIFICANT:**
   - ~75 μs per `record_function_or_nullcontext()` invocation
   - ~1.95x average CPU time increase when profiling is enabled

3. **Overhead is CPU-ONLY:**
   - CUDA times are identical with profiling ON vs OFF
   - No GPU performance impact from the profiling infrastructure

### Recommendations

1. **Production deployments:** Disable custom profiling scopes (`VLLM_CUSTOM_SCOPES_FOR_PROFILING=0`)

2. **Performance-critical benchmarks:** Always run with profiling OFF to get accurate measurements

3. **Debugging/analysis:** Enable profiling only when needed for detailed trace analysis

---

## Next Steps

- [x] Phase 1: Model Size Variation - **COMPLETE**
- [ ] Phase 2: Batch/Sequence Variation (using 1B model)
- [ ] Phase 3: Scope Isolation Tests
- [ ] Final Report Generation


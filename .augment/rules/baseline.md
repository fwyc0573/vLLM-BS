---
type: "always_apply"
description: "targets and rules of vllm running"
---

I am developing an LLM inference simulator. I have completed the main workflow of the simulator and now need to run the current vLLM project (version 0.10.2) to perform comprehensive testing and comparison across various aspects. Our high-level goal is to conduct detailed profiling and tracing using existing tools (e.g., PyTorch Profiler, Nsight Systems) and/or by adding print/log statements within vLLM code.

## Key Information

### 1. Environment Setup
- **Build Method**: Built from source
- **Conda Environment**: `vllm-bs-0.10.2` located at `/research/d1/gds/ytyang/anaconda3/envs/vllm-bs-0.10.2`
- **Requirement**: Always activate this conda environment before running vLLM commands

### 2. Three Critical Profiling Dimensions

Please focus on the following three aspects when modifying code, running experiments, and generating outputs:

#### a. **Operator-Level Profiling**
- **Target**: Individual operator performance (execution time)
- **Key Operators**: 
  - Flash Attention (during prefill and decode phases)
  - MLP operations
  - MoE (Mixture of Experts) structure operations
- **Required Information for Each Operator**:
  - Input characteristics: batch composition, request details, and parameters that may affect execution speed (e.g., sequence length, batch size, number of tokens)
  - Execution time (latency in milliseconds or microseconds)

#### b. **Flow-Level Profiling**
- **Target**: Critical control flow decisions and behaviors that significantly impact execution time and memory allocation
- **Key Components**:
  - Scheduling logic and decisions
  - Admission control mechanisms
  - Batching strategies (continuous batching, dynamic batching)
  - PagedAttention memory management
  - Synchronization points (sync/async operations, barriers)
  - GPU execution flow

#### c. **System-Level Profiling**
- **Target**: Overall performance metrics and per-request time breakdown
- **Overall Metrics**:
  - Throughput (requests/second, tokens/second)
  - Latency (TTFT - Time To First Token, TPOT - Time Per Output Token, E2E latency)
  - Memory usage (GPU memory, KV cache memory)
- **Per-Request Time Breakdown** (this is critical):
  - **Prefill Phase**:
    - Compute time
    - Communication time (if applicable)
    - Waiting time (queuing, scheduling delays)
  - **Decode Phase**:
    - Compute time per decode step
    - Communication time per decode step (if applicable)
    - Waiting time per decode step
    - Track each individual decode step separately
  - **KV Cache Transfer Time** (in disaggregated prefill-decode scenarios)
  - **Other Non-Negligible Components**: Identify and measure any other time components that contribute significantly to total latency

### 3. Configuration Parameters to Record
Document all key parameters that influence overall performance and behavior to enable controlled comparison with the simulator, including but not limited to:
- Model configuration (model name, size, architecture parameters)
- Batch size and batching strategy
- Sequence lengths (input/output)
- Scheduling policy
- Memory management settings (block size, GPU memory utilization)
- Parallelism settings (tensor parallel, pipeline parallel)
- Quantization settings (if any)

## Default Test Scenarios

Focus on the following scenarios by default:
0. **Engine Version Priority**:
- **ALWAYS use vLLM v1 engine** (`vllm/v1`) instead of v0 to ensure optimal performance
- If a specific scenario **cannot** be implemented using the v1 engine due to technical limitations or missing features, you **MUST**:
  - Explicitly notify me before proceeding
  - Explain the specific reason why v1 cannot be used
  - Wait for my approval before falling back to v0
1. **Architecture**: Disaggregated prefill-decode scenario (separate prefill and decode instances)
2. **Models**: 
   - MoE model: Mixtral 8x7B MoE
   - OR lightweight dense models (focus on small models initially)
   - **Model Cache Location**: If model download is required, save cache to `/research/d1/gds/ytyang/yichengfeng/frontier/sota-infer-engine/vllm/`
3. **Prefix Caching**: Disabled (do not enable prefix caching)
4. **Hardware Environment**: 
   - GPU: A800 (8 GPUs available)
   - Driver Version: 575.57.08
   - CUDA Version: 12.9

## Testing Approach

Follow an incremental, bottom-up testing methodology:
- Start with simple, small-scale tests
- Gradually increase complexity
- Do NOT rush to expand functionality scope
- Progress from simple to complex scenarios for testing and validation
- Validate each component thoroughly before moving to the next level of complexity

## Deliverables Expected

For each test run, provide:
1. Detailed profiling data for all three levels (operator, flow, system)
2. Per-request time breakdown with all components identified
3. Configuration parameters used
4. Comparison-ready metrics that can be validated against the simulator
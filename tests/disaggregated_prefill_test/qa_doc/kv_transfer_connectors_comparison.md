## Modification History

| Date       | Summary of Changes                                                                 |
|------------|--------------------------------------------------------------------------------------|
| 2025-12-12 | Documented v1 chunked prefill behavior and standardized env-driven configs for fair KV connector comparison |

## 目标

对 `SharedStorageConnector` 与 `P2pNcclConnector` 做性能对比时，**确保除 KV transfer connector 类型外，其它关键 runtime 配置尽可能一致**，并避免 warmup 对 KV/cache 的污染导致结果偏置。

## 关键结论（vLLM v1）

### 1) Chunked prefill（v1 generate）

- **现状**：vLLM v1 的 generate 任务默认启用 chunked prefill（属于 v1 运行时行为，不是测试脚本单独开启）。
- **处理策略（当前阶段）**：暂时接受“v1 下 chunked prefill 无法关闭”的事实（等待后续处理），本轮对比只保证：
  - **workload 不触发实际 chunk splitting**（即每个 prompt 的 prefill token 数量远小于 `max_num_batched_tokens`，使得 chunked prefill 实际上只走单 chunk）。
- **建议设置**（用于避免触发实际 chunking）：
  - `PREFILL_TOKENS=1024`
  - `NUM_REQUESTS=4`
  - v1 默认 `max_num_batched_tokens`（A800/H100 类）通常为 `16384`，因此总 prefill token `NUM_REQUESTS * PREFILL_TOKENS = 4096`，远小于阈值，通常不会发生“多 chunk 的 partial prefill”。

### 2) Prefix caching

- **对比要求**：必须禁用 prefix caching，避免不同 connector 下 cache reuse 干扰对比。
- **实现方式**：通过 shell export 环境变量统一控制（不在 Python 脚本中写死）。

### 3) CUDA graph（decode phase）

- **对比要求**：decode 阶段启用 CUDA graph（更接近 v1 性能路径）。
- **实现方式**：decode 侧 `enforce_eager=False`；prefill 侧可按需求使用 eager（但需两边一致）。

## 统一的环境变量配置（推荐由 sh 设置）

以下 env 建议由各个 test shell 统一 export：

- **v1 engine**
  - `VLLM_USE_V1=1`
  - `VLLM_ENABLE_V1_MULTIPROCESSING=0`

- **v1 generate defaults（显式化）**
  - `VLLM_V1_ENABLE_CHUNKED_PREFILL=1`  (v1 generate required)
  - `VLLM_V1_ENABLE_PREFIX_CACHING=0`   (required for fair comparison)

- **profiling（默认不开 stack）**
  - `VLLM_TORCH_PROFILER_WITH_STACK=0`
  - `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1`

## Warmup 污染规避

### SharedStorageConnector

- **问题**：warmup 会在 `local_storage/` 中写入 KV artifacts，若不清理，正式测量会复用这些文件，导致结果偏高。
- **修复策略**：warmup 结束后删除并重建 `kv_storage_path`（例如 `local_storage/`），确保正式 run 不从 warmup KV 中获益。

### P2pNcclConnector

- **问题**：warmup 可能导致 “KV transfer / runtime cache” 对正式测量产生偏置。
- **修复策略**：
  - warmup 使用与正式测量不同的请求集合（不同 seed），降低 warmup KV 直接命中的可能性；
  - 仍保持与 SharedStorage 一致的 `NUM_REQUESTS/PREFILL_TOKENS/DECODE_TOKENS` 以确保可比性。



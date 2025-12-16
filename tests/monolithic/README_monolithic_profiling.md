## Modification History

| Date | Summary of Changes |
|------------|---------------------------------------------|
| 2025-12-15 | Added monolithic profiling launcher and usage doc |

## 概述
- 单机单实例（Monolithic）离线推理 profiling，复用 `examples/offline_inference/simple_profiling.py` 并接入 request generator。
- 依赖 vLLM v1 引擎，禁用 chunked prefill 与 prefix caching，单 GPU 运行（默认 GPU 0）。
- 支持 torch profiler（与参考脚本一致的环境变量），输出日志到 `tests/monolithic/offline_monolithic_profiling.log`，可选 profiler traces 到 `tests/monolithic/profiles/`。

## 快速运行
```bash
# 示例：默认配置、GPU 0、启用 profiling
CUDA_VISIBLE_DEVICES=0 \
  tests/monolithic/offline_monolithic_profiling.sh \
  --num-requests 128 --prefill-tokens 512 --decode-tokens 2 \
  --seed 42 --warmup-iters 3 --profile
```
- 如需自定义 GPU：`--gpu 1` 或 `GPU_ID=1`。
- 如需关闭 profiling，删除 `--profile` 即可（默认关闭）。

## 关键配置参数
- 引擎与内存
  - `VLLM_USE_V1=1`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`
  - `VLLM_V1_ENABLE_CHUNKED_PREFILL=0`, `VLLM_V1_ENABLE_PREFIX_CACHING=0`
  - `GPU_MEMORY_UTILIZATION`（默认 `0.7`）
- 模型与工作负载
  - 默认模型：`unsloth/Llama-3.2-1B-Instruct`
  - `--num-requests`（默认 128）
  - `--prefill-tokens` / `--decode-tokens`（默认 512 / 2）
  - `--seed`（默认 42），`--warmup-iters`（默认 3）
- Profiling（可选）
  - `--profile` 开启时自动设置：`VLLM_TORCH_PROFILER_DIR`, `VLLM_TORCH_PROFILER_WITH_STACK=0`, `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1`
  - `--profile-max-decode-tokens` 控制 trace 规模（默认 256）

## 预期输出
- 日志：`tests/monolithic/offline_monolithic_profiling.log`（包含命令、配置、耗时、生成结果）。
- Profiler：若开启，trace 文件落在 `tests/monolithic/profiles/`。

## 与解耦式（Disaggregated）方案的差异
- 本脚本单实例同时执行 prefill 与 decode，无 KV 传输、无 producer/consumer 角色、无端口/同步文件配置。
- 无需 `KVTransferConfig`，也不需要双 GPU；仅设置单个 `CUDA_VISIBLE_DEVICES`。
- 仍沿用 request generator 构造固定长度 workload，但将 prefill 与 decode 合并在同一进程执行。
- 保持参考脚本的 profiling 环境变量与 warm-up 逻辑，配置更精简、参数更少。

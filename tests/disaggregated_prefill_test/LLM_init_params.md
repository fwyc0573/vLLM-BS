# Disaggregated Prefill-Decode LLM 初始化参数说明

## 修改记录
| 版本 | 日期 | 作者 | 变更 |
| --- | --- | --- | --- |
| 0.1 | 2025-12-09 | GitHub Copilot | 新增文档，整理 prefill/decode LLM 初始化参数与一致性说明 |

## 范围
- 基于 vLLM v1 引擎，示例脚本：
  - `examples/offline_inference/disaggregated-prefill-v1/prefill_example.py`
  - `examples/offline_inference/disaggregated-prefill-v1/decode_example.py`
- 仅覆盖两处 `LLM()` 初始化参数，并讨论二者在解耦预填-解码架构下的关系。

## Prefill 实例参数（`prefill_example.py`）
| 参数 | 值 | 作用 | 备注 |
| --- | --- | --- | --- |
| model | `meta-llama/Llama-3.2-1B-Instruct` | 选择模型权重与配置 | 需与 decode 实例一致以保证 KV 兼容 |
| enforce_eager | `True` | 强制使用 eager，避免 CUDA graph 编译开销 | 便于性能分析与一致性 |
| gpu_memory_utilization | `0.8` | 限制模型+KV 最高显存占用比例 | 影响可分配 KV blocks，需结合显存调节 |
| enable_prefix_caching | `False` | 关闭前缀缓存 | 显式禁用，避免自动缓存带来的行为差异 |
| enable_chunked_prefill | `False` | 关闭分块预填 | 确保一次性处理完整提示 |
| kv_transfer_config | `SharedStorageConnector`, role=`kv_both`, path=`local_storage` | 配置 KV 传输（共享存储） | Prefill 写入 KV，供 Decode 读取 |
| sampling_params | `max_tokens=1`, `temperature=0`, `top_p=0.95` | 每请求生成 1 个 token，用于构造后续 decode 的提示 | 请求级参数，与 LLM 初始化无全局冲突 |

## Decode 实例参数（`decode_example.py`）
| 参数 | 值 | 作用 | 备注 |
| --- | --- | --- | --- |
| model | `meta-llama/Llama-3.2-1B-Instruct` | 选择模型权重与配置 | 必须与 prefill 完全一致 |
| enforce_eager | `True` | 同上 | 保持一致性，便于对齐性能 |
| gpu_memory_utilization | `0.8` | 同上 | 与 prefill 可独立设置，但建议一致以避免容量预估偏差 |
| max_num_batched_tokens | `64` | 单步调度的总 token 上限 | 仅 decode 脚本设置；控制吞吐/显存峰值 |
| max_num_seqs | `128` | 单步调度的序列条数上限 | 仅 decode 脚本设置；控制批内请求数 |
| enable_prefix_caching | `False` | 同上 | 与 prefill 对齐 |
| enable_chunked_prefill | `False` | 同上 | 与 prefill 对齐 |
| kv_transfer_config | `SharedStorageConnector`, role=`kv_both`, path=`local_storage` | 读取 prefill 产出的 KV | 路径需与 prefill 一致 |
| sampling_params | `max_tokens=10`, `ignore_eos=True`, `temperature=0`, `top_p=0.95` | 每请求生成 10 个 token，忽略 EOS | 请求级参数 |

## Prefill vs Decode 差异
- Decode 额外设置了调度上限：`max_num_batched_tokens=64`、`max_num_seqs=128`，prefill 未显式设置（由默认推断）。
- 两者的采样参数不同：prefill 生成 1 token；decode 生成 10 token 并忽略 EOS。
- 其余核心参数（模型、eager、GPU 利用率、前缀缓存/分块预填开关、KV 传输配置）保持一致。

## 参数独立性与一致性分析
- **独立性**：prefill 与 decode 分别通过各自的 `LLM()` 创建独立引擎实例（独立进程/生命周期），参数不自动共享。
- **潜在冲突风险**（需手动保持一致）：
  - 模型与 tokenizer 相关：`model`、`tokenizer`、RoPE/滑动窗口等上下文设定必须一致，否则 KV 不兼容。
  - KV/内存相关：`block_size`、`cache_dtype`、`gpu_memory_utilization`、`kv_cache_memory_bytes` 若差异过大，可能导致一侧无法装载或容量估计不符。
  - 并行度：`tensor_parallel_size`、`pipeline_parallel_size` 应对齐，否则分片布局不匹配，KV 读取会失败或精度不可控。
  - 传输配置：`kv_transfer_config`（connector 类型、路径、角色）必须一致，否则 decode 无法找到或解析 KV。
- **现有校验机制**：vLLM 目前未对“跨实例”参数做自动一致性校验。每个 `LLM` 只验证自身配置（见 `vllm/entrypoints/llm.py` 构造 `EngineArgs` 并交给 `LLMEngine.from_engine_args`）。
- **风险与处理**：若参数不一致，常见表现是 decode 阶段加载/读取 KV 失败或生成错误。需要人工确保关键参数对齐；可通过统一的配置文件/环境变量注入，或在启动脚本中复用相同的参数集合。

## 建议
- 将共享的关键参数（模型名、并行度、KV 配置、cache_dtype、block_size、prefix-caching/分块预填开关）集中到单一配置来源，供 prefill 与 decode 同时引用。
- 对调度参数（`max_num_batched_tokens`、`max_num_seqs`）可按阶段需求单独调优，但需考虑显存与吞吐折衷。
- 运行前检查日志：确认加载的模型、并行度、KV 传输路径一致；若需要，可在两脚本中打印 `llm.llm_engine.vllm_config` 的关键字段做对比。

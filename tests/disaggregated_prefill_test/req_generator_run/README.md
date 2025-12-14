# vLLM Request Generator 使用说明

## 修改记录
| 版本 | 日期 | 作者 | 变更 |
| --- | --- | --- | --- |
| 0.1 | 2025-12-10 | GitHub Copilot | 初始文档 |
| 0.2 | 2025-12-10 | GitHub Copilot | 添加 KV 同步功能文档（--enable-kv-sync），更新配置参数说明 |
| 0.3 | 2025-12-11 | Claude | 添加 P2pNcclConnector 测试脚本和多 GPU 分布式推理文档 |

---

## 概述

vLLM Request Generator 是一个与 vLLM 运行时兼容的请求生成器组件，能够：
- 控制请求数量、输入长度 (prefill)、输出长度 (decode)
- 复用 Frontier 项目的请求生成逻辑
- 生成可用于 vLLM 离线推理的请求对象

本目录包含在 Disaggregated Prefill-Decode 架构下使用请求生成器的测试脚本。

---

## 快速开始

### 1. 运行默认测试

```bash
cd tests/disaggregated_prefill_test/req_generator_run
chmod +x run_test.sh
./run_test.sh
```

默认配置：
- 4 个请求
- 每个请求 1024 个 prefill tokens
- 每个请求 64 个 decode tokens

### 2. 自定义参数

```bash
# 更多请求
./run_test.sh --num-requests 10

# 调整长度
./run_test.sh --prefill-tokens 2048 --decode-tokens 128

# 启用 profiling
./run_test.sh --profile

# 组合参数
./run_test.sh --num-requests 8 --prefill-tokens 512 --decode-tokens 256 --profile
```

### 3. 环境变量

```bash
# 指定 GPU
GPU_ID=0 ./run_test.sh

# 跳过 warmup
WARMUP_ITERS=0 ./run_test.sh
```

---

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `run_test.sh` | SharedStorageConnector 测试脚本，执行 prefill + decode 两阶段（单 GPU） |
| `offline_P2pNcclConnector_test.sh` | P2pNcclConnector 测试脚本，执行分布式 prefill + decode（双 GPU） |
| `prefill_with_generator.py` | Prefill 阶段脚本（SharedStorageConnector），使用请求生成器 |
| `decode_with_generator.py` | Decode 阶段脚本（SharedStorageConnector），读取 prefill 输出 |
| `README.md` | 本文档 |

运行后生成的文件：
| 文件 | 说明 |
| --- | --- |
| `output.txt` | Prefill 输出的 token IDs (JSON 格式) |
| `metadata.json` | 请求元数据 (长度配置等) |
| `test_output.log` | 完整运行日志 |
| `local_storage/` | KV cache 存储目录 |
| `profiles/` | Profiling 输出目录 (启用时) |

---

## P2pNcclConnector 分布式测试

### 概述

P2pNcclConnector 使用 NCCL 进行 GPU 间直接 KV cache 传输，需要两个独立的 GPU：
- **Prefill GPU**: 执行 prefill 计算，生成 KV cache
- **Decode GPU**: 接收 KV cache，执行 decode 生成

与 SharedStorageConnector 的主要区别：

| 特性 | SharedStorageConnector | P2pNcclConnector |
| --- | --- | --- |
| 进程模型 | 顺序执行，单进程 | 并发执行，双进程 |
| KV 传输 | 文件系统 | NCCL GPU-to-GPU |
| GPU 需求 | 单 GPU | 双 GPU |
| 同步机制 | 文件标记 | multiprocessing.Event |
| 角色配置 | `kv_role="kv_both"` | `kv_producer` / `kv_consumer` |

### 运行 P2pNcclConnector 测试

```bash
cd tests/disaggregated_prefill_test/req_generator_run

# 默认配置（使用 GPU 3 和 4）
./offline_P2pNcclConnector_test.sh

# 启用 profiling
./offline_P2pNcclConnector_test.sh --profile

# 自定义 GPU
./offline_P2pNcclConnector_test.sh --gpu-prefill 0 --gpu-decode 1

# 自定义工作负载
./offline_P2pNcclConnector_test.sh --num-requests 8 --prefill-tokens 512 --decode-tokens 128
```

### 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--num-requests` | 4 | 请求数量 |
| `--prefill-tokens` | 1024 | 每请求 prefill tokens |
| `--decode-tokens` | 64 | 每请求 decode tokens |
| `--seed` | 42 | 随机种子 |
| `--warmup-iters` | 3 | Warmup 迭代次数 |
| `--profile` | - | 启用 profiling |
| `--model` | meta-llama/Llama-3.2-1B-Instruct | 模型路径 |
| `--gpu-prefill` | 3 | Prefill GPU ID |
| `--gpu-decode` | 4 | Decode GPU ID |
| `--gpu-memory-utilization` | 0.7 | GPU 显存利用率 |
| `--max-model-len` | 4096 | 最大模型长度 |
| `--prefill-timeout` | 300 | Prefill 超时（秒） |
| `--decode-timeout` | 600 | Decode 超时（秒） |

### Profiler 输出

启用 profiling 后，trace 文件保存在 `profiles/` 目录：
- Prefill 和 Decode worker 的 trace 使用不同的 worker name 前缀区分
- 可使用 TensorBoard 查看：`tensorboard --logdir profiles/`

**Profiler 环境变量说明**（默认配置）：

| 变量 | 值 | 说明 |
| --- | --- | --- |
| `VLLM_TORCH_PROFILER_DIR` | profiles/ | Trace 输出目录 |
| `VLLM_TORCH_PROFILER_WITH_STACK` | 0 | 禁用调用栈（减少开销） |
| `VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY` | 0 | 禁用内存分析 |
| `VLLM_TORCH_PROFILER_RECORD_SHAPES` | 0 | 禁用形状记录 |
| `VLLM_CUSTOM_SCOPES_FOR_PROFILING` | 1 | 启用自定义 scope |

### 数据传递机制

使用 `multiprocessing.Manager().dict()` 共享请求数据：

```python
# 主进程创建共享数据
manager = Manager()
shared_data = manager.dict()
shared_data["prompts"] = manager.list(prompts)
shared_data["metadata"] = manager.list(metadata)

# 子进程读取
prompts = list(shared_data["prompts"])
```

---

## 请求生成器 API

### 基本用法

```python
from vllm.request_generator import (
    VLLMRequestGenerator,
    RequestGeneratorConfig,
    FixedLengthConfig,
)

# 创建配置
config = RequestGeneratorConfig(
    num_requests=10,
    length_config=FixedLengthConfig(
        prefill_tokens=1024,
        decode_tokens=128,
    ),
    seed=42,
)

# 创建生成器
generator = VLLMRequestGenerator(config)

# 生成请求
requests = generator.generate()

# 用于 vLLM
prompts = [r.prompt for r in requests]
sampling_params = [r.sampling_params for r in requests]

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")
outputs = llm.generate(prompts, sampling_params)
```

### 长度分布类型

#### 1. Fixed (固定长度)
所有请求具有相同的输入/输出长度。

```python
from vllm.request_generator import FixedLengthConfig

config = FixedLengthConfig(
    prefill_tokens=1024,  # 输入长度
    decode_tokens=128,    # 输出长度
)
```

#### 2. Uniform (均匀分布)
总长度在 [min, max] 范围内均匀采样，按比例分配。

```python
from vllm.request_generator import UniformLengthConfig

config = UniformLengthConfig(
    min_tokens=512,               # 最小总长度
    max_tokens=4096,              # 最大总长度
    prefill_to_decode_ratio=8.0,  # P/D 比例
)
```

#### 3. Zipf (Zipf 分布)
模拟真实工作负载的长尾分布。

```python
from vllm.request_generator import ZipfLengthConfig

config = ZipfLengthConfig(
    theta=0.6,                    # Zipf 参数
    min_tokens=512,
    max_tokens=4096,
    prefill_to_decode_ratio=8.0,
)
```

### 便捷函数

```python
from vllm.request_generator.vllm_request_generator import create_generator

# 快速创建固定长度生成器
generator = create_generator(
    num_requests=10,
    prefill_tokens=1024,
    decode_tokens=128,
)

# 直接获取 prompts 和 sampling_params
prompts, params = generator.get_prompts_and_params()
```

### 工作负载摘要

```python
summary = generator.get_workload_summary()
print(summary)
# {
#     'num_requests': 10,
#     'prefill_tokens': {'min': 1024, 'max': 1024, 'mean': 1024.0, 'total': 10240},
#     'decode_tokens': {'min': 128, 'max': 128, 'mean': 128.0, 'total': 1280},
#     ...
# }
```

---

## 注意事项

### 1. 精确长度控制
- 使用 Token ID 模式 (`use_token_ids=True`) 可精确控制输入长度
- 设置 `force_exact_output_length=True` 确保输出达到指定长度
- 这会自动设置 `ignore_eos=True`、`stop=None`、`stop_token_ids=None`

### 2. Frontier 依赖
- 默认路径：`/research/d1/gds/ytyang/yichengfeng/frontier`
- 可通过环境变量 `FRONTIER_PATH` 覆盖
- 也可在 `RequestGeneratorConfig` 中设置 `frontier_path`

### 3. KV Transfer 同步（新增）
在 disaggregated 架构中，decode 集群需要等待 prefill 完成 KV cache 传输。

**启用 KV 同步**：
```bash
# Prefill 端（完成后创建标记文件）
python prefill_with_generator.py --enable-kv-sync

# Decode 端（等待标记文件）
python decode_with_generator.py --enable-kv-sync --kv-sync-timeout 60
```

**相关参数**：
| 参数 | 说明 |
| --- | --- |
| `--enable-kv-sync` | 启用 KV 传输同步 |
| `--kv-storage-path` | KV 存储路径（默认：local_storage） |
| `--kv-sync-timeout` | 等待超时秒数（默认：60） |

### 4. 与 Disaggregated 架构的配合
- Prefill 阶段：生成 1 个 token，主要执行 KV cache 计算
- Decode 阶段：读取 prefill 输出，生成指定数量的 token
- KV cache 通过 `SharedStorageConnector` 在两阶段间传递

---

## 故障排除

### 问题：Frontier 导入失败
```
ImportError: Failed to import Frontier modules
```
**解决**：确保 Frontier 项目存在，或设置 `FRONTIER_PATH` 环境变量：
```bash
export FRONTIER_PATH=/path/to/frontier
```

### 问题：输出长度不符合预期
**解决**：确保配置设置了：
- `force_exact_output_length=True`（默认已启用）
- 或手动设置 `SamplingParams`：
  - `ignore_eos=True`
  - `stop=None`
  - `stop_token_ids=None`

### 问题：KV cache 读取失败 (FileNotFoundError)
**症状**：
```
FileNotFoundError: No such file or directory: local_storage/<hash>/model.layers.0.self_attn.attn.safetensors
```

**原因**：`SharedStorageConnector` 的 hash 计算逻辑：
- `_found_match_for_request`: 使用 `align_to_block_size(len-1, block_size)` 检查
- `make_meta` 和加载: 使用 `align_to_block_size(len, block_size)` 计算

当 prompt 长度从 1024 变为 1025 时，hash 会不同。

**解决**：确保 decode 阶段使用与 prefill 阶段**相同长度**的 prompt：
- Prefill 保存原始 prompt（不含生成的 token）
- Decode 使用原始 prompt 长度

**验证 hash 一致性**：
```python
import hashlib, torch

def align_to_block_size(n, bs): return (n-1) // bs * bs

# Prefill prompt (1024 tokens)
prefill_hash = hashlib.md5(
    torch.tensor(prompt[:align_to_block_size(1024, 16)]).numpy().tobytes()
).hexdigest()

# Decode prompt (should be same 1024 tokens)
decode_hash = hashlib.md5(
    torch.tensor(prompt[:align_to_block_size(1024, 16)]).numpy().tobytes()
).hexdigest()

assert prefill_hash == decode_hash
```

### 问题：KV cache 目录存在但文件为空
**原因**：decode 阶段尝试加载时创建了目录（在 `_generate_foldername_debug` 中 `create_folder=True`），但 prefill 从未保存到该 hash 路径。

**解决**：清理 `local_storage/` 目录后重新运行 prefill。

如果使用 KV 同步，确保：
```bash
# Prefill 端启用同步
python prefill_with_generator.py --enable-kv-sync

# Decode 端启用同步并设置超时
python decode_with_generator.py --enable-kv-sync --kv-sync-timeout 120
```

### 问题：KV 同步超时
```
TimeoutError: Timeout waiting for KV transfer
```
**解决**：
- 确保 prefill 端运行时启用了 `--enable-kv-sync`
- 增加 decode 端的 `--kv-sync-timeout` 值
- 检查 `local_storage/.kv_transfer_complete` 文件是否存在

### 问题：P2pNcclConnector 进程超时
```
TimeoutError: Timeout waiting for prefill after 300s
```
**解决**：
- 检查两个 GPU 是否可用且有足够显存
- 增加 `--prefill-timeout` 和 `--decode-timeout` 值
- 确保模型能够加载到指定 GPU

### 问题：NCCL 连接失败
```
NCCL error: unhandled system error
```
**解决**：
- 检查两个 GPU 之间是否支持 P2P 通信
- 设置 `export NCCL_P2P_DISABLE=1` 禁用 P2P（可能降低性能）
- 检查防火墙是否阻止了端口通信

### 问题：GPU 显存不足
```
CUDA out of memory
```
**解决**：
- 降低 `--gpu-memory-utilization`（如 0.5）
- 减少 `--max-model-len`
- 使用更小的模型
- 清理其他占用 GPU 显存的进程

### 问题：端口冲突
```
RuntimeError: Could not find a free port block for KV transfer
```
**解决**：
- 设置环境变量指定不同端口：`export KV_BASE_PORT=15000`
- 检查并释放占用端口的进程

---

## 后续扩展

- **在线模式**：支持按到达时间分布的请求（使用 Poisson/Gamma 分布）
- **真实数据集**：从 ShareGPT 等数据集采样提示
- **性能报告**：自动收集并汇总性能指标
- **多节点支持**：扩展 P2pNcclConnector 到跨节点通信

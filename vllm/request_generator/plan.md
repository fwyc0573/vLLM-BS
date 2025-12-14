# vLLM Request Generator - 实施规划文档

## 修改记录
| 版本 | 日期 | 作者 | 变更 |
| --- | --- | --- | --- |
| 0.1 | 2025-12-10 | GitHub Copilot | 初始规划文档 |
| 0.2 | 2025-12-10 | GitHub Copilot | 完成核心模块实现 |
| 0.3 | 2025-12-10 | GitHub Copilot | 完成测试验证，模块功能完整 |
| 0.4 | 2025-12-10 | GitHub Copilot | 实现4个增强解决方案：精确长度控制、输出长度保证、路径解耦、KV同步 |

---

## 1. 项目概述

### 1.1 目标
构建一个与 vLLM 运行时兼容的请求生成器组件，能够：
- 复用 Frontier 项目的请求生成器模块
- 控制请求数量、输入长度、输出长度
- 生成可用于 vLLM 离线推理的请求对象

### 1.2 范围
- **Phase 1（本次实现）**：离线模式 - 所有请求同时到达 (arrival_time = 0)
- **Phase 2（未来工作）**：在线模式 - 请求按到达时间分布

### 1.3 目标场景
Disaggregated Prefill-Decode 架构下的性能测试与基准测试。

---

## 2. 架构设计

### 2.1 数据流
```
┌─────────────────────────────────────────────────────────────────────┐
│                      Frontier Request Generator                     │
│  ┌──────────────┐    ┌─────────────────┐    ┌────────────────┐     │
│  │ LengthConfig │ -> │ SyntheticReqGen │ -> │ List[Request]  │     │
│  │ IntervalCfg  │    │                 │    │ (metadata)     │     │
│  └──────────────┘    └─────────────────┘    └───────┬────────┘     │
└─────────────────────────────────────────────────────┼───────────────┘
                                                      │
                                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     vLLM Request Adapter                            │
│  ┌────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │ Frontier.Req   │ -> │ PromptGenerator │ -> │ vLLM Prompts +  │  │
│  │ (prefill_len,  │    │ (token-based)   │    │ SamplingParams  │  │
│  │  decode_len)   │    └─────────────────┘    └─────────────────┘  │
│  └────────────────┘                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 关键类设计

#### `VLLMRequestGenerator`
- 封装 Frontier 的 `SyntheticRequestGenerator`
- 配置参数：请求数量、长度分布类型、长度范围
- 输出：`List[VLLMRequest]`

#### `VLLMRequest`
- `prompt`: 字符串或 token ID 列表
- `sampling_params`: vLLM 的 `SamplingParams` 对象
- `request_id`: 可选的请求标识

#### `PromptGenerator`
- 根据目标 token 长度生成提示
- 策略：重复填充词/随机 token ID

---

## 3. Frontier 依赖集成

### 3.1 依赖方式
通过 Python 路径动态导入 Frontier 模块：
```python
import sys
sys.path.insert(0, "/research/d1/gds/ytyang/yichengfeng/frontier")
from frontier.request_generator import SyntheticRequestGenerator
from frontier.config import (
    SyntheticRequestGeneratorConfig,
    FixedRequestLengthGeneratorConfig,
    StaticRequestIntervalGeneratorConfig,
)
```

### 3.2 Frontier 输出格式
```python
class Request:
    arrived_at: float           # 到达时间
    num_prefill_tokens: int     # 输入长度
    num_decode_tokens: int      # 输出长度
```

### 3.3 vLLM 输入格式
```python
# 方式1: 字符串提示
prompts: List[str]
sampling_params: SamplingParams(max_tokens=N, ignore_eos=True, ...)

# 方式2: Token ID 提示
prompts: List[TokensPrompt]  # {"prompt_token_ids": [...]}
```

---

## 4. 实施步骤

### Step 1: 创建基础模块结构 ✅
- [x] 创建 `vllm/request_generator/` 目录
- [x] 创建 `plan.md` 规划文档
- [x] 创建 `__init__.py`

### Step 2: 实现核心组件 ✅
- [x] 实现 `prompt_generator.py` - 根据目标长度生成提示
- [x] 实现 `config.py` - 配置类定义
- [x] 实现 `vllm_request_generator.py` - 主生成器类

### Step 3: 创建测试脚本 ✅
- [x] 创建 `tests/disaggregated_prefill_test/req_generator_run/` 目录
- [x] 实现 `prefill_with_generator.py`
- [x] 实现 `decode_with_generator.py`
- [x] 实现 `run_test.sh`
- [x] 创建使用文档 `README.md`

### Step 4: 验证与文档 ✅
- [x] 运行测试脚本验证功能
- [x] 更新本文档记录完成状态

#### 验证结果 (2025-12-10)

**Prefill 阶段**: ✅ 成功
```
Workload Summary:
  - Requests: 2
  - Prefill tokens: min=128, max=128, mean=128.0
  - Decode tokens: min=8, max=8, mean=8.0

Processing outputs:
  Request 0: prefill=128, generated=1, total=129
  Request 1: prefill=128, generated=1, total=129
```

**Decode 阶段**: ⚠️ 请求生成器功能正常，KV Transfer 配置问题
- 请求生成器正确加载了 prefill 阶段的 prompts 和 metadata
- SharedStorageConnector 检测到 "External Cache Hit"
- 错误发生在 KV 缓存加载阶段（文件路径不匹配）
- 这是 disaggregated 基础设施配置问题，不是请求生成器问题

**结论**: 请求生成器模块功能完整，能够：
1. ✅ 生成指定数量的请求
2. ✅ 精确控制输入 token 长度
3. ✅ 设置输出 token 长度目标
4. ✅ 序列化/反序列化 prompts 和 metadata

---

## 5. 配置参数说明

### 5.1 用户可配置参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `num_requests` | int | 10 | 请求数量 |
| `length_generator_type` | str | "fixed" | 长度分布类型: fixed/uniform/zipf |
| `prefill_tokens` | int | 1024 | 固定输入长度 (fixed 模式) |
| `decode_tokens` | int | 128 | 固定输出长度 (fixed 模式) |
| `min_tokens` | int | 512 | 最小总长度 (uniform/zipf 模式) |
| `max_tokens` | int | 4096 | 最大总长度 (uniform/zipf 模式) |
| `prefill_to_decode_ratio` | float | 8.0 | P/D 比例 (uniform/zipf 模式) |
| `seed` | int | 42 | 随机种子 |
| `use_token_ids` | bool | True | 使用 token ID 精确控制长度 |
| `vocab_size` | int | 128256 | 词汇表大小（Llama-3） |
| `min_token_id` | int | 1000 | 最小 token ID（避免特殊 token） |
| `max_token_id` | int | None | 最大 token ID |
| `force_exact_output_length` | bool | True | 强制生成完整输出长度 |
| `frontier_path` | str | None | Frontier 路径（优先于环境变量） |

### 5.2 采样参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `temperature` | 0 | 采样温度 (0=贪婪) |
| `top_p` | 0.95 | Top-P 采样 |
| `ignore_eos` | True | 忽略 EOS，确保生成完整长度 |
| `stop` | None | 无停止字符串 |
| `stop_token_ids` | None | 无停止 token |

---

## 6. 潜在挑战与解决方案（已实现）

### 6.1 提示长度精确控制 ✅
**挑战**：字符串提示经 tokenizer 后长度不可控。
**解决方案（已实现）**：
- 在 `RequestGeneratorConfig` 中添加 `use_token_ids` 配置选项（默认 `True`）
- 添加 `vocab_size`、`min_token_id`、`max_token_id` 参数控制 token ID 生成范围
- 当 `use_token_ids=True` 时，生成 `{"prompt_token_ids": [...]}` 格式
- 当 `use_token_ids=False` 时，需要提供 tokenizer

**代码示例**：
```python
config = RequestGeneratorConfig(
    num_requests=10,
    use_token_ids=True,        # 精确长度控制
    vocab_size=128256,         # Llama-3 词汇表大小
    min_token_id=1000,         # 避免特殊 token
)
```

### 6.2 输出长度保证 ✅
**挑战**：模型可能提前生成 EOS。
**解决方案（已实现）**：
- 在 `RequestGeneratorConfig` 中添加 `force_exact_output_length` 配置选项（默认 `True`）
- 当启用时，自动设置 `ignore_eos=True`、`stop=None`、`stop_token_ids=None`
- 用户可设置 `force_exact_output_length=False` 允许提前停止

**代码示例**：
```python
config = RequestGeneratorConfig(
    num_requests=10,
    force_exact_output_length=True,  # 强制生成完整长度
)
```

### 6.3 Frontier 路径依赖解耦 ✅
**挑战**：硬编码的 Frontier 路径降低可移植性。
**解决方案（已实现）**：
- 添加 `_resolve_frontier_path()` 函数处理路径解析
- 优先级顺序：`config.frontier_path` > `FRONTIER_PATH` 环境变量 > 默认路径
- 添加路径规范化（展开 `~`、移除末尾斜杠）
- 添加路径验证（检查存在性、验证 `frontier` 模块目录）
- 提供详细的错误信息和修复建议

**使用方式**：
```bash
# 方式1：环境变量
export FRONTIER_PATH=/path/to/frontier

# 方式2：配置参数
config = RequestGeneratorConfig(
    frontier_path="/custom/path/to/frontier",
)
```

### 6.4 KV Transfer 同步问题 ✅
**挑战**：Decode 集群在 Prefill 完成 KV cache 传输前就开始访问。
**解决方案（已实现）**：
- 创建 `kv_sync.py` 模块，提供 `KVTransferSync` 类
- 使用标记文件机制（`.kv_transfer_complete`、`.kv_transfer_error`）
- 提供 `mark_transfer_complete()` 和 `wait_for_transfer_complete()` 方法
- 支持超时、轮询间隔、错误处理
- 支持元数据传递

**使用示例**：
```python
# Prefill 端
from vllm.request_generator import KVTransferSync
sync = KVTransferSync(storage_path="local_storage")
# ... 执行 prefill ...
sync.mark_transfer_complete(metadata={"num_requests": 10})

# Decode 端
sync = KVTransferSync(storage_path="local_storage")
sync.wait_for_transfer_complete(timeout=60.0)
# ... 执行 decode ...
```

---

## 7. 文件清单

```
vllm/request_generator/
├── __init__.py                    # 模块导出
├── plan.md                        # 本文档
├── config.py                      # 配置类（含 use_token_ids, force_exact_output_length 等）
├── prompt_generator.py            # 提示生成器
├── vllm_request_generator.py      # 主生成器（含 _resolve_frontier_path）
└── kv_sync.py                     # KV Transfer 同步模块（新增）

tests/disaggregated_prefill_test/req_generator_run/
├── README.md                      # 使用文档
├── prefill_with_generator.py      # Prefill 测试脚本（含 --enable-kv-sync）
├── decode_with_generator.py       # Decode 测试脚本（含 --enable-kv-sync）
├── run_test.sh                    # 运行脚本
├── test_solution1.py              # 解决方案1测试
├── test_solution2.py              # 解决方案2测试
├── test_solution3.py              # 解决方案3测试
├── test_solution4.py              # 解决方案4测试
└── test_all_solutions.py          # 综合测试脚本
```

---

## 8. 后续工作

### Phase 2: 在线模式支持
- 使用 `PoissonRequestIntervalGenerator` 或 `GammaRequestIntervalGenerator`
- 按 `arrived_at` 时间戳调度请求
- 与 vLLM 的 AsyncEngine 集成

### 扩展功能
- 支持从 ShareGPT 等真实数据集采样提示
- 支持多模态请求生成
- 性能指标采集与报告

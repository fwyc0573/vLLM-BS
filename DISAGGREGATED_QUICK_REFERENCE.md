# Disaggregated Prefill-Decode 快速参考

## 核心概念（1分钟速览）

**什么是 Disaggregated Prefill-Decode？**

一种推理架构，将请求处理分为两个阶段：
- **Prefill** (处理长上下文）：一次性处理所有输入 tokens，产生 KV cache
- **Decode** (生成 tokens）：逐个生成 tokens，复用 KV cache

```
Prefill: 5000 tokens in 1 step  →  KV Cache
Decode:  1 token per step (max_tokens=100)  ←  KV Cache
```

---

## 代码流程（5步快速理解）

### 步骤 1: 配置与初始化
```python
from vllm import LLM
from vllm.config import KVTransferConfig

# 关键：启用 KV 分离
kv_config = KVTransferConfig(
    kv_connector="SharedStorageConnector",      # 传输机制
    kv_role="kv_both",                          # producer+consumer
    kv_connector_extra_config={"shared_storage_path": "local_storage"}
)

llm = LLM(model="...", kv_transfer_config=kv_config)
```

**配置含义：**
| 字段 | 值 | 作用 |
|------|-----|------|
| `kv_connector` | `SharedStorageConnector` | 使用本地文件系统传输 KV |
| `kv_role` | `kv_both` | 此实例既产生 KV 也消费 KV |
| `shared_storage_path` | `local_storage/` | KV 存储目录 |

---

### 步骤 2: 提交请求
```python
prompts = ["The capital of France is"]
outputs = llm.generate(prompts, sampling_params)
```

**内部流程：**
```
generate()
  → _add_request()
    → llm_engine.add_request()
      → Processor.process()  # tokenize
        → EngineCoreClient.add_request()
```

---

### 步骤 3: Prefill 执行（一次性处理 prompt）

**Prefill Instance 执行：**

```python
# 伪代码 - 实际位置：vllm/v1/worker/gpu_worker.py
input_ids = tokenize(prompt)        # [101, 2054, 2003, ...]  shape: (5,)
outputs = model(input_ids)
logits = outputs.logits              # shape: (5, vocab_size)
kv_cache = outputs.past_key_values  # shape: (num_layers, 2, num_heads, 5, head_dim)

# 写入共享存储
store.put_kv(request_id=123, kv=kv_cache)

# 采样第一个 token
first_token = sample(logits[-1])     # 只看最后位置的 logits
```

**时间成本：** O(长度) - 但只执行一次

---

### 步骤 4: KV Cache 传输

**SharedStorageConnector 流程：**

```python
# Prefill 实例端
with open("local_storage/kv_req_123.bin", "wb") as f:
    torch.save(kv_cache, f)  # ← 序列化保存

# Decode 实例端
kv_cache = torch.load("local_storage/kv_req_123.bin")  # ← 反序列化加载
kv_cache = kv_cache.to("cuda")  # 恢复到显存
```

**优势：** 
- 本地传输（无网络延迟）
- 不占用 Prefill 实例显存

---

### 步骤 5: Decode 自回归循环（逐 token 生成）

**Decode Instance 执行（max_tokens 次）：**

```python
# 伪代码 - 实际位置：vllm/v1/worker/gpu_worker.py
for step in range(max_tokens):
    # 获取最后生成的 token
    current_token = sequence[-1]  # 形状：(1,)
    
    # 从存储读取 KV（每步需要）
    kv_cache = store.get_kv(request_id)
    
    # 前向传播：仅处理 1 个 token
    outputs = model(
        input_ids=current_token.unsqueeze(0),  # shape: (1, 1)
        past_key_values=kv_cache                # ← 复用
    )
    
    logits = outputs.logits  # shape: (1, vocab_size)
    next_token = sample(logits[0])
    
    sequence.append(next_token)
    
    if next_token == EOS_TOKEN:
        break
```

**关键特性：**
- 每步处理 1 个 token（高效）
- KV cache 大小固定（不增长）
- 时间成本：O(max_tokens) - 但每步很快

---

## 对比：为什么要分离？

### Traditional（非分离）

```
单实例处理全程：
┌─────────────────────────┐
│ GPU 显存                 │
├─────────────────────────┤
│ Model Weights (20GB)    │
│ KV Cache (整个长度)     │ ← 占用大
│ Compute Buffer          │
└─────────────────────────┘

问题：
❌ 显存压力大（长序列卡死）
❌ Prefill 和 Decode 交织，性能差
❌ 批处理能力弱
```

### Disaggregated（分离）

```
两个独立实例：

Prefill Instance          Decode Instance
┌──────────────┐         ┌──────────────┐
│ GPU0 80GB    │         │ GPU1 80GB    │
├──────────────┤         ├──────────────┤
│ Weights(20GB)│         │ Weights(20GB)│
│ Long KV(50GB)│         │ Tiny KV(1GB) │
│             │         │             │
│[Prefill在]  │         │ [Decode在]   │
│[此完全运行] │         │[轻松运行]    │
│             │         │             │
│[完成后释放]  │ ─KV──> │[加载并使用]  │
│             │         │             │
│ [处理新请求]│         │ [生成序列]   │
└──────────────┘         └──────────────┘

优势：
✅ 显存使用率高（可多处理长序列）
✅ 独立优化（Prefill用Flash-Attn，Decode用小批量）
✅ 两实例可并行（不相互阻塞）
✅ 吞吐量提升 3-5x
```

---

## 核心数据流

```python
# 输入
prompts = [
    "The capital of France is",   # 长度: 5
    "Hello world",                 # 长度: 2
]
max_tokens = 1

# Prefill 输出
logits_step1 = {
    req_1: (5, vocab_size),  # 5 个位置的 logits
    req_2: (2, vocab_size),  # 2 个位置的 logits
}
first_tokens = {req_1: 2054, req_2: 3932}  # sampled
kv_caches = {
    req_1: (L, 2, H, 5, D),    # 长度 5 的 KV
    req_2: (L, 2, H, 2, D),    # 长度 2 的 KV
}

# 写入存储
store["kv_req_1"] = kv_caches[req_1]
store["kv_req_2"] = kv_caches[req_2]

# Decode 输入（第 1 步）
current_tokens = {req_1: 2054, req_2: 3932}
loaded_kv = {
    req_1: store["kv_req_1"],
    req_2: store["kv_req_2"],
}

# Decode 输出（第 1 步）= 完成（因为 max_tokens=1）
next_tokens = {req_1: 8293, req_2: 5428}

# 最终输出
results = [
    {"prompt": "The capital of France is", "output": "France"},
    {"prompt": "Hello world", "output": "friend"}
]
```

---

## 关键文件导航

| 用途 | 文件 | 关键类/函数 |
|------|------|-----------|
| **高级 API** | `vllm/entrypoints/llm.py` | `LLM.generate()` |
| **V1 引擎** | `vllm/v1/engine/llm_engine.py` | `LLMEngine.step()` |
| **模型执行** | `vllm/v1/worker/gpu_worker.py` | `execute_model()` |
| **KV 配置** | `vllm/config/kv_transfer.py` | `KVTransferConfig` |
| **KV 传输** | `vllm/distributed/kv_transfer/...` | `SharedStorageConnector` |
| **输入处理** | `vllm/v1/engine/processor.py` | `Processor.process()` |
| **输出处理** | `vllm/v1/engine/output_processor.py` | `OutputProcessor` |

---

## 常见问题

### Q1: Prefill 和 Decode 需要两台机器吗？
**A:** 不需要。可以是：
- 同一台机器的两个 GPU
- 同一台机器的两个进程
- 不同机器（需 TorchDistributedConnector 或 P2pNcclConnector）

### Q2: 为什么要使用 SharedStorageConnector？
**A:** 
- 最简单的配置（本地文件系统）
- 适合单机多 GPU
- 避免网络开销

### Q3: KV Cache 需要多少存储空间？
**A:**
```
KV size = num_layers × 2 × num_heads × seq_len × head_dim
        = 32 × 2 × 32 × 5000 × 128 bytes
        ≈ 50 GB (单请求，长序列)
```
使用 SharedStorageConnector 时，确保有足够的本地磁盘空间。

### Q4: 比 Traditional 快多少？
**A:**
- **Prefill 时间：** 减少 10-20%（可用更大批处理）
- **Decode 时间：** 减少 30-50%（显存压力小）
- **整体吞吐量：** 增加 2-5x（取决于工作负载）
- **延迟：** 减少 15-30%（特别是长序列）

---

## 调试技巧

### 检查 KV Transfer 是否生效
```python
import os
os.environ["VLLM_USE_V1"] = "1"

from vllm import LLM
from vllm.config import KVTransferConfig

llm = LLM(
    model="meta-llama/Llama-3.2-1B",
    kv_transfer_config=KVTransferConfig(
        kv_connector="SharedStorageConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"shared_storage_path": "/tmp/kv_store"}
    )
)

# 运行生成
outputs = llm.generate(["Hello world"])

# 检查存储文件
print(os.listdir("/tmp/kv_store"))  # 应该看到 kv_req_*.bin 文件
```

### 性能监测
```python
import time

start = time.time()
outputs = llm.generate(long_prompts, sampling_params)
end = time.time()

print(f"总耗时: {end - start:.2f}s")
print(f"吞吐量: {sum(len(p) for p in long_prompts) / (end - start):.0f} tokens/s")
```

---

## 总结对照表

| 方面 | Traditional | Disaggregated |
|------|-----------|----------------|
| **架构** | 单实例处理全程 | Prefill/Decode 分离 |
| **显存** | 高（需存储完整KV） | 低（分阶段释放） |
| **吞吐量** | 受单阶段限制 | 2-5x 提升 |
| **延迟** | 高（长序列卡） | 低（异步并行） |
| **配置复杂度** | 简单 | 中等 |
| **适用场景** | 通用 | 长上下文推理 |


# Disaggregated Prefill-Decode 工作流解读

## 📋 概览

Disaggregated Prefill-Decode 是一种推理优化架构，将**长上下文填充阶段**（Prefill）和**短序列生成阶段**（Decode）分离到不同的计算实例中，通过 KV Cache 传输实现解耦。

```
┌──────────────────┐         KV Cache Transfer      ┌──────────────────┐
│  Prefill         │  ───────────────────────────>  │  Decode          │
│  Instance        │                                │  Instance        │
│  (Producer)      │  <────────────────────────────  │  (Consumer)      │
└──────────────────┘                                 └──────────────────┘
```

---

## 🔍 核心代码流程解析

### 第一步：初始化（Initialization）

#### 代码位置：`prefill_example.py` - Line 24-33
```python
llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    enforce_eager=True,
    gpu_memory_utilization=0.8,
    kv_transfer_config=KVTransferConfig(
        kv_connector="SharedStorageConnector",
        kv_role="kv_both",  # 既是 producer 也是 consumer
        kv_connector_extra_config={"shared_storage_path": "local_storage"},
    ),
)
```

**核心配置项解析：**

| 配置项 | 值 | 含义 |
|--------|-----|------|
| `kv_connector` | `SharedStorageConnector` | 使用本地共享存储作为 KV 传输通道 |
| `kv_role` | `"kv_both"` | 此实例既产生 KV（Prefill），也消费 KV（Decode） |
| `shared_storage_path` | `"local_storage"` | KV 缓存的本地存储路径 |

**工作原理：**

1. **`KVTransferConfig` 初始化**（`vllm/config/kv_transfer.py`）
   ```python
   @dataclass
   class KVTransferConfig:
       kv_connector: Optional[str]        # 传输机制类型
       kv_role: Optional[KVRole]          # 角色：producer/consumer/both
       kv_rank: Optional[int]             # 实例排名（0=prefill, 1=decode）
       kv_parallel_size: int = 1          # 并行度（1P1D情况下=1）
       kv_connector_extra_config: dict    # 连接器专用配置
   ```

2. **`LLM` 类初始化**（`vllm/entrypoints/llm.py`）
   ```
   LLM.__init__()
   ├─ 创建 EngineArgs（参数对象）
   ├─ 调用 LLMEngine.from_engine_args()
   └─ 创建引擎核心和处理器
   ```

3. **引擎选择**（`vllm/engine/llm_engine.py`）
   ```python
   if envs.VLLM_USE_V1:
       from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine
       LLMEngine = V1LLMEngine  # ← 使用 V1 引擎
   ```

---

### 第二步：请求提交（Request Submission）

#### 代码位置：`prefill_example.py` - Line 37-41
```python
outputs = llm.generate(
    prompts,
    sampling_params,
)
```

**工作链路：**

```
llm.generate()
├─ 验证和归一化输入
├─ 为每个 prompt 创建请求
├─ 调用 _validate_and_add_requests()
│  └─ 对每个 prompt 调用 _add_request()
│     └─ 调用 llm_engine.add_request()
│
└─ 调用 _run_engine()
   └─ 循环执行 llm_engine.step()
      └─ 返回结果列表
```

**详细步骤：**

#### 2.1 请求预处理
```python
# vllm/entrypoints/llm.py - LLM._add_request()
def _add_request(
    self,
    prompt: PromptType,
    params: SamplingParams,
    lora_request: Optional[LoRARequest] = None,
):
    # 1. 对 prompt 进行 tokenize
    # 2. 创建内部请求 ID
    # 3. 调用引擎的 add_request()
    self.llm_engine.add_request(
        request_id=request_id,
        prompt=prompt,
        params=params,
        lora_request=lora_request,
    )
```

#### 2.2 V1 引擎请求处理
```python
# vllm/v1/engine/llm_engine.py
class LLMEngine:
    def add_request(self, ...):
        # 1. 调用 Processor 转换输入
        request = self.processor.process_request(...)
        
        # 2. 提交给 EngineCoreClient
        self.engine_core.add_request(request)
```

---

### 第三步：Prefill 阶段（KV 生成）

#### 工作机制

**在 Prefill 实例上执行：**

```python
# vllm/v1/worker/gpu_worker.py
class GPUWorker:
    def execute_model(self, execute_model_request):
        """
        Prefill 阶段的模型执行
        """
        # 1. 准备输入：prompt tokens
        # 2. 执行前向传播：获取 logits 和中间状态
        # 3. 生成 KV cache（重要！）
        
        # 伪代码:
        outputs = self.model(input_ids, attention_mask)
        past_key_values = outputs.past_key_values  # ← KV cache
        
        return output_with_kv_cache
```

**Prefill 的关键产物：**

| 产物 | 说明 |
|------|------|
| **KV Cache** | K 和 V 的缓存，大小取决于上下文长度 |
| **Logits** | 下一个 token 的概率分布 |
| **First Token** | 采样得到的第一个生成 token |

---

### 第四步：KV Cache 传输（KV Transfer）

#### 配置路径

```
KVTransferConfig (配置)
    ↓
ensure_kv_transfer_initialized() (初始化)
    ↓
KVConnector (传输管道)
    ├─ SharedStorageConnector (本地存储)
    ├─ TorchDistributedConnector (分布式)
    └─ P2pNcclConnector (NCCL点对点)
```

#### SharedStorageConnector 工作流

```python
# vllm/distributed/kv_transfer/kv_connector/v1/shared_storage_connector.py

class SharedStorageConnector(KVConnectorBase_V1):
    def put_kv_cache(self, kv_cache_data):
        """
        Prefill 实例：将 KV cache 存入共享存储
        """
        # 1. 序列化 KV cache 张量
        # 2. 写入 local_storage/ 目录
        # 3. 生成元数据记录
        with open(f"{shared_storage_path}/kv_request_{request_id}.bin", "wb"):
            torch.save(kv_cache_data, file)
    
    def get_kv_cache(self, request_id):
        """
        Decode 实例：从共享存储读取 KV cache
        """
        # 1. 查找元数据
        # 2. 从磁盘读取 KV cache
        # 3. 恢复到显存中
        kv_cache = torch.load(f"{shared_storage_path}/kv_request_{request_id}.bin")
        return kv_cache
```

**传输时序图：**

```
时间轴 →

Prefill Instance              Decode Instance
────────────────             ───────────────
1. 执行 Prefill
   ├─ prompt tokenize       
   ├─ forward pass
   └─ 生成 KV cache
                             
2. KV 写入共享存储
   put_kv_cache() ──────────>
                             
                             3. KV 读取
                                get_kv_cache()
                                ├─ 读入 KV
                                └─ 恢复到显存
                             
                             4. Decode step 1
                                ├─ 使用 KV cache
                                ├─ forward(first_token, kv)
                                └─ 生成 next_token
```

---

### 第五步：Decode 阶段（自回归生成）

#### 核心流程

```python
# vllm/v1/worker/gpu_worker.py - Decode 步骤

class GPUWorker:
    def execute_model_for_decode(self, execute_model_request):
        """
        Decode 阶段：一次生成一个 token
        """
        # 1. 获取已生成 token（从前一步或初始）
        current_token = sequence.tokens[-1]  # 最后一个 token
        
        # 2. 加载 KV cache（从共享存储）
        kv_cache = kv_connector.get_kv_cache(request_id)
        
        # 3. 执行模型前向传播
        logits = self.model(
            input_ids=current_token.unsqueeze(0),  # shape: [1, 1]
            past_key_values=kv_cache,              # ← 复用的 KV cache
            use_cache=True
        )
        
        # 4. 采样获得下一个 token
        next_token = sample(logits, params=sampling_params)
        
        # 5. 累积新的 KV（用于下一步）
        sequence.append(next_token)
        
        return next_token
```

**Decode 的优势：**

| 方面 | 优势 |
|------|------|
| **计算效率** | 仅处理 1 个 token，矩阵运算更高效 |
| **内存节省** | Prefill 实例可释放，不需同时驻留 |
| **吞吐量** | 两实例独立并行，无相互阻塞 |
| **延迟** | 减少 Prefill 实例的显存压力 |

---

### 第六步：循环生成（Autoregressive Loop）

```python
# vllm/v1/engine/llm_engine.py

def step(self):
    """单次迭代步骤"""
    
    # 对每个活跃请求执行一步
    for request in active_requests:
        if request.is_prefill_phase:
            # Prefill: 一次处理所有 prompt tokens
            prefill_output = model.forward(
                input_ids=prompt_tokens,
                use_cache=True
            )
            request.kv_cache = prefill_output.past_key_values
            request.phase = "decode"  # ← 转换到 decode 阶段
            kv_connector.put_kv_cache(request.id, request.kv_cache)
            
        else:  # Decode phase
            # Decode: 只处理上一步生成的 token
            decode_output = model.forward(
                input_ids=previous_token,
                past_key_values=kv_cache,
                use_cache=True
            )
            next_token = sample(decode_output.logits)
            request.append_token(next_token)
            
            if is_finished(next_token):
                request.finish()
    
    return request_outputs
```

**生成循环可视化：**

```
步骤 1: Prefill
┌─────────────────────┐
│ "The capital of"    │ (5 tokens)
│ → KV cache          │
└─────────────────────┘

步骤 2: Decode (token 1)
┌─────────────────────┐
│ KV cache (cached)   │
│ + "France" (1 token)│
│ → logits            │
└─────────────────────┘

步骤 3: Decode (token 2)
┌─────────────────────┐
│ KV cache (updated)  │
│ + "is" (1 token)    │
│ → logits            │
└─────────────────────┘

...继续直到 EOS token
```

---

### 第七步：结果聚合和返回

```python
# vllm/entrypoints/llm.py - LLM.generate()

def generate(self, prompts, sampling_params):
    # 1. 添加所有请求
    self._validate_and_add_requests(prompts, sampling_params)
    
    # 2. 运行引擎直到完成
    outputs = self._run_engine(use_tqdm=True)
    
    # 3. 处理输出
    # vllm/v1/engine/output_processor.py
    request_outputs = [
        RequestOutput(
            request_id=req_id,
            prompt=original_prompt,
            outputs=[
                CompletionOutput(
                    text="France",  # 生成的文本
                    tokens=[...],
                    logprobs=[...]
                )
            ],
            usage=UsageStats(...)
        )
        for req_id in finished_requests
    ]
    
    return request_outputs
```

---

## 🏗️ 架构对比：Disaggregated vs Traditional

### Traditional（非分离）

```
单实例处理全程
┌────────────────────────┐
│ Prefill + Decode       │ ← 同一 GPU
│ (所有阶段)             │
└────────────────────────┘

缺点：
- GPU显存压力大
- Prefill 和 Decode 队列阻塞
- 难以优化不同阶段的计算策略
```

### Disaggregated（分离）

```
两个独立实例

┌──────────────┐        ┌──────────────┐
│ Prefill      │        │ Decode       │
│ 实例 (GPU0)  │        │ 实例 (GPU1)  │
└──────┬───────┘        └──────┬───────┘
       │                       │
       │ KV Cache Transfer     │
       ├──────────────────────>│
       │                       │
       │<──────────────────────┤
       │ (同步点)              │
       ▼                       ▼
    释放显存              继续生成
    处理新请求           下一个 token
```

**优势总结：**

| 指标 | Traditional | Disaggregated |
|------|------------|---------------|
| **显存使用** | 高（需存储所有KV） | 低（分阶段释放） |
| **吞吐量** | 受单阶段限制 | 两阶段独立并行 |
| **延迟** | 高（长上下文卡） | 低（Prefill快速处理） |
| **扩展性** | 困难 | 容易（调整实例数） |

---

## 📊 数据流示意

```
┌─────────────────────────────────────────────────────────────────┐
│                      Input Prompts                              │
│  ["The capital of", "Hello my", "What is", ...]                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
             ┌──────────────────────┐
             │ Tokenization         │
             │ (word → token IDs)   │
             └──────────────────────┘
                         │
                         ▼
     ┌───────────────────────────────────────┐
     │     Prefill Instance                  │
     │                                       │
     │ Input: [101, 2054, 2003, ...] (5tok) │
     │ ↓ Model Forward Pass                  │
     │ Output:                               │
     │ - logits: [vocab_size]                │
     │ - KV cache: [(L, 2, H, S, D)]         │
     │   L=layers, H=heads, S=seq_len, D=dim│
     └────────────┬────────────────────────┘
                  │
                  ▼
         ┌─────────────────────┐
         │ KV Cache Storage    │
         │ (SharedStorage)     │
         │ request_123.bin     │
         └─────────────────────┘
                  │
                  ▼
     ┌───────────────────────────────────────┐
     │     Decode Instance                   │
     │                                       │
     │ Load KV cache from storage            │
     │ Input: [last_token] (1 token)         │
     │ ↓ Model Forward Pass                  │
     │ Output:                               │
     │ - logits: [vocab_size]                │
     │ - next_token_id: 2054                 │
     │ - updated KV cache                    │
     └────────────┬────────────────────────┘
                  │
                  ▼
            ┌─────────────┐
            │ Sampling    │
            │ next token  │
            └──────┬──────┘
                   │
                   ▼
         ┌──────────────────┐
         │ Sequence Update  │
         │ tokens += [2054] │
         └────────┬─────────┘
                  │
            ┌─────▼─────┐
            │ Finished? │
            └─────┬─────┘
              Yes │   No
                  │    └─────────┐
                  │              ▼
                  │         Loop: Decode...
                  │
                  ▼
         ┌──────────────────┐
         │ Output Result    │
         │ "The capital of  │
         │  France is..."   │
         └──────────────────┘
```

---

## 🔧 关键组件映射表

| 组件 | 文件路径 | 职责 |
|------|---------|------|
| **LLM** | `vllm/entrypoints/llm.py` | 高级 API，用户接口 |
| **LLMEngine(V1)** | `vllm/v1/engine/llm_engine.py` | 核心引擎，请求管理 |
| **EngineCoreClient** | `vllm/v1/engine/core_client.py` | 引擎核心客户端 |
| **Processor** | `vllm/v1/engine/processor.py` | 输入处理（tokenize等） |
| **OutputProcessor** | `vllm/v1/engine/output_processor.py` | 输出处理（detokenize等） |
| **GPUWorker** | `vllm/v1/worker/gpu_worker.py` | 模型执行 |
| **KVTransferConfig** | `vllm/config/kv_transfer.py` | KV传输配置 |
| **SharedStorageConnector** | `vllm/distributed/kv_transfer/...` | KV传输实现 |
| **SamplingParams** | `vllm/sampling_params.py` | 采样参数 |

---

## 🎯 执行示例追踪

### 示例 Prompts
```python
prompts = [
    "Hi " * 1000 + "Hello, my name is",  # 长上下文
    "Hey " * 500 + "Your name is",        # 中上下文
]
sampling_params = SamplingParams(
    temperature=0,
    top_p=0.95,
    max_tokens=1  # 只生成 1 个 token
)
```

### 执行时序

```
时间 → 

[T=0]   Prefill Instance                Decode Instance
        ├─ Prompt 1: tokenize (1005)
        ├─ Prompt 2: tokenize (505)
        └─ Forward pass (both)
           ├─ Output: logits, KV
           ├─ Sample: "France", "Yuki"
           └─ Write KV to storage

[T=1]                                    ├─ Read KV (req1, req2)
                                         ├─ Input: ["France"], ["Yuki"]
                                         ├─ Forward: single token
                                         └─ [max_tokens=1, DONE]

[T=2]   ✓ Finished: Req1                ✓ Finished: Req2
        Output: "Hello, my name is      Output: "Your name is Yuki"
                 France"
```

### 返回结果示例

```python
[
    RequestOutput(
        request_id="req-1",
        prompt="Hi Hi Hi ... Hello, my name is",
        outputs=[
            CompletionOutput(
                text="France",
                tokens=[2054],
                logprobs=[-0.23]
            )
        ],
        usage=UsageStats(
            prompt_tokens=1005,
            completion_tokens=1,
            total_tokens=1006
        )
    ),
    RequestOutput(
        request_id="req-2",
        prompt="Hey Hey ... Your name is",
        outputs=[
            CompletionOutput(
                text="Yuki",
                tokens=[6142],
                logprobs=[-0.15]
            )
        ],
        usage=UsageStats(
            prompt_tokens=505,
            completion_tokens=1,
            total_tokens=506
        )
    )
]
```

---

## 🚀 性能优化要点

### 1. **Prefill 优化**
- 批处理多个 prompt
- 使用 Flash Attention
- 启用 CUDA Graph

### 2. **KV 传输优化**
- 共享存储本地化（避免网络）
- 异步读写
- 压缩 KV cache（FP8 量化）

### 3. **Decode 优化**
- 推测解码（Speculative Decoding）
- 批处理多序列
- 使用较小的显存

### 4. **负载均衡**
- 动态调整 Prefill/Decode 实例数
- 根据请求长度调度
- 监控队列长度

---

## 📝 总结

**Disaggregated Prefill-Decode 的核心思想：**

1. **分离阶段**：Prefill（处理长上下文）和 Decode（生成序列）在不同实例上运行
2. **异步传输**：通过 KV Cache 共享存储实现解耦
3. **独立优化**：每个阶段可针对性优化（批处理、精度、并行度等）
4. **降低延迟**：避免长序列对短生成的阻塞，提升整体吞吐量

这种架构特别适合长上下文推理场景，如 RAG、长文档处理等。

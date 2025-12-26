# vLLM v1引擎 Prefill-Decode分离场景下Token生成行为调研报告

## 调研背景

在PD-disaggregation（Prefill-Decode分离）模式下，我们观察到vLLM记录了2次ADMISSION事件（prefill + first decode token），而Frontier仅记录1次ADMISSION（仅prefill）。本调研旨在通过分析vLLM v1引擎的实际代码实现，明确以下问题：

1. Prefill阶段完成后是否生成/输出新token？
2. "First Token"在vLLM代码库中的语义定义

---

## 调研问题1：Prefill到Decode的Token转换分析

### 核心发现

**结论：Prefill阶段完成后会生成1个token，该token与KV cache一起传输到decode实例。Decode实例的第一次decode step操作的上下文大小是11 tokens（input_length + 1 prefill生成的token）。**

### 代码证据

#### 1.1 Prefill阶段的Token生成

**文件：`vllm/v1/worker/gpu_model_runner.py`**
**行号：2000-2160**

```python
@torch.inference_mode()
def execute_model(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors: Optional[IntermediateTensors] = None,
) -> Union[ModelRunnerOutput, AsyncModelRunnerOutput, IntermediateTensors]:
    # ... 预处理代码 ...
    
    with record_function_or_nullcontext("Sample"):
        sampler_output = self._sample(logits, spec_decode_metadata)
    
    # ... 后处理代码 ...
    
    output = ModelRunnerOutput(
        req_ids=req_ids_output_copy,
        req_id_to_index=req_id_to_index_output_copy,
        sampled_token_ids=valid_sampled_token_ids,  # <-- 采样的token IDs
        logprobs=logprobs_lists,
        prompt_logprobs_dict=prompt_logprobs_dict,
        pooler_output=[],
        kv_connector_output=kv_connector_output,
        num_nans_in_logits=num_nans_in_logits,
    )
```

**关键点**：`execute_model`方法在每次forward pass后都会调用`_sample`方法生成token，无论是prefill还是decode阶段。

#### 1.2 Scheduler中的Token处理

**文件：`vllm/v1/core/sched/scheduler.py`**
**行号：920-1050**

```python
def update_from_output(
    self,
    scheduler_output: SchedulerOutput,
    model_runner_output: ModelRunnerOutput,
) -> dict[int, EngineCoreOutputs]:
    sampled_token_ids = model_runner_output.sampled_token_ids
    # ...
    
    for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
        # ...
        req_index = model_runner_output.req_id_to_index[req_id]
        generated_token_ids = sampled_token_ids[
            req_index] if sampled_token_ids else []
        
        # ...
        
        new_token_ids = generated_token_ids
        # ...
        
        # Check for stop and update request status.
        if new_token_ids:
            new_token_ids, stopped = self._update_request_with_output(
                request, new_token_ids)
```

**关键点**：Scheduler的`update_from_output`方法处理来自model runner的采样token，并将其添加到请求的输出token列表中。

#### 1.3 KV Cache传输时的Token状态

**文件：`vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py`**
**行号：370-420**

```python
def request_finished(
    self,
    request: "Request",
    block_ids: list[int],
) -> tuple[bool, Optional[dict[str, Any]]]:
    """
    Once a request is finished, determine whether request blocks
    should be freed now or will be sent asynchronously and freed later.
    """
    # ...
    
    return delay_free_blocks, dict(
        do_remote_prefill=True,
        do_remote_decode=False,
        remote_block_ids=block_ids,  # <-- 包含prefill生成的token的KV cache
        remote_engine_id=self.engine_id,
        remote_host=self.side_channel_host,
        remote_port=self.side_channel_port,
        tp_size=self.vllm_config.parallel_config.tensor_parallel_size)
```

**关键点**：当prefill完成时，`request_finished`方法返回包含所有block IDs的KV传输参数，这些blocks包含了prefill阶段生成的token的KV cache。

#### 1.4 Decode实例接收KV Cache后的处理

**文件：`vllm/v1/core/sched/scheduler.py`**
**行号：1310-1335**

```python
def _update_waiting_for_remote_kv(self, request: Request) -> bool:
    """
    KV Connector: check if the request_id is finished_recving.
    """
    assert self.connector is not None
    if request.request_id not in self.finished_recving_kv_req_ids:
        return False

    # Now that the blocks are ready, actually cache them.
    (block_ids, ) = self.kv_cache_manager.get_block_ids(request.request_id)
    num_computed_tokens = len(block_ids) * self.block_size
    # Handle the case where num request tokens less than one block.
    num_computed_tokens = min(num_computed_tokens, request.num_tokens)
    if num_computed_tokens == request.num_tokens:
        num_computed_tokens -= 1  # <-- 关键：确保至少有1个token需要计算
    # This will cache the blocks iff caching is enabled.
    self.kv_cache_manager.cache_blocks(request, num_computed_tokens)

    # Update the request state for scheduling.
    request.num_computed_tokens = num_computed_tokens
```

**关键点**：当decode实例接收到KV cache后，`num_computed_tokens`被设置为传输的blocks对应的token数量。代码中的`num_computed_tokens -= 1`确保至少有1个token需要在decode实例上计算。

### 1.5 Prefill示例代码分析

**文件：`examples/offline_inference/disaggregated_prefill.py`**
**行号：130-145**

```python
def run_prefill(args: argparse.Namespace):
    # ...
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)
    # ...
    llm.generate(prompts, sampling_params)
```

**关键点**：Prefill节点使用`max_tokens=1`，这意味着prefill阶段会生成1个token。

---

## 调研问题2："First Token"语义澄清

### 核心发现

**结论：在vLLM代码库中，"first token"指的是prefill阶段结束时生成的第一个输出token，即TTFT（Time To First Token）测量的是从请求到达到prefill完成并生成第一个token的时间。**

### 代码证据

#### 2.1 TTFT指标实现

**文件：`vllm/v1/metrics/stats.py`**
**行号：110-135**

```python
def update_from_output(self, output: "EngineCoreOutput",
                       engine_core_timestamp: float, is_prefilling: bool,
                       prompt_len: int, req_stats: RequestStateStats,
                       lora_stats: Optional[LoRAStats]):
    num_new_generation_tokens = len(output.new_token_ids)

    self.num_generation_tokens += num_new_generation_tokens
    if is_prefilling:
        self.num_prompt_tokens += prompt_len

        first_token_latency = self._time_since(req_stats.arrival_time)
        self.time_to_first_tokens_iter.append(first_token_latency)
        req_stats.first_token_latency = first_token_latency

    # ...
    
    # Process the batch-level "new tokens" engine core event
    if is_prefilling:
        req_stats.first_token_ts = engine_core_timestamp  # <-- first token时间戳
    else:
        itl = engine_core_timestamp - req_stats.last_token_ts
        self.inter_token_latencies_iter.append(itl)
```

**关键点**：
- `is_prefilling`为True时记录`first_token_latency`
- `first_token_ts`在prefill阶段完成时设置
- TTFT = `iteration_timestamp - arrival_time`（当`is_prefilling`为True时）

#### 2.2 Output Processor中的is_prefilling状态

**文件：`vllm/v1/engine/output_processor.py`**
**行号：410-420**

```python
# 在process_outputs方法中
kv_transfer_params = engine_core_output.kv_transfer_params
req_state.num_cached_tokens = engine_core_output.num_cached_tokens
req_state.is_prefilling = False  # <-- prefill完成后设置为False
```

**关键点**：`is_prefilling`状态在处理第一个输出token后被设置为False，这意味着：
- 第一个token生成时`is_prefilling = True`
- 后续token生成时`is_prefilling = False`

#### 2.3 Prometheus指标定义

**文件：`vllm/v1/metrics/loggers.py`**
**行号：367-380**

```python
histogram_time_to_first_token = self._histogram_cls(
    name="vllm:time_to_first_token_seconds",
    documentation="Histogram of time to first token in seconds.",
    buckets=[
        0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
        0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float("inf")
    ],
    labelnames=labelnames)
```

#### 2.4 Tracing中的TTFT属性

**文件：`vllm/v1/engine/output_processor.py`**
**行号：480-495**

```python
def do_tracing(self, engine_core_output: EngineCoreOutput,
               req_state: RequestState,
               iteration_stats: Optional[IterationStats]) -> None:
    # ...
    prefill_time = metrics.first_token_ts - metrics.scheduled_ts
    decode_time = metrics.last_token_ts - metrics.first_token_ts
    # ...
    span.set_attribute(
        SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN,
        metrics.first_token_latency)
```

**关键点**：
- `prefill_time` = `first_token_ts` - `scheduled_ts`
- `decode_time` = `last_token_ts` - `first_token_ts`
- 这清楚地表明first token是prefill阶段的输出

#### 2.5 v0引擎中的first_token_time定义

**文件：`vllm/sequence.py`**
**行号：775-785**

```python
def maybe_set_first_token_time(self, time: float) -> None:
    """Sets the first token time for Request level timings."""
    # Note: in a case where a sequence_group is swapped and
    #   recomputed, the time between iterations is counted
    #   in TPOT, rather than recalculating TTFT (since from the )
    #   POV of the user, there is simply a long generation delay.
    if (self.metrics.first_token_time is None
            and self.first_seq.get_output_len() == 1):
        self.metrics.first_token_time = time
```

**关键点**：`first_token_time`在`output_len == 1`时设置，即第一个输出token生成时。

---

## 与ADMISSION事件差异的关联分析

### vLLM的ADMISSION事件记录逻辑

**文件：`vllm/v1/core/sched/scheduler.py`**
**行号：540-560**

```python
# Flow validation: log WAITING request admission
_allocated_blocks = self.kv_cache_manager.get_blocks(
    request.request_id)
_num_blocks = sum(len(group) for group in _allocated_blocks.blocks) if _allocated_blocks else 0
_log_flow(
    f"[ADMISSION] req={request.request_id}, "
    f"num_tokens={num_new_tokens}, "
    f"running_count={len(self.running)}, "
    f"token_budget_remaining={token_budget}, "
    f"blocks_allocated={_num_blocks}"
)
```

### 分析

在PD-disaggregation模式下，vLLM记录2次ADMISSION事件的原因：

1. **第一次ADMISSION（Prefill实例）**：
   - 请求首次从WAITING队列进入RUNNING队列
   - 分配KV cache blocks用于prefill
   - 执行prefill并生成第一个token

2. **第二次ADMISSION（Decode实例）**：
   - 请求在decode实例上从WAITING_FOR_REMOTE_KVS状态转换为RUNNING状态
   - 接收远程KV cache
   - 开始decode阶段

### Frontier实现建议

如果Frontier仅记录1次ADMISSION（prefill），这可能是因为：
1. Frontier将整个请求生命周期视为单一admission
2. 或者Frontier的事件记录粒度不同

**建议**：Frontier应该考虑是否需要区分prefill admission和decode admission，以便更准确地追踪请求在分离架构中的生命周期。

---

## 总结

### 问题1答案

| 问题 | 答案 |
|------|------|
| Prefill完成后是否生成token？ | **是**，生成1个token |
| 该token是否传输到decode实例？ | **是**，与KV cache一起传输 |
| Decode实例第一次decode的上下文大小？ | **Option B: 11 tokens** (input_length=10 + prefill生成的1个token) |

### 问题2答案

| 问题 | 答案 |
|------|------|
| "First token"指什么？ | **Option A**：Prefill阶段结束时生成/输出的token |
| TTFT测量的是什么？ | 从请求到达到prefill完成并生成第一个token的时间 |

### 关键代码引用汇总

| 功能 | 文件 | 关键行号 |
|------|------|----------|
| Token采样 | `vllm/v1/worker/gpu_model_runner.py` | 2000-2160 |
| Token处理 | `vllm/v1/core/sched/scheduler.py` | 920-1050 |
| KV传输 | `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py` | 370-420 |
| TTFT计算 | `vllm/v1/metrics/stats.py` | 110-135 |
| is_prefilling状态 | `vllm/v1/engine/output_processor.py` | 410-420 |
| ADMISSION日志 | `vllm/v1/core/sched/scheduler.py` | 540-560 |

---

## 附录：调研方法

1. 使用`grepSearch`搜索关键术语：`TTFT`, `first_token`, `kv_transfer`, `disagg`, `is_prefill`
2. 阅读核心模块源代码：scheduler, model_runner, output_processor, metrics
3. 分析disaggregated prefill示例代码
4. 交叉验证v0和v1引擎的实现差异

---

*调研日期：2024年12月21日*
*vLLM版本：0.10.2*

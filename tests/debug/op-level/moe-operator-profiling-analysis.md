## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2025-12-19 | Initial comprehensive analysis of MoE operator profiling semantic correspondence between simulator and vLLM; documented vLLM instrumentation locations and recommended scope alignment strategy. |

## MoE Operator Profiling Analysis (Simulator ↔ vLLM)

本文件记录并固化关于 MoE (Mixture of Experts) operator-level profiling 的关键结论与操作指南，覆盖以下 5 个问题点（**1-5 全部内容不遗漏**），用于后续在 vLLM 中做 scope instrumentation、跑 profiling、并与 simulator 预测结果进行一致性对比。

---

## 0. Scope and Target

- **Target model**: `microsoft/Phi-tiny-MoE-instruct`
- **vLLM engine**: vLLM v1 engine
- **Simulator modules**:
  - `frontier/profiling/moe/moe_wrapper.py`
  - `frontier/profiling/moe/moe_impl.py`
  - `frontier/profiling/moe/moe_vllm_kernel.py`
  - `frontier/profiling/utils/record_function_tracer.py`
- **vLLM instrumentation files**:
  - `vllm/model_executor/models/phimoe.py`
  - `vllm/model_executor/layers/fused_moe/layer.py`
  - `vllm/model_executor/layers/fused_moe/fused_moe.py`
  - (辅助理解) `vllm/model_executor/layers/fused_moe/moe_align_block_size.py`

---

## 1) Semantic Correspondence Analysis

本节对照 simulator 与 vLLM 的真实执行流，判断 `moe_gating / moe_shuffling / moe_grouped_gemm` 语义是否一致、实现是否一致，并解释理由（带明确 code reference）。

### 1.1 Simulator: Ground-truth operator semantics (implementation-defined)

#### (A) `moe_gating` (Router/gating network)

Simulator 的 `moe_gating` 计时域 **包含**：
- Router linear (`nn.Linear(hidden_dim, num_experts)`)
- `torch.topk` 选择 top-k experts
- `F.softmax` 对 routing weights 归一化

Code reference:

```71:83:/research/d1/gds/ytyang/yichengfeng/frontier/frontier/profiling/moe/moe_impl.py
        with self.gating_timer:
            # Compute gating logits using native PyTorch
            logits = self.gate(hidden_states)  # [num_tokens, num_experts]

            # Top-K selection
            routing_weights, selected_experts = torch.topk(
                logits, self.router_topk, dim=-1
            )

            # Softmax normalization
            routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
```

**结论（语义）**：simulator 的 `moe_gating` 是“完整 routing decision”（logits → topk → softmax）的一体化 scope。

#### (B) `moe_shuffling` (Local token shuffling)

Simulator 的 `moe_shuffling` 计时域 **包含**：
- `selected_experts` flatten（topk expansion）
- `torch.argsort(expert_ids)`：按 expert id 排序（核心 shuffling）
- `hidden_states[...]` gather：按排序结果做 token reorder

Code reference:

```117:129:/research/d1/gds/ytyang/yichengfeng/frontier/frontier/profiling/moe/moe_impl.py
        with self.shuffling_timer:
            expert_ids = selected_experts.flatten()  # [num_tokens * router_topk]
            token_ids = torch.arange(num_tokens, device=hidden_states.device).repeat_interleave(self.router_topk)
            sorted_indices = torch.argsort(expert_ids)
            shuffled_token_ids = token_ids[sorted_indices]
            shuffled_states = hidden_states[shuffled_token_ids]
```

**结论（语义）**：simulator 的 `moe_shuffling` 本质是“按 expert 分组的 token 重排（sort + gather）”，并且强调是 local shuffling（不含跨设备 all-to-all）。

#### (C) `moe_grouped_gemm` (Expert computation)

Simulator 的 `moe_grouped_gemm` 计时域（默认 loop mode）**包含专家 FFN 的全流程计算**：
- up projection
- activation（例如 `SiluAndMul`）
- down projection
- 在 per-expert loop 下逐 expert 处理，然后 concat

Code reference:

```238:269:/research/d1/gds/ytyang/yichengfeng/frontier/frontier/profiling/moe/moe_impl.py
        with self.grouped_gemm_timer:
            outputs = []
            start_idx = 0
            for expert_id, num_tokens_for_expert in enumerate(expert_allocation):
                if num_tokens_for_expert > 0:
                    end_idx = start_idx + num_tokens_for_expert
                    expert_input = hidden_states[start_idx:end_idx]
                    if self.use_gated:
                        up_output, _ = self.experts[expert_id]["up_proj"](expert_input)
                        intermediate = self.experts[expert_id]["act_fn"](up_output)  # SiluAndMul
                        expert_output, _ = self.experts[expert_id]["down_proj"](intermediate)
```

**结论（语义）**：simulator 的 `moe_grouped_gemm` 是“专家侧 FFN compute 总体”的单一 scope（up→act→down），即使实现是 loop，它测量边界是“专家计算整体”。

### 1.2 vLLM: Real execution flow for PhiMoE (`microsoft/Phi-tiny-MoE-instruct`)

#### (A) `moe_gating` in vLLM is split into two *separate* sites

vLLM routing 在 PhiMoE 路径上被拆为两段：

1) **Router logits linear layer**（模型侧 gate linear）在 `PhiMoE.forward`：

```286:294:vllm/model_executor/models/phimoe.py
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # router_logits: (num_tokens, n_experts)
        with record_function_or_nullcontext("moe_gating"):
            router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states, router_logits)
        return final_hidden_states.view(orig_shape)
```

2) **Top-k selection + routing weights (softmax-like behavior)** 在 `FusedMoE.select_experts(...)`，其内部会调用 `fused_topk`/`grouped_topk`（最终走 custom op / GPU kernel）：

```1473:1509:vllm/model_executor/layers/fused_moe/layer.py
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk

        with record_function_or_nullcontext("moe_gating"):
            ...
            elif custom_routing_function is None:
                topk_weights, topk_ids, token_expert_indices = fused_topk(
                    hidden_states=hidden_states,
                    gating_output=router_logits,
                    topk=top_k,
                    renormalize=renormalize,
                    indices_type=indices_type,
                )
```

**语义对应判断**：
- simulator 的 `moe_gating` = (linear logits) + (topk) + (softmax)
- vLLM 当前 `moe_gating` 分散为两处，但**合起来**等价于 simulator 的语义

**一致性注意（关键）**：
- `RecordFunctionTracer` 会按 `user_annotation` 的 `name` 聚合统计（同名 scope 的 kernel 时间会累加）。
- 因此比较时应把 vLLM 中两个 `moe_gating` 片段按名字聚合后与 simulator 的单一 `moe_gating` 对齐。

参考：`RecordFunctionTracer.get_operation_time_stats()` 聚合逻辑（按 `event["name"]` 收集统计）。

```60:112:/research/d1/gds/ytyang/yichengfeng/frontier/frontier/profiling/utils/record_function_tracer.py
    def get_operation_time_stats(self, debug=False):
        stats = {}
        ...
        for event in trace:
            if not ("cat" in event and event["cat"] == "user_annotation"):
                continue
            ...
            name = event["name"].replace("vidur_", "")
            ...
            if name not in stats:
                stats[name] = []
            stats[name].append(cuda_time * 1e-3)  # ms
```

#### (B) `moe_shuffling` semantic match exists in vLLM via `moe_align_block_size`

vLLM 的“shuffling / routing-induced reorder”由 `moe_align_block_size` 承担：它从 `topk_ids` 生成 `sorted_token_ids` 与 `expert_ids`（内部调用 `ops.moe_align_block_size`）。

```12:87:/research/d1/gds/ytyang/yichengfeng/frontier/sota-infer-engine/vllm/vllm/model_executor/layers/fused_moe/moe_align_block_size.py
def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    pad_sorted_ids: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ...
    ops.moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids,
                             expert_ids, num_tokens_post_pad)
    ...
    return sorted_ids, expert_ids, num_tokens_post_pad
```

语义对齐结论：
- simulator: `argsort(expert_ids)` + gather
- vLLM: `ops.moe_align_block_size(...)` 生成等价的“按 expert 分组的 token 顺序”（更偏 kernelized/fused）
- 两者语义一致（local shuffling），但实现不同（Python tensor ops vs custom op）

#### (C) `moe_grouped_gemm` semantic match exists but is split into two GEMM calls

vLLM 的专家 FFN 结构为两段 GEMM（up/w13 + down/w2）与 activation/merge（详见第4/第5节）。

---

## 2) Missing Shuffling Profiling

### 2.1 Confirm: Shuffling exists in vLLM MoE

**存在**。核心在 `moe_align_block_size(topk_ids, ...)`，其内部调用 `ops.moe_align_block_size` 生成 `sorted_token_ids`（即 token reorder / alignment）。

其语义对应 simulator 的 `MoETokenShuffler`。

### 2.2 Precise location: where shuffling occurs in vLLM execution flow

在 vLLM fused MoE 执行中（两条路径）都会调用 `moe_align_block_size`：
- `fused_experts_impl(...)` 的 chunked path
- `TritonExperts.apply(...)` 的 modular kernel path

### 2.3 Instrumentation added: `record_function_or_nullcontext("moe_shuffling")`

已在 `vllm/model_executor/layers/fused_moe/fused_moe.py` 中，精准包住 `moe_align_block_size(...)`（确保 scope 语义接近 simulator 的“sort+gather”）。

验证片段（chunked path）：

```1636:1647:vllm/model_executor/layers/fused_moe/fused_moe.py
        with record_function_or_nullcontext("moe_expert"):
            ...
            with record_function_or_nullcontext("moe_shuffling"):
                sorted_token_ids, expert_ids, num_tokens_post_padded = (
                    moe_align_block_size(curr_topk_ids, config['BLOCK_SIZE_M'],
                                         global_num_experts, expert_map))
```

验证片段（modular path）：

```2003:2007:vllm/model_executor/layers/fused_moe/fused_moe.py
        with record_function_or_nullcontext("moe_expert"):
            with record_function_or_nullcontext("moe_shuffling"):
                sorted_token_ids, expert_ids, num_tokens_post_padded = (
                    moe_align_block_size(topk_ids, config['BLOCK_SIZE_M'],
                                         global_num_experts, expert_map))
```

---

## 3) Performance Bottleneck Validation

本节评估 simulator 的假设：“MoE FFN 的主要耗时来自 `grouped_gemm`, `gating`, `shuffling`”，并指出 vLLM 实际实现里是否还有其它显著操作应纳入测量。

### 3.1 Is the simulator assumption reasonable?

总体合理（常见 GPU MoE/FFN 性能结构）：
- **`moe_grouped_gemm`**：核心是两次大矩阵乘（按 token-expert grouping 的 grouped GEMM），通常是主耗时来源，compute-heavy。
- **`moe_gating`**：router logits linear + topk/softmax，复杂度较低但在 token 数/experts 数大时可能显著。
- **`moe_shuffling`**：token reorder / alignment，多为 memory-bound；一般小于 GEMM，但可能在极端 token 数或特定实现中可见。

### 3.2 Other significant operations in vLLM that may need measurement

基于 vLLM `fused_experts_impl` 的真实执行流（参见第5节引用片段），除上述三项外，还有：
- **Activation**：例如 `torch.ops._C.silu_and_mul(...)`（常有可见成本）
- **Quantize/scale prep**：`moe_kernel_quantize_input(...)`（可能触发额外 kernel）
- **Combine/reduce**：`ops.moe_sum(...)`（将 top-k expert outputs 合并回 token 输出）
- **Distributed communication (if EP/DP/all2all enabled)**：这通常应作为通信项单独测量；simulator 也强调 cross-device shuffling 属于 communication，而不是 local shuffling。

结论建议（实操）：
- 如果目标是与 simulator 的三项特征严格对齐，至少应明确：vLLM 的 activation/quantize/moe_sum 是否被归入某个 simulator 项（否则会出现“ground truth 时间组成不同”的偏差）。

---

## 4) Duplicate `moe_grouped_gemm` Profiling in vLLM

你观察到在 `vllm/model_executor/layers/fused_moe/fused_moe.py` 内 `moe_grouped_gemm` 被包了两次，这不是重复调用同一个阶段，而是 **MoE expert FFN 本身需要两次 GEMM**。

### 4.1 Why `grouped_gemm` is invoked twice

MoE expert FFN（以 gated FFN/SwiGLU 为例）结构是：
1) **GEMM #1**: up projection / w1 or w13（输出通常为 `2 * intermediate_size`）
2) **Activation**: `silu_and_mul` 把 gate/value 融合回 `intermediate_size`
3) **GEMM #2**: down projection / w2（回到 `hidden_size`）
4) **Combine**: `ops.moe_sum` 合并 top-k outputs

因此，vLLM 里两个 `moe_grouped_gemm` 分别对应：
- 第一次 `invoke_fused_moe_kernel(..., w1/w13, ...)`：expert up projection
- 第二次 `invoke_fused_moe_kernel(..., w2, ...)`：expert down projection

这不是不同 expert group，也不是 prefill/decode phase，而是同一个 MoE 层内部的两段线性。

### 4.2 Concrete vLLM evidence (chunked path)

```1649:1722:vllm/model_executor/layers/fused_moe/fused_moe.py
            with record_function_or_nullcontext("moe_grouped_gemm"):
                invoke_fused_moe_kernel(... w1 ... )
            ...
            # activation between the two GEMMs
            if activation == "silu" and is_act_and_mul:
                torch.ops._C.silu_and_mul(...)
            ...
            with record_function_or_nullcontext("moe_grouped_gemm"):
                invoke_fused_moe_kernel(... w2 ... )
```

同理，modular path 在 `TritonExperts.apply(...)` 也保持相同结构（见第5节第二段引用）。

---

## 5) `moe_expert` Profiling Scope in vLLM

本节解释 `moe_expert` 代表什么、是否覆盖“整个 FFN 结构处理时间”、其 scope 包含哪些操作，以及与 `moe_gating/moe_shuffling/moe_grouped_gemm` 的包含关系。

### 5.1 What `moe_expert` represents

`moe_expert` 是 vLLM 侧“专家计算整体 scope”（expert-side end-to-end within a MoE layer），用于包住：
- local shuffling/alignment（`moe_align_block_size`）
- 两次 grouped GEMM（up + down）
- activation（例如 `silu_and_mul`）
- 量化/scale 处理（如 `moe_kernel_quantize_input`）
- 合并输出（`ops.moe_sum`）

### 5.2 Does `moe_expert` measure the entire MoE FFN processing time?

它测量的是 **MoE 专家侧 FFN 的整体处理时间**，但 **不包含 routing/gating 的时间**（router logits linear + topk/softmax），这些在 `moe_gating` scope 中。

换句话说：
- `moe_expert` ≈ (shuffling + expert GEMM/activation + combine) 的整体
- `moe_gating` ≈ routing decision（可能分散为两个同名区间，需要 name 聚合）

### 5.3 Concrete vLLM evidence (what is included inside `moe_expert`)

chunked path（`fused_experts_impl`）：

```1636:1725:vllm/model_executor/layers/fused_moe/fused_moe.py
        with record_function_or_nullcontext("moe_expert"):
            qcurr_hidden_states, a1q_scale = moe_kernel_quantize_input(...)
            with record_function_or_nullcontext("moe_shuffling"):
                sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(...)
            with record_function_or_nullcontext("moe_grouped_gemm"):
                invoke_fused_moe_kernel(... w1 ...)
            # activation (included in moe_expert, not in moe_grouped_gemm)
            torch.ops._C.silu_and_mul(...)
            qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(...)
            with record_function_or_nullcontext("moe_grouped_gemm"):
                invoke_fused_moe_kernel(... w2 ...)
            ops.moe_sum(...)
```

modular path（`TritonExperts.apply`）：

```2003:2068:vllm/model_executor/layers/fused_moe/fused_moe.py
        with record_function_or_nullcontext("moe_expert"):
            with record_function_or_nullcontext("moe_shuffling"):
                sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(...)
            with record_function_or_nullcontext("moe_grouped_gemm"):
                invoke_fused_moe_kernel(... w1 ...)
            self.activation(...)
            qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(...)
            with record_function_or_nullcontext("moe_grouped_gemm"):
                invoke_fused_moe_kernel(... w2 ...)
            ops.moe_sum(...)
```

### 5.4 Relation to other profiling scopes

建议用“包含关系”理解（便于后续对齐 simulator 与 predictor）：
- **`moe_gating`**：routing（logits linear + topk/softmax）。在 vLLM 中当前拆成两处同名 scope，比较时按 name 聚合。
- **`moe_shuffling`**：token reorder/alignment（vLLM 对应 `moe_align_block_size`/`ops.moe_align_block_size`）。
- **`moe_grouped_gemm`**：两段 GEMM（up 和 down），在 vLLM 中被分别包了两次（不是重复，是结构需要）。
- **`moe_expert`**：专家侧 end-to-end（含 shuffling + 两段 GEMM + activation + quantize + moe_sum）。

---

## Actionable Notes for Consistency with Simulator (Important)

### A) Aggregation rule for comparing `moe_gating`

由于 vLLM 的 routing 被拆为两段（模型侧 linear + fused_topk/grouped_topk），当前 instrumentation 用相同 name `moe_gating` 标注两个 scope。
在比较时应按 name 聚合（累加）后与 simulator 的单一 `moe_gating` 对齐。

### B) Potential boundary mismatch for `moe_grouped_gemm` (strict alignment warning)

simulator 的 `moe_grouped_gemm`（loop mode）语义是“专家 FFN compute 总体”（up→act→down）。
而 vLLM 当前 `moe_grouped_gemm` 仅包住两次 GEMM，各自独立；activation 与 `moe_sum` 落在 `moe_expert` 内。

如果你的 simulator 训练/预测模块严格把 activation 也算入 `moe_grouped_gemm`，则需要调整 vLLM instrumentation 边界（例如将 `moe_grouped_gemm` 变成一个更大 scope，包住 w1 GEMM + activation + w2 GEMM，并明确排除/包含 shuffling 与 moe_sum 的策略），否则会导致 ground truth 组成不一致。



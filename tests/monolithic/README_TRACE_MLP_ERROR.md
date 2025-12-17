# MLP 算子执行时间误差分析脚本

## 概述

`trace_mlp_error.py` 用于自动从 PyTorch profiler 的 Chrome trace 文件中拆分 **PREFILL** 和 **DECODE** 阶段，并与 simulator/predictor 的预测值进行对比，生成误差分析表。

**主要用途**：
- 验证 MLP 相关算子的 execution time predictor 的精度
- 按 phase 隔离地分析误差（而不是混合的平均值）
- 支持批量校准：可接受自定义预测值 JSON，快速迭代优化

## 快速开始

### 基础用法（使用默认预测值）

```bash
python tests/monolithic/trace_mlp_error.py \
  --trace tests/monolithic/profiles/proj186_1427354.1765880405044290957.pt.trace.json.gz
```

**输出示例**：
```
op                           phase       pred_ms    actual_ms   abs_err_ms  rel_err_pct
post_attention_layernorm     PREFILL    0.037268     0.037828     0.000560       -1.48%
mlp_up_proj                  PREFILL    0.168175     1.262392     1.094217      -86.68%
mlp_act                      PREFILL    0.053029     0.202950     0.149921      -73.87%
mlp_down_proj                PREFILL    0.114113     0.656241     0.542128      -82.61%
post_attention_layernorm     DECODE     0.024690     0.003944     0.020746      526.02%
mlp_up_proj                  DECODE     0.055024     0.054404     0.000620        1.14%
mlp_act                      DECODE     0.010961     0.007064     0.003897       55.17%
mlp_down_proj                DECODE     0.030401     0.030858     0.000457       -1.48%
```

### 使用自定义预测值

1. 创建预测 JSON 文件 (`predictions.json`)：
```json
[
  {"op": "post_attention_layernorm", "phase": "PREFILL", "pred_ms": 0.038},
  {"op": "mlp_up_proj", "phase": "PREFILL", "pred_ms": 1.26},
  {"op": "mlp_act", "phase": "PREFILL", "pred_ms": 0.203},
  {"op": "mlp_down_proj", "phase": "PREFILL", "pred_ms": 0.656},
  {"op": "post_attention_layernorm", "phase": "DECODE", "pred_ms": 0.004},
  {"op": "mlp_up_proj", "phase": "DECODE", "pred_ms": 0.054},
  {"op": "mlp_act", "phase": "DECODE", "pred_ms": 0.007},
  {"op": "mlp_down_proj", "phase": "DECODE", "pred_ms": 0.031}
]
```

2. 运行脚本：
```bash
python tests/monolithic/trace_mlp_error.py \
  --trace tests/monolithic/profiles/proj186_1427354.1765880405044290957.pt.trace.json.gz \
  --pred-json predictions.json
```

### 导出 CSV

```bash
python tests/monolithic/trace_mlp_error.py \
  --trace tests/monolithic/profiles/proj186_1427354.1765880405044290957.pt.trace.json.gz \
  --csv-out /tmp/mlp_errors.csv
```

## 输出格式

| 列 | 含义 |
|---|---|
| **op** | 算子名称（`post_attention_layernorm`, `mlp_up_proj`, `mlp_act`, `mlp_down_proj`） |
| **phase** | 执行阶段（`PREFILL` 或 `DECODE`） |
| **pred_ms** | 预测执行时间（毫秒） |
| **actual_ms** | 实测执行时间，取该 phase 内所有调用的平均值（毫秒） |
| **abs_err_ms** | 绝对误差 = `\|pred - actual\|` |
| **rel_err_pct** | 相对误差百分比 = `(pred - actual) / actual × 100%`；负数=underpredict，正数=overpredict |

## 关键特性

### 自动 Phase 识别
脚本通过检测 `attn_prefill` 和 `attn_decode` markers 来自动分类 Forward 步骤：
- **PREFILL**：包含 `attn_prefill` 事件的 Forward（通常高 token 数，M~4096）
- **DECODE**：包含 `attn_decode` 事件的 Forward（通常低 token 数，M~8）

### 统计汇总
对每个 op 在该 phase 内的所有调用计算：
- **平均值** (mean)：用于输出表
- **最小/最大** (min/max)：衡量稳定性
- **百分位数** (p50/p90)：评估分布

### 多 trace 支持
支持处理：
- `.pt.trace.json.gz`（gzip 压缩，通常由 vLLM profiler 生成）
- `.pt.trace.json`（未压缩 JSON）

## 批量校准工作流示例

```bash
#!/bin/bash
# 对所有 profiler trace 运行错误分析

for trace_file in tests/monolithic/profiles/*.pt.trace.json.gz; do
  echo "Processing $trace_file"
  python tests/monolithic/trace_mlp_error.py \
    --trace "$trace_file" \
    --csv-out "/tmp/$(basename "$trace_file" .pt.trace.json.gz)_errors.csv"
done

# 合并所有 CSV（便于后续统计）
cat /tmp/*_errors.csv | sort | uniq > /tmp/all_errors.csv
```

## 语义一致性提示

- **per-layer 平均**：输出表中的 `actual_ms` 是该 phase 内所有 layer 调用的均值。这与 simulator 中 **`layer_id=0` 的预测** 对齐（假设同一 model 内不同 layer 的执行时间接近）。

- **Fused op**：
  - `mlp_up_proj`：实际对应 **fused gate+up projection**，输出维度=`2×intermediate_size`
  - `mlp_act`：**fused SiLU+mul**（通常一个 custom CUDA kernel）
  - 确保 predictor 使用的是正确的 op 维度定义

- **GPU 测量**：表中的时间来自 `gpu_user_annotation` 事件，代表 PyTorch profiler 归因到该 scope 的总 GPU 时间（包括 kernel + 可能的同步开销）。

## 故障排除

### 找不到 trace 文件
```
FileNotFoundError: [Errno 2] No such file or directory: '...'
```
**解决方案**：确保 trace 路径正确，通常在 `tests/monolithic/profiles/` 下，文件名包含 PID 和时间戳。

### 没有 Forward 事件
```
SystemExit: No Forward gpu_user_annotation events found in trace.
```
**可能原因**：
- Profiler 没有成功启动
- 检查 `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1` 是否被设置
- 检查 trace 文件是否被正确保存

### 相对误差为 NaN 或 inf
- 通常表示 `actual_ms ≈ 0`（op 没有被执行或执行时间极短）
- 检查 predictor 格式是否包含所有 4 个 op × 2 个 phase = 8 个条目

## 进阶：集成到 calibration 流程

1. **生成多个预测变体**（例如不同的 model 参数）
2. **对每个变体运行此脚本**，导出 CSV
3. **聚合 CSV 并计算平均误差** → 选择最优变体
4. **更新 predictor 权重/参数**，迭代优化

示例 Python 脚本（`calibrate_predictor.py`）：
```python
import subprocess
import pandas as pd

predictions = [
    {"variant": "v1", "json": "pred_v1.json"},
    {"variant": "v2", "json": "pred_v2.json"},
]

for pred in predictions:
    result = subprocess.run(
        ["python", "tests/monolithic/trace_mlp_error.py",
         "--trace", "tests/monolithic/profiles/...",
         "--pred-json", pred["json"],
         "--csv-out", f"/tmp/{pred['variant']}_errors.csv"],
        capture_output=True, text=True
    )
    df = pd.read_csv(f"/tmp/{pred['variant']}_errors.csv")
    mean_abs_err = df["abs_err_ms"].mean()
    print(f"{pred['variant']}: mean_abs_err = {mean_abs_err:.6f}")
```

## 附录：默认预测值来源

默认预测值对应于 simulator 在以下配置下的输出：
- **Model**：unsloth/Llama-3.2-1B-Instruct
- **batch_size**：8
- **dtype**：bfloat16
- **GPU**：A100/H100（假设）
- **PREFILL**：512 tokens/request, 4096 total tokens
- **DECODE**：1 token/request, 8 total tokens

见 `sklearn_execution_time_predictor.py` log：
```
[OP-TRACE][PREFILL][MLP] batch_id=0, layer_id=0, num_tokens=4096, ...
[OP-TRACE][DECODE][MLP] batch_id=9, layer_id=0, num_tokens=8, ...
```

---

**最后修改**：2025-12-16  
**脚本位置**：`tests/monolithic/trace_mlp_error.py`




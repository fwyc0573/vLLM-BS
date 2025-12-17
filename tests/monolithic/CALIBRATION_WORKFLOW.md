# MLP Predictor 校准工作流

## 概述

此文档描述如何使用新增的分析工具对 MLP 算子的 execution time predictor 进行校准。工作流分为三个阶段：

1. **数据采集**：运行 profiler 生成 trace
2. **误差分析**：拆分 PREFILL/DECODE 并计算预测误差
3. **迭代优化**：根据误差调整 predictor 参数

## 完整工作流

### 阶段 1: 数据采集

#### 1.1 运行 profiler

```bash
cd tests/monolithic
./offline_monolithic_profiling.sh --profile --model unsloth/Llama-3.2-1B-Instruct --gpu 7
```

关键参数：
- `--profile`：启用 PyTorch profiler
- `--gpu 7`：指定 GPU
- `--num-requests 8`：batch size
- `--prefill-tokens 512`：prefill 长度
- `--decode-tokens 2`：decode 长度

**输出**：生成 `profiles/*.pt.trace.json.gz` trace 文件

#### 1.2 验证 trace 质量

```bash
# 检查 trace 文件是否有效
python trace_mlp_error.py --trace profiles/proj186_1427354.1765880405044290957.pt.trace.json.gz
```

预期输出：显示 PREFILL 和 DECODE 阶段的错误表（默认预测值）

### 阶段 2: 误差分析

#### 2.1 单个 trace 分析

```bash
# 基础分析（使用默认预测值）
python trace_mlp_error.py \
  --trace profiles/proj186_1427354.1765880405044290957.pt.trace.json.gz

# 导出为 CSV
python trace_mlp_error.py \
  --trace profiles/proj186_1427354.1765880405044290957.pt.trace.json.gz \
  --csv-out /tmp/single_trace_errors.csv
```

#### 2.2 批量分析

```bash
# 处理所有 trace，聚合统计
python batch_calibrate_mlp.py \
  --trace-dir profiles \
  --output-csv /tmp/aggregated_errors.csv
```

**输出示例**：
```
======================================================================
AGGREGATED ERROR STATISTICS
======================================================================

Mean absolute error: 0.115042 ms
Mean relative error: inf%  ← (某些 op 零时间，需调查)
Max abs error: 1.094217 ms
Min abs error: 0.000457 ms

TOP 5 WORST CASES (by relative error)
- mlp_up_proj PREFILL: -86.68% (severely underpredict)
- mlp_down_proj PREFILL: -82.61% (severely underpredict)
...
```

### 阶段 3: 迭代优化

#### 3.1 根据误差找根因

从上面的结果看，**PREFILL 阶段的 MLP 操作严重 underpredict** (~-82% to -87%)。可能的原因：

| 现象 | 可能原因 | 检查方法 |
|------|--------|--------|
| PREFILL MLP 时间远大于预测 | Fused 算子维度不对 | `llama.py` 中 `gate_up_proj` 的输出维度 |
| DECODE 相对误差小 | 小 M 区间拟合较好 | 检查 predictor 训练数据范围 |
| DECODE layernorm overpredict | 小 M 下 overhead 占比高 | 用 Nsight 检查 kernel 启动开销 |

#### 3.2 修改 predictor

假设问题是 MLP 操作的 output 维度计算错误。修改步骤：

1. **检查 vLLM 实际维度**：
```python
# vllm/model_executor/models/llama.py
# 行 75-81
self.gate_up_proj = MergedColumnParallelLinear(
    input_size=hidden_size,  # 2048
    output_sizes=[intermediate_size] * 2,  # 2x8192 = 16384
    ...
)
```

2. **更新 predictor 的输入特征**：
   - 如果 predictor 使用 `output_size=intermediate_size`，改为 `output_size=2*intermediate_size`
   - 重新训练或用线性拟合调整系数

3. **生成新预测值**：
```json
[
  {"op": "mlp_up_proj", "phase": "PREFILL", "pred_ms": 1.26},
  ...
]
```

#### 3.3 验证修正

```bash
# 用新预测值重新分析
python batch_calibrate_mlp.py \
  --trace-dir profiles \
  --pred-json updated_predictions.json \
  --output-csv /tmp/after_fix_errors.csv

# 对比修正前后
echo "=== Before ===" && tail -3 /tmp/aggregated_errors.csv | grep mlp_up_proj
echo "=== After ===" && tail -3 /tmp/after_fix_errors.csv | grep mlp_up_proj
```

## 关键数据点

### 当前误差汇总（4 个 trace）

| Op | Phase | Avg Pred | Avg Actual | Avg Rel Err |
|---|---|---:|---:|---:|
| mlp_up_proj | PREFILL | 0.168 | 0.482 | -84% |
| mlp_up_proj | DECODE | 0.055 | 0.028 | +97% |
| mlp_down_proj | PREFILL | 0.114 | 0.264 | -57% |
| mlp_down_proj | DECODE | 0.030 | 0.016 | +90% |
| mlp_act | PREFILL | 0.053 | 0.077 | -31% |
| mlp_act | DECODE | 0.011 | 0.004 | +175% |

**模式识别**：
- **PREFILL**：主要 underpredict（除 layernorm 较准）
- **DECODE**：相对误差方向不一（mix of under/over）
- **layernorm**：PREFILL 有时 overpredict，DECODE 严重 overpredict

### 推荐优化方向

1. **M 维度缩放**：
   - PREFILL 高 M（~4096）下需要平方项或指数项
   - DECODE 低 M（~8）下主要是 launch overhead

2. **分段模型**（推荐）：
   ```
   time_prefill(M) = A + B*M + C*M^2
   time_decode(M) = launch_overhead + B_small*M
   ```

3. **逐 op 校准**：
   - `mlp_up_proj` 单独处理（输出维度最大）
   - `mlp_act` 单独处理（kernel 特性不同）

## 工具速查表

| 任务 | 命令 |
|------|------|
| 单个 trace 分析 | `python trace_mlp_error.py --trace <file>` |
| 导出单个 CSV | `python trace_mlp_error.py --trace <file> --csv-out <csv>` |
| 批量分析 | `python batch_calibrate_mlp.py --trace-dir profiles` |
| 自定义预测 | `python batch_calibrate_mlp.py --trace-dir profiles --pred-json preds.json` |
| 合并多个 CSV | `python -c "import pandas as pd; df = pd.concat([pd.read_csv(f) for f in ['a.csv', 'b.csv']]); df.to_csv('merged.csv')"` |

## 输出文件说明

### 从 `trace_mlp_error.py` 的 CSV

```csv
op,phase,pred_ms,actual_ms,abs_err_ms,rel_err_pct,trace_file
mlp_up_proj,PREFILL,0.168175,1.262392,1.094217,-86.678068,proj186_1427354...
...
```

**用途**：细粒度调试，找出哪个 trace/op/phase 的问题最严重

### 从 `batch_calibrate_mlp.py` 的聚合输出

```
Mean absolute error: 0.115042 ms
Median relative error: inf%
Max abs error: 1.094217 ms
```

**用途**：评估全局 predictor 质量，决定是否需要大规模修改

## 最佳实践

### ✅ 推荐做法

1. **定期采集数据**：每次更新 predictor 前后都运行 profiler
2. **分阶段验证**：先验证单个 op，再看全局
3. **保存历史**：`/tmp/calibration_v1.csv`, `v2.csv`, etc.
4. **根本分析**：不仅看误差数字，还要理解 Why（通过 Nsight/代码审查）
5. **增量优化**：一次修一个 op 或一个 phase，避免变量过多

### ❌ 常见陷阱

1. **忽视 phase 差异**：PREFILL 和 DECODE 的特性差异很大
2. **单个 trace 过拟合**：用 batch 平均值而不是单个最优值
3. **混淆 layer 平均**：输出表的 `actual_ms` 是该 phase 内所有 layer 的平均
4. **没有控制变量**：修改 predictor 时不同时修改 vLLM 代码（或反之）

## 后续步骤

1. **收集更多 traces**：
   - 不同 batch_size（4, 8, 16）
   - 不同 seq_len（256, 512, 1024）
   - 不同 GPU（H100, A100, L40S）

2. **建立 baseline**：
   - 运行 10 次 profiler，求平均/stddev
   - 这是 predictor 评估的"ground truth"

3. **自动化 CI**：
   - 每次提交时运行 `batch_calibrate_mlp.py`
   - 若 mean_rel_err > threshold，拒绝合并

4. **发表结果**：
   - 写一份"Frontier Simulator 精度评估"的报告
   - 包含上述汇总表、TOP-5 误差、优化建议

---

**最后更新**：2025-12-16  
**相关脚本**：
- `trace_mlp_error.py`
- `batch_calibrate_mlp.py`
- `README_TRACE_MLP_ERROR.md`




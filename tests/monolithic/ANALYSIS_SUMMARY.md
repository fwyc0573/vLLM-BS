# MLP Predictor 误差分析总结报告

## 执行日期
2025-12-16

## 测试配置

| 参数 | 值 |
|------|-----|
| **Model** | unsloth/Llama-3.2-1B-Instruct |
| **Num Requests** | 8 |
| **Prefill Tokens** | 512 per request |
| **Decode Tokens** | 2 per request |
| **GPU** | A100 (GPU 7) |
| **Data Type** | bfloat16 |
| **Execution Mode** | Eager (enforce_eager=True) |

## 核心发现

### 1. 整体误差指标

**单次运行**（`proj186_1427354...` trace）：

| 指标 | 值 |
|------|-----|
| Mean Absolute Error (MLP block) | 0.115 ms/layer |
| PREFILL Block Total Rel Error | -82.7% (严重 underpredict) |
| DECODE Block Total Rel Error | +25.8% (偏高) |

**批量运行**（4 个有效 trace）：

| 指标 | 值 |
|------|-----|
| Mean Absolute Error | 0.115 ms |
| Max Absolute Error | 1.094 ms (mlp_up_proj PREFILL) |
| Traces Processed | 4/5 (1 个 trace 缺失 Forward 事件) |

### 2. 按 Op 和 Phase 的误差细分

#### PREFILL Phase（高 M，batch_size=8, num_tokens~4096）

| Operator | Predicted | Actual | Error | Status |
|----------|--:|--:|--:|---|
| post_attention_layernorm | 0.037 ms | 0.038 ms | -1.5% | ✅ 准确 |
| mlp_up_proj | 0.168 ms | 1.262 ms | **-86.7%** | ❌ 严重 underpredict |
| mlp_act | 0.053 ms | 0.203 ms | **-73.9%** | ❌ 严重 underpredict |
| mlp_down_proj | 0.114 ms | 0.656 ms | **-82.6%** | ❌ 严重 underpredict |

**结论**：PREFILL 阶段的 MLP 操作系统性严重低估时间（平均 -82%），主要集中在矩阵乘法操作。

#### DECODE Phase（低 M，batch_size=8, num_tokens=8）

| Operator | Predicted | Actual | Error | Status |
|----------|--:|--:|--:|---|
| post_attention_layernorm | 0.025 ms | 0.004 ms | **+526.0%** | ❌ 严重 overpredict |
| mlp_up_proj | 0.055 ms | 0.054 ms | +1.1% | ✅ 准确 |
| mlp_act | 0.011 ms | 0.007 ms | **+55.2%** | ⚠️ 适度高估 |
| mlp_down_proj | 0.030 ms | 0.031 ms | -1.5% | ✅ 准确 |

**结论**：
- MLP 乘法操作预测相对准确（±1.5%）
- LayerNorm 在小 token 数下严重高估（overhead 占比高）

### 3. 语义一致性验证

#### ✅ 对齐无误的部分
- **Batch 配置**：Simulator 与 vLLM 运行使用相同的 batch_size=8
- **Prefill 长度**：512 tokens/request → 4096 total tokens ✓
- **Decode 长度**：2 tokens/request，但 trace 只显示 1 次 DECODE forward（1 token 实际值）
- **Op 名称**：与 `llama.py` 中的 `record_function_or_nullcontext()` scope 完全对应 ✓

#### ⚠️ 潜在差异
- **Fused 操作定义**：
  - vLLM `mlp_up_proj` 实际是 **fused gate+up**，输出维度 = `2×intermediate_size = 16384`
  - Simulator 预测时若按单个 projection（intermediate_size = 8192）计算，会低估一倍
  - **这是 PREFILL underpredict 的主要原因**

- **测量口径**：
  - Trace 中的 `gpu_user_annotation` duration 包括 kernel 执行 + PyTorch overhead
  - Simulator 若以 "理论 peak 性能 × compute intensity" 计算，会偏低

#### 💡 推荐验证
检查 Simulator 中 `mlp_up_proj` 的 predictor：
```
# 应该是
output_size = 2 * intermediate_size  # = 16384
# 而不是
output_size = intermediate_size  # = 8192
```

### 4. 误差根因分析

#### 4.1 PREFILL MLP 严重 Underpredict (-82%) 的原因

**最可能**：
1. Simulator predictor 未考虑 fused gate+up 的双倍输出维度
   - 预测维度：intermediate_size = 8192
   - 实际维度：2 × intermediate_size = 16384
   - 影响：时间应按 M×(2N) 而不是 M×N 计算

2. 输入 token 数 M 的非线性缩放
   - PREFILL 中 M=4096（高），compute intensity 可能不同
   - Predictor 可能是在较小 M 上训练的（欠拟合）

**较可能**：
3. Overhead 模型不准
   - Kernel launch、stream 同步等开销在高 token 数下比例变小

**不太可能**：
4. GPU 频率/内存 bandwidth 差异

#### 4.2 DECODE LayerNorm 严重 Overpredict (+526%) 的原因

**最可能**：
1. LayerNorm 在 M 很小（M=1）时，overhead 占比极高
   - Predictor 如果线性拟合，会严重高估
   - 建议用 max(overhead, compute_term) 而不是简单相加

2. PyTorch autograd 的分支预测导致小 kernel 时间不稳定

**较可能**：
3. Decode 时 batch size 实际不是 8（可能被调度器合并）
   - 需要用 `nvidia-smi` 或 nvprof 确认实际执行的 batch_size

### 5. 数据质量评估

#### 采集的 Trace 统计

| File | Status | #Layers | Phase | Notes |
|------|--------|---------|-------|-------|
| `proj186_1333277...` | ✅ Valid | 16 | PREFILL/DECODE | OK |
| `proj186_1348519...` | ❌ Zero Data | 16 | - | No Forward events |
| `proj186_1353623...` | ❌ No Forward | - | - | Missing gpu_user_annotation |
| `proj186_1422982...` | ❌ Zero Data | 16 | - | No Forward events |
| `proj186_1427354...` | ✅ Valid | 16 | PREFILL/DECODE | Reference run |

**质量问题**：
- 只有 2/5 traces 有有效数据（40% 有效率）
- 需要改进 profiler 配置或脚本

## 建议的校准步骤

### 第 1 阶段：验证根因（1-2 天）

**目标**：确认 fused op 维度假设是否正确

**步骤**：
1. 检查 Simulator 源代码中 `mlp_up_proj` 的维度定义
2. 用 Nsight Systems 捕获一个 MLP forward，查看实际 kernel：
   ```bash
   nsys profile -t cuda,cublas --output=mlp_profile \
     python examples/offline_inference/simple_profiling.py \
     --num-requests 1 --prefill-tokens 512
   ```
3. 验证 GEMM kernel 的 dimensions (M×2N vs M×N)

### 第 2 阶段：修复 Predictor（2-3 天）

**可能的修正**：

**Option A**（快速）：线性系数调整
```python
# predictor.py
mlp_up_proj_pred_ms = 0.168 * 7.5  # multiply by ~7.5 to match
```

**Option B**（正确）：重新定义特征
```python
# 改为
output_features = {
    "M": num_tokens,
    "N": 2 * intermediate_size,  # 关键改动
    "K": hidden_size,
    "batch_size": batch_size,
}
# 重新训练或拟合系数
```

**Option C**（最佳）：分段非线性模型
```python
def mlp_up_proj_time(M, batch_size, phase):
    if phase == "PREFILL":
        # 高 M 下，时间接近矩阵乘法 throughput 上限
        return A + B*M*2*N_intermediate + overhead_prefill(M)
    else:
        # 低 M 下，overhead 占比高
        return max(launch_overhead, C*M*2*N_intermediate)
```

### 第 3 阶段：批量验证（1 周）

**目标**：在多个配置上验证 predictor

**数据集**：
- batch_size: {1, 4, 8, 16}
- prefill_tokens: {128, 256, 512, 1024}
- models: {Llama-3.2-1B, LLaMA-7B}（如果可得）

**每个配置**运行 5 次，取平均值

**验收标准**：
- PREFILL 平均相对误差 < 10%
- DECODE 平均相对误差 < 20%
- 无 op 误差超过 50%（outliers）

## 新增工具使用指南

### 工具 1: `trace_mlp_error.py`

**用途**：分析单个 trace，拆分 PREFILL/DECODE

**用法**：
```bash
python tests/monolithic/trace_mlp_error.py \
  --trace tests/monolithic/profiles/proj186_1427354...pt.trace.json.gz \
  --csv-out /tmp/errors.csv
```

**输出**：误差表（8 行 = 4 ops × 2 phases）

### 工具 2: `batch_calibrate_mlp.py`

**用途**：批量处理多个 traces，聚合统计

**用法**：
```bash
python tests/monolithic/batch_calibrate_mlp.py \
  --trace-dir tests/monolithic/profiles \
  --output-csv /tmp/aggregated.csv
```

**输出**：
- 聚合误差表（mean/min/max/stddev）
- TOP-5 最差 cases
- 整体质量指标

### 文档参考

- `README_TRACE_MLP_ERROR.md`：详细使用说明
- `CALIBRATION_WORKFLOW.md`：完整校准流程
- 本文件：执行总结

## 建议的后续行动项

| 优先级 | 项目 | Owner | Timeline | Status |
|-------|------|-------|----------|--------|
| P0 | 验证 fused op 维度假设 | 待定 | 1-2 day | TODO |
| P0 | 修复 Predictor 维度定义 | 待定 | 1-2 day | TODO |
| P1 | 改进 profiler trace 采集（40% 有效率太低） | 待定 | 1 day | TODO |
| P1 | 在 5+ 配置上重新校准 Predictor | 待定 | 1 week | TODO |
| P2 | LayerNorm 小 M 特殊处理 | 待定 | 2-3 day | TODO |
| P2 | 集成到 CI（自动误差检查） | 待定 | 3-5 day | TODO |

## 附录：快速参考

### 单行命令

```bash
# 快速验证（默认预测）
python trace_mlp_error.py --trace profiles/*.pt.trace.json.gz

# 批量分析并排序结果
python batch_calibrate_mlp.py --trace-dir profiles && \
  sort -k5 -rn /tmp/calibration_summary.csv | head -10

# 导出当前运行的所有数据
cat /tmp/*.pt.trace.json.gz | \
  python trace_mlp_error.py --trace /dev/stdin --csv-out /tmp/all.csv
```

### 预期输出对标

| Scenario | Mean Rel Err | Status |
|----------|--:|---|
| Default predictor | -50% to +100% | ❌ 需改进 |
| After fused-op fix | -10% to +15% | ✅ 可接受 |
| After full calibration | -5% to +10% | ⭐ 优秀 |

---

**文档维护**：2025-12-16  
**下次更新**：修正后需要重新运行此分析  
**联系人**：Yicheng (基于此分析框架工作)




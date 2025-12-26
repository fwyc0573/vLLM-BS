# vLLM模型加载和配置指南

本文档记录了在使用vLLM进行模型推理时的关键配置选项，特别是针对避免权重下载、使用随机权重以及自定义模型配置的方法。

## 1. 模型权重缓存路径

### 默认缓存路径
当执行脚本时，如果需要下载模型权重，缓存会存储在：
```
~/.cache/huggingface/hub/
```

### 缓存机制原理
- vLLM使用HuggingFace的`snapshot_download`函数下载模型
- 缓存路径由HuggingFace的缓存机制决定
- 相关代码位于：`vllm/model_executor/model_loader/weight_utils.py`

### 自定义缓存路径
可以通过以下环境变量自定义缓存路径：

```bash
# 方法1：设置HuggingFace主目录
export HF_HOME=/your/custom/cache/path

# 方法2：直接设置缓存目录
export TRANSFORMERS_CACHE=/your/custom/cache/path
```

### 验证缓存路径
```bash
# 查看当前缓存配置
python -c "import huggingface_hub; print(huggingface_hub.constants.HF_HUB_CACHE)"
```

## 2. 使用随机权重避免下载

### 背景
vLLM提供了`DummyModelLoader`来使用随机权重，这对于：
- 性能测试和profiling
- 架构验证
- 避免大模型下载时间
- 专注于forward耗时测量

### 实现方法

#### 方法1：通过命令行参数
在调用`simple_profiling.py`时添加`--load-format dummy`：

```bash
python examples/offline_inference/simple_profiling.py \
    --load-format dummy \
    --model microsoft/Phi-tiny-MoE-instruct \
    --num-requests 1 \
    --prefill-tokens 512 \
    --decode-tokens 32
```

#### 方法2：修改脚本文件
在`tests/monolithic/offline_monolithic_profiling.sh`中修改python调用：

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" \
python "$EXAMPLE_DIR/simple_profiling.py" \
    --load-format dummy \
    --num-requests "$NUM_REQUESTS" \
    --prefill-tokens "$PREFILL_TOKENS" \
    --decode-tokens "$DECODE_TOKENS" \
    --seed "$SEED" \
    --warmup-iters "$WARMUP_ITERS" \
    --model "$MODEL" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    $PROFILE_ARG \
    $EXTRA_ARGS
```

### 技术细节
- 使用`DummyModelLoader`类（位于`vllm/model_executor/model_loader/dummy_loader.py`）
- 调用`initialize_dummy_weights`函数生成随机权重
- 权重值范围：-1e-3 到 1e-3
- 使用固定种子确保一致性

## 3. 修改config.json参数

### 使用hf_overrides参数
vLLM支持通过`--hf-overrides`参数在运行时修改模型配置，无需修改原始config.json文件。

#### 示例：修改num_local_experts为2

**方法1：命令行直接指定**
```bash
tests/monolithic/offline_monolithic_profiling.sh \
    --model microsoft/Phi-tiny-MoE-instruct \
    --gpu 5 \
    --hf-overrides '{"num_local_experts": 2}'
```

**方法2：修改脚本文件**
```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" \
python "$EXAMPLE_DIR/simple_profiling.py" \
    --hf-overrides '{"num_local_experts": 2}' \
    --model "$MODEL" \
    # ... 其他参数
```

**方法3：通过环境变量**
```bash
export HF_OVERRIDES='{"num_local_experts": 2}'
# 然后运行脚本
```

### 支持的配置参数
常见的可修改参数包括：
- `num_local_experts`：MoE模型的专家数量
- `num_experts_per_tok`：每个token路由的专家数
- `max_position_embeddings`：最大位置编码长度
- `hidden_size`：隐藏层维度
- 其他HuggingFace配置参数

### 技术实现
- 配置覆盖在`vllm/transformers_utils/config.py`中的`get_config`函数实现
- 支持字典形式和函数形式的覆盖
- 在模型加载前应用配置修改

## 4. 完整解决方案

### 综合使用示例
结合避免下载权重和自定义配置的完整命令：

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" \
python "$EXAMPLE_DIR/simple_profiling.py" \
    --load-format dummy \
    --hf-overrides '{"num_local_experts": 2}' \
    --num-requests "$NUM_REQUESTS" \
    --prefill-tokens "$PREFILL_TOKENS" \
    --decode-tokens "$DECODE_TOKENS" \
    --seed "$SEED" \
    --warmup-iters "$WARMUP_ITERS" \
    --model "$MODEL" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    $PROFILE_ARG \
    $EXTRA_ARGS
```

### 修改offline_monolithic_profiling.sh脚本
在脚本中添加新的参数支持：

```bash
# 在CLI parsing部分添加
--load-format) LOAD_FORMAT="$2"; shift 2 ;;
--hf-overrides) HF_OVERRIDES="$2"; shift 2 ;;

# 在运行部分添加参数
LOAD_FORMAT_ARG=""
if [[ -n "$LOAD_FORMAT" ]]; then
    LOAD_FORMAT_ARG="--load-format $LOAD_FORMAT"
fi

HF_OVERRIDES_ARG=""
if [[ -n "$HF_OVERRIDES" ]]; then
    HF_OVERRIDES_ARG="--hf-overrides '$HF_OVERRIDES'"
fi

# 在python调用中添加
python "$EXAMPLE_DIR/simple_profiling.py" \
    $LOAD_FORMAT_ARG \
    $HF_OVERRIDES_ARG \
    # ... 其他参数
```

## 5. 使用场景和注意事项

### 适用场景
- **性能基准测试**：专注于计算性能而非模型精度
- **架构验证**：验证模型结构和内存使用
- **开发调试**：快速迭代和测试
- **CI/CD流水线**：避免大文件下载

### 注意事项
1. **随机权重**：输出结果无意义，仅用于性能测试
2. **配置兼容性**：确保修改的配置参数与模型架构兼容
3. **内存使用**：配置修改可能影响内存需求
4. **缓存清理**：定期清理HuggingFace缓存以节省磁盘空间

### 相关文件路径
- 模型加载器：`vllm/model_executor/model_loader/`
- 配置处理：`vllm/transformers_utils/config.py`
- 权重工具：`vllm/model_executor/model_loader/weight_utils.py`
- 示例脚本：`examples/offline_inference/simple_profiling.py`

## 6. 故障排除

### 常见问题
1. **权重下载失败**：检查网络连接和HuggingFace访问权限
2. **配置参数无效**：验证参数名称和值的正确性
3. **内存不足**：调整`--gpu-memory-utilization`参数
4. **模型不兼容**：确认模型支持指定的配置修改

### 调试命令
```bash
# 检查模型配置
python -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('model_name'))"

# 验证vLLM加载
python -c "from vllm import LLM; llm = LLM('model_name', load_format='dummy')"
```

---

*文档创建时间：2024年12月*  
*适用vLLM版本：v0.10.2+*
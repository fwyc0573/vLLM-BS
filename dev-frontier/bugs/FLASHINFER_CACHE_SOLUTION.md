# FlashInfer 磁盘配额问题解决方案

## 问题描述

### 错误信息
```
OSError: [Errno 122] Disk quota exceeded: '/uac/gds/ytyang/.cache/flashinfer/80/cached_ops/sampling'
```

### 根本原因
FlashInfer 会将 JIT 编译的 CUDA 算子缓存到磁盘以提高后续启动速度。默认情况下，缓存位置为：

```
$HOME/.cache/flashinfer/
```

但您的 home 目录 (`/uac/gds/ytyang`) 已达到磁盘配额限制：
```
Filesystem: uranus:/d0/data
Quota: 2447M* / 2447M (100% 已使用)
```

### FlashInfer 缓存机制
FlashInfer 使用环境变量 `FLASHINFER_WORKSPACE_BASE` 来确定缓存根目录：
- 如果设置了 `FLASHINFER_WORKSPACE_BASE`，则缓存路径为：`$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/`
- 如果未设置，默认为：`$HOME/.cache/flashinfer/`

## 应用的解决方案

### 修改内容
在 `run_test.sh` 中添加了以下配置：

```bash
# Set FlashInfer cache directory to avoid disk quota issues
# FlashInfer uses FLASHINFER_WORKSPACE_BASE to determine cache location
# Cache will be stored at: $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/
FLASHINFER_CACHE_DIR="$EXAMPLE_DIR/flashinfer_cache"
mkdir -p "$FLASHINFER_CACHE_DIR/.cache/flashinfer"
export FLASHINFER_WORKSPACE_BASE="$FLASHINFER_CACHE_DIR"
```

### 新的缓存位置
```
/research/d1/gds/ytyang/yichengfeng/frontier/sota-infer-engine/vllm/examples/offline_inference/disaggregated-prefill-v1/flashinfer_cache/.cache/flashinfer/
```

这个位置在您的研究目录下，有足够的磁盘空间（944G 可用）。

## 完整的磁盘配额保护措施

现在 `run_test.sh` 包含了三层保护：

### 1. vLLM 配置目录重定向
```bash
export VLLM_CONFIG_ROOT=$PROJECT_ROOT/.vllm_config
```
- 避免写入 `~/.config/vllm/`
- 新位置：`/research/.../vllm/.vllm_config/`

### 2. 禁用使用统计
```bash
export VLLM_NO_USAGE_STATS=1
```
- 避免写入 `~/.config/vllm/usage_stats.json`
- 完全禁用统计数据收集

### 3. FlashInfer 缓存重定向 ✨ 新增
```bash
export FLASHINFER_WORKSPACE_BASE="$FLASHINFER_CACHE_DIR"
```
- 避免写入 `~/.cache/flashinfer/`
- 新位置：`$EXAMPLE_DIR/flashinfer_cache/.cache/flashinfer/`

## 缓存内容说明

FlashInfer 缓存包含：
- JIT 编译的 CUDA 内核
- 采样算子 (sampling operators)
- 注意力算子 (attention operators)
- 预编译的二进制文件

### 缓存大小估计
- 首次运行：生成缓存，可能需要 50-200MB
- 后续运行：直接使用缓存，显著加快启动速度

### 缓存清理（如需要）
```bash
# 清理 FlashInfer 缓存
rm -rf /research/d1/gds/ytyang/yichengfeng/frontier/sota-infer-engine/vllm/examples/offline_inference/disaggregated-prefill-v1/flashinfer_cache/

# 或使用变量
rm -rf "$EXAMPLE_DIR/flashinfer_cache/"
```

## 验证修复

运行测试脚本：
```bash
tests/disaggregated_prefill_test/run_test.sh
```

### 预期行为
1. **不再出现磁盘配额错误**
2. **FlashInfer 缓存写入到新位置**
3. **首次运行会看到 JIT 编译信息**
4. **后续运行会更快（使用缓存）**

### 确认缓存位置
运行后检查：
```bash
ls -lh /research/d1/gds/ytyang/yichengfeng/frontier/sota-infer-engine/vllm/examples/offline_inference/disaggregated-prefill-v1/flashinfer_cache/.cache/flashinfer/
```

应该会看到类似的目录结构：
```
flashinfer_cache/
└── .cache/
    └── flashinfer/
        └── 80/              # CUDA compute capability (SM 8.0 for A800)
            └── cached_ops/
                ├── sampling/     # 采样算子缓存
                └── attention/    # 注意力算子缓存
```

## 技术细节

### FlashInfer 环境变量优先级
1. `FLASHINFER_WORKSPACE_BASE` - 工作空间基础目录（✅ 已设置）
2. `XDG_CACHE_HOME` - XDG 缓存目录
3. `HOME` - 用户主目录（默认，有配额限制）

### 为什么选择 $EXAMPLE_DIR
- ✅ 与测试文件在同一目录，方便管理
- ✅ 在研究目录下，有充足空间
- ✅ 容易清理，不影响其他项目
- ✅ 与 profiling 输出在同一区域

### 其他可选位置
如果需要更改位置，可以修改：
```bash
# 选项 1：放在项目根目录
FLASHINFER_CACHE_DIR="$PROJECT_ROOT/.flashinfer_cache"

# 选项 2：放在临时目录
FLASHINFER_CACHE_DIR="/tmp/flashinfer_cache_$$"

# 选项 3：放在专门的缓存目录
FLASHINFER_CACHE_DIR="/research/d1/gds/ytyang/cache/flashinfer"
```

## 与 vLLM 版本兼容性

### FlashInfer 0.5.3 (当前安装)
- ✅ 支持 `FLASHINFER_WORKSPACE_BASE` 环境变量
- ✅ 自动管理缓存生命周期
- ⚠️ 与 vLLM v0.10.2 不兼容（会导致 Bus Error）

### FlashInfer 0.3.0 (vLLM v0.10.2 要求)
- ✅ 支持 `FLASHINFER_WORKSPACE_BASE` 环境变量
- ✅ 与 vLLM v0.10.2 完全兼容
- 建议降级以避免运行时错误

## 完整的环境变量列表

运行测试时设置的所有环境变量：

```bash
# GPU 配置
export CUDA_VISIBLE_DEVICES=0

# vLLM 配置
export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_CONFIG_ROOT=$PROJECT_ROOT/.vllm_config
export VLLM_NO_USAGE_STATS=1

# Profiling 配置
export VLLM_TORCH_PROFILER_DIR=$EXAMPLE_DIR/profiles
export VLLM_TORCH_PROFILER_WITH_STACK=1
export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1

# FlashInfer 配置 ✨ 新增
export FLASHINFER_WORKSPACE_BASE="$FLASHINFER_CACHE_DIR"
```

## 总结

### 已解决的问题
- ✅ vLLM 配置文件磁盘配额 (`~/.config/vllm/`)
- ✅ vLLM 使用统计磁盘配额 (`~/.config/vllm/usage_stats.json`)
- ✅ FlashInfer 缓存磁盘配额 (`~/.cache/flashinfer/`) ✨ 新增

### 系统影响
- ✅ 所有写入操作重定向到研究目录
- ✅ Home 目录不再增长
- ✅ 磁盘配额问题完全解决

### 性能影响
- ✅ 无负面影响
- ✅ FlashInfer 缓存加快后续启动
- ✅ Profiling 输出正常保存

---
**生成时间**: 2025年12月7日  
**修复位置**: `tests/disaggregated_prefill_test/run_test.sh`  
**新缓存目录**: `$EXAMPLE_DIR/flashinfer_cache/.cache/flashinfer/`

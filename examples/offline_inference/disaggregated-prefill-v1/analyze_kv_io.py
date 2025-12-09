# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
测试脚本：分析 KV Transfer 的保存（Save）vs 加载（Load）开销
用于精确定位等待时间来源
"""

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
import os
import time
import shutil

# profiling codes
os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"
os.environ["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "1"


def read_prompts():
    context = "Hi " * 1000
    context2 = "Hey " * 500
    return [
        context + "Hello, my name is",
        context + "The capital of France is",
        context2 + "Your name is",
        context2 + "The capital of China is",
    ]


def clean_kv_cache():
    """清理 KV cache 存储目录"""
    storage_path = "local_storage"
    if os.path.exists(storage_path):
        shutil.rmtree(storage_path)
        print(f"🧹 已清理 KV cache 目录: {storage_path}")
    os.makedirs(storage_path, exist_ok=True)


def run_kv_save_test():
    """测试 KV Cache 保存开销（无缓存命中）"""
    print("=" * 60)
    print("测试 1: KV Cache 保存 (Save) - 无缓存命中")
    print("=" * 60)
    
    # 清理旧缓存，强制重新保存
    clean_kv_cache()
    
    prompts = read_prompts()
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    llm = LLM(
        model="meta-llama/Llama-3.2-1B-Instruct",
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        kv_transfer_config=KVTransferConfig(
            kv_connector="SharedStorageConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": "local_storage"},
        ),
    )

    # 不做 warmup，直接测试（会触发 KV save）
    llm.start_profile()
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start
    llm.stop_profile()
    
    # 检查生成的 KV cache 文件
    kv_files = []
    for root, dirs, files in os.walk("local_storage"):
        for f in files:
            if f.endswith(".safetensors"):
                kv_files.append(os.path.join(root, f))
    
    total_size = sum(os.path.getsize(f) for f in kv_files)
    
    print(f"\n✅ KV Save 总耗时: {elapsed:.3f}s")
    print(f"   KV cache 文件数: {len(kv_files)}")
    print(f"   KV cache 总大小: {total_size / 1024 / 1024:.2f} MB")
    print(f"   I/O 带宽: {total_size / elapsed / 1024 / 1024:.2f} MB/s")
    
    del llm
    return elapsed, len(kv_files), total_size


def run_kv_load_test():
    """测试 KV Cache 加载开销（有缓存命中）"""
    print("\n" + "=" * 60)
    print("测试 2: KV Cache 加载 (Load) - 有缓存命中")
    print("=" * 60)
    
    prompts = read_prompts()
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    llm = LLM(
        model="meta-llama/Llama-3.2-1B-Instruct",
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        kv_transfer_config=KVTransferConfig(
            kv_connector="SharedStorageConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": "local_storage"},
        ),
    )

    # Warmup 确保缓存已存在
    _ = llm.generate(prompts[:1], sampling_params)
    
    llm.start_profile()
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start
    llm.stop_profile()
    
    print(f"\n✅ KV Load 总耗时: {elapsed:.3f}s")
    
    del llm
    return elapsed


def run_baseline_test():
    """基准测试：无 KV Transfer"""
    print("\n" + "=" * 60)
    print("测试 3: 基准 (无 KV Transfer)")
    print("=" * 60)
    
    prompts = read_prompts()
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    llm = LLM(
        model="meta-llama/Llama-3.2-1B-Instruct",
        enforce_eager=True,
        gpu_memory_utilization=0.8,
    )

    # Warmup
    _ = llm.generate(prompts[:1], sampling_params)
    
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start
    
    print(f"\n✅ 基准总耗时: {elapsed:.3f}s")
    
    del llm
    return elapsed


def main():
    import torch
    
    print("\n" + "🔬" * 30)
    print("KV Transfer 性能分析: Save vs Load")
    print("🔬" * 30 + "\n")
    
    torch.cuda.empty_cache()
    
    # 测试 1: 基准（无 KV Transfer）
    time_baseline = run_baseline_test()
    
    torch.cuda.empty_cache()
    
    # 测试 2: KV Save（清理缓存后首次运行）
    time_save, num_files, total_size = run_kv_save_test()
    
    torch.cuda.empty_cache()
    
    # 测试 3: KV Load（有缓存命中）
    time_load = run_kv_load_test()
    
    # 对比分析
    print("\n" + "=" * 60)
    print("📊 对比分析")
    print("=" * 60)
    print(f"基准 (无 KV Transfer):     {time_baseline:.3f}s")
    print(f"KV Save (首次/无缓存):     {time_save:.3f}s  (+{time_save - time_baseline:.3f}s)")
    print(f"KV Load (有缓存命中):      {time_load:.3f}s  (+{time_load - time_baseline:.3f}s)")
    print(f"")
    print(f"📁 KV Cache 统计:")
    print(f"   文件数: {num_files}")
    print(f"   总大小: {total_size / 1024 / 1024:.2f} MB")
    print(f"")
    print(f"⏱️ 开销分解:")
    print(f"   Save 开销: {time_save - time_baseline:.3f}s ({(time_save - time_baseline) / time_baseline * 100:.1f}%)")
    print(f"   Load 开销: {time_load - time_baseline:.3f}s ({(time_load - time_baseline) / time_baseline * 100:.1f}%)")
    print(f"   Save vs Load: Save 比 Load 慢 {time_save / time_load:.1f}x")


if __name__ == "__main__":
    main()

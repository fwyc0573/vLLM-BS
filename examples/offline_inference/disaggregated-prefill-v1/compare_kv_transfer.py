# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
测试脚本：对比有/无 KV Transfer 的性能差异
用于验证等待时间来源
"""

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
import os
import time

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


def run_with_kv_transfer():
    """运行带 KV Transfer 的测试"""
    print("=" * 60)
    print("测试 1: 启用 KV Transfer (SharedStorageConnector)")
    print("=" * 60)
    
    prompts = read_prompts()
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    llm = LLM(
        model="meta-llama/Llama-3.2-1B-Instruct",
        enforce_eager=True,  # 使用 eager 模式便于对比
        gpu_memory_utilization=0.8,
        kv_transfer_config=KVTransferConfig(
            kv_connector="SharedStorageConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": "local_storage"},
        ),
    )

    # Warmup
    _ = llm.generate(prompts[:1], sampling_params)
    
    llm.start_profile()
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start
    llm.stop_profile()
    
    print(f"\n✅ 带 KV Transfer 总耗时: {elapsed:.3f}s")
    print(f"   输入 tokens: {sum(len(o.prompt_token_ids) for o in outputs)}")
    print(f"   吞吐量: {sum(len(o.prompt_token_ids) for o in outputs) / elapsed:.1f} toks/s")
    
    del llm
    return elapsed


def run_without_kv_transfer():
    """运行不带 KV Transfer 的测试"""
    print("\n" + "=" * 60)
    print("测试 2: 禁用 KV Transfer (无 SharedStorageConnector)")
    print("=" * 60)
    
    prompts = read_prompts()
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    llm = LLM(
        model="meta-llama/Llama-3.2-1B-Instruct",
        enforce_eager=True,  # 使用 eager 模式便于对比
        gpu_memory_utilization=0.8,
        # 不设置 kv_transfer_config
    )

    # Warmup
    _ = llm.generate(prompts[:1], sampling_params)
    
    llm.start_profile()
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start
    llm.stop_profile()
    
    print(f"\n✅ 无 KV Transfer 总耗时: {elapsed:.3f}s")
    print(f"   输入 tokens: {sum(len(o.prompt_token_ids) for o in outputs)}")
    print(f"   吞吐量: {sum(len(o.prompt_token_ids) for o in outputs) / elapsed:.1f} toks/s")
    
    del llm
    return elapsed


def main():
    import torch
    torch.cuda.empty_cache()
    
    # 先运行无 KV Transfer
    time_no_kv = run_without_kv_transfer()
    
    torch.cuda.empty_cache()
    
    # 再运行有 KV Transfer
    time_with_kv = run_with_kv_transfer()
    
    # 对比分析
    print("\n" + "=" * 60)
    print("对比分析")
    print("=" * 60)
    print(f"无 KV Transfer: {time_no_kv:.3f}s")
    print(f"有 KV Transfer: {time_with_kv:.3f}s")
    print(f"差异 (KV I/O 开销): {time_with_kv - time_no_kv:.3f}s")
    print(f"开销比例: {(time_with_kv - time_no_kv) / time_no_kv * 100:.1f}%")


if __name__ == "__main__":
    main()

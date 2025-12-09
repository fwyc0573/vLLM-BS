#!/usr/bin/env python3
"""Test script to verify vLLM import works correctly."""

import sys
import os

# Redirect output to file
output_file = "/research/d1/gds/ytyang/yichengfeng/frontier/sota-infer-engine/vllm/tests/import_test_output.txt"
sys.stdout = open(output_file, 'w')
sys.stderr = sys.stdout

def main():
    print("=" * 60)
    print("Testing vLLM Import")
    print("=" * 60)
    
    print(f"\nPython: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Test basic imports
    print("\n--- Testing basic imports ---")
    
    try:
        import torch
        print(f"[OK] torch: {torch.__version__}")
        print(f"     CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"     CUDA version: {torch.version.cuda}")
            print(f"     GPU count: {torch.cuda.device_count()}")
    except Exception as e:
        print(f"[FAIL] torch: {e}")
        return 1
    
    try:
        from vllm import LLM, SamplingParams
        print(f"[OK] vllm.LLM imported")
        print(f"[OK] vllm.SamplingParams imported")
    except Exception as e:
        print(f"[FAIL] vllm import: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    try:
        from vllm.config import KVTransferConfig
        print(f"[OK] vllm.config.KVTransferConfig imported")
    except Exception as e:
        print(f"[FAIL] KVTransferConfig import: {e}")
        return 1
    
    try:
        import vllm
        print(f"\nvLLM location: {vllm.__file__}")
        print(f"vLLM version: {vllm.__version__}")
    except Exception as e:
        print(f"[WARN] Could not get vllm info: {e}")
    
    print("\n" + "=" * 60)
    print("All imports successful!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())


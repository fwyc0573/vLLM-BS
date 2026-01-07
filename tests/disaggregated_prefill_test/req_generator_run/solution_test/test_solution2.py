#!/usr/bin/env python3
"""Test script for Solution 2: Output length guarantee."""

import sys
import os
# Dynamically resolve vLLM path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_VLLM_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR))))))
sys.path.insert(0, _VLLM_PATH)

from vllm.request_generator import (
    VLLMRequestGenerator,
    RequestGeneratorConfig,
    FixedLengthConfig,
)

def test_force_exact_output_length_true():
    """Test 1: force_exact_output_length=True (default) - should set ignore_eos."""
    print('=== Test 1: force_exact_output_length=True (Default) ===')
    config = RequestGeneratorConfig(
        num_requests=2,
        length_config=FixedLengthConfig(prefill_tokens=128, decode_tokens=64),
        force_exact_output_length=True,  # Explicit
    )
    generator = VLLMRequestGenerator(config)
    requests = generator.generate()

    all_passed = True
    for r in requests:
        sp = r.sampling_params
        print(f'  Request {r.request_id}:')
        print(f'    max_tokens: {sp.max_tokens} (expected: {r.decode_tokens})')
        print(f'    ignore_eos: {sp.ignore_eos} (expected: True)')
        print(f'    stop: {sp.stop} (expected: empty or None)')
        print(f'    stop_token_ids: {sp.stop_token_ids} (expected: empty or None)')
        
        # Verify settings
        if sp.max_tokens != r.decode_tokens:
            print(f'    ERROR: max_tokens mismatch!')
            all_passed = False
        if not sp.ignore_eos:
            print(f'    ERROR: ignore_eos should be True!')
            all_passed = False
        if sp.stop and len(sp.stop) > 0:
            print(f'    ERROR: stop should be empty!')
            all_passed = False
        if sp.stop_token_ids and len(sp.stop_token_ids) > 0:
            print(f'    ERROR: stop_token_ids should be empty!')
            all_passed = False
    return all_passed


def test_force_exact_output_length_false():
    """Test 2: force_exact_output_length=False - should not set ignore_eos."""
    print('\n=== Test 2: force_exact_output_length=False ===')
    config = RequestGeneratorConfig(
        num_requests=2,
        length_config=FixedLengthConfig(prefill_tokens=128, decode_tokens=64),
        force_exact_output_length=False,  # Allow early stopping
    )
    generator = VLLMRequestGenerator(config)
    requests = generator.generate()

    all_passed = True
    for r in requests:
        sp = r.sampling_params
        print(f'  Request {r.request_id}:')
        print(f'    max_tokens: {sp.max_tokens} (expected: {r.decode_tokens})')
        print(f'    ignore_eos: {sp.ignore_eos} (expected: False/default)')
        
        # Verify settings - ignore_eos should be False (or default)
        if sp.max_tokens != r.decode_tokens:
            print(f'    ERROR: max_tokens mismatch!')
            all_passed = False
        if sp.ignore_eos:
            print(f'    ERROR: ignore_eos should be False when force_exact_output_length=False!')
            all_passed = False
    return all_passed


if __name__ == "__main__":
    test1 = test_force_exact_output_length_true()
    test2 = test_force_exact_output_length_false()
    
    print()
    if test1 and test2:
        print('✓ Solution 2 PASSED!')
        sys.exit(0)
    else:
        print('✗ Solution 2 FAILED!')
        sys.exit(1)

#!/usr/bin/env python3
"""Test script for Solution 1: Precise prompt length control."""

import sys
sys.path.insert(0, "/research/d1/gds/ytyang/yichengfeng/frontier/sota-infer-engine/vllm")

from vllm.request_generator import (
    VLLMRequestGenerator,
    RequestGeneratorConfig,
    FixedLengthConfig,
)

def test_token_id_mode():
    """Test 1: Token ID mode (precise control) - default."""
    print('=== Test 1: Token ID Mode (Default) ===')
    config = RequestGeneratorConfig(
        num_requests=2,
        length_config=FixedLengthConfig(prefill_tokens=128, decode_tokens=8),
        use_token_ids=True,  # Explicit
    )
    generator = VLLMRequestGenerator(config)
    requests = generator.generate()

    all_passed = True
    for r in requests:
        prompt = r.prompt
        print(f'  Request {r.request_id}:')
        print(f'    Type: {type(prompt).__name__}')
        if isinstance(prompt, dict):
            has_ids = "prompt_token_ids" in prompt
            token_count = len(prompt["prompt_token_ids"]) if has_ids else 0
            match = token_count == r.prefill_tokens
            print(f'    Has prompt_token_ids: {has_ids}')
            print(f'    Token count: {token_count}')
            print(f'    Expected: {r.prefill_tokens}')
            print(f'    Match: {match}')
            if not match:
                all_passed = False
        else:
            print(f'    Text prompt (length may vary)')
            all_passed = False
    return all_passed


def test_text_mode_validation():
    """Test 2: Config validation (text mode without tokenizer)."""
    print('\n=== Test 2: Config validation (text mode without tokenizer) ===')
    try:
        config2 = RequestGeneratorConfig(
            num_requests=2,
            length_config=FixedLengthConfig(prefill_tokens=128, decode_tokens=8),
            use_token_ids=False,  # Text mode
        )
        generator2 = VLLMRequestGenerator(config2)  # Should fail
        print('  ERROR: Should have raised ValueError')
        return False
    except ValueError as e:
        print(f'  Correctly raised ValueError: {str(e)[:80]}...')
        return True


if __name__ == "__main__":
    test1 = test_token_id_mode()
    test2 = test_text_mode_validation()
    
    print()
    if test1 and test2:
        print('✓ Solution 1 PASSED!')
        sys.exit(0)
    else:
        print('✗ Solution 1 FAILED!')
        sys.exit(1)

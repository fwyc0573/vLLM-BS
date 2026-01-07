#!/usr/bin/env python3
"""Test script for Solution 3: Frontier path dependency decoupling."""

import os
import sys
# Dynamically resolve vLLM path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_VLLM_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR))))))
sys.path.insert(0, _VLLM_PATH)

from vllm.request_generator import (
    VLLMRequestGenerator,
    RequestGeneratorConfig,
    FixedLengthConfig,
)
from vllm.request_generator.vllm_request_generator import (
    _resolve_frontier_path,
    DEFAULT_FRONTIER_PATH,
)


def test_default_path():
    """Test 1: Default path resolution."""
    print('=== Test 1: Default Path Resolution ===')
    # Clear env var if set
    original_env = os.environ.pop("FRONTIER_PATH", None)
    
    try:
        path = _resolve_frontier_path(None)
        print(f'  Resolved path: {path}')
        print(f'  Expected: {DEFAULT_FRONTIER_PATH}')
        print(f'  Exists: {os.path.exists(path)}')
        
        # Check normalization (no trailing slash)
        if path.endswith('/'):
            print(f'  ERROR: Path should not end with slash!')
            return False
        
        return path == os.path.normpath(DEFAULT_FRONTIER_PATH)
    finally:
        # Restore env var
        if original_env:
            os.environ["FRONTIER_PATH"] = original_env


def test_env_var_override():
    """Test 2: Environment variable override."""
    print('\n=== Test 2: Environment Variable Override ===')
    
    # Set custom env var (use default path since we know it exists)
    custom_path = DEFAULT_FRONTIER_PATH + "/"  # Add trailing slash to test normalization
    original_env = os.environ.get("FRONTIER_PATH")
    os.environ["FRONTIER_PATH"] = custom_path
    
    try:
        path = _resolve_frontier_path(None)
        print(f'  FRONTIER_PATH set to: {custom_path}')
        print(f'  Resolved path: {path}')
        print(f'  Trailing slash removed: {not path.endswith("/")}')
        
        return not path.endswith("/") and os.path.exists(path)
    finally:
        # Restore env var
        if original_env:
            os.environ["FRONTIER_PATH"] = original_env
        else:
            os.environ.pop("FRONTIER_PATH", None)


def test_config_override():
    """Test 3: Config path overrides env var."""
    print('\n=== Test 3: Config Path Override ===')
    
    # Set env var to something different
    original_env = os.environ.get("FRONTIER_PATH")
    os.environ["FRONTIER_PATH"] = "/some/other/path"
    
    try:
        # Config path should take priority
        config_path = DEFAULT_FRONTIER_PATH
        path = _resolve_frontier_path(config_path)
        print(f'  FRONTIER_PATH: /some/other/path')
        print(f'  config.frontier_path: {config_path}')
        print(f'  Resolved path: {path}')
        print(f'  Used config path (priority): {path == os.path.normpath(config_path)}')
        
        return path == os.path.normpath(config_path)
    finally:
        # Restore env var
        if original_env:
            os.environ["FRONTIER_PATH"] = original_env
        else:
            os.environ.pop("FRONTIER_PATH", None)


def test_invalid_path_error():
    """Test 4: Invalid path raises error."""
    print('\n=== Test 4: Invalid Path Error ===')
    
    try:
        _resolve_frontier_path("/nonexistent/path/to/frontier")
        print('  ERROR: Should have raised ValueError!')
        return False
    except ValueError as e:
        error_msg = str(e)
        print(f'  Correctly raised ValueError')
        print(f'  Error contains helpful info:')
        print(f'    - Path info: {"nonexistent" in error_msg}')
        print(f'    - Fix suggestions: {"FRONTIER_PATH" in error_msg}')
        return "FRONTIER_PATH" in error_msg


def test_generator_uses_resolved_path():
    """Test 5: Generator stores resolved path."""
    print('\n=== Test 5: Generator Uses Resolved Path ===')
    
    config = RequestGeneratorConfig(
        num_requests=1,
        length_config=FixedLengthConfig(prefill_tokens=64, decode_tokens=8),
        frontier_path=DEFAULT_FRONTIER_PATH,
    )
    generator = VLLMRequestGenerator(config)
    
    print(f'  _resolved_frontier_path: {generator._resolved_frontier_path}')
    print(f'  Path in sys.path: {generator._resolved_frontier_path in sys.path}')
    
    return (
        hasattr(generator, '_resolved_frontier_path') 
        and generator._resolved_frontier_path in sys.path
    )


if __name__ == "__main__":
    tests = [
        ("Default path", test_default_path),
        ("Env var override", test_env_var_override),
        ("Config override", test_config_override),
        ("Invalid path error", test_invalid_path_error),
        ("Generator uses resolved path", test_generator_uses_resolved_path),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f'  EXCEPTION: {e}')
            results.append((name, False))
    
    print()
    print('=' * 50)
    all_passed = all(r for _, r in results)
    for name, passed in results:
        status = '✓' if passed else '✗'
        print(f'  {status} {name}')
    
    print()
    if all_passed:
        print('✓ Solution 3 PASSED!')
        sys.exit(0)
    else:
        print('✗ Solution 3 FAILED!')
        sys.exit(1)

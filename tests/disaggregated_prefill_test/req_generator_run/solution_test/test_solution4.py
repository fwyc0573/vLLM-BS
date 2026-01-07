#!/usr/bin/env python3
"""Test script for Solution 4: KV Transfer synchronization."""

import os
import sys
import time
import tempfile
import shutil
from threading import Thread

# Dynamically resolve vLLM path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_VLLM_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR))))))
sys.path.insert(0, _VLLM_PATH)

from vllm.request_generator.kv_sync import KVTransferSync


def test_marker_creation():
    """Test 1: Marker file creation and detection."""
    print('=== Test 1: Marker File Creation ===')
    
    with tempfile.TemporaryDirectory() as tmpdir:
        sync = KVTransferSync(storage_path=tmpdir)
        
        # Initially not complete
        print(f'  Before mark: is_complete={sync.is_transfer_complete()}')
        if sync.is_transfer_complete():
            print('  ERROR: Should not be complete initially!')
            return False
        
        # Mark complete
        sync.mark_transfer_complete(metadata={"test": "data"})
        
        print(f'  After mark: is_complete={sync.is_transfer_complete()}')
        if not sync.is_transfer_complete():
            print('  ERROR: Should be complete after marking!')
            return False
        
        # Check marker file exists
        print(f'  Marker file exists: {sync.marker_file.exists()}')
        
        # Check metadata
        meta = sync.get_transfer_metadata()
        print(f'  Metadata retrieved: {meta}')
        if meta.get("test") != "data":
            print('  ERROR: Metadata not preserved!')
            return False
        
        return True


def test_error_marker():
    """Test 2: Error marker handling."""
    print('\n=== Test 2: Error Marker Handling ===')
    
    with tempfile.TemporaryDirectory() as tmpdir:
        sync = KVTransferSync(storage_path=tmpdir)
        
        # Mark error
        sync.mark_transfer_error("Test error message")
        
        # Check error
        error = sync.has_transfer_error()
        print(f'  Error message: {error}')
        if "Test error" not in error:
            print('  ERROR: Error message not preserved!')
            return False
        
        # Should not be complete when error exists
        print(f'  is_complete with error: {sync.is_transfer_complete()}')
        if sync.is_transfer_complete():
            print('  ERROR: Should not be complete when error exists!')
            return False
        
        return True


def test_wait_success():
    """Test 3: Wait for completion (success case)."""
    print('\n=== Test 3: Wait for Completion (Success) ===')
    
    with tempfile.TemporaryDirectory() as tmpdir:
        sync = KVTransferSync(storage_path=tmpdir)
        
        # Mark complete in background after delay
        def mark_after_delay():
            time.sleep(0.5)
            sync.mark_transfer_complete()
        
        thread = Thread(target=mark_after_delay)
        thread.start()
        
        # Wait for completion
        start = time.time()
        result = sync.wait_for_transfer_complete(timeout=5.0, poll_interval=0.1, verbose=False)
        elapsed = time.time() - start
        
        thread.join()
        
        print(f'  Wait returned: {result}')
        print(f'  Elapsed time: {elapsed:.2f}s')
        
        if not result:
            print('  ERROR: Wait should return True!')
            return False
        
        if elapsed < 0.3:
            print('  ERROR: Wait returned too quickly (should have waited for marker)!')
            return False
        
        return True


def test_wait_timeout():
    """Test 4: Wait for completion (timeout case)."""
    print('\n=== Test 4: Wait for Completion (Timeout) ===')
    
    with tempfile.TemporaryDirectory() as tmpdir:
        sync = KVTransferSync(storage_path=tmpdir)
        
        # Don't mark complete - should timeout
        start = time.time()
        try:
            sync.wait_for_transfer_complete(timeout=1.0, poll_interval=0.1, verbose=False)
            print('  ERROR: Should have raised TimeoutError!')
            return False
        except TimeoutError as e:
            elapsed = time.time() - start
            print(f'  Correctly raised TimeoutError after {elapsed:.2f}s')
            print(f'  Error message contains timeout info: {"timeout" in str(e).lower()}')
            return True


def test_wait_error():
    """Test 5: Wait for completion (error case)."""
    print('\n=== Test 5: Wait for Completion (Error) ===')
    
    with tempfile.TemporaryDirectory() as tmpdir:
        sync = KVTransferSync(storage_path=tmpdir)
        
        # Mark error
        sync.mark_transfer_error("Transfer failed!")
        
        try:
            sync.wait_for_transfer_complete(timeout=5.0, poll_interval=0.1, verbose=False)
            print('  ERROR: Should have raised RuntimeError!')
            return False
        except RuntimeError as e:
            print(f'  Correctly raised RuntimeError')
            print(f'  Error contains message: {"Transfer failed" in str(e)}')
            return "Transfer failed" in str(e)


def test_cleanup():
    """Test 6: Cleanup removes markers."""
    print('\n=== Test 6: Cleanup ===')
    
    with tempfile.TemporaryDirectory() as tmpdir:
        sync = KVTransferSync(storage_path=tmpdir)
        
        # Create both markers
        sync.mark_transfer_complete()
        sync.mark_transfer_error("test")
        
        print(f'  Before cleanup: complete={sync.marker_file.exists()}, error={sync.error_file.exists()}')
        
        # Cleanup
        sync.cleanup()
        
        print(f'  After cleanup: complete={sync.marker_file.exists()}, error={sync.error_file.exists()}')
        
        if sync.marker_file.exists() or sync.error_file.exists():
            print('  ERROR: Markers should be removed after cleanup!')
            return False
        
        return True


if __name__ == "__main__":
    tests = [
        ("Marker creation", test_marker_creation),
        ("Error marker", test_error_marker),
        ("Wait success", test_wait_success),
        ("Wait timeout", test_wait_timeout),
        ("Wait error", test_wait_error),
        ("Cleanup", test_cleanup),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f'  EXCEPTION: {e}')
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print()
    print('=' * 50)
    all_passed = all(r for _, r in results)
    for name, passed in results:
        status = '✓' if passed else '✗'
        print(f'  {status} {name}')
    
    print()
    if all_passed:
        print('✓ Solution 4 PASSED!')
        sys.exit(0)
    else:
        print('✗ Solution 4 FAILED!')
        sys.exit(1)

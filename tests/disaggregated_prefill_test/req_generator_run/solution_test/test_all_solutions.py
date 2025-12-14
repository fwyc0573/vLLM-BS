#!/usr/bin/env python3
"""Run all solution tests."""

import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

tests = [
    ("Solution 1: Precise prompt length control", "test_solution1.py"),
    ("Solution 2: Output length guarantee", "test_solution2.py"),
    ("Solution 3: Frontier path decoupling", "test_solution3.py"),
    ("Solution 4: KV Transfer synchronization", "test_solution4.py"),
]

results = []
for name, script in tests:
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print('='*60)
    
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
    )
    results.append((name, result.returncode == 0))

print(f"\n{'='*60}")
print("SUMMARY")
print('='*60)

for name, passed in results:
    status = '✓' if passed else '✗'
    print(f"  {status} {name}")

all_passed = all(r for _, r in results)
print()
if all_passed:
    print("✓ ALL SOLUTIONS PASSED!")
    sys.exit(0)
else:
    print("✗ SOME SOLUTIONS FAILED!")
    sys.exit(1)

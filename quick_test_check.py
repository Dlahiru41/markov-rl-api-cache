"""
Quick integration test runner to check fixes.
"""

import subprocess
import sys
from pathlib import Path

def run_test(test_spec):
    """Run a specific test."""
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', test_spec, '-v', '--tb=line'],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )
    return result.returncode == 0, result.stdout, result.stderr

print("="*70)
print("RUNNING INTEGRATION TESTS (AFTER FIXES)")
print("="*70)
print()

# Test 1: Environment Basic Tests
print("1. Testing Environment Basics...")
success, stdout, stderr = run_test('tests/integration/test_environment.py::TestEnvironmentBasics')

if success:
    print("   [PASSED]")
    # Count passed tests
    lines = stdout.split('\n')
    for line in lines:
        if 'passed' in line.lower():
            print(f"   {line.strip()}")
else:
    print("   [FAILED]")
    # Show failures
    for line in stdout.split('\n')[-20:]:
        if line.strip():
            print(f"   {line}")

print()

# Test 2: Cache Operations
print("2. Testing Cache Operations...")
success2, stdout2, stderr2 = run_test('tests/integration/test_cache_system.py::TestCacheOperations::test_cache_and_retrieve')

if success2:
    print("   [PASSED]")
    for line in stdout2.split('\n'):
        if 'passed' in line.lower():
            print(f"   {line.strip()}")
else:
    print("   [FAILED]")
    for line in stdout2.split('\n')[-10:]:
        if line.strip():
            print(f"   {line}")

print()

# Test 3: Run a quick count of all tests
print("3. Collecting all integration tests...")
result = subprocess.run(
    [sys.executable, '-m', 'pytest', 'tests/integration/', '--collect-only', '-q'],
    capture_output=True,
    text=True,
    cwd=Path(__file__).parent
)

if result.returncode == 0:
    output = result.stdout
    # Count tests
    test_count = output.count('test_')
    print(f"   [OK] Found approximately {test_count} tests")
else:
    print("   [WARNING] Collection had issues")

print()
print("="*70)
print("SUMMARY")
print("="*70)
print(f"Environment Tests: {'[PASSED]' if success else '[FAILED]'}")
print(f"Cache Tests: {'[PASSED]' if success2 else '[FAILED]'}")
print()
print("To run all tests:")
print("  python -m pytest tests/integration/ -v")
print("="*70)


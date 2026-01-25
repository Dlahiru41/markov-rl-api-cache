"""
Comprehensive test runner for cache backend tests.

This script runs all cache-related tests and provides a summary.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    """Run all cache backend tests."""
    print("="*70)
    print("CACHE BACKEND TEST SUITE")
    print("="*70)

    results = []

    # 1. Unit tests for cache backend
    success = run_command(
        ["pytest", "tests/unit/test_cache_backend.py", "-v"],
        "Unit Tests - Cache Backend (InMemory)"
    )
    results.append(("Cache Backend Unit Tests", success))

    # 2. Unit tests for Redis backend
    success = run_command(
        ["pytest", "tests/unit/test_redis_backend.py", "-v"],
        "Unit Tests - Redis Backend"
    )
    results.append(("Redis Backend Unit Tests", success))

    # 3. Integration tests for Redis (requires Redis server)
    success = run_command(
        ["pytest", "tests/integration/test_redis_integration.py", "-v"],
        "Integration Tests - Redis Backend (requires Redis server)"
    )
    results.append(("Redis Integration Tests", success))

    # 4. Backend comparison tests
    success = run_command(
        ["pytest", "tests/integration/test_cache_backend_comparison.py", "-v"],
        "Integration Tests - Backend Comparison"
    )
    results.append(("Backend Comparison Tests", success))

    # 5. Performance tests
    success = run_command(
        ["pytest", "tests/performance/test_cache_performance.py", "-v", "-s"],
        "Performance Tests"
    )
    results.append(("Performance Tests", success))

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, success in results:
        status = "[SUCCESS] PASSED" if success else "[ERROR] FAILED"
        print(f"{status} - {test_name}")

    # Overall result
    all_passed = all(success for _, success in results)

    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("[WARNING]  SOME TESTS FAILED")
    print("="*70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())


"""
Script to run integration tests and capture results properly.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_tests(test_path, description):
    """Run tests and capture results."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}\n")

    result = subprocess.run(
        [sys.executable, '-m', 'pytest', test_path, '-v', '--tb=short'],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return result.returncode == 0

def main():
    print("="*70)
    print("  INTEGRATION TESTS - EXECUTION REPORT")
    print(f"  {datetime.now()}")
    print("="*70)

    tests_to_run = [
        ('tests/integration/test_environment.py::TestEnvironmentBasics', 'Environment Basic Tests'),
        ('tests/integration/test_cache_system.py::TestCacheOperations', 'Cache Operations Tests'),
        ('tests/integration/test_simulator.py::TestServiceInteraction', 'Simulator Service Tests'),
    ]

    results = {}

    for test_path, description in tests_to_run:
        success = run_tests(test_path, description)
        results[description] = success

    # Summary
    print("\n" + "="*70)
    print("  TEST EXECUTION SUMMARY")
    print("="*70)

    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed

    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {status}: {test_name}")

    print("")
    print(f"  Total: {len(results)} test suites")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print("="*70)

    if failed == 0:
        print("\n🎉 ALL TEST SUITES PASSED!")
        return 0
    else:
        print("\n⚠️  SOME TEST SUITES FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())


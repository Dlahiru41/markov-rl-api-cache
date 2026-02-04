"""
Test runner script for integration tests.

This script provides convenient commands to run integration tests with
various configurations and options.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"\n❌ {description} FAILED")
        return False
    else:
        print(f"\n✅ {description} PASSED")
        return True


def main():
    """Main test runner."""
    tests_dir = Path(__file__).parent

    print("""
╔════════════════════════════════════════════════════════════════════╗
║                   Integration Test Runner                          ║
║                                                                    ║
║  Comprehensive integration tests for markov-rl-api-cache          ║
╚════════════════════════════════════════════════════════════════════╝
    """)

    # Check if pytest is available
    result = subprocess.run("pytest --version", shell=True, capture_output=True)
    if result.returncode != 0:
        print("❌ pytest not found. Please install: pip install pytest pytest-cov pytest-xdist")
        sys.exit(1)

    print("✅ pytest found")

    # Menu
    print("\nSelect test suite to run:")
    print("1. All integration tests (default)")
    print("2. Environment tests only")
    print("3. Training loop tests only")
    print("4. Cache system tests only")
    print("5. Simulator tests only")
    print("6. Full pipeline tests only")
    print("7. Quick smoke test (fast)")
    print("8. All tests with coverage")
    print("9. All tests with parallel execution")

    choice = input("\nEnter choice (1-9) or press Enter for default: ").strip()

    if not choice:
        choice = "1"

    # Run selected tests
    success = True

    if choice == "1":
        success = run_command(
            "pytest tests/integration/ -v",
            "All Integration Tests"
        )

    elif choice == "2":
        success = run_command(
            "pytest tests/integration/test_environment.py -v",
            "Environment Tests"
        )

    elif choice == "3":
        success = run_command(
            "pytest tests/integration/test_training_loop.py -v",
            "Training Loop Tests"
        )

    elif choice == "4":
        success = run_command(
            "pytest tests/integration/test_cache_system.py -v",
            "Cache System Tests"
        )

    elif choice == "5":
        success = run_command(
            "pytest tests/integration/test_simulator.py -v",
            "Simulator Tests"
        )

    elif choice == "6":
        success = run_command(
            "pytest tests/integration/test_full_pipeline.py -v",
            "Full Pipeline Tests"
        )

    elif choice == "7":
        success = run_command(
            "pytest tests/integration/test_environment.py::TestEnvironmentBasics -v",
            "Quick Smoke Test"
        )

    elif choice == "8":
        success = run_command(
            "pytest tests/integration/ -v --cov=src --cov-report=html --cov-report=term",
            "All Tests With Coverage"
        )
        if success:
            print("\n📊 Coverage report generated in htmlcov/index.html")

    elif choice == "9":
        success = run_command(
            "pytest tests/integration/ -n auto -v",
            "All Tests With Parallel Execution"
        )

    else:
        print("Invalid choice")
        sys.exit(1)

    # Summary
    print("\n" + "="*70)
    if success:
        print("✅ ALL TESTS PASSED")
        print("="*70)
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        print("="*70)
        sys.exit(1)


if __name__ == "__main__":
    main()


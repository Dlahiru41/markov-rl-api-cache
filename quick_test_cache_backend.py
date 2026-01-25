#!/usr/bin/env python
"""
Quick test runner for cache backend tests.

Usage:
    python quick_test_cache.py              # Run all unit tests
    python quick_test_cache.py --all        # Run all tests
    python quick_test_cache.py --unit       # Run unit tests only
    python quick_test_cache.py --integration # Run integration tests
    python quick_test_cache.py --performance # Run performance tests
    python quick_test_cache.py --validate   # Quick validation
    python quick_test_cache.py --coverage   # Run with coverage
"""

import sys
import subprocess
import argparse


def run_tests(command, description):
    """Run tests with a command."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"{'='*70}\n")

    result = subprocess.run(command, shell=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Quick test runner for cache backend tests"
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Run all tests'
    )
    parser.add_argument(
        '--unit', action='store_true',
        help='Run unit tests only'
    )
    parser.add_argument(
        '--integration', action='store_true',
        help='Run integration tests only'
    )
    parser.add_argument(
        '--performance', action='store_true',
        help='Run performance tests only'
    )
    parser.add_argument(
        '--coverage', action='store_true',
        help='Run with coverage report'
    )
    parser.add_argument(
        '--validate', action='store_true',
        help='Quick validation without pytest'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # If no arguments, run unit tests
    if not any([args.all, args.unit, args.integration,
                args.performance, args.coverage, args.validate]):
        args.unit = True

    verbose = '-v' if args.verbose else ''
    success = True

    if args.validate:
        success = run_tests(
            'python validate_cache_tests.py',
            'Quick Validation'
        )

    elif args.coverage:
        success = run_tests(
            f'pytest tests/ {verbose} --cov=src/cache --cov-report=term --cov-report=html',
            'All Tests with Coverage'
        )
        if success:
            print("\n✅ Coverage report generated in htmlcov/index.html")

    elif args.all:
        success = run_tests(
            'python run_cache_tests.py',
            'All Test Categories'
        )

    elif args.unit:
        success = run_tests(
            f'pytest tests/unit/ {verbose}',
            'Unit Tests'
        )

    elif args.integration:
        success = run_tests(
            f'pytest tests/integration/ {verbose}',
            'Integration Tests (requires Redis)'
        )

    elif args.performance:
        success = run_tests(
            f'pytest tests/performance/ {verbose} -s',
            'Performance Tests'
        )

    # Print result
    print(f"\n{'='*70}")
    if success:
        print("✅ Tests PASSED")
    else:
        print("❌ Tests FAILED")
    print(f"{'='*70}\n")

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())


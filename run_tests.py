"""
Test runner script for Markov chain test suite.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --fast       # Run only fast tests (skip slow)
    python run_tests.py --coverage   # Run with coverage report
"""

import sys
import subprocess

def run_tests(args=None):
    """Run the test suite with optional arguments."""
    if args is None:
        args = sys.argv[1:]

    cmd = ['python', '-m', 'pytest', 'tests/unit/test_markov.py', '-v']

    if '--fast' in args:
        cmd.extend(['-m', 'not slow'])
        print("Running fast tests only (excluding slow tests)...")
    elif '--coverage' in args:
        cmd.extend(['--cov=src/markov', '--cov-report=html', '--cov-report=term'])
        print("Running tests with coverage report...")
    else:
        print("Running all tests...")

    result = subprocess.run(cmd)
    return result.returncode

if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)


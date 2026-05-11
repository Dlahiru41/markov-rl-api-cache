#!/usr/bin/env python3
"""
Chapter 8 Testing Report Generator
Runs all tests and generates comprehensive evaluation report.

Usage:
    python run_chapter_8_evaluation.py
    python run_chapter_8_evaluation.py --quick      # Unit + functional only
    python run_chapter_8_evaluation.py --full       # All tests (slow)
    python run_chapter_8_evaluation.py --nfr-only   # Non-functional only
"""

import subprocess
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
import datetime

@dataclass
class TestResult:
    category: str
    passed: int
    failed: int
    skipped: int
    duration_seconds: float
    coverage_percent: float = None

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.skipped

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0
        return self.passed / self.total * 100

class TestRunner:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.results: List[TestResult] = []
        self.start_time = None
        self.end_time = None

    def run_test_suite(self, category: str, test_path: str, extra_args: List[str] = None) -> TestResult:
        """Run a pytest suite and collect results."""
        cmd = ["pytest", test_path, "-v", "--tb=short", "-q"]
        if extra_args:
            cmd.extend(extra_args)

        print(f"\n{'='*70}")
        print(f"Running {category} Tests")
        print(f"{'='*70}")

        try:
            start = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            elapsed = time.time() - start

            # Parse pytest output
            output = result.stdout + result.stderr
            passed = output.count(" PASSED")
            failed = output.count(" FAILED")
            skipped = output.count(" SKIPPED")

            test_result = TestResult(
                category=category,
                passed=passed,
                failed=failed,
                skipped=skipped,
                duration_seconds=elapsed
            )

            print(f"✓ {test_result.passed} passed, {test_result.failed} failed, "
                  f"{test_result.skipped} skipped in {elapsed:.1f}s")

            if result.returncode != 0 and test_result.failed > 0:
                print("\n❌ FAILURES DETECTED:")
                print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

            self.results.append(test_result)
            return test_result

        except subprocess.TimeoutExpired:
            print(f"❌ Timeout after 600s")
            return TestResult(category, 0, 1, 0, 600)
        except Exception as e:
            print(f"❌ Error: {e}")
            return TestResult(category, 0, 1, 0, 0)

    def run_coverage_analysis(self) -> float:
        """Run coverage analysis."""
        print(f"\n{'='*70}")
        print("Running Coverage Analysis")
        print(f"{'='*70}")

        try:
            result = subprocess.run(
                ["pytest", "tests/", "--cov=src", "--cov-report=term-missing", "-q"],
                capture_output=True,
                text=True,
                timeout=300
            )

            # Extract coverage percentage
            output = result.stdout
            for line in output.split('\n'):
                if 'TOTAL' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        coverage = float(parts[-1].rstrip('%'))
                        print(f"✓ Code Coverage: {coverage:.1f}%")
                        return coverage

            return 0.0

        except Exception as e:
            print(f"❌ Coverage analysis failed: {e}")
            return 0.0

    def generate_report(self):
        """Generate comprehensive test report."""
        print(f"\n\n{'='*70}")
        print("CHAPTER 8 TESTING & EVALUATION SUMMARY")
        print(f"{'='*70}\n")

        # Summary table
        print(f"{'Category':<25} {'Passed':<10} {'Failed':<10} {'Pass %':<10} {'Duration':<12}")
        print("-" * 70)

        total_passed = 0
        total_failed = 0
        total_time = 0

        for result in self.results:
            total_passed += result.passed
            total_failed += result.failed
            total_time += result.duration_seconds

            print(f"{result.category:<25} {result.passed:<10} {result.failed:<10} "
                  f"{result.pass_rate:>7.1f}%  {result.duration_seconds:>8.1f}s")

        print("-" * 70)
        print(f"{'TOTAL':<25} {total_passed:<10} {total_failed:<10} "
              f"{(total_passed/(total_passed+total_failed)*100 if total_passed+total_failed else 0):>7.1f}%  "
              f"{total_time:>8.1f}s")

        # Overall status
        print(f"\n{'='*70}")
        if total_failed == 0:
            print("✓✓✓ ALL TESTS PASSED ✓✓✓")
            print(f"Total: {total_passed} tests in {total_time:.1f}s")
        else:
            print(f"❌ {total_failed} TESTS FAILED")
            print(f"Passed: {total_passed}/{total_passed+total_failed}")
        print(f"{'='*70}\n")

        return total_failed == 0

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Chapter 8 testing suite and generate report"
    )
    parser.add_argument("--quick", action="store_true", help="Run quick tests only (unit + functional)")
    parser.add_argument("--full", action="store_true", help="Run full test suite including slow tests")
    parser.add_argument("--nfr-only", action="store_true", help="Run non-functional tests only")
    parser.add_argument("--functional-only", action="store_true", help="Run functional tests only")
    parser.add_argument("--coverage", action="store_true", help="Include coverage analysis")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    runner = TestRunner(verbose=args.verbose)

    print("\n" + "="*70)
    print("CHAPTER 8: TESTING & EVALUATION INFRASTRUCTURE")
    print("Markov RL API Cache Gateway Project")
    print(f"Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Determine what to run
    if args.quick:
        print("\n[QUICK MODE] Running unit + functional tests only\n")
        runner.run_test_suite("Unit Tests", "tests/unit/", ["--maxfail=3"])
        runner.run_test_suite("Functional Tests", "tests/functional/", ["--maxfail=3"])

    elif args.nfr_only:
        print("\n[NFR ONLY] Running non-functional tests\n")
        runner.run_test_suite("Non-Functional Tests", "tests/nonfunctional/")

    elif args.functional_only:
        print("\n[FUNCTIONAL ONLY] Running functional tests\n")
        runner.run_test_suite("Functional Tests", "tests/functional/")

    elif args.full:
        print("\n[FULL MODE] Running all tests with coverage\n")
        runner.run_test_suite("Unit Tests", "tests/unit/")
        runner.run_test_suite("Functional Tests", "tests/functional/")
        runner.run_test_suite("Non-Functional Tests", "tests/nonfunctional/")
        runner.run_test_suite("Integration Tests", "tests/integration/")
        runner.run_test_suite("Model Tests", "tests/model/")
        if args.coverage:
            runner.run_coverage_analysis()

    else:
        # Default: comprehensive but reasonable
        print("\n[DEFAULT MODE] Running standard test suite\n")
        runner.run_test_suite("Unit Tests", "tests/unit/")
        runner.run_test_suite("Functional Tests", "tests/functional/")
        runner.run_test_suite("Non-Functional Tests", "tests/nonfunctional/")
        runner.run_test_suite("Integration Tests", "tests/integration/")
        if args.coverage:
            runner.run_coverage_analysis()

    # Generate report
    success = runner.generate_report()

    # Return appropriate exit code
    return 0 if success else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏸ Testing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        sys.exit(1)


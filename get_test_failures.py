"""Run tests and get failures."""
import subprocess
import sys

# Run tests with summary
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/integration/",
     "-v", "--tb=line", "--no-header", "-q"],
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='replace'
)

print("=" * 80)
print("TEST RESULTS")
print("=" * 80)
print(result.stdout)
if result.stderr:
    print("\nERRORS:")
    print(result.stderr)

print(f"\nReturn code: {result.returncode}")

# Count failures
lines = result.stdout.split('\n')
failed_tests = [l for l in lines if 'FAILED' in l]
passed_tests = [l for l in lines if 'PASSED' in l]

print(f"\nPassed: {len(passed_tests)}")
print(f"Failed: {len(failed_tests)}")

if failed_tests:
    print("\nFailed tests:")
    for test in failed_tests[:20]:  # First 20
        print(f"  - {test}")


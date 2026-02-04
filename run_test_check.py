"""Quick script to run tests and capture output."""
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/integration/", "-v", "--tb=short"],
    capture_output=True,
    text=True
)

print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print(f"\nReturn code: {result.returncode}")

# Write to file
with open("test_run_output.txt", "w", encoding="utf-8") as f:
    f.write("STDOUT:\n")
    f.write(result.stdout)
    f.write("\n\nSTDERR:\n")
    f.write(result.stderr)
    f.write(f"\n\nReturn code: {result.returncode}\n")

print("\nOutput saved to test_run_output.txt")


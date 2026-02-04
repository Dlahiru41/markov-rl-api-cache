"""
Comprehensive test validation that writes results to a file.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Output file
output_file = Path(__file__).parent / "test_validation_results.txt"

def log(message):
    """Log to both console and file."""
    print(message)
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

# Clear output file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(f"Integration Tests Validation - {datetime.now()}\n")
    f.write("="*70 + "\n\n")

log("INTEGRATION TESTS VALIDATION")
log("="*70)
log("")

# 1. Check Python version
log("1. Python Version")
log(f"   Version: {sys.version}")
log(f"   Executable: {sys.executable}")
log("")

# 2. Check pytest
log("2. Checking pytest...")
try:
    import pytest
    log(f"   ✅ pytest version: {pytest.__version__}")
    pytest_available = True
except ImportError as e:
    log(f"   ❌ pytest not available: {e}")
    pytest_available = False
log("")

# 3. Check required packages
log("3. Checking required packages...")
packages_to_check = [
    ('numpy', 'numpy'),
    ('torch', 'torch'),
    ('gymnasium', 'gymnasium'),
    ('scipy', 'scipy'),
]

missing_packages = []
for package_name, import_name in packages_to_check:
    try:
        pkg = __import__(import_name)
        version = getattr(pkg, '__version__', 'unknown')
        log(f"   ✅ {package_name}: {version}")
    except ImportError:
        log(f"   ❌ {package_name}: NOT INSTALLED")
        missing_packages.append(package_name)
log("")

# 4. Check if source modules exist and can be imported
log("4. Checking source module imports...")
source_imports = [
    ('CachingEnv', 'src.integration.gym_environment', 'CachingEnv'),
    ('DQNAgent', 'src.rl.agents.dqn_agent', 'DQNAgent'),
    ('CacheManager', 'src.cache.cache_manager', 'CacheManager'),
    ('MarkovPredictor', 'src.markov.predictor', 'MarkovPredictor'),
    ('Trainer', 'src.rl.training.trainer', 'Trainer'),
]

import_errors = []
for name, module, cls in source_imports:
    try:
        mod = __import__(module, fromlist=[cls])
        getattr(mod, cls)
        log(f"   ✅ {name} from {module}")
    except Exception as e:
        log(f"   ❌ {name} from {module}: {str(e)[:50]}")
        import_errors.append(name)
log("")

# 5. Check test files exist
log("5. Checking test files...")
test_files = [
    'tests/integration/conftest.py',
    'tests/integration/test_environment.py',
    'tests/integration/test_training_loop.py',
    'tests/integration/test_cache_system.py',
    'tests/integration/test_simulator.py',
    'tests/integration/test_full_pipeline.py',
]

missing_files = []
for test_file in test_files:
    file_path = Path(test_file)
    if file_path.exists():
        size = file_path.stat().st_size
        log(f"   ✅ {test_file} ({size} bytes)")
    else:
        log(f"   ❌ {test_file} NOT FOUND")
        missing_files.append(test_file)
log("")

# 6. Syntax check test files
log("6. Checking Python syntax of test files...")
import py_compile

syntax_errors = []
for test_file in test_files:
    if Path(test_file).exists():
        try:
            py_compile.compile(test_file, doraise=True)
            log(f"   ✅ {test_file}")
        except py_compile.PyCompileError as e:
            log(f"   ❌ {test_file}: Syntax error")
            syntax_errors.append(test_file)
log("")

# 7. Try running pytest collection
log("7. Testing pytest collection...")
if pytest_available and not import_errors and not syntax_errors:
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', 'tests/integration/', '--collect-only', '-q'],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=Path(__file__).parent
        )

        if result.returncode == 0:
            output_lines = result.stdout.strip().split('\n')
            test_count = 0
            for line in output_lines:
                if 'test' in line.lower():
                    test_count += 1
            log(f"   ✅ Pytest can collect tests")
            log(f"   Found approximately {test_count} test items")
            if result.stdout:
                log(f"   Output preview: {result.stdout[:200]}")
        else:
            log(f"   ⚠️  Pytest collection had issues")
            if result.stderr:
                log(f"   Error: {result.stderr[:200]}")
    except Exception as e:
        log(f"   ⚠️  Could not run pytest: {str(e)[:100]}")
else:
    log("   ⚠️  Skipping (prerequisites not met)")
log("")

# 8. Summary
log("="*70)
log("VALIDATION SUMMARY")
log("="*70)

issues = []
if missing_packages:
    issues.append(f"Missing packages: {', '.join(missing_packages)}")
if import_errors:
    issues.append(f"Import errors: {', '.join(import_errors)}")
if missing_files:
    issues.append(f"Missing files: {', '.join(missing_files)}")
if syntax_errors:
    issues.append(f"Syntax errors: {', '.join(syntax_errors)}")
if not pytest_available:
    issues.append("pytest not available")

if not issues:
    log("✅ ALL CHECKS PASSED!")
    log("")
    log("Your integration test environment is ready!")
    log("")
    log("To run tests:")
    log("  python -m pytest tests/integration/ -v")
    log("  python run_integration_tests.py")
else:
    log("❌ ISSUES FOUND:")
    for issue in issues:
        log(f"  - {issue}")
    log("")
    log("Please fix these issues before running tests.")

log("")
log(f"Results written to: {output_file}")
log("="*70)

print(f"\n✅ Validation complete! Results saved to: {output_file}")
print(f"You can view the full report in: {output_file.absolute()}")


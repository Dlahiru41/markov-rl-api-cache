"""
Validation script for integration tests setup.

This script checks that all dependencies are installed and the test
environment is properly configured.
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version is 3.8+."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"✅ {package_name} installed")
        return True
    except ImportError:
        print(f"❌ {package_name} not installed")
        return False


def check_pytest():
    """Check pytest installation."""
    try:
        import pytest
        print(f"✅ pytest {pytest.__version__} installed")
        return True
    except ImportError:
        print("❌ pytest not installed")
        return False


def check_project_structure():
    """Check that required directories exist."""
    base_dir = Path(__file__).parent

    required_dirs = [
        base_dir / "src",
        base_dir / "tests" / "integration",
        base_dir / "src" / "integration",
        base_dir / "src" / "rl",
        base_dir / "src" / "cache",
    ]

    all_exist = True
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"✅ {dir_path.relative_to(base_dir)} exists")
        else:
            print(f"❌ {dir_path.relative_to(base_dir)} missing")
            all_exist = False

    return all_exist


def check_test_files():
    """Check that test files exist."""
    base_dir = Path(__file__).parent
    test_dir = base_dir / "tests" / "integration"

    required_files = [
        "conftest.py",
        "test_environment.py",
        "test_training_loop.py",
        "test_cache_system.py",
        "test_simulator.py",
        "test_full_pipeline.py",
    ]

    all_exist = True
    for file_name in required_files:
        file_path = test_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name} exists")
        else:
            print(f"❌ {file_name} missing")
            all_exist = False

    return all_exist


def check_imports():
    """Check that main modules can be imported."""
    imports_to_check = [
        ("src.integration.gym_environment", "CachingEnv"),
        ("src.rl.agents.dqn_agent", "DQNAgent"),
        ("src.cache.cache_manager", "CacheManager"),
        ("src.markov.predictor", "MarkovPredictor"),
    ]

    all_ok = True
    for module_name, class_name in imports_to_check:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ Can import {class_name} from {module_name}")
        except Exception as e:
            print(f"❌ Cannot import {class_name} from {module_name}: {e}")
            all_ok = False

    return all_ok


def run_quick_test():
    """Run a quick smoke test."""
    try:
        result = subprocess.run(
            ["pytest", "tests/integration/", "--collect-only", "-q"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            # Count collected tests
            output = result.stdout
            if "test" in output.lower():
                print("✅ Tests can be collected by pytest")
                return True
            else:
                print("⚠️  Pytest runs but no tests collected")
                return True
        else:
            print(f"❌ Pytest collection failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Could not run pytest: {e}")
        return False


def main():
    """Run all validation checks."""
    print("="*70)
    print("  Integration Tests Setup Validation")
    print("="*70)
    print()

    checks = {
        "Python Version": check_python_version(),
        "pytest": check_pytest(),
        "numpy": check_package("numpy"),
        "torch": check_package("torch"),
        "gymnasium": check_package("gymnasium"),
        "Project Structure": check_project_structure(),
        "Test Files": check_test_files(),
    }

    print()
    print("="*70)
    print("  Advanced Checks")
    print("="*70)
    print()

    # Only check imports if basic packages are installed
    if checks["numpy"] and checks["torch"] and checks["gymnasium"]:
        checks["Module Imports"] = check_imports()
    else:
        print("⚠️  Skipping import checks (install dependencies first)")
        checks["Module Imports"] = None

    if checks["pytest"] and checks["Test Files"]:
        checks["Pytest Collection"] = run_quick_test()
    else:
        print("⚠️  Skipping pytest test (install pytest first)")
        checks["Pytest Collection"] = None

    # Summary
    print()
    print("="*70)
    print("  Validation Summary")
    print("="*70)
    print()

    passed = sum(1 for v in checks.values() if v is True)
    failed = sum(1 for v in checks.values() if v is False)
    skipped = sum(1 for v in checks.values() if v is None)

    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    if skipped > 0:
        print(f"⚠️  Skipped: {skipped}")

    print()

    if failed > 0:
        print("❌ VALIDATION FAILED")
        print()
        print("Please fix the issues above before running tests.")
        print()
        print("Quick fixes:")
        print("  - Install dependencies: pip install -r requirements_integration_tests.txt")
        print("  - Install package: pip install -e .")
        print("  - Check file paths are correct")
        sys.exit(1)
    else:
        print("✅ ALL CHECKS PASSED!")
        print()
        print("Your integration test environment is ready!")
        print()
        print("Next steps:")
        print("  1. Run tests: python run_integration_tests.py")
        print("  2. Or use pytest: pytest tests/integration/ -v")
        print("  3. With coverage: pytest tests/integration/ --cov=src --cov-report=html")
        sys.exit(0)


if __name__ == "__main__":
    main()


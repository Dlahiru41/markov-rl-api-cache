"""
Comprehensive Exception Checker

This script checks for common exceptions in the test files.
"""

import sys
import traceback

print("\n" + "=" * 70)
print("CHECKING FOR EXCEPTIONS IN TEST MODULES")
print("=" * 70 + "\n")

exceptions_found = []

# Check 1: PyYAML availability
print("1. Checking PyYAML installation...")
try:
    import yaml
    print("   [OK] PyYAML is installed")
except ImportError as e:
    print(f"   [FAIL] PyYAML NOT INSTALLED: {e}")
    exceptions_found.append(("PyYAML Import", str(e)))

# Check 2: FastAPI availability
print("\n2. Checking FastAPI installation...")
try:
    import fastapi
    print("   [OK] FastAPI is installed")
except ImportError as e:
    print(f"   [FAIL] FastAPI NOT INSTALLED: {e}")
    exceptions_found.append(("FastAPI Import", str(e)))

# Check 3: Faker availability
print("\n3. Checking Faker installation...")
try:
    from faker import Faker
    print("   [OK] Faker is installed")
except ImportError as e:
    print(f"   [FAIL] Faker NOT INSTALLED: {e}")
    exceptions_found.append(("Faker Import", str(e)))

# Check 4: aiohttp availability
print("\n4. Checking aiohttp installation...")
try:
    import aiohttp
    print("   [OK] aiohttp is installed")
except ImportError as e:
    print(f"   [FAIL] aiohttp NOT INSTALLED: {e}")
    exceptions_found.append(("aiohttp Import", str(e)))

# Check 5: httpx availability
print("\n5. Checking httpx installation...")
try:
    import httpx
    print("   [OK] httpx is installed")
except ImportError as e:
    print(f"   [FAIL] httpx NOT INSTALLED: {e}")
    exceptions_found.append(("httpx Import", str(e)))

# Check 6: Base service import
print("\n6. Checking BaseService import...")
try:
    from simulator.services.base_service import BaseService, ServiceConfig
    print("   [OK] BaseService imports successfully")
except Exception as e:
    print(f"   [FAIL] ERROR: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    exceptions_found.append(("BaseService Import", str(e)))

# Check 7: Failure injector import
print("\n7. Checking FailureInjector import...")
try:
    from simulator.failures.injector import FailureInjector, FailureScenario, CascadeSimulator
    print("   [OK] FailureInjector imports successfully")
except Exception as e:
    print(f"   [FAIL] ERROR: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    exceptions_found.append(("FailureInjector Import", str(e)))

# Check 8: Traffic generator import
print("\n8. Checking TrafficGenerator import...")
try:
    from simulator.traffic.generator import TrafficGenerator, TrafficProfile
    print("   [OK] TrafficGenerator imports successfully")
except Exception as e:
    print(f"   [FAIL] ERROR: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    exceptions_found.append(("TrafficGenerator Import", str(e)))

# Check 9: E-commerce services import
print("\n9. Checking E-commerce services import...")
try:
    from simulator.services.ecommerce import UserService
    print("   [OK] E-commerce services import successfully")
except Exception as e:
    print(f"   [FAIL] ERROR: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    exceptions_found.append(("E-commerce Services Import", str(e)))

# Check 10: YAML scenarios file
print("\n10. Checking scenarios.yaml file...")
try:
    import os
    yaml_path = 'simulator/failures/scenarios.yaml'
    if os.path.exists(yaml_path):
        print(f"   [OK] scenarios.yaml exists at {yaml_path}")
        with open(yaml_path, 'r') as f:
            scenarios = yaml.safe_load(f)
            print(f"   [OK] scenarios.yaml is valid YAML ({len(scenarios.get('scenarios', []))} scenarios)")
    else:
        print(f"   [FAIL] scenarios.yaml NOT FOUND at {yaml_path}")
        exceptions_found.append(("YAML File", "File not found"))
except Exception as e:
    print(f"   [FAIL] ERROR: {e}")
    exceptions_found.append(("YAML File", str(e)))

# Summary
print("\n" + "=" * 70)
if exceptions_found:
    print(f"[ERROR] FOUND {len(exceptions_found)} EXCEPTION(S)")
    print("=" * 70)
    print("\nExceptions:")
    for i, (module, error) in enumerate(exceptions_found, 1):
        print(f"\n{i}. {module}")
        print(f"   Error: {error}")

    print("\n" + "-" * 70)
    print("SOLUTION:")
    print("-" * 70)
    print("\nInstall missing packages:")
    print("  pip install -r requirements.txt")
    print("\nOr install individually:")
    if any("PyYAML" in e[0] for e in exceptions_found):
        print("  pip install pyyaml")
    if any("FastAPI" in e[0] for e in exceptions_found):
        print("  pip install fastapi uvicorn")
    if any("Faker" in e[0] for e in exceptions_found):
        print("  pip install faker")
    if any("aiohttp" in e[0] for e in exceptions_found):
        print("  pip install aiohttp")
    if any("httpx" in e[0] for e in exceptions_found):
        print("  pip install httpx")

    sys.exit(1)
else:
    print("[SUCCESS] NO EXCEPTIONS FOUND - All modules load successfully!")
    print("=" * 70)
    print("\n[OK] All dependencies are installed")
    print("[OK] All imports work correctly")
    print("[OK] Test files should run without exceptions")
    print("\nYou can now run:")
    print("  python test_failure_injection.py")
    print("  python test_traffic_generator.py")
    print("  python test_ecommerce_unit.py")
    sys.exit(0)


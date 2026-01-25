"""
Diagnostic script to check for exceptions in test files
"""

import sys
import traceback

print("=" * 70)
print("DIAGNOSTIC: Checking test imports and basic functionality")
print("=" * 70)

# Test 1: Import failure injector
print("\nTest 1: Import failure injector...")
try:
    from simulator.failures.injector import FailureInjector, FailureScenario, CascadeSimulator
    print("[OK] SUCCESS: Imports working")
except Exception as e:
    print(f"[FAIL] FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 2: Import base service
print("\nTest 2: Import base service...")
try:
    from simulator.services.base_service import BaseService, ServiceConfig
    print("[OK] SUCCESS: BaseService imports working")
except Exception as e:
    print(f"[FAIL] FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 3: Create a simple scenario
print("\nTest 3: Create FailureScenario...")
try:
    scenario = FailureScenario(
        name="test",
        description="Test scenario",
        affected_services=["test-service"],
        failure_type="latency",
        parameters={"multiplier": 2.0},
        duration_seconds=10
    )
    print(f"[OK] SUCCESS: Created scenario '{scenario.name}'")
except Exception as e:
    print(f"[FAIL] FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 4: Create a service and injector
print("\nTest 4: Create service and injector...")
try:
    config = ServiceConfig(name="test-service", port=9999)
    service = BaseService(config)
    services = {"test-service": service}
    injector = FailureInjector(services)
    print("[OK] SUCCESS: Created injector")
except Exception as e:
    print(f"[FAIL] FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 5: Inject latency
print("\nTest 5: Inject latency spike...")
try:
    injector.inject_latency_spike("test-service", multiplier=3.0, duration=5)
    active = injector.get_active_failures()
    print(f"[OK] SUCCESS: Injected latency, active failures: {len(active)}")
except Exception as e:
    print(f"[FAIL] FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 6: Load YAML scenarios
print("\nTest 6: Load YAML scenarios...")
try:
    scenarios = FailureScenario.from_yaml('simulator/failures/scenarios.yaml')
    print(f"[OK] SUCCESS: Loaded {len(scenarios)} scenarios")
except Exception as e:
    print(f"[FAIL] FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 7: Create cascade simulator
print("\nTest 7: Create cascade simulator...")
try:
    deps = {
        'service-a': ['service-b'],
        'service-b': ['service-c'],
        'service-c': []
    }
    simulator = CascadeSimulator(deps)
    timeline = simulator.simulate_cascade('service-b', 'latency', duration=30)
    print(f"[OK] SUCCESS: Generated cascade with {len(timeline)} events")
except Exception as e:
    print(f"[FAIL] FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("[SUCCESS] ALL DIAGNOSTIC TESTS PASSED")
print("=" * 70)
print("\nNo exceptions found. System is working correctly.")


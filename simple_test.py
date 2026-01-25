"""
Simple test that writes results to a file
"""

output_file = "test_results.txt"

with open(output_file, "w") as f:
    f.write("Starting tests...\n\n")

    # Test 1: Import FailureInjector
    f.write("Test 1: Importing FailureInjector...\n")
    try:
        from simulator.failures.injector import FailureInjector, FailureScenario
        f.write("[OK] SUCCESS\n\n")
    except Exception as e:
        f.write(f"[FAIL] FAILED: {e}\n")
        import traceback
        f.write(traceback.format_exc())
        f.write("\n\n")

    # Test 2: Import BaseService
    f.write("Test 2: Importing BaseService...\n")
    try:
        from simulator.services.base_service import BaseService, ServiceConfig
        f.write("[OK] SUCCESS\n\n")
    except Exception as e:
        f.write(f"[FAIL] FAILED: {e}\n")
        import traceback
        f.write(traceback.format_exc())
        f.write("\n\n")

    # Test 3: Create scenario
    f.write("Test 3: Creating FailureScenario...\n")
    try:
        scenario = FailureScenario(
            name="test",
            description="Test",
            affected_services=["test"],
            failure_type="latency",
            parameters={"multiplier": 2.0}
        )
        f.write(f"[OK] SUCCESS: {scenario.name}\n\n")
    except Exception as e:
        f.write(f"[FAIL] FAILED: {e}\n")
        import traceback
        f.write(traceback.format_exc())
        f.write("\n\n")

    # Test 4: Create injector
    f.write("Test 4: Creating FailureInjector...\n")
    try:
        config = ServiceConfig(name="test", port=9999)
        service = BaseService(config)
        injector = FailureInjector({"test": service})
        f.write("[OK] SUCCESS\n\n")
    except Exception as e:
        f.write(f"[FAIL] FAILED: {e}\n")
        import traceback
        f.write(traceback.format_exc())
        f.write("\n\n")

    # Test 5: Inject failure
    f.write("Test 5: Injecting latency spike...\n")
    try:
        injector.inject_latency_spike("test", multiplier=3.0)
        active = injector.get_active_failures()
        f.write(f"[OK] SUCCESS: {len(active)} active failures\n\n")
    except Exception as e:
        f.write(f"[FAIL] FAILED: {e}\n")
        import traceback
        f.write(traceback.format_exc())
        f.write("\n\n")

    f.write("Tests complete. Check test_results.txt for details.\n")

print(f"Tests complete. Results written to {output_file}")


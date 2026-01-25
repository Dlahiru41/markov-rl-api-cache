"""
Validation script for Failure Injection System

Tests the failure injector according to the requirements.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from simulator.failures.injector import FailureInjector, FailureScenario, CascadeSimulator
from simulator.services.base_service import BaseService, ServiceConfig


def test_failure_scenario_dataclass():
    """Test FailureScenario dataclass."""
    print("\n" + "=" * 70)
    print("TEST 1: FailureScenario Dataclass")
    print("=" * 70)

    # Create scenario
    scenario = FailureScenario(
        name="payment_slowdown",
        description="Payment service experiencing high latency",
        affected_services=["payment-service"],
        failure_type="latency",
        parameters={"multiplier": 5.0},
        duration_seconds=60,
        start_delay_seconds=5
    )

    print(f"[OK] Scenario created: {scenario.name}")
    print(f"  Description: {scenario.description}")
    print(f"  Type: {scenario.failure_type}")
    print(f"  Affected services: {scenario.affected_services}")
    print(f"  Parameters: {scenario.parameters}")
    print(f"  Duration: {scenario.duration_seconds}s")
    print(f"  Start delay: {scenario.start_delay_seconds}s")

    assert scenario.name == "payment_slowdown"
    assert scenario.failure_type == "latency"
    print("[OK] FailureScenario dataclass working\n")


def test_load_scenarios_from_yaml():
    """Test loading scenarios from YAML."""
    print("=" * 70)
    print("TEST 2: Load Scenarios from YAML")
    print("=" * 70)

    scenarios = FailureScenario.from_yaml('simulator/failures/scenarios.yaml')

    print(f"[OK] Loaded {len(scenarios)} scenarios:")
    for scenario in scenarios[:5]:  # Show first 5
        print(f"  - {scenario.name}: {scenario.description}")

    # Verify specific scenarios
    scenario_names = [s.name for s in scenarios]
    expected = [
        'payment_gateway_slow',
        'database_connection_exhaustion',
        'memory_pressure',
        'full_cascade',
        'partial_outage'
    ]

    for name in expected:
        assert name in scenario_names, f"Missing scenario: {name}"
        print(f"[OK] Found scenario: {name}")

    print("[OK] All scenarios loaded successfully\n")


def test_failure_injector_basic():
    """Test basic FailureInjector functionality."""
    print("=" * 70)
    print("TEST 3: FailureInjector Basic Operations")
    print("=" * 70)

    # Create mock services
    services = {}
    for name in ['user-service', 'payment-service', 'product-service']:
        config = ServiceConfig(name=name, port=8000 + len(services))
        services[name] = BaseService(config)

    print(f"[OK] Created {len(services)} mock services")

    # Create injector
    injector = FailureInjector(services)
    print("[OK] FailureInjector created")

    # Test latency injection
    injector.inject_latency_spike('payment-service', multiplier=5.0, duration=60)
    assert injector.is_failure_active('payment-service')
    print("[OK] Latency spike injected")

    # Check active failures
    active = injector.get_active_failures()
    print(f"[OK] Active failures: {len(active)}")
    for failure in active:
        print(f"  - {failure['service']}: {failure['type']} "
              f"(elapsed: {failure['elapsed']:.1f}s)")

    assert len(active) > 0
    print("[OK] Active failures tracked correctly")

    # Test partial failure
    injector.inject_partial_failure('user-service', error_rate=0.3, status_code=503)
    assert injector.is_failure_active('user-service')
    print("[OK] Partial failure injected")

    # Restore one service
    injector.restore('payment-service')
    assert not injector.is_failure_active('payment-service')
    print("[OK] Service restored")

    # Restore all
    injector.restore_all()
    assert len(injector.get_active_failures()) == 0
    print("[OK] All services restored\n")


def test_failure_types():
    """Test different failure types."""
    print("=" * 70)
    print("TEST 4: Different Failure Types")
    print("=" * 70)

    # Create services
    services = {}
    for name in ['service-a', 'service-b', 'service-c']:
        config = ServiceConfig(name=name, port=8000 + len(services))
        services[name] = BaseService(config)

    injector = FailureInjector(services)

    # Test latency
    injector.inject_latency_spike('service-a', multiplier=3.0)
    service_a = services['service-a']
    assert service_a._latency_multiplier == 3.0
    print("[OK] Latency injection verified")

    # Test error rate
    injector.inject_partial_failure('service-b', error_rate=0.5)
    service_b = services['service-b']
    assert service_b._failure_rate_override == 0.5
    print("[OK] Error injection verified")

    # Test timeout
    injector.inject_timeout('service-c', timeout_rate=0.3)
    service_c = services['service-c']
    assert service_c._latency_multiplier == 100.0  # Very high = timeout
    print("[OK] Timeout injection verified")

    # Test network partition
    services['service-a'].register_service('service-b', 'http://localhost:8001')
    assert 'service-b' in services['service-a'].service_registry

    injector.inject_network_partition('service-a', 'service-b')
    assert 'service-b' not in services['service-a'].service_registry
    print("[OK] Network partition verified")

    injector.restore_all()
    print("[OK] All failure types working\n")


def test_cascade_simulation():
    """Test cascade failure simulation."""
    print("=" * 70)
    print("TEST 5: Cascade Failure Simulation")
    print("=" * 70)

    # Define service dependencies
    service_dependencies = {
        'order-service': ['payment-service', 'cart-service', 'inventory-service'],
        'cart-service': ['product-service', 'user-service'],
        'product-service': ['inventory-service'],
        'user-service': ['auth-service'],
        'payment-service': [],
        'inventory-service': [],
        'auth-service': [],
    }

    simulator = CascadeSimulator(service_dependencies)
    print("[OK] CascadeSimulator created")

    # Simulate cascade
    timeline = simulator.simulate_cascade('payment-service', 'latency', duration=120)

    print(f"[OK] Simulated cascade: {len(timeline)} events")
    print("\nCascade timeline:")
    for event in timeline[:10]:  # Show first 10 events
        print(f"  t={event['time']:3d}s: {event['service']:<20} - {event['impact']}")

    assert len(timeline) > 0
    assert timeline[0]['service'] == 'payment-service'
    print("\n[OK] Cascade simulation working")

    # Test cascade path
    paths = simulator.get_cascade_path('payment-service')
    print(f"\n[OK] Found {len(paths)} cascade paths from payment-service")
    for i, path in enumerate(paths[:3], 1):
        print(f"  Path {i}: {' -> '.join(path)}")

    # Test critical services
    critical = simulator.get_critical_services()
    print(f"\n[OK] Critical services (most depended upon):")
    for service, count in critical[:5]:
        print(f"  - {service}: {count} dependents")

    print("\n[OK] Cascade simulation complete\n")


def test_cascade_risk_detection():
    """Test cascade risk detection."""
    print("=" * 70)
    print("TEST 6: Cascade Risk Detection")
    print("=" * 70)

    service_dependencies = {
        'service-a': ['service-b'],
        'service-b': ['service-c'],
        'service-c': [],
    }

    simulator = CascadeSimulator(service_dependencies)

    # Test with normal metrics
    normal_metrics = {
        'service-a': {'latency_p95': 100, 'error_rate': 0.01},
        'service-b': {'latency_p95': 150, 'error_rate': 0.02},
        'service-c': {'latency_p95': 80, 'error_rate': 0.01},
    }

    risk, at_risk = simulator.detect_cascade_risk(normal_metrics)
    print(f"Normal conditions - Risk score: {risk:.2f}, At-risk services: {at_risk}")
    assert risk < 0.5
    print("[OK] Normal conditions detected")

    # Test with degraded metrics
    degraded_metrics = {
        'service-a': {'latency_p95': 1500, 'error_rate': 0.15},
        'service-b': {'latency_p95': 2000, 'error_rate': 0.20},
        'service-c': {'latency_p95': 100, 'error_rate': 0.02},
    }

    risk, at_risk = simulator.detect_cascade_risk(degraded_metrics)
    print(f"Degraded conditions - Risk score: {risk:.2f}, At-risk services: {at_risk}")
    assert risk > 0.5
    assert len(at_risk) > 0
    print("[OK] High-risk conditions detected")

    print("[OK] Risk detection working\n")


def test_scenario_injection():
    """Test injecting a complete scenario."""
    print("=" * 70)
    print("TEST 7: Scenario Injection")
    print("=" * 70)

    # Create services
    services = {
        'payment-service': BaseService(ServiceConfig(name='payment-service', port=8006)),
    }

    injector = FailureInjector(services)

    # Load and inject scenario
    scenarios = FailureScenario.from_yaml('simulator/failures/scenarios.yaml')
    payment_scenario = next(s for s in scenarios if s.name == 'payment_gateway_slow')

    print(f"Injecting scenario: {payment_scenario.name}")
    injector.inject_scenario(payment_scenario)

    # Verify injection
    assert injector.is_failure_active('payment-service')
    active = injector.get_active_failures()
    assert len(active) > 0
    print(f"[OK] Scenario injected: {active[0]['type']} on {active[0]['service']}")

    injector.restore_all()
    print("[OK] Scenario injection working\n")


def test_validation_example():
    """Test the exact validation example from requirements."""
    print("=" * 70)
    print("TEST 8: Validation Example (from requirements)")
    print("=" * 70)

    # Create services
    services = {
        'payment-service': BaseService(ServiceConfig(name='payment-service', port=8006)),
        'user-service': BaseService(ServiceConfig(name='user-service', port=8001)),
    }

    # Add dependencies for cascade
    services['payment-service'].config.dependencies = []
    services['user-service'].config.dependencies = ['payment-service']

    # Create injector with services
    injector = FailureInjector(services)
    print("[OK] Injector created")

    # Inject a latency spike
    injector.inject_latency_spike('payment-service', multiplier=5.0, duration=60)
    print(f"Active failures: {injector.get_active_failures()}")
    assert len(injector.get_active_failures()) > 0
    print("[OK] Latency spike injected")

    # Inject partial failure
    injector.inject_partial_failure('user-service', error_rate=0.3, status_code=503)
    print("[OK] Partial failure injected")

    # After testing, restore
    injector.restore('payment-service')
    assert not injector.is_failure_active('payment-service')
    print("[OK] Service restored")

    # Test cascade simulation
    service_dependencies = {
        'user-service': ['payment-service'],
        'payment-service': [],
    }
    simulator = CascadeSimulator(service_dependencies)
    timeline = simulator.simulate_cascade('payment-service', 'latency', duration=120)
    print("\nCascade timeline:")
    for event in timeline:
        print(f"  t={event['time']}s: {event['service']} - {event['impact']}")
    assert len(timeline) > 0
    print("[OK] Cascade simulation working")

    # Restore everything
    injector.restore_all()
    assert len(injector.get_active_failures()) == 0
    print("[OK] All restored")

    print("\n[OK] Validation example passed\n")


def main():
    """Run all tests."""
    print("\n")
    print("=" + "=" * 68 + "=")
    print("|" + " " * 15 + "FAILURE INJECTION SYSTEM TESTS" + " " * 23 + "|")
    print("=" + "=" * 68 + "╝")

    try:
        test_failure_scenario_dataclass()
        test_load_scenarios_from_yaml()
        test_failure_injector_basic()
        test_failure_types()
        test_cascade_simulation()
        test_cascade_risk_detection()
        test_scenario_injection()
        test_validation_example()

        print("=" * 70)
        print("[SUCCESS] ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("Summary:")
        print("  [OK] FailureScenario dataclass working")
        print("  [OK] Load scenarios from YAML working")
        print("  [OK] FailureInjector basic operations working")
        print("  [OK] All failure types (latency/error/timeout/partition) working")
        print("  [OK] Cascade simulation working")
        print("  [OK] Cascade risk detection working")
        print("  [OK] Scenario injection working")
        print("  [OK] Validation example passed")
        print()
        print("Failure injection system is ready to use!")
        print()
        print("Available failure types:")
        print("  - Latency spike (make service slower)")
        print("  - Partial failure (service returns errors)")
        print("  - Timeout (service hangs)")
        print("  - Cascade failure (propagates through dependencies)")
        print("  - Network partition (service unreachable)")
        print()
        print("Pre-defined scenarios:")
        print("  - payment_gateway_slow")
        print("  - database_connection_exhaustion")
        print("  - memory_pressure")
        print("  - full_cascade")
        print("  - partial_outage")
        print("  - And 10 more in scenarios.yaml")
        print()

        return True

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


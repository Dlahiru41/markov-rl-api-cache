"""
Failure Injection - Practical Usage Examples

Demonstrates how to use the failure injection system for testing.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from simulator.failures.injector import FailureInjector, FailureScenario, CascadeSimulator
from simulator.services.base_service import BaseService, ServiceConfig


def example_basic_injection():
    """Example 1: Basic failure injection."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Failure Injection")
    print("=" * 70 + "\n")

    # Create mock services
    services = {
        'payment-service': BaseService(ServiceConfig(name='payment-service', port=8006)),
        'user-service': BaseService(ServiceConfig(name='user-service', port=8001)),
    }

    print("Services created:")
    for name in services.keys():
        print(f"  - {name}")
    print()

    # Create injector
    injector = FailureInjector(services)
    print("[OK] FailureInjector created\n")

    # Inject latency spike
    print("Injecting 5x latency spike to payment-service...")
    injector.inject_latency_spike('payment-service', multiplier=5.0, duration=10)

    # Check status
    print(f"[OK] Is payment-service affected? {injector.is_failure_active('payment-service')}")
    print(f"[OK] Payment service latency multiplier: {services['payment-service']._latency_multiplier}x")
    print()

    # Show active failures
    print("Active failures:")
    for failure in injector.get_active_failures():
        print(f"  - {failure['service']}: {failure['type']} "
              f"(elapsed: {failure['elapsed']:.1f}s, duration: {failure['duration']}s)")
    print()

    # Restore
    print("Restoring payment-service...")
    injector.restore('payment-service')
    print(f"[OK] Is payment-service affected? {injector.is_failure_active('payment-service')}")
    print()


def example_multiple_failures():
    """Example 2: Multiple simultaneous failures."""
    print("=" * 70)
    print("EXAMPLE 2: Multiple Simultaneous Failures")
    print("=" * 70 + "\n")

    services = {
        'service-a': BaseService(ServiceConfig(name='service-a', port=8001)),
        'service-b': BaseService(ServiceConfig(name='service-b', port=8002)),
        'service-c': BaseService(ServiceConfig(name='service-c', port=8003)),
    }

    injector = FailureInjector(services)

    # Inject different failures
    print("Injecting multiple failures:")
    print("  1. service-a: 3x latency")
    injector.inject_latency_spike('service-a', multiplier=3.0)

    print("  2. service-b: 50% error rate")
    injector.inject_partial_failure('service-b', error_rate=0.5, status_code=503)

    print("  3. service-c: 30% timeouts")
    injector.inject_timeout('service-c', timeout_rate=0.3)
    print()

    # Show all active failures
    failures = injector.get_active_failures()
    print(f"[OK] Total active failures: {len(failures)}")
    for failure in failures:
        print(f"  - {failure['service']}: {failure['type']}")
    print()

    # Restore all
    print("Restoring all services...")
    injector.restore_all()
    print(f"[OK] Active failures: {len(injector.get_active_failures())}")
    print()


def example_cascade_simulation():
    """Example 3: Cascade failure simulation."""
    print("=" * 70)
    print("EXAMPLE 3: Cascade Failure Simulation")
    print("=" * 70 + "\n")

    # Define realistic service dependencies
    service_dependencies = {
        'frontend': ['api-gateway'],
        'api-gateway': ['order-service', 'user-service'],
        'order-service': ['payment-service', 'inventory-service'],
        'user-service': ['auth-service', 'database'],
        'payment-service': ['external-payment-gateway'],
        'inventory-service': ['database'],
        'auth-service': ['database'],
        'database': [],
        'external-payment-gateway': [],
    }

    print("Service dependency graph:")
    for service, deps in service_dependencies.items():
        if deps:
            print(f"  {service} → {', '.join(deps)}")
    print()

    # Create simulator
    simulator = CascadeSimulator(service_dependencies)
    print("[OK] CascadeSimulator created\n")

    # Simulate cascade from payment service
    print("Simulating cascade from payment-service failure...")
    timeline = simulator.simulate_cascade('payment-service', 'latency', duration=60)

    print(f"\n[OK] Generated {len(timeline)} cascade events\n")
    print("Cascade timeline:")
    for event in timeline:
        severity_icon = "[CRITICAL]" if event['severity'] == 'critical' else "[HIGH]"
        print(f"  {severity_icon} t={event['time']:3d}s: {event['service']:<30} {event['impact']}")
    print()

    # Show cascade paths
    print("Cascade propagation paths from payment-service:")
    paths = simulator.get_cascade_path('payment-service')
    for i, path in enumerate(paths, 1):
        print(f"  Path {i}: {' → '.join(path)}")
    print()


def example_critical_services():
    """Example 4: Identify critical services."""
    print("=" * 70)
    print("EXAMPLE 4: Critical Services Analysis")
    print("=" * 70 + "\n")

    service_dependencies = {
        'frontend': ['api-gateway'],
        'api-gateway': ['user-service', 'product-service', 'order-service'],
        'order-service': ['payment-service', 'inventory-service', 'user-service'],
        'product-service': ['inventory-service', 'cache'],
        'user-service': ['auth-service', 'database'],
        'payment-service': ['database'],
        'inventory-service': ['database'],
        'auth-service': ['database'],
        'cache': ['database'],
        'database': [],
    }

    simulator = CascadeSimulator(service_dependencies)

    # Find critical services
    critical = simulator.get_critical_services()

    print("Most critical services (most depended upon):")
    print(f"{'Rank':<6} {'Service':<25} {'Dependents':<12} {'Risk Level'}")
    print("-" * 60)

    for i, (service, count) in enumerate(critical, 1):
        if count >= 4:
            risk = "[CRITICAL] CRITICAL"
        elif count >= 2:
            risk = "[HIGH] HIGH"
        else:
            risk = "[MODERATE] MODERATE"

        print(f"{i:<6} {service:<25} {count:<12} {risk}")

    print()
    print("Recommendation: Monitor critical services closely and ensure:")
    print("  - High availability (redundancy, failover)")
    print("  - Circuit breakers to prevent cascade")
    print("  - Aggressive caching to reduce load")
    print()


def example_risk_detection():
    """Example 5: Cascade risk detection."""
    print("=" * 70)
    print("EXAMPLE 5: Cascade Risk Detection")
    print("=" * 70 + "\n")

    service_dependencies = {
        'service-a': ['service-b', 'service-c'],
        'service-b': ['service-d'],
        'service-c': ['service-d'],
        'service-d': [],
    }

    simulator = CascadeSimulator(service_dependencies)

    # Scenario 1: Normal conditions
    print("Scenario 1: Normal operating conditions")
    normal_metrics = {
        'service-a': {'latency_p95': 100, 'error_rate': 0.01, 'queue_depth': 10},
        'service-b': {'latency_p95': 150, 'error_rate': 0.02, 'queue_depth': 15},
        'service-c': {'latency_p95': 120, 'error_rate': 0.01, 'queue_depth': 12},
        'service-d': {'latency_p95': 80, 'error_rate': 0.01, 'queue_depth': 8},
    }

    risk, at_risk = simulator.detect_cascade_risk(normal_metrics)
    print(f"  Risk score: {risk:.2f} ({risk*100:.0f}%)")
    print(f"  At-risk services: {at_risk if at_risk else 'None'}")
    print(f"  Status: {'[OK] HEALTHY' if risk < 0.3 else '[WARNING] WARNING'}")
    print()

    # Scenario 2: Degraded conditions
    print("Scenario 2: Degraded conditions (service-d struggling)")
    degraded_metrics = {
        'service-a': {'latency_p95': 200, 'error_rate': 0.05, 'queue_depth': 50},
        'service-b': {'latency_p95': 400, 'error_rate': 0.08, 'queue_depth': 80},
        'service-c': {'latency_p95': 350, 'error_rate': 0.07, 'queue_depth': 70},
        'service-d': {'latency_p95': 1500, 'error_rate': 0.15, 'queue_depth': 150},
    }

    risk, at_risk = simulator.detect_cascade_risk(degraded_metrics)
    print(f"  Risk score: {risk:.2f} ({risk*100:.0f}%)")
    print(f"  At-risk services: {', '.join(at_risk)}")
    print(f"  Status: {'[CRITICAL] CRITICAL' if risk > 0.7 else '[HIGH] WARNING'}")
    print()

    # Scenario 3: Critical conditions
    print("Scenario 3: Critical conditions (cascade imminent)")
    critical_metrics = {
        'service-a': {'latency_p95': 3000, 'error_rate': 0.25, 'queue_depth': 200},
        'service-b': {'latency_p95': 2500, 'error_rate': 0.20, 'queue_depth': 180},
        'service-c': {'latency_p95': 2800, 'error_rate': 0.22, 'queue_depth': 190},
        'service-d': {'latency_p95': 5000, 'error_rate': 0.40, 'queue_depth': 300},
    }

    risk, at_risk = simulator.detect_cascade_risk(critical_metrics)
    print(f"  Risk score: {risk:.2f} ({risk*100:.0f}%)")
    print(f"  At-risk services: {', '.join(at_risk)}")
    print(f"  Status: [CRITICAL] CASCADE LIKELY")
    print(f"  Action: Enable circuit breakers, shed load, increase cache TTL")
    print()


def example_scenario_from_yaml():
    """Example 6: Load and inject scenario from YAML."""
    print("=" * 70)
    print("EXAMPLE 6: Pre-defined Scenarios from YAML")
    print("=" * 70 + "\n")

    # Load scenarios
    scenarios = FailureScenario.from_yaml('simulator/failures/scenarios.yaml')

    print(f"Loaded {len(scenarios)} pre-defined scenarios:\n")
    for i, scenario in enumerate(scenarios[:5], 1):
        print(f"{i}. {scenario.name}")
        print(f"   {scenario.description}")
        print(f"   Type: {scenario.failure_type}, Duration: {scenario.duration_seconds}s")
        print()

    # Create services and injector
    services = {
        'payment-service': BaseService(ServiceConfig(name='payment-service', port=8006)),
    }
    injector = FailureInjector(services)

    # Inject a scenario
    payment_scenario = next(s for s in scenarios if s.name == 'payment_gateway_slow')

    print(f"Injecting scenario: {payment_scenario.name}")
    injector.inject_scenario(payment_scenario)

    # Check results
    print(f"[OK] Scenario injected")
    print(f"[OK] Payment service latency: {services['payment-service']._latency_multiplier}x")
    print(f"[OK] Active failures: {len(injector.get_active_failures())}")

    # Cleanup
    injector.restore_all()
    print()


def example_network_partition():
    """Example 7: Network partition simulation."""
    print("=" * 70)
    print("EXAMPLE 7: Network Partition")
    print("=" * 70 + "\n")

    # Create services
    services = {
        'order-service': BaseService(ServiceConfig(name='order-service', port=8005)),
        'payment-service': BaseService(ServiceConfig(name='payment-service', port=8006)),
    }

    # Register payment service in order service
    services['order-service'].register_service('payment-service', 'http://localhost:8006')

    print("Initial state:")
    print(f"  order-service can reach: {list(services['order-service'].service_registry.keys())}")
    print()

    # Inject partition
    injector = FailureInjector(services)
    print("Injecting network partition: order-service cannot reach payment-service")
    injector.inject_network_partition('order-service', 'payment-service', duration=30)

    print(f"  order-service can reach: {list(services['order-service'].service_registry.keys())}")
    print(f"  [OK] payment-service is unreachable from order-service")
    print()

    print("Effect: Order service will fail when trying to process payments")
    print("Expected behavior: Should use fallback/queue orders for later processing")
    print()

    injector.restore_all()


def main():
    """Run all examples."""
    print("\n")
    print("=" + "=" * 68 + "=")
    print("|" + " " * 15 + "FAILURE INJECTION EXAMPLES" + " " * 27 + "|")
    print("=" + "=" * 68 + "╝")

    try:
        example_basic_injection()
        example_multiple_failures()
        example_cascade_simulation()
        example_critical_services()
        example_risk_detection()
        example_scenario_from_yaml()
        example_network_partition()

        print("=" * 70)
        print("[SUCCESS] ALL EXAMPLES COMPLETED")
        print("=" * 70)
        print()
        print("Key Takeaways:")
        print("  1. Easy to inject various failure types")
        print("  2. Can simulate realistic cascade failures")
        print("  3. Identify critical services that need monitoring")
        print("  4. Detect cascade risk before it happens")
        print("  5. Use pre-defined scenarios from YAML")
        print("  6. Test network partitions and fallback behavior")
        print()
        print("Use these techniques to:")
        print("  - Test cache resilience during failures")
        print("  - Validate circuit breaker logic")
        print("  - Ensure graceful degradation")
        print("  - Measure cache effectiveness during incidents")
        print()

    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


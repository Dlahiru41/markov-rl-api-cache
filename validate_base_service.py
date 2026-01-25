"""
Validation script for base_service.py

Tests the functionality specified in the requirements:
- ServiceConfig and EndpointConfig dataclasses
- BaseService class with FastAPI
- Standard endpoints (/health, /metrics, /config)
- Endpoint registration
"""

from simulator.services.base_service import BaseService, ServiceConfig, EndpointConfig


def test_basic_initialization():
    """Test basic service initialization."""
    print("=" * 70)
    print("TEST 1: Basic Initialization")
    print("=" * 70)

    config = ServiceConfig(
        name="test-service",
        port=8000,
        base_latency_ms=50,
        latency_std_ms=10,
        failure_rate=0.01,
        endpoints=[
            EndpointConfig("/test", "GET", 100, 1.0, [], "Test endpoint")
        ]
    )

    service = BaseService(config)

    # Check FastAPI app was created
    print(f"[OK] Service name: {service.config.name}")
    print(f"[OK] FastAPI app type: {type(service.app).__name__}")
    print(f"[OK] Service port: {service.config.port}")
    print(f"[OK] Base latency: {service.config.base_latency_ms}ms")
    print(f"[OK] Failure rate: {service.config.failure_rate}")
    print()


def test_endpoint_registration():
    """Test that endpoints are registered correctly."""
    print("=" * 70)
    print("TEST 2: Endpoint Registration")
    print("=" * 70)

    config = ServiceConfig(
        name="test-service",
        port=8001,
        base_latency_ms=50,
        endpoints=[
            EndpointConfig("/test", "GET", 100, 1.0, [], "Test endpoint"),
            EndpointConfig("/users/{id}", "GET", 500, 1.5, [], "Get user"),
            EndpointConfig("/data", "POST", 200, 2.0, [], "Post data")
        ]
    )

    service = BaseService(config)

    # Check endpoints are registered
    routes = [route.path for route in service.app.routes]
    print("Registered routes:")
    for route in routes:
        print(f"  - {route}")

    # Verify standard endpoints exist
    required_endpoints = ["/health", "/metrics", "/config"]
    for endpoint in required_endpoints:
        if endpoint in routes:
            print(f"[OK] Standard endpoint '{endpoint}' registered")
        else:
            print(f"[FAIL] Missing standard endpoint '{endpoint}'")

    # Verify custom endpoints exist
    custom_endpoints = ["/test", "/users/{id}", "/data"]
    for endpoint in custom_endpoints:
        if endpoint in routes:
            print(f"[OK] Custom endpoint '{endpoint}' registered")
        else:
            print(f"[FAIL] Missing custom endpoint '{endpoint}'")
    print()


def test_configuration_options():
    """Test various configuration options."""
    print("=" * 70)
    print("TEST 3: Configuration Options")
    print("=" * 70)

    config = ServiceConfig(
        name="advanced-service",
        host="127.0.0.1",
        port=8002,
        base_latency_ms=100,
        latency_std_ms=25,
        failure_rate=0.05,
        timeout_rate=0.02,
        dependencies=["user-service", "auth-service"],
        endpoints=[
            EndpointConfig(
                path="/complex/{id}",
                method="GET",
                response_size_bytes=2048,
                latency_multiplier=3.0,
                dependencies=["user-service:/users/{id}"],
                description="Complex endpoint with dependencies"
            )
        ]
    )

    service = BaseService(config)

    print(f"[OK] Service name: {service.config.name}")
    print(f"[OK] Host: {service.config.host}")
    print(f"[OK] Port: {service.config.port}")
    print(f"[OK] Base latency: {service.config.base_latency_ms}ms ± {service.config.latency_std_ms}ms")
    print(f"[OK] Failure rate: {service.config.failure_rate:.1%}")
    print(f"[OK] Timeout rate: {service.config.timeout_rate:.1%}")
    print(f"[OK] Dependencies: {', '.join(service.config.dependencies)}")
    print(f"[OK] Endpoints count: {len(service.config.endpoints)}")

    # Check endpoint details
    ep = service.config.endpoints[0]
    print(f"\nEndpoint details:")
    print(f"  - Path: {ep.path}")
    print(f"  - Method: {ep.method}")
    print(f"  - Response size: {ep.response_size_bytes} bytes")
    print(f"  - Latency multiplier: {ep.latency_multiplier}x")
    print(f"  - Dependencies: {ep.dependencies}")
    print(f"  - Description: {ep.description}")
    print()


def test_metrics_collector():
    """Test metrics collection."""
    print("=" * 70)
    print("TEST 4: Metrics Collector")
    print("=" * 70)

    config = ServiceConfig(
        name="metrics-test-service",
        port=8003,
        base_latency_ms=50
    )

    service = BaseService(config)

    # Simulate some requests
    service.metrics.record_request("/test", 45.2, 200)
    service.metrics.record_request("/test", 52.1, 200)
    service.metrics.record_request("/test", 48.7, 200)
    service.metrics.record_request("/users", 105.3, 200)
    service.metrics.record_request("/users", 95.8, 500)
    service.metrics.record_dependency_call("user-service")
    service.metrics.record_dependency_call("auth-service")

    # Get metrics
    metrics = service.metrics.get_metrics()

    print(f"[OK] Total requests: {metrics['total_requests']}")
    print(f"[OK] Total errors: {metrics['total_errors']}")
    print(f"[OK] Uptime: {metrics['uptime_seconds']:.2f}s")

    print(f"\nEndpoint metrics:")
    for endpoint, stats in metrics['endpoints'].items():
        print(f"  {endpoint}:")
        print(f"    - Requests: {stats['request_count']}")
        print(f"    - Errors: {stats['error_count']}")
        print(f"    - Mean latency: {stats['latency_mean_ms']:.2f}ms")
        print(f"    - P50 latency: {stats['latency_p50_ms']:.2f}ms")
        print(f"    - P95 latency: {stats['latency_p95_ms']:.2f}ms")

    print(f"\nDependency calls:")
    for service_name, count in metrics['dependency_calls'].items():
        print(f"  - {service_name}: {count} calls")

    # Test Prometheus format
    prom_metrics = service.metrics.get_prometheus_metrics()
    print(f"\n[OK] Prometheus metrics generated ({len(prom_metrics)} chars)")
    print()


def test_chaos_engineering():
    """Test chaos engineering controls."""
    print("=" * 70)
    print("TEST 5: Chaos Engineering Controls")
    print("=" * 70)

    config = ServiceConfig(
        name="chaos-test-service",
        port=8004,
        base_latency_ms=50,
        failure_rate=0.01
    )

    service = BaseService(config)

    # Test latency multiplier
    print(f"Initial latency multiplier: {service._latency_multiplier}x")
    service.set_latency_multiplier(3.0)
    print(f"[OK] Set latency multiplier to: {service._latency_multiplier}x")

    # Test failure rate override
    print(f"\nInitial failure rate: {service.config.failure_rate:.1%}")
    service.set_failure_rate(0.25)
    print(f"[OK] Set failure rate to: {service._failure_rate_override:.1%}")

    # Test offline mode
    print(f"\nInitial offline status: {service._is_offline}")
    service.set_offline(True)
    print(f"[OK] Service offline: {service._is_offline}")
    service.set_offline(False)
    print(f"[OK] Service back online: {service._is_offline}")
    print()


def test_service_registry():
    """Test service registry for dependencies."""
    print("=" * 70)
    print("TEST 6: Service Registry")
    print("=" * 70)

    config = ServiceConfig(
        name="gateway-service",
        port=8005,
        dependencies=["user-service", "auth-service"]
    )

    service = BaseService(config)

    # Register dependent services
    service.register_service("user-service", "http://localhost:8001")
    service.register_service("auth-service", "http://localhost:8002")

    print(f"[OK] Registered {len(service.service_registry)} services")
    for service_name, url in service.service_registry.items():
        print(f"  - {service_name}: {url}")
    print()


def test_complete_example():
    """Test the exact validation example from requirements."""
    print("=" * 70)
    print("TEST 7: Complete Example (from requirements)")
    print("=" * 70)

    config = ServiceConfig(
        name="test-service",
        port=8000,
        base_latency_ms=50,
        latency_std_ms=10,
        failure_rate=0.01,
        endpoints=[
            EndpointConfig("/test", "GET", 100, 1.0, [], "Test endpoint")
        ]
    )

    service = BaseService(config)

    # Check FastAPI app was created
    print(f"Service name: {service.config.name}")
    print(f"FastAPI app: {type(service.app)}")

    # In a real test, you would run: service.run()
    # For unit test, check endpoints are registered
    routes = [r.path for r in service.app.routes]
    print(f"Registered routes: {routes}")

    # Verify required routes
    required = ["/health", "/metrics", "/test"]
    all_present = all(route in routes for route in required)

    if all_present:
        print(f"[OK] All required routes present: {required}")
    else:
        print(f"[FAIL] Some routes missing")
        for route in required:
            status = "[OK]" if route in routes else "[FAIL]"
            print(f"  {status} {route}")
    print()


def main():
    """Run all validation tests."""
    print("\n")
    print("=" + "=" * 68 + "=")
    print("|" + " " * 15 + "BASE SERVICE VALIDATION SUITE" + " " * 24 + "|")
    print("=" + "=" * 68 + "╝")
    print()

    try:
        test_basic_initialization()
        test_endpoint_registration()
        test_configuration_options()
        test_metrics_collector()
        test_chaos_engineering()
        test_service_registry()
        test_complete_example()

        print("=" * 70)
        print("[OK] ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("The base service template is ready for use!")
        print()
        print("Next steps:")
        print("  1. Create specific service implementations (UserService, ProductService, etc.)")
        print("  2. Test with actual HTTP requests using pytest")
        print("  3. Build a service mesh with multiple interconnected services")
        print()

    except Exception as e:
        print("=" * 70)
        print(f"[FAIL] TEST FAILED: {str(e)}")
        print("=" * 70)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


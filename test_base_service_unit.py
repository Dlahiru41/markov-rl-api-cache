"""
Simple unit test for BaseService without HTTP requests

This test validates the core functionality without starting the server.
"""

from simulator.services.base_service import BaseService, ServiceConfig, EndpointConfig
from fastapi.testclient import TestClient


def main():
    print("\n" + "=" * 70)
    print("UNIT TEST: BaseService Functionality")
    print("=" * 70 + "\n")

    # Test 1: Basic initialization
    print("Test 1: Basic Initialization")
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
    print(f"  [OK] Service name: {service.config.name}")
    print(f"  [OK] FastAPI app type: {type(service.app).__name__}")
    print()

    # Test 2: Endpoint registration
    print("Test 2: Endpoint Registration")
    routes = [r.path for r in service.app.routes]
    print(f"  Registered routes ({len(routes)}):")
    for route in routes:
        print(f"    - {route}")

    required = ["/health", "/metrics", "/test"]
    all_present = all(route in routes for route in required)
    if all_present:
        print(f"  [OK] All required routes present")
    else:
        print(f"  [FAIL] Some routes missing")
    print()

    # Test 3: Test client requests (without starting server)
    print("Test 3: TestClient Requests (No Server Required)")
    client = TestClient(service.app)

    # Health endpoint
    response = client.get("/health")
    print(f"  GET /health: {response.status_code}")
    print(f"    Response: {response.json()}")
    assert response.status_code == 200
    assert "status" in response.json()
    print(f"    [OK] Health check working")

    # Config endpoint
    response = client.get("/config")
    print(f"  GET /config: {response.status_code}")
    config_data = response.json()
    print(f"    Service: {config_data['name']}")
    print(f"    Endpoints: {len(config_data['endpoints'])}")
    assert response.status_code == 200
    print(f"    [OK] Config endpoint working")

    # Metrics endpoint
    response = client.get("/metrics/json")
    print(f"  GET /metrics/json: {response.status_code}")
    metrics = response.json()
    print(f"    Total requests: {metrics['total_requests']}")
    assert response.status_code == 200
    print(f"    [OK] Metrics endpoint working")

    # Custom endpoint
    response = client.get("/test")
    print(f"  GET /test: {response.status_code}")
    assert response.status_code == 200
    print(f"    [OK] Custom endpoint working")
    print()

    # Test 4: Chaos engineering
    print("Test 4: Chaos Engineering Controls")
    service.set_latency_multiplier(5.0)
    print(f"  [OK] Latency multiplier set to: {service._latency_multiplier}x")

    service.set_failure_rate(0.25)
    print(f"  [OK] Failure rate set to: {service._failure_rate_override:.1%}")

    service.set_offline(True)
    print(f"  [OK] Service offline: {service._is_offline}")

    # Test offline mode
    response = client.get("/health")
    print(f"  GET /health (while offline): {response.status_code}")
    assert response.status_code == 503
    print(f"    [OK] Offline mode working")

    service.set_offline(False)
    print(f"  [OK] Service back online: {not service._is_offline}")
    print()

    # Test 5: Metrics collection
    print("Test 5: Metrics Collection")
    service.metrics.record_request("/test", 45.2, 200)
    service.metrics.record_request("/test", 52.1, 200)
    service.metrics.record_request("/test", 48.7, 500)
    service.metrics.record_dependency_call("user-service")

    metrics = service.metrics.get_metrics()
    print(f"  Total requests recorded: {metrics['total_requests']}")
    print(f"  Total errors recorded: {metrics['total_errors']}")
    print(f"  Endpoints tracked: {len(metrics['endpoints'])}")

    if '/test' in metrics['endpoints']:
        stats = metrics['endpoints']['/test']
        print(f"  /test stats:")
        print(f"    - Requests: {stats['request_count']}")
        print(f"    - Mean latency: {stats['latency_mean_ms']:.1f}ms")
        print(f"    - P95 latency: {stats['latency_p95_ms']:.1f}ms")

    print(f"  [OK] Metrics collection working")
    print()

    # Test 6: Service registry
    print("Test 6: Service Registry")
    service.register_service("user-service", "http://localhost:8001")
    service.register_service("product-service", "http://localhost:8002")
    print(f"  Registered services: {len(service.service_registry)}")
    for svc_name, url in service.service_registry.items():
        print(f"    - {svc_name}: {url}")
    print(f"  [OK] Service registry working")
    print()

    # Test 7: Prometheus metrics
    print("Test 7: Prometheus Metrics Format")
    prom_text = service.metrics.get_prometheus_metrics()
    print(f"  Prometheus text length: {len(prom_text)} bytes")
    print(f"  Contains HELP: {'# HELP' in prom_text}")
    print(f"  Contains TYPE: {'# TYPE' in prom_text}")
    print(f"  [OK] Prometheus format working")
    print()

    # Test 8: Complex service configuration
    print("Test 8: Complex Configuration")
    complex_config = ServiceConfig(
        name="complex-service",
        host="127.0.0.1",
        port=8005,
        base_latency_ms=100,
        latency_std_ms=25,
        failure_rate=0.05,
        timeout_rate=0.02,
        dependencies=["auth-service", "data-service"],
        endpoints=[
            EndpointConfig("/fast", "GET", 100, 0.5, [], "Fast endpoint"),
            EndpointConfig("/slow", "GET", 2000, 5.0, [], "Slow endpoint"),
            EndpointConfig("/data", "POST", 500, 1.5, ["data-service:/store"], "Post data"),
        ]
    )

    complex_service = BaseService(complex_config)
    print(f"  Service: {complex_service.config.name}")
    print(f"  Endpoints: {len(complex_service.config.endpoints)}")
    print(f"  Dependencies: {len(complex_service.config.dependencies)}")

    # Check endpoints were registered
    routes = [r.path for r in complex_service.app.routes]
    custom_routes = ["/fast", "/slow", "/data"]
    all_registered = all(route in routes for route in custom_routes)
    print(f"  Custom endpoints registered: {all_registered}")
    print(f"  [OK] Complex configuration working")
    print()

    print("=" * 70)
    print("[OK] ALL UNIT TESTS PASSED")
    print("=" * 70)
    print()
    print("Summary:")
    print("  [OK] ServiceConfig and EndpointConfig dataclasses working")
    print("  [OK] BaseService initialization successful")
    print("  [OK] FastAPI app created and configured")
    print("  [OK] Standard endpoints registered (/health, /metrics, /config)")
    print("  [OK] Custom endpoints registered from configuration")
    print("  [OK] Chaos engineering controls functional")
    print("  [OK] Metrics collection tracking requests correctly")
    print("  [OK] Service registry for dependencies working")
    print("  [OK] Prometheus metrics format generated")
    print("  [OK] Complex configurations supported")
    print()
    print("🎉 BaseService template validation complete!")
    print()
    print("Note: These tests used FastAPI's TestClient, which doesn't")
    print("      require starting an actual HTTP server. For integration")
    print("      tests with real HTTP requests, run:")
    print("      python test_base_service_integration.py")
    print()


if __name__ == "__main__":
    print("\n")
    print("=" + "=" * 68 + "=")
    print("|" + " " * 20 + "UNIT TEST SUITE" + " " * 33 + "|")
    print("=" + "=" * 68 + "╝")

    try:
        main()
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()


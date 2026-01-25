"""
Integration test: Start a service and make real HTTP requests

This test demonstrates:
1. Starting a BaseService in a background thread
2. Making actual HTTP requests to standard endpoints
3. Testing latency simulation
4. Testing chaos engineering controls
5. Checking metrics collection
"""

import time
import threading
import httpx
from simulator.services.base_service import BaseService, ServiceConfig, EndpointConfig


def test_service_with_real_requests():
    """Test the service with actual HTTP requests."""

    print("\n" + "=" * 70)
    print("INTEGRATION TEST: Real HTTP Requests")
    print("=" * 70 + "\n")

    # Create a test service
    config = ServiceConfig(
        name="integration-test-service",
        port=9001,  # Use a different port to avoid conflicts
        base_latency_ms=30,
        latency_std_ms=5,
        failure_rate=0.0,  # Start with no failures
        endpoints=[
            EndpointConfig("/test/fast", "GET", 100, 0.5, [], "Fast endpoint"),
            EndpointConfig("/test/slow", "GET", 1000, 3.0, [], "Slow endpoint"),
            EndpointConfig("/test/data", "POST", 500, 1.0, [], "Post data"),
        ]
    )

    service = BaseService(config)

    # Start service in background thread
    print("Starting service in background thread...")
    server_thread = threading.Thread(target=lambda: service.run(log_level="warning"))
    server_thread.daemon = True
    server_thread.start()

    # Wait for service to start
    time.sleep(2)
    print("[OK] Service started\n")

    base_url = f"http://localhost:{config.port}"

    try:
        # Test 1: Health endpoint
        print("Test 1: Health Check")
        response = httpx.get(f"{base_url}/health", timeout=5.0)
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.json()}")
        assert response.status_code == 200
        assert response.json()["status"] in ["healthy", "degraded"]
        print("  [OK] Passed\n")

        # Test 2: Config endpoint
        print("Test 2: Service Configuration")
        response = httpx.get(f"{base_url}/config", timeout=5.0)
        print(f"  Status: {response.status_code}")
        config_data = response.json()
        print(f"  Service name: {config_data['name']}")
        print(f"  Endpoints: {len(config_data['endpoints'])}")
        assert response.status_code == 200
        assert config_data["name"] == "integration-test-service"
        print("  [OK] Passed\n")

        # Test 3: Make requests to custom endpoints
        print("Test 3: Custom Endpoints")

        # Fast endpoint (should be quick)
        start = time.time()
        response = httpx.get(f"{base_url}/test/fast", timeout=5.0)
        latency = (time.time() - start) * 1000
        print(f"  /test/fast: {response.status_code} ({latency:.1f}ms)")
        assert response.status_code == 200

        # Slow endpoint (should be slower)
        start = time.time()
        response = httpx.get(f"{base_url}/test/slow", timeout=5.0)
        latency = (time.time() - start) * 1000
        print(f"  /test/slow: {response.status_code} ({latency:.1f}ms)")
        assert response.status_code == 200
        assert latency > 50  # Should be slower due to latency_multiplier=3.0

        # POST endpoint
        response = httpx.post(f"{base_url}/test/data", json={"test": "data"}, timeout=5.0)
        print(f"  /test/data (POST): {response.status_code}")
        assert response.status_code == 200
        print("  [OK] Passed\n")

        # Test 4: Metrics after requests
        print("Test 4: Metrics Collection")
        response = httpx.get(f"{base_url}/metrics/json", timeout=5.0)
        metrics = response.json()
        print(f"  Total requests: {metrics['total_requests']}")
        print(f"  Total errors: {metrics['total_errors']}")
        print(f"  Endpoints tracked: {len(metrics['endpoints'])}")
        assert metrics['total_requests'] >= 3  # At least our 3 test requests
        print("  [OK] Passed\n")

        # Test 5: Chaos engineering - Latency
        print("Test 5: Chaos Engineering - Latency Multiplier")
        # Set high latency
        response = httpx.post(f"{base_url}/chaos/latency", params={"multiplier": 10.0}, timeout=5.0)
        print(f"  Set latency multiplier: {response.json()}")

        # Test that requests are now slower
        start = time.time()
        response = httpx.get(f"{base_url}/test/fast", timeout=10.0)
        latency = (time.time() - start) * 1000
        print(f"  Request with 10x latency: {latency:.1f}ms")
        assert latency > 100  # Should be much slower now
        print("  [OK] Passed\n")

        # Test 6: Chaos engineering - Failure rate
        print("Test 6: Chaos Engineering - Failure Rate")
        # Set high failure rate
        response = httpx.post(f"{base_url}/chaos/failure-rate", params={"rate": 0.8}, timeout=5.0)
        print(f"  Set failure rate: {response.json()}")

        # Make multiple requests, some should fail
        failures = 0
        for i in range(10):
            try:
                response = httpx.get(f"{base_url}/test/fast", timeout=5.0)
                if response.status_code >= 400:
                    failures += 1
            except:
                failures += 1

        print(f"  Failures: {failures}/10 requests")
        assert failures > 0  # With 80% failure rate, we should see some failures
        print("  [OK] Passed\n")

        # Test 7: Prometheus metrics
        print("Test 7: Prometheus Metrics Format")
        response = httpx.get(f"{base_url}/metrics", timeout=5.0)
        prom_text = response.text
        print(f"  Prometheus metrics size: {len(prom_text)} bytes")
        assert "# HELP" in prom_text
        assert "# TYPE" in prom_text
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        print("  [OK] Passed\n")

        # Test 8: Service offline
        print("Test 8: Chaos Engineering - Offline Mode")
        # Take service offline
        response = httpx.post(f"{base_url}/chaos/offline", params={"offline": True}, timeout=5.0)
        print(f"  Set offline: {response.json()}")

        # Next request should fail with 503
        try:
            response = httpx.get(f"{base_url}/test/fast", timeout=5.0)
            assert response.status_code == 503
            print(f"  Request while offline: {response.status_code} (Service Unavailable)")
        except httpx.HTTPStatusError as e:
            print(f"  Request while offline: {e.response.status_code} (Service Unavailable)")

        # Bring back online
        response = httpx.post(f"{base_url}/chaos/offline", params={"offline": False}, timeout=5.0)
        print(f"  Set online: {response.json()}")

        # Should work now
        response = httpx.get(f"{base_url}/test/fast", timeout=5.0)
        assert response.status_code == 200
        print(f"  Request after coming back online: {response.status_code}")
        print("  [OK] Passed\n")

        print("=" * 70)
        print("[OK] ALL INTEGRATION TESTS PASSED")
        print("=" * 70)
        print()
        print("Summary:")
        print("  [OK] Service successfully started in background")
        print("  [OK] Standard endpoints working (/health, /metrics, /config)")
        print("  [OK] Custom endpoints registered and responding")
        print("  [OK] Latency simulation working (fast vs slow endpoints)")
        print("  [OK] Metrics collection tracking all requests")
        print("  [OK] Chaos engineering controls functional:")
        print("    - Latency multiplier increases response time")
        print("    - Failure rate injection working")
        print("    - Offline mode blocks requests")
        print("  [OK] Prometheus metrics format correct")
        print()
        print("🎉 The BaseService template is fully functional!")
        print()

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        print("Note: Service will stop when the script exits (daemon thread)")


if __name__ == "__main__":
    print("\n")
    print("=" + "=" * 68 + "=")
    print("|" + " " * 18 + "INTEGRATION TEST SUITE" + " " * 29 + "|")
    print("=" + "=" * 68 + "╝")

    test_service_with_real_requests()

    print("\nPress Ctrl+C to exit...")
    try:
        # Keep the main thread alive so the service keeps running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")


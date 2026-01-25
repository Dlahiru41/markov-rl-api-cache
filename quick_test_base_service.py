"""Simple quick test for base service."""
import sys
sys.path.insert(0, '.')

try:
    from simulator.services.base_service import BaseService, ServiceConfig, EndpointConfig
    print("[OK] Imports successful")

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
    print("[OK] ServiceConfig created")

    service = BaseService(config)
    print(f"[OK] BaseService initialized: {service.config.name}")
    print(f"[OK] FastAPI app type: {type(service.app).__name__}")

    routes = [r.path for r in service.app.routes]
    print(f"[OK] Routes registered: {len(routes)}")
    print(f"Routes: {routes}")

    print("\n[OK][OK][OK] ALL CHECKS PASSED [OK][OK][OK]")

except Exception as e:
    print(f"[FAIL] Error: {e}")
    import traceback
    traceback.print_exc()


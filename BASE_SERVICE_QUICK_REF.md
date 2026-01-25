# Base Service Template - Quick Reference

## Installation

```bash
pip install fastapi uvicorn httpx python-dotenv
```

## Basic Usage

```python
from simulator.services.base_service import BaseService, ServiceConfig, EndpointConfig

# Create service configuration
config = ServiceConfig(
    name="my-service",
    port=8001,
    base_latency_ms=50,
    latency_std_ms=10,
    failure_rate=0.01,
    endpoints=[
        EndpointConfig("/api/resource/{id}", "GET", 500, 1.0, [], "Get resource")
    ]
)

# Create and run service
service = BaseService(config)
service.run()  # Start on http://0.0.0.0:8001
```

## Standard Endpoints (Auto-generated)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check status |
| `/metrics` | GET | Prometheus metrics (text) |
| `/metrics/json` | GET | JSON metrics |
| `/config` | GET | Service configuration |
| `/chaos/latency` | POST | Set latency multiplier |
| `/chaos/failure-rate` | POST | Set failure rate |
| `/chaos/offline` | POST | Take offline/online |

## Configuration Options

### ServiceConfig Parameters

```python
ServiceConfig(
    name="service-name",           # Required: Service identifier
    port=8000,                      # Required: Port number
    host="0.0.0.0",                 # Optional: Bind address
    base_latency_ms=100.0,          # Average response time
    latency_std_ms=20.0,            # Latency variance
    failure_rate=0.0,               # Error probability (0.0-1.0)
    timeout_rate=0.0,               # Timeout probability (0.0-1.0)
    dependencies=[],                # List of service names
    endpoints=[]                    # List of EndpointConfig
)
```

### EndpointConfig Parameters

```python
EndpointConfig(
    path="/api/users/{id}",        # Required: URL pattern
    method="GET",                   # Required: HTTP method
    response_size_bytes=1024,      # Required: Response size
    latency_multiplier=1.0,        # Optional: Multiply base latency
    dependencies=[],               # Optional: Dependency endpoints
    description=""                 # Optional: Human description
)
```

## Service-to-Service Calls

```python
# Register dependent service
service.register_service("user-service", "http://localhost:8001")

# Make HTTP call
result = await service.call_service(
    service_name="user-service",
    endpoint="/users/123",
    method="GET",
    params={"include": "profile"}
)
```

## Chaos Engineering

```python
# Increase latency by 5x
service.set_latency_multiplier(5.0)

# Set 25% failure rate
service.set_failure_rate(0.25)

# Take service offline
service.set_offline(True)

# Bring back online
service.set_offline(False)
```

## Metrics Access

```python
# Get metrics programmatically
metrics = service.metrics.get_metrics()
print(f"Total requests: {metrics['total_requests']}")
print(f"Error rate: {metrics['total_errors'] / metrics['total_requests']:.2%}")

# Per-endpoint stats
for endpoint, stats in metrics['endpoints'].items():
    print(f"{endpoint}: {stats['latency_mean_ms']:.1f}ms (p95: {stats['latency_p95_ms']:.1f}ms)")

# Prometheus format
prom_text = service.metrics.get_prometheus_metrics()
```

## HTTP Client Usage

```python
import httpx

# Health check
response = httpx.get("http://localhost:8001/health")
print(response.json())  # {"status": "healthy", ...}

# Get metrics
response = httpx.get("http://localhost:8001/metrics/json")
metrics = response.json()

# Chaos control
httpx.post("http://localhost:8001/chaos/latency", params={"multiplier": 3.0})
httpx.post("http://localhost:8001/chaos/failure-rate", params={"rate": 0.20})
httpx.post("http://localhost:8001/chaos/offline", params={"offline": True})
```

## Custom Service Implementation

```python
from simulator.services.base_service import BaseService, ServiceConfig, endpoint

class UserService(BaseService):
    def __init__(self):
        config = ServiceConfig(name="user-service", port=8001)
        super().__init__(config)
        self._register_custom_endpoints()
    
    def _register_custom_endpoints(self):
        @self.app.get("/users/{user_id}")
        async def get_user(user_id: str):
            return {"id": user_id, "name": "John Doe"}
        
        @self.app.post("/users")
        async def create_user(user: dict):
            return {"id": "new-id", **user}

# Use it
service = UserService()
service.run()
```

## Multi-Service Setup

```python
import threading

# Create services
services = [
    BaseService(ServiceConfig(name="service-1", port=8001)),
    BaseService(ServiceConfig(name="service-2", port=8002)),
    BaseService(ServiceConfig(name="service-3", port=8003)),
]

# Run in separate threads
threads = []
for service in services:
    thread = threading.Thread(target=service.run)
    thread.daemon = True
    thread.start()
    threads.append(thread)

# Keep running
for thread in threads:
    thread.join()
```

## Realistic Configuration Examples

### Fast Service (Cache, Auth)
```python
ServiceConfig(
    name="cache-service",
    port=8001,
    base_latency_ms=10,
    latency_std_ms=3,
    failure_rate=0.001
)
```

### Medium Service (API, Business Logic)
```python
ServiceConfig(
    name="api-service",
    port=8002,
    base_latency_ms=50,
    latency_std_ms=15,
    failure_rate=0.01
)
```

### Slow Service (Database, ML)
```python
ServiceConfig(
    name="ml-service",
    port=8003,
    base_latency_ms=200,
    latency_std_ms=50,
    failure_rate=0.03
)
```

## Testing with pytest

```python
import pytest
from fastapi.testclient import TestClient
from simulator.services.base_service import BaseService, ServiceConfig

@pytest.fixture
def test_service():
    config = ServiceConfig(name="test", port=8000)
    return BaseService(config)

def test_health_endpoint(test_service):
    client = TestClient(test_service.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in ["healthy", "degraded"]

def test_metrics_endpoint(test_service):
    client = TestClient(test_service.app)
    response = client.get("/metrics/json")
    assert response.status_code == 200
    assert "total_requests" in response.json()
```

## Common Patterns

### Service with Database Dependency
```python
config = ServiceConfig(
    name="order-service",
    port=8003,
    base_latency_ms=100,
    latency_multiplier_by_endpoint={
        "/orders": 1.0,           # Read: fast
        "/orders/search": 3.0,    # Search: slow
        "/orders/report": 10.0    # Report: very slow
    }
)
```

### Service Mesh Configuration
```python
# Gateway service that routes to others
gateway = BaseService(ServiceConfig(name="gateway", port=8000))
gateway.register_service("users", "http://localhost:8001")
gateway.register_service("products", "http://localhost:8002")
gateway.register_service("orders", "http://localhost:8003")

# Gateway can now proxy requests
@gateway.app.get("/api/users/{id}")
async def get_user(id: str):
    return await gateway.call_service("users", f"/users/{id}")
```

## Performance Testing Setup

```python
# Service with realistic load characteristics
config = ServiceConfig(
    name="api-gateway",
    port=8000,
    base_latency_ms=20,
    latency_std_ms=5,
    failure_rate=0.005,
    timeout_rate=0.001,
    endpoints=[
        EndpointConfig("/fast", "GET", 100, 0.5, [], "Fast endpoint"),
        EndpointConfig("/medium", "GET", 500, 1.0, [], "Medium endpoint"),
        EndpointConfig("/slow", "GET", 2000, 5.0, [], "Slow endpoint"),
    ]
)

service = BaseService(config)

# Gradually increase load
for multiplier in [1.0, 2.0, 3.0, 5.0]:
    service.set_latency_multiplier(multiplier)
    # Run load test...
    print(f"Load test at {multiplier}x latency")
```

## Files Reference

- `simulator/services/base_service.py` - Main implementation
- `example_service_simulator.py` - Complete examples
- `validate_base_service.py` - Test suite
- `quick_test_base_service.py` - Quick validation
- `BASE_SERVICE_README.md` - Full documentation

## Key Features Summary

✅ **Automatic latency simulation** - Normal distribution with configurable mean/std  
✅ **Failure injection** - Random errors based on failure_rate  
✅ **Timeout simulation** - Configurable timeout probability  
✅ **Metrics collection** - Request count, latency percentiles, errors  
✅ **Prometheus export** - Standard metrics format  
✅ **Dependency tracking** - Service mesh support  
✅ **Chaos engineering** - Runtime control of latency/failures  
✅ **Health checks** - Standard /health endpoint  
✅ **Easy configuration** - Dataclass-based setup  
✅ **FastAPI powered** - Modern async Python web framework  

---

**Ready to use!** Start building your microservice simulator now.


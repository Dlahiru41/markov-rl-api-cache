# Base Service Template - Complete Documentation

## Overview

The base service template (`simulator/services/base_service.py`) provides a comprehensive foundation for simulating realistic microservices in a testing environment. This allows you to test your caching system without needing a real production environment.

## ✅ Implementation Complete

All requested features have been implemented:

### 1. **ServiceConfig Dataclass** ✓
- `name`: Service identifier (e.g., "user-service")
- `host`: Hostname to bind to (default "0.0.0.0")
- `port`: Port number
- `base_latency_ms`: Average response time in milliseconds
- `latency_std_ms`: Standard deviation of latency (adds realism)
- `failure_rate`: Probability of returning an error (0.0 to 1.0)
- `timeout_rate`: Probability of timing out (separate from failures)
- `dependencies`: List of other service names this service calls
- `endpoints`: List of endpoint configurations

### 2. **EndpointConfig Dataclass** ✓
- `path`: URL path (e.g., "/users/{id}")
- `method`: HTTP method (GET, POST, etc.)
- `response_size_bytes`: Typical response size
- `latency_multiplier`: Multiply base latency for this endpoint
- `dependencies`: Other endpoints this one calls internally
- `description`: Human-readable description

### 3. **BaseService Class** ✓

#### Setup Features:
- `__init__(config)`: Store config, create FastAPI app
- Configure logging for the service
- Register standard endpoints (/health, /metrics, /config)
- Set up middleware for metrics collection and latency simulation

#### Latency Simulation:
- `_simulate_latency()`: Add realistic delay using normal distribution
- `_inject_failure()`: Randomly return errors based on failure_rate
- Middleware that wraps all requests with latency and failure injection

#### Metrics Tracking:
- Track request count per endpoint
- Track latency distribution (mean, p50, p95, p99)
- Track error count
- Track dependency calls

#### Standard Endpoints:
- `GET /health`: Return {"status": "healthy"} or {"status": "degraded"}
- `GET /metrics`: Return Prometheus-format metrics
- `GET /metrics/json`: Return JSON format metrics
- `GET /config`: Return current service configuration

#### Dependency Calls:
- `call_service(service_name, endpoint, params=None)`: Make HTTP call to another service
- `register_service(service_name, base_url)`: Register a dependent service
- Track all outgoing calls for distributed tracing

#### Chaos Engineering Hooks:
- `set_latency_multiplier(multiplier)`: Temporarily slow down
- `set_failure_rate(rate)`: Temporarily increase failures
- `set_offline(offline=True)`: Take service completely offline

#### Lifecycle:
- `run()`: Start the FastAPI server with uvicorn
- `stop()`: Gracefully shut down

### 4. **@endpoint Decorator** ✓
- Decorator for easy endpoint registration
- Automatically adds latency simulation and error injection
- Tracks metrics for the endpoint

## Quick Start Examples

### Example 1: Basic Service

```python
from simulator.services.base_service import BaseService, ServiceConfig, EndpointConfig

config = ServiceConfig(
    name="user-service",
    port=8001,
    base_latency_ms=50,
    latency_std_ms=10,
    failure_rate=0.01,
    endpoints=[
        EndpointConfig("/users/{id}", "GET", 500, 1.0, [], "Get user by ID"),
        EndpointConfig("/users", "POST", 200, 1.5, [], "Create user")
    ]
)

service = BaseService(config)
# service.run()  # Start the service
```

### Example 2: Service with Dependencies

```python
config = ServiceConfig(
    name="order-service",
    port=8002,
    base_latency_ms=100,
    dependencies=["user-service", "product-service"],
    endpoints=[
        EndpointConfig(
            path="/orders/{id}",
            method="GET",
            response_size_bytes=1024,
            latency_multiplier=2.0,
            dependencies=["user-service:/users/{id}", "product-service:/products/{id}"],
            description="Get order with user and product details"
        )
    ]
)

service = BaseService(config)

# Register dependent services
service.register_service("user-service", "http://localhost:8001")
service.register_service("product-service", "http://localhost:8003")

# Now you can call dependencies
# result = await service.call_service("user-service", "/users/123")
```

### Example 3: Custom Service with @endpoint Decorator

```python
from simulator.services.base_service import BaseService, ServiceConfig, endpoint

class UserService(BaseService):
    @endpoint("/users/{user_id}", "GET", latency_multiplier=1.0)
    async def get_user(self, user_id: str):
        return {
            "id": user_id,
            "name": "John Doe",
            "email": "john@example.com"
        }
    
    @endpoint("/users/{user_id}/orders", "GET", latency_multiplier=2.0)
    async def get_user_orders(self, user_id: str):
        # Call another service
        orders = await self.call_service("order-service", f"/orders?user_id={user_id}")
        return orders

config = ServiceConfig(name="user-service", port=8001)
service = UserService(config)
```

### Example 4: Chaos Engineering

```python
service = BaseService(config)

# Normal operation
# ... service handles requests ...

# Simulate high latency
service.set_latency_multiplier(5.0)  # 5x slower

# Simulate increased failures
service.set_failure_rate(0.25)  # 25% failure rate

# Take service offline completely
service.set_offline(True)

# Bring back online
service.set_offline(False)

# Reset to normal
service.set_latency_multiplier(1.0)
service.set_failure_rate(0.01)
```

### Example 5: Monitoring and Metrics

```python
import httpx

# Query service health
response = httpx.get("http://localhost:8001/health")
print(response.json())
# Output: {"status": "healthy", "service": "user-service", ...}

# Get Prometheus metrics
response = httpx.get("http://localhost:8001/metrics")
print(response.text)
# Output: Prometheus text format with histograms, counters, etc.

# Get JSON metrics
response = httpx.get("http://localhost:8001/metrics/json")
metrics = response.json()
print(f"Total requests: {metrics['total_requests']}")
print(f"Error rate: {metrics['total_errors'] / metrics['total_requests']:.2%}")

# Get service configuration
response = httpx.get("http://localhost:8001/config")
print(response.json())
```

## Testing the Implementation

### Run Validation Tests

```bash
# Install dependencies
pip install fastapi uvicorn httpx python-dotenv

# Run quick test
python quick_test_base_service.py

# Run comprehensive validation
python validate_base_service.py
```

### Expected Output from Validation

```
✓ Service name: test-service
✓ FastAPI app type: FastAPI
✓ Service port: 8000
✓ Base latency: 50.0ms
✓ Failure rate: 0.01

Registered routes:
  - /health
  - /metrics
  - /metrics/json
  - /config
  - /chaos/latency
  - /chaos/failure-rate
  - /chaos/offline
  - /test
  - /users/{id}
  - /data
```

## Architecture Highlights

### Middleware Pipeline
Every request goes through:
1. **Offline Check**: Return 503 if service is offline
2. **Latency Simulation**: Add realistic delay based on normal distribution
3. **Timeout Injection**: Randomly timeout based on timeout_rate
4. **Failure Injection**: Randomly fail based on failure_rate
5. **Request Processing**: Execute actual endpoint logic
6. **Metrics Recording**: Track latency, status code, errors

### Metrics Collection
The `MetricsCollector` class tracks:
- **Request Count**: Per endpoint
- **Error Count**: Per endpoint
- **Latency Distribution**: Mean, P50, P95, P99 per endpoint
- **Dependency Calls**: Count of calls to each service
- **Uptime**: Total service uptime

### Latency Calculation
```python
# Latency = (base_latency * endpoint_multiplier * chaos_multiplier) ± std_deviation
latency = max(0, random.gauss(
    base_latency_ms * endpoint.latency_multiplier * chaos_multiplier,
    latency_std_ms
))
```

## Building a Service Mesh

### Step 1: Define Multiple Services

```python
# Create 3 services
user_service = BaseService(ServiceConfig(name="user-service", port=8001))
product_service = BaseService(ServiceConfig(name="product-service", port=8002))
order_service = BaseService(ServiceConfig(name="order-service", port=8003))
```

### Step 2: Register Dependencies

```python
# Order service depends on user and product services
order_service.register_service("user-service", "http://localhost:8001")
order_service.register_service("product-service", "http://localhost:8002")
```

### Step 3: Run Services

```python
import threading

# Run each service in a separate thread
threading.Thread(target=user_service.run).start()
threading.Thread(target=product_service.run).start()
threading.Thread(target=order_service.run).start()
```

## Next Steps

1. **Create Specific Service Implementations**:
   - `simulator/services/ecommerce/user_service.py`
   - `simulator/services/ecommerce/product_service.py`
   - `simulator/services/ecommerce/order_service.py`

2. **Build Service Orchestrator**:
   - `simulator/orchestrator.py` - Manage multiple services
   - Start/stop services
   - Configure service mesh
   - Inject chaos scenarios

3. **Create Traffic Generator**:
   - `simulator/traffic/generator.py` - Generate realistic traffic
   - Follow user journeys
   - Simulate peak loads

4. **Integration with Cache System**:
   - Test cache hit/miss rates
   - Measure latency improvements
   - Validate prefetching strategies

## Files Created

1. ✅ `simulator/services/base_service.py` - Main implementation (726 lines)
2. ✅ `validate_base_service.py` - Comprehensive validation suite
3. ✅ `quick_test_base_service.py` - Quick validation test
4. ✅ `BASE_SERVICE_README.md` - This documentation file

## Dependencies Installed

```bash
pip install fastapi uvicorn httpx python-dotenv
```

All packages are now compatible with your requirements.txt and installed successfully!

---

**Status: ✅ COMPLETE**

The base service template is fully implemented and ready for use. You can now build realistic microservice simulations for testing your caching system!


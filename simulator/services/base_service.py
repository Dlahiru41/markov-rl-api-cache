"""
Base Service Template for Microservice Simulation

This module provides a realistic microservice simulator that can be used to test
caching systems without a production environment. It includes:
- Realistic latency simulation with normal distribution
- Failure and timeout injection for chaos engineering
- Dependency tracking and service mesh simulation
- Prometheus-compatible metrics
- FastAPI-based HTTP endpoints

Example:
    config = ServiceConfig(
        name="user-service",
        port=8001,
        base_latency_ms=50,
        failure_rate=0.01,
        endpoints=[
            EndpointConfig("/users/{id}", "GET", 500, 1.0, [], "Get user by ID")
        ]
    )
    service = BaseService(config)
    service.run()
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from functools import wraps
from collections import defaultdict
import json

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import httpx


# ============================================================================
# Configuration Data Classes
# ============================================================================

@dataclass
class EndpointConfig:
    """Configuration for a single service endpoint.

    Attributes:
        path: URL path pattern (e.g., "/users/{id}")
        method: HTTP method (GET, POST, PUT, DELETE, etc.)
        response_size_bytes: Typical response size in bytes
        latency_multiplier: Multiply base service latency (e.g., 2.0 = twice as slow)
        dependencies: List of other endpoints this one calls (service_name:endpoint_path)
        description: Human-readable description of the endpoint
    """
    path: str
    method: str
    response_size_bytes: int
    latency_multiplier: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class ServiceConfig:
    """Configuration for a microservice.

    Attributes:
        name: Service identifier (e.g., "user-service")
        host: Hostname to bind to (default "0.0.0.0")
        port: Port number for the service
        base_latency_ms: Average response time in milliseconds
        latency_std_ms: Standard deviation of latency (adds realism)
        failure_rate: Probability of returning an error (0.0 to 1.0)
        timeout_rate: Probability of timing out (separate from failures)
        dependencies: List of other service names this service calls
        endpoints: List of endpoint configurations
    """
    name: str
    port: int
    host: str = "0.0.0.0"
    base_latency_ms: float = 100.0
    latency_std_ms: float = 20.0
    failure_rate: float = 0.0
    timeout_rate: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    endpoints: List[EndpointConfig] = field(default_factory=list)


# ============================================================================
# Metrics Collection
# ============================================================================

class MetricsCollector:
    """Collects and aggregates service metrics."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.request_count: Dict[str, int] = defaultdict(int)
        self.error_count: Dict[str, int] = defaultdict(int)
        self.latencies: Dict[str, List[float]] = defaultdict(list)
        self.dependency_calls: Dict[str, int] = defaultdict(int)
        self.start_time = time.time()

    def record_request(self, endpoint: str, latency_ms: float, status_code: int):
        """Record a completed request."""
        self.request_count[endpoint] += 1
        self.latencies[endpoint].append(latency_ms)
        if status_code >= 400:
            self.error_count[endpoint] += 1

    def record_dependency_call(self, service_name: str):
        """Record an outgoing dependency call."""
        self.dependency_calls[service_name] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        uptime = time.time() - self.start_time

        metrics = {
            "service": self.service_name,
            "uptime_seconds": uptime,
            "total_requests": sum(self.request_count.values()),
            "total_errors": sum(self.error_count.values()),
            "endpoints": {}
        }

        for endpoint in self.request_count:
            latencies = self.latencies[endpoint]
            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)

            metrics["endpoints"][endpoint] = {
                "request_count": self.request_count[endpoint],
                "error_count": self.error_count[endpoint],
                "error_rate": self.error_count[endpoint] / max(1, self.request_count[endpoint]),
                "latency_mean_ms": sum(latencies) / n if n > 0 else 0,
                "latency_p50_ms": sorted_latencies[n // 2] if n > 0 else 0,
                "latency_p95_ms": sorted_latencies[int(n * 0.95)] if n > 0 else 0,
                "latency_p99_ms": sorted_latencies[int(n * 0.99)] if n > 0 else 0,
            }

        metrics["dependency_calls"] = dict(self.dependency_calls)

        return metrics

    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus text format."""
        lines = []

        # Request count
        lines.append(f"# HELP {self.service_name}_requests_total Total number of requests")
        lines.append(f"# TYPE {self.service_name}_requests_total counter")
        for endpoint, count in self.request_count.items():
            safe_endpoint = endpoint.replace("/", "_").replace("{", "").replace("}", "")
            lines.append(f'{self.service_name}_requests_total{{endpoint="{endpoint}"}} {count}')

        # Error count
        lines.append(f"# HELP {self.service_name}_errors_total Total number of errors")
        lines.append(f"# TYPE {self.service_name}_errors_total counter")
        for endpoint, count in self.error_count.items():
            lines.append(f'{self.service_name}_errors_total{{endpoint="{endpoint}"}} {count}')

        # Latency histograms
        lines.append(f"# HELP {self.service_name}_latency_ms Request latency in milliseconds")
        lines.append(f"# TYPE {self.service_name}_latency_ms histogram")
        for endpoint, latencies in self.latencies.items():
            if latencies:
                sorted_latencies = sorted(latencies)
                n = len(sorted_latencies)
                p50 = sorted_latencies[n // 2]
                p95 = sorted_latencies[int(n * 0.95)]
                p99 = sorted_latencies[int(n * 0.99)]
                lines.append(f'{self.service_name}_latency_ms{{endpoint="{endpoint}",quantile="0.5"}} {p50}')
                lines.append(f'{self.service_name}_latency_ms{{endpoint="{endpoint}",quantile="0.95"}} {p95}')
                lines.append(f'{self.service_name}_latency_ms{{endpoint="{endpoint}",quantile="0.99"}} {p99}')

        # Dependency calls
        lines.append(f"# HELP {self.service_name}_dependency_calls_total Total dependency calls")
        lines.append(f"# TYPE {self.service_name}_dependency_calls_total counter")
        for service, count in self.dependency_calls.items():
            lines.append(f'{self.service_name}_dependency_calls_total{{service="{service}"}} {count}')

        return "\n".join(lines)


# ============================================================================
# Base Service Implementation
# ============================================================================

class BaseService:
    """Base class for simulated microservices.

    Provides realistic behavior including:
    - Latency simulation with configurable distribution
    - Random failures and timeouts
    - Dependency tracking
    - Metrics collection (Prometheus-compatible)
    - Standard health and config endpoints
    - Chaos engineering controls

    Subclasses should use the @endpoint decorator to register custom endpoints.
    """

    def __init__(self, config: ServiceConfig):
        """Initialize the service with configuration.

        Args:
            config: ServiceConfig instance with all service parameters
        """
        self.config = config
        self.app = FastAPI(title=config.name, version="1.0.0")
        self.metrics = MetricsCollector(config.name)
        self.logger = self._setup_logging()

        # Chaos engineering controls
        self._latency_multiplier = 1.0
        self._failure_rate_override: Optional[float] = None
        self._is_offline = False

        # HTTP client for dependency calls
        self.http_client: Optional[httpx.AsyncClient] = None

        # Service registry (for dependency resolution)
        self.service_registry: Dict[str, str] = {}  # service_name -> base_url

        # Register middleware and standard endpoints
        self._setup_middleware()
        self._register_standard_endpoints()
        self._register_configured_endpoints()

        self.logger.info(f"Service '{config.name}' initialized on {config.host}:{config.port}")

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the service."""
        logger = logging.getLogger(f"service.{self.config.name}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {self.config.name} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _setup_middleware(self):
        """Set up middleware for latency simulation and metrics."""

        @self.app.middleware("http")
        async def process_request(request: Request, call_next):
            """Middleware to add latency simulation and metrics collection."""

            # Check if service is offline
            if self._is_offline:
                return JSONResponse(
                    status_code=503,
                    content={"error": "Service temporarily unavailable"}
                )

            start_time = time.time()
            endpoint = request.url.path

            try:
                # Simulate latency before processing
                await self._simulate_latency(endpoint)

                # Check for injected timeout
                if self._should_timeout():
                    await asyncio.sleep(30)  # Simulate timeout
                    return JSONResponse(
                        status_code=504,
                        content={"error": "Gateway timeout"}
                    )

                # Check for injected failure
                failure_response = self._inject_failure()
                if failure_response:
                    latency_ms = (time.time() - start_time) * 1000
                    self.metrics.record_request(endpoint, latency_ms, failure_response.status_code)
                    return failure_response

                # Process the actual request
                response = await call_next(request)

                # Record metrics
                latency_ms = (time.time() - start_time) * 1000
                self.metrics.record_request(endpoint, latency_ms, response.status_code)

                self.logger.debug(
                    f"{request.method} {endpoint} -> {response.status_code} ({latency_ms:.2f}ms)"
                )

                return response

            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                self.metrics.record_request(endpoint, latency_ms, 500)
                self.logger.error(f"Error processing {endpoint}: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={"error": "Internal server error", "detail": str(e)}
                )

    async def _simulate_latency(self, endpoint: str):
        """Simulate realistic latency using normal distribution.

        Args:
            endpoint: The endpoint path to determine latency multiplier
        """
        # Find endpoint config to get latency multiplier
        latency_multiplier = 1.0
        for ep_config in self.config.endpoints:
            if ep_config.path in endpoint or endpoint in ep_config.path:
                latency_multiplier = ep_config.latency_multiplier
                break

        # Calculate latency with normal distribution
        base_latency = self.config.base_latency_ms * latency_multiplier * self._latency_multiplier
        latency = max(0, random.gauss(base_latency, self.config.latency_std_ms))

        # Convert to seconds and sleep
        await asyncio.sleep(latency / 1000.0)

    def _should_timeout(self) -> bool:
        """Determine if this request should timeout."""
        return random.random() < self.config.timeout_rate

    def _inject_failure(self) -> Optional[JSONResponse]:
        """Randomly inject failures based on failure rate.

        Returns:
            JSONResponse with error if failure should be injected, None otherwise
        """
        failure_rate = (
            self._failure_rate_override
            if self._failure_rate_override is not None
            else self.config.failure_rate
        )

        if random.random() < failure_rate:
            # Choose a random error type
            error_types = [
                (500, "Internal server error"),
                (503, "Service unavailable"),
                (429, "Too many requests"),
            ]
            status_code, message = random.choice(error_types)

            self.logger.warning(f"Injecting failure: {status_code} - {message}")

            return JSONResponse(
                status_code=status_code,
                content={"error": message, "injected": True}
            )

        return None

    def _register_standard_endpoints(self):
        """Register standard endpoints (health, metrics, config)."""

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            # Determine health status based on recent error rate
            metrics = self.metrics.get_metrics()
            total_requests = metrics["total_requests"]
            total_errors = metrics["total_errors"]

            if total_requests > 0:
                error_rate = total_errors / total_requests
                status = "healthy" if error_rate < 0.1 else "degraded"
            else:
                status = "healthy"

            return {
                "status": status,
                "service": self.config.name,
                "uptime_seconds": metrics["uptime_seconds"],
                "total_requests": total_requests,
                "error_rate": total_errors / max(1, total_requests)
            }

        @self.app.get("/metrics")
        async def metrics():
            """Prometheus-compatible metrics endpoint."""
            prometheus_text = self.metrics.get_prometheus_metrics()
            return Response(content=prometheus_text, media_type="text/plain")

        @self.app.get("/metrics/json")
        async def metrics_json():
            """JSON format metrics endpoint."""
            return self.metrics.get_metrics()

        @self.app.get("/config")
        async def config():
            """Return current service configuration."""
            return {
                "name": self.config.name,
                "host": self.config.host,
                "port": self.config.port,
                "base_latency_ms": self.config.base_latency_ms,
                "latency_std_ms": self.config.latency_std_ms,
                "failure_rate": self.config.failure_rate,
                "timeout_rate": self.config.timeout_rate,
                "dependencies": self.config.dependencies,
                "endpoints": [
                    {
                        "path": ep.path,
                        "method": ep.method,
                        "description": ep.description
                    }
                    for ep in self.config.endpoints
                ],
                "chaos_controls": {
                    "latency_multiplier": self._latency_multiplier,
                    "failure_rate_override": self._failure_rate_override,
                    "is_offline": self._is_offline
                }
            }

        # Chaos engineering control endpoints
        @self.app.post("/chaos/latency")
        async def set_latency(multiplier: float):
            """Temporarily adjust latency multiplier."""
            self.set_latency_multiplier(multiplier)
            return {"latency_multiplier": self._latency_multiplier}

        @self.app.post("/chaos/failure-rate")
        async def set_failure(rate: float):
            """Temporarily adjust failure rate."""
            self.set_failure_rate(rate)
            return {"failure_rate": self._failure_rate_override or self.config.failure_rate}

        @self.app.post("/chaos/offline")
        async def set_offline(offline: bool = True):
            """Take service offline or bring it back online."""
            self.set_offline(offline)
            return {"offline": self._is_offline}

    def _register_configured_endpoints(self):
        """Register endpoints defined in configuration."""
        for ep_config in self.config.endpoints:
            # Create a simple default handler
            async def default_handler(ep=ep_config):
                """Default handler that returns mock data."""
                return {
                    "endpoint": ep.path,
                    "method": ep.method,
                    "description": ep.description,
                    "data": "Mock response data",
                    "size_bytes": ep.response_size_bytes
                }

            # Register the endpoint
            if ep_config.method.upper() == "GET":
                self.app.get(ep_config.path)(default_handler)
            elif ep_config.method.upper() == "POST":
                self.app.post(ep_config.path)(default_handler)
            elif ep_config.method.upper() == "PUT":
                self.app.put(ep_config.path)(default_handler)
            elif ep_config.method.upper() == "DELETE":
                self.app.delete(ep_config.path)(default_handler)

    # ========================================================================
    # Dependency Management
    # ========================================================================

    def register_service(self, service_name: str, base_url: str):
        """Register a dependent service's location.

        Args:
            service_name: Name of the service
            base_url: Base URL (e.g., "http://localhost:8001")
        """
        self.service_registry[service_name] = base_url
        self.logger.info(f"Registered service '{service_name}' at {base_url}")

    async def call_service(
        self,
        service_name: str,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make an HTTP call to another service.

        Args:
            service_name: Name of the service to call
            endpoint: Endpoint path (e.g., "/users/123")
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            json_data: JSON body data

        Returns:
            Response data as dictionary

        Raises:
            HTTPException: If service is not registered or call fails
        """
        if service_name not in self.service_registry:
            raise HTTPException(
                status_code=503,
                detail=f"Service '{service_name}' not registered"
            )

        base_url = self.service_registry[service_name]
        url = f"{base_url}{endpoint}"

        # Record the dependency call
        self.metrics.record_dependency_call(service_name)
        self.logger.debug(f"Calling {service_name}: {method} {url}")

        # Create client if needed
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=30.0)

        try:
            if method.upper() == "GET":
                response = await self.http_client.get(url, params=params)
            elif method.upper() == "POST":
                response = await self.http_client.post(url, params=params, json=json_data)
            elif method.upper() == "PUT":
                response = await self.http_client.put(url, params=params, json=json_data)
            elif method.upper() == "DELETE":
                response = await self.http_client.delete(url, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error calling {service_name}: {e.response.status_code}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Dependency call failed: {str(e)}"
            )
        except Exception as e:
            self.logger.error(f"Error calling {service_name}: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail=f"Service call failed: {str(e)}"
            )

    # ========================================================================
    # Chaos Engineering Controls
    # ========================================================================

    def set_latency_multiplier(self, multiplier: float):
        """Temporarily increase or decrease latency.

        Args:
            multiplier: Factor to multiply base latency by (e.g., 2.0 = twice as slow)
        """
        self._latency_multiplier = multiplier
        self.logger.info(f"Latency multiplier set to {multiplier}x")

    def set_failure_rate(self, rate: float):
        """Temporarily adjust failure rate.

        Args:
            rate: Failure probability (0.0 to 1.0), or None to use config default
        """
        if rate < 0.0 or rate > 1.0:
            raise ValueError("Failure rate must be between 0.0 and 1.0")
        self._failure_rate_override = rate
        self.logger.info(f"Failure rate set to {rate:.2%}")

    def set_offline(self, offline: bool = True):
        """Take service completely offline or bring it back online.

        Args:
            offline: True to take offline, False to bring back online
        """
        self._is_offline = offline
        status = "offline" if offline else "online"
        self.logger.warning(f"Service set to {status}")

    # ========================================================================
    # Lifecycle Management
    # ========================================================================

    def run(self, **kwargs):
        """Start the FastAPI server with uvicorn.

        Args:
            **kwargs: Additional arguments passed to uvicorn.run()
        """
        self.logger.info(f"Starting service '{self.config.name}'...")

        # Set default log_level if not provided in kwargs
        if 'log_level' not in kwargs:
            kwargs['log_level'] = 'info'

        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            **kwargs
        )

    async def stop(self):
        """Gracefully shut down the service."""
        self.logger.info(f"Shutting down service '{self.config.name}'...")

        # Close HTTP client
        if self.http_client:
            await self.http_client.aclose()

        self.logger.info("Service shutdown complete")


# ============================================================================
# Decorator for Easy Endpoint Registration
# ============================================================================

def endpoint(
    path: str,
    method: str = "GET",
    latency_multiplier: float = 1.0,
    response_size_bytes: int = 1024,
    description: str = ""
):
    """Decorator for registering custom service endpoints with automatic instrumentation.

    Args:
        path: URL path pattern (e.g., "/users/{id}")
        method: HTTP method
        latency_multiplier: Multiply base service latency
        response_size_bytes: Typical response size
        description: Human-readable description

    Example:
        class UserService(BaseService):
            @endpoint("/users/{user_id}", "GET", latency_multiplier=1.5)
            async def get_user(self, user_id: str):
                return {"id": user_id, "name": "John Doe"}
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Function is already wrapped by middleware, just call it
            return await func(self, *args, **kwargs)

        # Store metadata for service initialization
        wrapper._endpoint_config = EndpointConfig(
            path=path,
            method=method,
            response_size_bytes=response_size_bytes,
            latency_multiplier=latency_multiplier,
            description=description
        )

        return wrapper

    return decorator


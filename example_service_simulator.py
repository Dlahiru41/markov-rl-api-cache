"""
Example: Complete Microservice Simulation

This demonstrates how to use the BaseService template to create
a realistic e-commerce microservice environment for testing.
"""

import asyncio
from simulator.services.base_service import BaseService, ServiceConfig, EndpointConfig


def create_user_service():
    """Create a user service with typical endpoints."""
    config = ServiceConfig(
        name="user-service",
        port=8001,
        base_latency_ms=30,
        latency_std_ms=10,
        failure_rate=0.01,
        endpoints=[
            EndpointConfig(
                path="/users/{id}",
                method="GET",
                response_size_bytes=512,
                latency_multiplier=1.0,
                description="Get user by ID"
            ),
            EndpointConfig(
                path="/users",
                method="POST",
                response_size_bytes=256,
                latency_multiplier=1.5,
                description="Create new user"
            ),
            EndpointConfig(
                path="/users/{id}/profile",
                method="GET",
                response_size_bytes=2048,
                latency_multiplier=1.2,
                description="Get user profile with details"
            ),
        ]
    )
    return BaseService(config)


def create_product_service():
    """Create a product catalog service."""
    config = ServiceConfig(
        name="product-service",
        port=8002,
        base_latency_ms=50,
        latency_std_ms=15,
        failure_rate=0.02,
        endpoints=[
            EndpointConfig(
                path="/products/{id}",
                method="GET",
                response_size_bytes=1024,
                latency_multiplier=1.0,
                description="Get product details"
            ),
            EndpointConfig(
                path="/products/search",
                method="GET",
                response_size_bytes=4096,
                latency_multiplier=3.0,
                description="Search products (slow operation)"
            ),
            EndpointConfig(
                path="/products/{id}/reviews",
                method="GET",
                response_size_bytes=8192,
                latency_multiplier=2.5,
                description="Get product reviews"
            ),
            EndpointConfig(
                path="/products/{id}/inventory",
                method="GET",
                response_size_bytes=256,
                latency_multiplier=0.8,
                description="Check product inventory"
            ),
        ]
    )
    return BaseService(config)


def create_order_service():
    """Create an order processing service with dependencies."""
    config = ServiceConfig(
        name="order-service",
        port=8003,
        base_latency_ms=100,
        latency_std_ms=25,
        failure_rate=0.015,
        dependencies=["user-service", "product-service"],
        endpoints=[
            EndpointConfig(
                path="/orders/{id}",
                method="GET",
                response_size_bytes=2048,
                latency_multiplier=2.0,
                dependencies=[
                    "user-service:/users/{user_id}",
                    "product-service:/products/{product_id}"
                ],
                description="Get order with full details"
            ),
            EndpointConfig(
                path="/orders",
                method="POST",
                response_size_bytes=512,
                latency_multiplier=3.0,
                dependencies=[
                    "user-service:/users/{user_id}",
                    "product-service:/products/{product_id}/inventory"
                ],
                description="Create new order"
            ),
            EndpointConfig(
                path="/orders/user/{user_id}",
                method="GET",
                response_size_bytes=4096,
                latency_multiplier=2.5,
                description="Get all orders for a user"
            ),
        ]
    )
    return BaseService(config)


def create_recommendation_service():
    """Create a recommendation service (ML-based, slower)."""
    config = ServiceConfig(
        name="recommendation-service",
        port=8004,
        base_latency_ms=200,
        latency_std_ms=50,
        failure_rate=0.03,
        dependencies=["user-service", "product-service"],
        endpoints=[
            EndpointConfig(
                path="/recommendations/{user_id}",
                method="GET",
                response_size_bytes=4096,
                latency_multiplier=5.0,
                dependencies=[
                    "user-service:/users/{user_id}/profile",
                    "product-service:/products/search"
                ],
                description="Get personalized recommendations (ML inference)"
            ),
            EndpointConfig(
                path="/recommendations/trending",
                method="GET",
                response_size_bytes=2048,
                latency_multiplier=3.0,
                description="Get trending products"
            ),
        ]
    )
    return BaseService(config)


async def demonstrate_service_calls():
    """Demonstrate making service calls between microservices."""
    print("\n" + "=" * 70)
    print("DEMONSTRATING SERVICE-TO-SERVICE CALLS")
    print("=" * 70 + "\n")

    # Create services
    user_service = create_user_service()
    product_service = create_product_service()
    order_service = create_order_service()

    # Register dependencies in order service
    order_service.register_service("user-service", "http://localhost:8001")
    order_service.register_service("product-service", "http://localhost:8002")

    print("[OK] Services created and dependencies registered")
    print(f"  - {user_service.config.name} on port {user_service.config.port}")
    print(f"  - {product_service.config.name} on port {product_service.config.port}")
    print(f"  - {order_service.config.name} on port {order_service.config.port}")
    print()

    # Note: In real usage, you would start these services in separate processes
    # For this demo, we just show the configuration
    print("To run these services in production mode:")
    print("  1. Start each service in a separate terminal:")
    print("     python -c 'from example_service_simulator import create_user_service; create_user_service().run()'")
    print("  2. Or use the orchestrator to manage them all")
    print()


def demonstrate_chaos_engineering():
    """Demonstrate chaos engineering capabilities."""
    print("\n" + "=" * 70)
    print("DEMONSTRATING CHAOS ENGINEERING")
    print("=" * 70 + "\n")

    service = create_product_service()

    print(f"Service: {service.config.name}")
    print(f"Initial configuration:")
    print(f"  - Base latency: {service.config.base_latency_ms}ms")
    print(f"  - Failure rate: {service.config.failure_rate:.1%}")
    print(f"  - Status: Online")
    print()

    # Simulate different scenarios
    scenarios = [
        {
            "name": "High Latency Scenario",
            "description": "Simulate database slowdown",
            "actions": lambda s: s.set_latency_multiplier(5.0)
        },
        {
            "name": "Increased Failures",
            "description": "Simulate overload",
            "actions": lambda s: s.set_failure_rate(0.20)
        },
        {
            "name": "Service Outage",
            "description": "Complete service failure",
            "actions": lambda s: s.set_offline(True)
        },
    ]

    for scenario in scenarios:
        print(f"Scenario: {scenario['name']}")
        print(f"  Description: {scenario['description']}")
        scenario['actions'](service)
        print(f"  Status: Applied [OK]")
        print()

    # Reset
    service.set_latency_multiplier(1.0)
    service.set_failure_rate(0.01)
    service.set_offline(False)
    print("[OK] Service reset to normal operation")
    print()


def demonstrate_metrics():
    """Demonstrate metrics collection."""
    print("\n" + "=" * 70)
    print("DEMONSTRATING METRICS COLLECTION")
    print("=" * 70 + "\n")

    service = create_user_service()

    # Simulate some requests
    print("Simulating requests...")
    service.metrics.record_request("/users/123", 45.2, 200)
    service.metrics.record_request("/users/123", 52.1, 200)
    service.metrics.record_request("/users/456", 48.7, 200)
    service.metrics.record_request("/users/789", 105.3, 404)
    service.metrics.record_request("/users/profile", 95.8, 200)
    service.metrics.record_dependency_call("auth-service")
    service.metrics.record_dependency_call("auth-service")
    print()

    # Get metrics
    metrics = service.metrics.get_metrics()

    print(f"Service: {metrics['service']}")
    print(f"Uptime: {metrics['uptime_seconds']:.2f}s")
    print(f"Total Requests: {metrics['total_requests']}")
    print(f"Total Errors: {metrics['total_errors']}")
    print()

    print("Endpoint Breakdown:")
    for endpoint, stats in metrics['endpoints'].items():
        print(f"  {endpoint}:")
        print(f"    Requests: {stats['request_count']}")
        print(f"    Errors: {stats['error_count']} ({stats['error_rate']:.1%})")
        print(f"    Latency: {stats['latency_mean_ms']:.1f}ms (p95: {stats['latency_p95_ms']:.1f}ms)")
    print()

    print("Dependency Calls:")
    for dep_service, count in metrics['dependency_calls'].items():
        print(f"  {dep_service}: {count} calls")
    print()

    # Show Prometheus format sample
    prom_metrics = service.metrics.get_prometheus_metrics()
    print("Prometheus Metrics (sample):")
    print("-" * 70)
    for line in prom_metrics.split('\n')[:10]:
        print(f"  {line}")
    print(f"  ... ({len(prom_metrics)} total characters)")
    print()


def show_service_summary():
    """Show summary of all created services."""
    print("\n" + "=" * 70)
    print("SERVICE SUMMARY")
    print("=" * 70 + "\n")

    services = [
        create_user_service(),
        create_product_service(),
        create_order_service(),
        create_recommendation_service(),
    ]

    print(f"Total Services: {len(services)}\n")

    for service in services:
        print(f"📦 {service.config.name}")
        print(f"   Port: {service.config.port}")
        print(f"   Base Latency: {service.config.base_latency_ms}ms (±{service.config.latency_std_ms}ms)")
        print(f"   Failure Rate: {service.config.failure_rate:.1%}")
        print(f"   Endpoints: {len(service.config.endpoints)}")
        if service.config.dependencies:
            print(f"   Dependencies: {', '.join(service.config.dependencies)}")
        print()


def main():
    """Run all demonstrations."""
    print("\n")
    print("=" + "=" * 68 + "=")
    print("|" + " " * 15 + "MICROSERVICE SIMULATOR DEMO" + " " * 26 + "|")
    print("=" + "=" * 68 + "╝")

    # Run demonstrations
    show_service_summary()
    demonstrate_metrics()
    demonstrate_chaos_engineering()
    asyncio.run(demonstrate_service_calls())

    print("=" * 70)
    print("[OK] DEMONSTRATION COMPLETE")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("  1. Easy to create realistic microservices with BaseService")
    print("  2. Built-in latency simulation with normal distribution")
    print("  3. Automatic metrics collection (Prometheus-compatible)")
    print("  4. Chaos engineering controls for testing resilience")
    print("  5. Service-to-service dependency tracking")
    print()
    print("Next Steps:")
    print("  1. Start services: service.run()")
    print("  2. Generate traffic with simulator/traffic/generator.py")
    print("  3. Test your caching system against realistic workloads")
    print("  4. Monitor metrics and optimize cache strategies")
    print()


if __name__ == "__main__":
    main()


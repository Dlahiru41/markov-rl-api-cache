"""
Traffic Generator Runner - For Docker Container

Runs the traffic generator with the specified profile.
Profile is set via TRAFFIC_PROFILE environment variable.
"""

import os
import sys
import asyncio
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.traffic.generator import TrafficGenerator, TrafficProfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """Main entry point for traffic generator."""

    # Get profile from environment
    profile_name = os.environ.get('TRAFFIC_PROFILE', 'normal')
    profile_path = f'simulator/traffic/profiles/{profile_name}.yaml'

    logger.info(f"Starting traffic generator with profile: {profile_name}")

    # Load profile
    try:
        profile = TrafficProfile.from_yaml(profile_path)
        logger.info(f"Loaded profile: {profile.name}")
        logger.info(f"  - Target RPS: {profile.requests_per_second}")
        logger.info(f"  - Duration: {profile.duration_seconds}s")
    except Exception as e:
        logger.error(f"Failed to load profile: {e}")
        sys.exit(1)

    # Get service URLs from environment
    service_urls = {
        'user': os.environ.get('USER_SERVICE_URL', 'http://user-service:8001'),
        'auth': os.environ.get('AUTH_SERVICE_URL', 'http://auth-service:8002'),
        'product': os.environ.get('PRODUCT_SERVICE_URL', 'http://product-service:8003'),
        'cart': os.environ.get('CART_SERVICE_URL', 'http://cart-service:8004'),
        'order': os.environ.get('ORDER_SERVICE_URL', 'http://order-service:8005'),
        'payment': os.environ.get('PAYMENT_SERVICE_URL', 'http://payment-service:8006'),
        'inventory': os.environ.get('INVENTORY_SERVICE_URL', 'http://inventory-service:8007'),
    }

    logger.info("Service URLs:")
    for name, url in service_urls.items():
        logger.info(f"  - {name}: {url}")

    # Create generator
    generator = TrafficGenerator(profile, service_urls)

    # Start generating traffic
    try:
        await generator.start()
        logger.info("Traffic generation started")

        # Run until complete or interrupted
        while generator.running:
            await asyncio.sleep(5)

            # Log stats every 5 seconds
            stats = generator.get_stats()
            logger.info(
                f"Stats: {stats['total']} requests, "
                f"{stats['success_rate']:.1%} success, "
                f"{stats['throughput_rps']:.1f} RPS, "
                f"p95: {stats['latency_p95']:.0f}ms"
            )

        # Final stats
        stats = generator.get_stats()
        logger.info("Traffic generation complete")
        logger.info(f"Final stats:")
        logger.info(f"  - Total requests: {stats['total']}")
        logger.info(f"  - Successful: {stats['successful']}")
        logger.info(f"  - Failed: {stats['failed']}")
        logger.info(f"  - Success rate: {stats['success_rate']:.2%}")
        logger.info(f"  - Avg latency: {stats['latency_mean']:.0f}ms")
        logger.info(f"  - p95 latency: {stats['latency_p95']:.0f}ms")
        logger.info(f"  - Throughput: {stats['throughput_rps']:.1f} RPS")

        await generator.stop()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        await generator.stop()
    except Exception as e:
        logger.error(f"Error during traffic generation: {e}")
        import traceback
        traceback.print_exc()
        await generator.stop()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())


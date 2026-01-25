"""
Service Registry for Microservice Discovery

Tracks all services and their URLs, providing service discovery patterns.
Used by services to find and communicate with dependencies.
"""

from typing import Dict, Optional, List
import logging


class ServiceRegistry:
    """Central registry for service discovery in the microservice mesh.

    Provides methods to register, lookup, and manage service endpoints.
    Supports health checking and service status tracking.
    """

    def __init__(self):
        """Initialize the service registry."""
        self._services: Dict[str, str] = {}
        self._health_status: Dict[str, bool] = {}
        self.logger = logging.getLogger("ServiceRegistry")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def register(self, service_name: str, base_url: str):
        """Register a service with its base URL.

        Args:
            service_name: Name of the service (e.g., "user-service")
            base_url: Base URL including protocol, host, and port (e.g., "http://localhost:8001")
        """
        self._services[service_name] = base_url
        self._health_status[service_name] = True
        self.logger.info(f"Registered service '{service_name}' at {base_url}")

    def get_service_url(self, service_name: str) -> Optional[str]:
        """Get the base URL for a service.

        Args:
            service_name: Name of the service to lookup

        Returns:
            Base URL of the service, or None if not registered
        """
        url = self._services.get(service_name)
        if url is None:
            self.logger.warning(f"Service '{service_name}' not found in registry")
        return url

    def unregister(self, service_name: str):
        """Remove a service from the registry.

        Args:
            service_name: Name of the service to unregister
        """
        if service_name in self._services:
            del self._services[service_name]
            if service_name in self._health_status:
                del self._health_status[service_name]
            self.logger.info(f"Unregistered service '{service_name}'")

    def list_services(self) -> List[str]:
        """Get list of all registered service names.

        Returns:
            List of service names
        """
        return list(self._services.keys())

    def get_all_services(self) -> Dict[str, str]:
        """Get all registered services with their URLs.

        Returns:
            Dictionary mapping service names to URLs
        """
        return self._services.copy()

    def set_health_status(self, service_name: str, is_healthy: bool):
        """Update the health status of a service.

        Args:
            service_name: Name of the service
            is_healthy: True if service is healthy, False otherwise
        """
        if service_name in self._services:
            self._health_status[service_name] = is_healthy
            status = "healthy" if is_healthy else "unhealthy"
            self.logger.info(f"Service '{service_name}' marked as {status}")

    def is_healthy(self, service_name: str) -> bool:
        """Check if a service is healthy.

        Args:
            service_name: Name of the service

        Returns:
            True if service is registered and healthy, False otherwise
        """
        return self._health_status.get(service_name, False)

    def get_healthy_services(self) -> List[str]:
        """Get list of all healthy services.

        Returns:
            List of healthy service names
        """
        return [
            name for name, healthy in self._health_status.items()
            if healthy
        ]

    def clear(self):
        """Clear all registered services."""
        self._services.clear()
        self._health_status.clear()
        self.logger.info("Registry cleared")


# Global singleton instance
_global_registry = ServiceRegistry()


def get_registry() -> ServiceRegistry:
    """Get the global service registry instance.

    Returns:
        The global ServiceRegistry instance
    """
    return _global_registry


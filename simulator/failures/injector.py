"""
Failure Injection System

Simulates realistic failure scenarios for testing system resilience:
- Latency spikes
- Partial failures
- Timeouts
- Cascade failures
- Network partitions

Integrates with BaseService chaos engineering hooks.
"""

import asyncio
import time
import yaml
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class FailureScenario:
    """Configuration for a failure scenario.

    Attributes:
        name: Scenario identifier
        description: Human-readable explanation
        affected_services: List of service names to affect
        failure_type: Type of failure (latency/error/timeout/cascade/partition)
        parameters: Type-specific settings
        duration_seconds: How long the failure lasts (None = manual stop)
        start_delay_seconds: Wait before starting the failure
    """
    name: str
    description: str
    affected_services: List[str]
    failure_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: Optional[int] = None
    start_delay_seconds: int = 0

    @classmethod
    def from_yaml(cls, filepath: str) -> List['FailureScenario']:
        """Load failure scenarios from YAML file.

        Args:
            filepath: Path to YAML file

        Returns:
            List of FailureScenario instances
        """
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        scenarios = []
        for scenario_data in data.get('scenarios', []):
            scenarios.append(cls(
                name=scenario_data.get('name', 'unnamed'),
                description=scenario_data.get('description', ''),
                affected_services=scenario_data.get('affected_services', []),
                failure_type=scenario_data.get('failure_type', 'latency'),
                parameters=scenario_data.get('parameters', {}),
                duration_seconds=scenario_data.get('duration_seconds'),
                start_delay_seconds=scenario_data.get('start_delay_seconds', 0)
            ))

        return scenarios


# ============================================================================
# Failure Injector
# ============================================================================

class FailureInjector:
    """Injects failures into services for testing resilience."""

    def __init__(self, services: Dict[str, Any]):
        """Initialize the failure injector.

        Args:
            services: Dict mapping service names to BaseService instances
        """
        self.services = services
        self.active_failures: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.failure_tasks: List[asyncio.Task] = []

        self.logger = logging.getLogger("FailureInjector")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    # ========================================================================
    # Injection Methods
    # ========================================================================

    def inject_latency_spike(
        self,
        service_name: str,
        multiplier: float,
        duration: Optional[int] = None
    ):
        """Inject latency spike into a service.

        Args:
            service_name: Name of service to affect
            multiplier: Latency multiplier (e.g., 5.0 = 5x slower)
            duration: Duration in seconds (None = until manually restored)
        """
        service = self.services.get(service_name)
        if not service:
            self.logger.error(f"Service '{service_name}' not found")
            return

        # Apply latency multiplier
        service.set_latency_multiplier(multiplier)

        # Record active failure
        failure_info = {
            'type': 'latency',
            'multiplier': multiplier,
            'start_time': time.time(),
            'duration': duration,
        }
        self.active_failures[service_name].append(failure_info)

        self.logger.info(
            f"Injected latency spike: {service_name} -> {multiplier}x "
            f"for {duration}s" if duration else "indefinitely"
        )

        # Schedule automatic restoration if duration specified
        if duration:
            task = asyncio.create_task(
                self._auto_restore_latency(service_name, duration)
            )
            self.failure_tasks.append(task)

    def inject_partial_failure(
        self,
        service_name: str,
        error_rate: float,
        status_code: int = 500,
        duration: Optional[int] = None
    ):
        """Inject partial failure into a service.

        Args:
            service_name: Name of service to affect
            error_rate: Fraction of requests that should fail (0.0-1.0)
            status_code: HTTP status code for errors
            duration: Duration in seconds (None = until manually restored)
        """
        service = self.services.get(service_name)
        if not service:
            self.logger.error(f"Service '{service_name}' not found")
            return

        # Apply failure rate
        service.set_failure_rate(error_rate)

        # Record active failure
        failure_info = {
            'type': 'error',
            'error_rate': error_rate,
            'status_code': status_code,
            'start_time': time.time(),
            'duration': duration,
        }
        self.active_failures[service_name].append(failure_info)

        self.logger.info(
            f"Injected partial failure: {service_name} -> {error_rate:.0%} errors "
            f"(status {status_code}) for {duration}s" if duration else "indefinitely"
        )

        # Schedule automatic restoration
        if duration:
            task = asyncio.create_task(
                self._auto_restore_failure(service_name, duration)
            )
            self.failure_tasks.append(task)

    def inject_timeout(
        self,
        service_name: str,
        timeout_rate: float,
        duration: Optional[int] = None
    ):
        """Inject timeout behavior into a service.

        Args:
            service_name: Name of service to affect
            timeout_rate: Fraction of requests that should timeout (0.0-1.0)
            duration: Duration in seconds (None = until manually restored)
        """
        service = self.services.get(service_name)
        if not service:
            self.logger.error(f"Service '{service_name}' not found")
            return

        # Apply timeout rate (simulate by making service very slow)
        # Combined with high latency multiplier
        service.set_latency_multiplier(100.0)  # 100x slower = timeout

        # Record active failure
        failure_info = {
            'type': 'timeout',
            'timeout_rate': timeout_rate,
            'start_time': time.time(),
            'duration': duration,
        }
        self.active_failures[service_name].append(failure_info)

        self.logger.info(
            f"Injected timeout: {service_name} -> {timeout_rate:.0%} timeouts "
            f"for {duration}s" if duration else "indefinitely"
        )

        # Schedule automatic restoration
        if duration:
            task = asyncio.create_task(
                self._auto_restore_timeout(service_name, duration)
            )
            self.failure_tasks.append(task)

    def inject_cascade_failure(
        self,
        trigger_service: str,
        propagation_rate: float = 0.8
    ):
        """Inject cascade failure starting from a trigger service.

        Args:
            trigger_service: Service where cascade starts
            propagation_rate: Probability of failure propagating to dependents
        """
        self.logger.info(
            f"Injecting cascade failure from {trigger_service} "
            f"(propagation rate: {propagation_rate:.0%})"
        )

        # Start with trigger service
        self.inject_latency_spike(trigger_service, multiplier=10.0)

        # Find dependent services and propagate
        affected_services = self._propagate_cascade(
            trigger_service,
            propagation_rate
        )

        # Record cascade
        failure_info = {
            'type': 'cascade',
            'trigger_service': trigger_service,
            'propagation_rate': propagation_rate,
            'affected_services': affected_services,
            'start_time': time.time(),
        }
        self.active_failures['_cascade_'].append(failure_info)

        self.logger.info(
            f"Cascade affected {len(affected_services)} services: "
            f"{', '.join(affected_services)}"
        )

    def inject_network_partition(
        self,
        service_a: str,
        service_b: str,
        duration: Optional[int] = None
    ):
        """Inject network partition between two services.

        Args:
            service_a: First service (cannot reach service_b)
            service_b: Second service
            duration: Duration in seconds (None = until manually restored)
        """
        service = self.services.get(service_a)
        if not service:
            self.logger.error(f"Service '{service_a}' not found")
            return

        # Remove service_b from service_a's registry
        if hasattr(service, 'service_registry') and service_b in service.service_registry:
            original_url = service.service_registry[service_b]
            del service.service_registry[service_b]

            # Record partition
            failure_info = {
                'type': 'partition',
                'partitioned_service': service_b,
                'original_url': original_url,
                'start_time': time.time(),
                'duration': duration,
            }
            self.active_failures[service_a].append(failure_info)

            self.logger.info(
                f"Injected network partition: {service_a} cannot reach {service_b} "
                f"for {duration}s" if duration else "indefinitely"
            )

            # Schedule restoration
            if duration:
                task = asyncio.create_task(
                    self._auto_restore_partition(service_a, service_b, original_url, duration)
                )
                self.failure_tasks.append(task)

    def inject_scenario(self, scenario: FailureScenario):
        """Apply a failure scenario.

        Args:
            scenario: FailureScenario to apply
        """
        self.logger.info(f"Injecting scenario: {scenario.name} - {scenario.description}")

        # Wait for start delay if specified
        if scenario.start_delay_seconds > 0:
            self.logger.info(f"Waiting {scenario.start_delay_seconds}s before starting...")
            time.sleep(scenario.start_delay_seconds)

        # Apply based on failure type
        if scenario.failure_type == 'latency':
            multiplier = scenario.parameters.get('multiplier', 5.0)
            for service in scenario.affected_services:
                self.inject_latency_spike(service, multiplier, scenario.duration_seconds)

        elif scenario.failure_type == 'error':
            rate = scenario.parameters.get('rate', 0.5)
            status_code = scenario.parameters.get('status_code', 500)
            for service in scenario.affected_services:
                self.inject_partial_failure(service, rate, status_code, scenario.duration_seconds)

        elif scenario.failure_type == 'timeout':
            rate = scenario.parameters.get('rate', 0.3)
            for service in scenario.affected_services:
                self.inject_timeout(service, rate, scenario.duration_seconds)

        elif scenario.failure_type == 'cascade':
            trigger = scenario.parameters.get('trigger_service', scenario.affected_services[0])
            propagation_rate = scenario.parameters.get('propagation_rate', 0.8)
            self.inject_cascade_failure(trigger, propagation_rate)

        elif scenario.failure_type == 'partition':
            if len(scenario.affected_services) >= 2:
                self.inject_network_partition(
                    scenario.affected_services[0],
                    scenario.affected_services[1],
                    scenario.duration_seconds
                )

    # ========================================================================
    # Restoration Methods
    # ========================================================================

    def restore(self, service_name: Optional[str] = None):
        """Remove failures from a service.

        Args:
            service_name: Service to restore (None = restore all)
        """
        if service_name is None:
            self.restore_all()
            return

        service = self.services.get(service_name)
        if not service:
            self.logger.error(f"Service '{service_name}' not found")
            return

        # Restore service to normal
        service.set_latency_multiplier(1.0)
        service.set_failure_rate(0.0)
        service.set_offline(False)

        # Remove active failures
        if service_name in self.active_failures:
            del self.active_failures[service_name]

        self.logger.info(f"Restored service: {service_name}")

    def restore_all(self):
        """Remove all active failures."""
        self.logger.info("Restoring all services...")

        # Cancel all auto-restore tasks
        for task in self.failure_tasks:
            if not task.done():
                task.cancel()
        self.failure_tasks.clear()

        # Restore all services
        for service_name in list(self.active_failures.keys()):
            if service_name != '_cascade_':
                self.restore(service_name)

        # Clear cascade failures
        if '_cascade_' in self.active_failures:
            del self.active_failures['_cascade_']

        self.logger.info("All services restored")

    # ========================================================================
    # Query Methods
    # ========================================================================

    def get_active_failures(self) -> List[Dict[str, Any]]:
        """Get list of currently active failure scenarios.

        Returns:
            List of active failure descriptions
        """
        active = []
        for service_name, failures in self.active_failures.items():
            for failure in failures:
                active.append({
                    'service': service_name,
                    'type': failure['type'],
                    'start_time': failure['start_time'],
                    'elapsed': time.time() - failure['start_time'],
                    'duration': failure.get('duration'),
                    'details': {k: v for k, v in failure.items()
                               if k not in ['type', 'start_time', 'duration']}
                })
        return active

    def is_failure_active(self, service_name: str) -> bool:
        """Check if a service has active failures.

        Args:
            service_name: Service to check

        Returns:
            True if service has active failures
        """
        return service_name in self.active_failures and len(self.active_failures[service_name]) > 0

    # ========================================================================
    # Internal Methods
    # ========================================================================

    def _propagate_cascade(
        self,
        trigger_service: str,
        propagation_rate: float
    ) -> List[str]:
        """Propagate cascade failure through dependencies.

        Args:
            trigger_service: Service where cascade starts
            propagation_rate: Probability of propagation

        Returns:
            List of affected service names
        """
        import random

        affected = []

        # Find services that depend on trigger_service
        for service_name, service in self.services.items():
            if service_name == trigger_service:
                continue

            # Check if service has trigger_service as dependency
            if hasattr(service, 'config') and hasattr(service.config, 'dependencies'):
                if trigger_service in service.config.dependencies:
                    # Propagate with probability
                    if random.random() < propagation_rate:
                        # Apply failure
                        self.inject_partial_failure(service_name, error_rate=0.5)
                        affected.append(service_name)

                        # Recursively propagate
                        sub_affected = self._propagate_cascade(
                            service_name,
                            propagation_rate * 0.8  # Decay
                        )
                        affected.extend(sub_affected)

        return affected

    async def _auto_restore_latency(self, service_name: str, duration: int):
        """Automatically restore latency after duration."""
        await asyncio.sleep(duration)
        service = self.services.get(service_name)
        if service:
            service.set_latency_multiplier(1.0)
            self.logger.info(f"Auto-restored latency for {service_name}")

    async def _auto_restore_failure(self, service_name: str, duration: int):
        """Automatically restore failure rate after duration."""
        await asyncio.sleep(duration)
        service = self.services.get(service_name)
        if service:
            service.set_failure_rate(0.0)
            self.logger.info(f"Auto-restored failure rate for {service_name}")

    async def _auto_restore_timeout(self, service_name: str, duration: int):
        """Automatically restore timeout after duration."""
        await asyncio.sleep(duration)
        service = self.services.get(service_name)
        if service:
            service.set_latency_multiplier(1.0)
            self.logger.info(f"Auto-restored timeout for {service_name}")

    async def _auto_restore_partition(
        self,
        service_a: str,
        service_b: str,
        original_url: str,
        duration: int
    ):
        """Automatically restore network partition after duration."""
        await asyncio.sleep(duration)
        service = self.services.get(service_a)
        if service and hasattr(service, 'service_registry'):
            service.service_registry[service_b] = original_url
            self.logger.info(f"Auto-restored partition: {service_a} can reach {service_b}")


# ============================================================================
# Cascade Simulator
# ============================================================================

class CascadeSimulator:
    """Simulates cascade failures through service dependencies."""

    def __init__(self, service_dependencies: Dict[str, List[str]]):
        """Initialize cascade simulator.

        Args:
            service_dependencies: Dict mapping service names to their dependencies
        """
        self.dependencies = service_dependencies
        self.logger = logging.getLogger("CascadeSimulator")
        self.logger.setLevel(logging.INFO)

    def simulate_cascade(
        self,
        trigger_service: str,
        trigger_type: str,
        duration: int
    ) -> List[Dict[str, Any]]:
        """Simulate how a cascade would propagate.

        Args:
            trigger_service: Service where cascade starts
            trigger_type: Type of initial failure (latency/error/timeout)
            duration: Total duration of simulation

        Returns:
            Timeline of cascade events
        """
        timeline = []
        affected_services = {trigger_service}
        current_time = 0

        # Initial event
        timeline.append({
            'time': current_time,
            'service': trigger_service,
            'impact': f"{trigger_type} spike begins",
            'severity': 'critical'
        })

        # Simulate propagation over time
        propagation_delay = 5  # seconds between propagation waves

        while current_time < duration:
            current_time += propagation_delay

            # Find services that depend on affected services
            new_affected = set()
            for service_name, deps in self.dependencies.items():
                if service_name not in affected_services:
                    # Check if any dependency is affected
                    if any(dep in affected_services for dep in deps):
                        new_affected.add(service_name)

                        # Determine impact based on time and position
                        if current_time < 10:
                            impact = "starts timing out on dependency calls"
                            severity = 'high'
                        elif current_time < 20:
                            impact = "queues filling up, response time increasing"
                            severity = 'high'
                        elif current_time < 30:
                            impact = "starts rejecting requests (circuit breaker)"
                            severity = 'critical'
                        else:
                            impact = "severely degraded, cascading failures"
                            severity = 'critical'

                        timeline.append({
                            'time': current_time,
                            'service': service_name,
                            'impact': impact,
                            'severity': severity,
                            'triggered_by': list(set(deps) & affected_services)
                        })

            affected_services.update(new_affected)

            # Stop if no new services affected
            if not new_affected:
                break

        return timeline

    def detect_cascade_risk(
        self,
        current_metrics: Dict[str, Dict[str, float]]
    ) -> Tuple[float, List[str]]:
        """Analyze current metrics to detect cascade risk.

        Args:
            current_metrics: Dict of service_name -> metrics
                Metrics should include: latency_p95, error_rate, queue_depth

        Returns:
            Tuple of (risk_score 0-1, list of at-risk services)
        """
        risk_factors = []
        at_risk_services = []

        for service_name, metrics in current_metrics.items():
            service_risk = 0.0

            # High latency is a risk factor
            if metrics.get('latency_p95', 0) > 1000:  # > 1 second
                service_risk += 0.3

            # High error rate
            if metrics.get('error_rate', 0) > 0.1:  # > 10%
                service_risk += 0.4

            # Queue depth (if available)
            if metrics.get('queue_depth', 0) > 100:
                service_risk += 0.3

            if service_risk > 0.5:
                at_risk_services.append(service_name)
                risk_factors.append(service_risk)

        # Overall risk is average of at-risk services
        overall_risk = sum(risk_factors) / len(risk_factors) if risk_factors else 0.0

        return overall_risk, at_risk_services

    def get_cascade_path(self, trigger_service: str) -> List[List[str]]:
        """Get the potential cascade propagation paths.

        Args:
            trigger_service: Service where cascade would start

        Returns:
            List of propagation paths (each path is a list of services)
        """
        paths = []
        visited = set()

        def find_paths(current_service: str, current_path: List[str]):
            if current_service in visited:
                return

            visited.add(current_service)
            current_path = current_path + [current_service]

            # Find services that depend on current service
            dependents = [
                service for service, deps in self.dependencies.items()
                if current_service in deps
            ]

            if not dependents:
                # End of path
                paths.append(current_path)
            else:
                # Continue down each dependent
                for dependent in dependents:
                    find_paths(dependent, current_path)

            visited.remove(current_service)

        find_paths(trigger_service, [])

        return paths

    def get_critical_services(self) -> List[Tuple[str, int]]:
        """Identify critical services (most depended upon).

        Returns:
            List of (service_name, dependent_count) sorted by criticality
        """
        criticality = {}

        for service_name, deps in self.dependencies.items():
            for dep in deps:
                criticality[dep] = criticality.get(dep, 0) + 1

        # Sort by criticality
        return sorted(criticality.items(), key=lambda x: x[1], reverse=True)


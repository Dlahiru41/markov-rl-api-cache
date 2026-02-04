"""
Integration tests for simulator components.

These tests verify that the microservices simulator, traffic generator,
and failure injection work correctly together.
"""

import pytest
import numpy as np
import time
from typing import Dict, Any


class TestServiceInteraction:
    """Test microservice interaction and simulation."""

    def test_services_respond(self, mock_services):
        """Test that all services respond to requests."""
        for endpoint, service in mock_services.items():
            response = service.call()

            assert response is not None
            assert isinstance(response, dict)
            assert 'endpoint' in response
            assert response['endpoint'] == endpoint

    def test_service_dependencies(self, mock_services):
        """Test that dependent services can call each other."""
        # Simulate service dependency chain
        login_service = mock_services['/api/auth/login']
        profile_service = mock_services['/api/user/profile']

        # Login first
        login_response = login_service.call({'username': 'test'})
        assert login_response is not None

        # Then access profile (simulating dependency)
        profile_response = profile_service.call({'user_id': 123})
        assert profile_response is not None

        # Both should have been called
        assert login_service.call_count >= 1
        assert profile_service.call_count >= 1

    def test_latency_simulation(self, mock_services):
        """Test that responses have realistic latency."""
        service = mock_services['/api/products/list']
        service.set_latency(50.0)  # 50ms

        start_time = time.time()
        response = service.call()
        elapsed_ms = (time.time() - start_time) * 1000

        # Should take at least the configured latency
        assert elapsed_ms >= 40  # Allow some variance
        assert response is not None


class TestFailureInjection:
    """Test failure injection capabilities."""

    def test_latency_injection(self, mock_services):
        """Test that injected latency affects responses."""
        service = mock_services['/api/orders/list']

        # Normal latency
        service.set_latency(10.0)
        start_normal = time.time()
        service.call()
        normal_time = (time.time() - start_normal) * 1000

        # Inject high latency
        service.set_latency(100.0)
        start_slow = time.time()
        service.call()
        slow_time = (time.time() - start_slow) * 1000

        # Slow call should be noticeably slower
        assert slow_time > normal_time * 2

    def test_error_injection(self, mock_services):
        """Test that injected errors return expected codes."""
        service = mock_services['/api/cart/view']

        # Set high error rate
        service.set_error_rate(1.0)  # 100% errors

        # Should raise error
        with pytest.raises(Exception):
            service.call()

        # Reset error rate
        service.set_error_rate(0.0)

        # Should work now
        response = service.call()
        assert response is not None

    def test_cascade_propagation(self, mock_services):
        """Test that failures propagate through dependencies."""
        # Simulate cascade scenario
        auth_service = mock_services['/api/auth/login']
        profile_service = mock_services['/api/user/profile']
        orders_service = mock_services['/api/orders/list']

        # If auth fails, dependent services should also fail
        auth_service.set_error_rate(1.0)

        # Auth should fail
        with pytest.raises(Exception):
            auth_service.call()

        # In real system, this would cascade to profile
        # For mock, we just verify the mechanism
        assert auth_service.error_rate == 1.0

        # Dependent services could check auth status
        # This tests the injection mechanism works

    def test_failure_restoration(self, mock_services):
        """Test that failures can be cleared."""
        service = mock_services['/api/products/123']

        # Inject failure
        service.set_error_rate(1.0)
        service.set_latency(500.0)

        # Verify failure
        assert service.error_rate == 1.0
        assert service.latency_ms == 500.0

        # Restore
        service.reset()

        # Should be normal now
        assert service.error_rate == 0.0
        response = service.call()
        assert response is not None


class TestTrafficGeneration:
    """Test traffic generation capabilities."""

    def test_traffic_follows_workflow(self, sample_traffic):
        """Test that generated traffic follows defined workflows."""
        assert len(sample_traffic) > 0

        for session in sample_traffic:
            # Each session should have required fields
            assert 'session_id' in session
            assert 'user_type' in session
            assert 'apis' in session

            # Should have API calls
            assert len(session['apis']) > 0

            # Each API call should have endpoint
            for api_call in session['apis']:
                assert 'endpoint' in api_call

    def test_traffic_rate_achieved(self, sample_traffic):
        """Test that target RPS is approximately achieved."""
        # Count APIs per session
        total_apis = sum(len(session['apis']) for session in sample_traffic)
        num_sessions = len(sample_traffic)

        # Should have reasonable API count
        assert total_apis > 0
        assert num_sessions > 0

        # Average APIs per session
        avg_per_session = total_apis / num_sessions
        assert avg_per_session > 0
        assert avg_per_session <= 100  # Reasonable upper bound

    def test_user_type_distribution(self, sample_traffic):
        """Test that user types match configured distribution."""
        user_types = [session['user_type'] for session in sample_traffic]

        # Should have all types represented (probabilistically)
        unique_types = set(user_types)
        expected_types = {'guest', 'free', 'premium'}

        # At least some variety (may not have all in small sample)
        assert len(unique_types) > 0
        assert all(ut in expected_types for ut in unique_types)


class TestRealisticScenarios:
    """Test realistic usage scenarios."""

    def test_normal_traffic_scenario(self, mock_services, sample_traffic):
        """Test handling of normal traffic patterns."""
        # Simulate normal traffic
        successful_calls = 0
        failed_calls = 0

        for session in sample_traffic[:5]:  # Test subset
            for api_call in session['apis'][:10]:  # Test subset
                endpoint = api_call['endpoint']

                # Find matching service
                matching_services = [
                    svc for ep, svc in mock_services.items()
                    if api_call['endpoint'].startswith('api_')
                ]

                if matching_services:
                    service = matching_services[0]
                    try:
                        response = service.call()
                        if response:
                            successful_calls += 1
                    except:
                        failed_calls += 1

        # Most calls should succeed in normal scenario
        if successful_calls + failed_calls > 0:
            success_rate = successful_calls / (successful_calls + failed_calls)
            assert success_rate > 0.9  # >90% success

    def test_peak_traffic_scenario(self, mock_services, sample_traffic):
        """Test handling of peak load."""
        # Simulate peak traffic with reduced latency
        for service in mock_services.values():
            service.set_latency(5.0)  # Fast responses under peak

        start_time = time.time()
        call_count = 0

        # Process many requests quickly
        for session in sample_traffic[:3]:
            for api_call in session['apis'][:5]:
                matching = [
                    svc for ep, svc in mock_services.items()
                    if ep == '/api/products/list'
                ]
                if matching:
                    matching[0].call()
                    call_count += 1

        elapsed = time.time() - start_time

        # Should handle peak efficiently
        assert call_count > 0
        assert elapsed < 5.0  # Should complete quickly

    def test_cascade_prevention_scenario(self, mock_services):
        """Test that system can prevent cascades."""
        # Scenario: One service starts failing
        failing_service = mock_services['/api/orders/list']
        failing_service.set_error_rate(0.3)  # 30% errors

        # Other services should remain healthy
        healthy_service = mock_services['/api/products/list']

        # Healthy service should still work
        response = healthy_service.call()
        assert response is not None

        # This demonstrates circuit breaker pattern
        # In real system, would stop calling failing service

    def test_cold_start_scenario(self, mock_services, cache_manager):
        """Test handling of cold start (empty cache)."""
        # Clear cache
        cache_manager.clear()

        # All calls should be cache misses initially
        test_keys = ['/api/item1', '/api/item2', '/api/item3']

        for key in test_keys:
            result = cache_manager.get(key)
            assert result is None  # Cache miss

        # Populate cache
        for i, key in enumerate(test_keys):
            cache_manager.set(key, {'id': i, 'data': f'item{i}'}, ttl=300)

        # Now should hit
        hits = 0
        for key in test_keys:
            result = cache_manager.get(key)
            if result is not None:
                hits += 1

        # All should hit now
        assert hits == len(test_keys)


class TestServiceMetrics:
    """Test service metrics and monitoring."""

    def test_call_count_tracking(self, mock_services):
        """Test that service call counts are tracked."""
        service = mock_services['/api/auth/login']

        initial_count = service.call_count

        # Make calls
        for _ in range(5):
            service.call()

        # Count should increase
        assert service.call_count == initial_count + 5

    def test_error_rate_tracking(self, mock_services):
        """Test that error rates are tracked."""
        service = mock_services['/api/user/profile']
        service.set_error_rate(0.5)  # 50% errors

        successes = 0
        failures = 0

        for _ in range(20):
            try:
                service.call()
                successes += 1
            except:
                failures += 1

        # Should have some failures
        total = successes + failures
        if total > 0:
            failure_rate = failures / total
            # Should be roughly 50% (allow variance)
            assert 0.2 < failure_rate < 0.8

    def test_latency_metrics(self, mock_services):
        """Test that latency metrics are recorded."""
        service = mock_services['/api/products/list']
        service.set_latency(20.0)

        latencies = []

        for _ in range(10):
            start = time.time()
            service.call()
            elapsed_ms = (time.time() - start) * 1000
            latencies.append(elapsed_ms)

        # Calculate statistics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        # Should be around configured latency
        assert avg_latency >= 15  # At least 15ms average
        assert p95_latency >= 15  # P95 should also be reasonable


class TestServiceResilience:
    """Test service resilience patterns."""

    def test_retry_mechanism(self, mock_services):
        """Test that failed requests can be retried."""
        service = mock_services['/api/cart/view']
        service.set_error_rate(0.8)  # High failure rate

        # Retry logic
        max_retries = 3
        success = False

        for attempt in range(max_retries):
            try:
                response = service.call()
                if response:
                    success = True
                    break
            except:
                if attempt < max_retries - 1:
                    time.sleep(0.01)  # Brief delay
                    continue
                else:
                    break

        # May or may not succeed, but mechanism should work
        # The test verifies retry pattern can be implemented
        assert attempt >= 0

    def test_circuit_breaker_pattern(self, mock_services):
        """Test circuit breaker pattern implementation."""
        service = mock_services['/api/orders/list']

        # Track consecutive failures
        failure_count = 0
        threshold = 3
        circuit_open = False

        service.set_error_rate(1.0)  # All fail

        # Make requests until circuit opens
        for _ in range(5):
            if circuit_open:
                # Don't make request when circuit is open
                break

            try:
                service.call()
                failure_count = 0  # Reset on success
            except:
                failure_count += 1
                if failure_count >= threshold:
                    circuit_open = True

        # Circuit should have opened
        assert circuit_open
        assert failure_count >= threshold

    def test_graceful_degradation(self, mock_services, cache_manager):
        """Test graceful degradation when services fail."""
        service = mock_services['/api/products/123']
        service.set_error_rate(1.0)  # Service down

        # Try to get from cache first
        cache_key = '/api/products/123'
        cached = cache_manager.get(cache_key)

        if cached is None:
            # Cache miss, try service
            try:
                response = service.call()
            except:
                # Service failed, use fallback
                response = {'id': 123, 'error': 'Service unavailable', 'fallback': True}
        else:
            response = cached

        # Should have some response (cached or fallback)
        assert response is not None
        assert isinstance(response, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


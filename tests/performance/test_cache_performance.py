"""
Performance tests for cache backends.

These tests compare performance characteristics of different backends
and validate that they meet performance requirements.
"""

import pytest
import time
import statistics
from typing import List, Callable
from src.cache.backend import InMemoryBackend, CacheBackend
from src.cache.redis_backend import RedisBackend, RedisConfig, REDIS_AVAILABLE


def check_redis_available():
    """Check if Redis server is available."""
    if not REDIS_AVAILABLE:
        return False

    config = RedisConfig(socket_timeout=1.0)
    backend = RedisBackend(config)
    result = backend.connect()
    if result:
        backend.disconnect()
    return result


def measure_operation(func: Callable, iterations: int = 1000) -> dict:
    """
    Measure operation performance.

    Returns:
        Dict with timing statistics (mean, median, min, max, total)
    """
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds

    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'min': min(times),
        'max': max(times),
        'total': sum(times),
        'iterations': iterations
    }


@pytest.fixture
def inmemory_backend():
    """Fixture for InMemory backend."""
    backend = InMemoryBackend(max_size_bytes=100 * 1024 * 1024)  # 100MB
    yield backend


@pytest.fixture
def redis_backend():
    """Fixture for Redis backend."""
    if not check_redis_available():
        pytest.skip("Redis not available")

    config = RedisConfig(
        key_prefix="perf_test:",
        socket_timeout=5.0
    )
    backend = RedisBackend(config)

    if not backend.connect():
        pytest.skip("Cannot connect to Redis")

    backend.clear()
    yield backend

    backend.clear()
    backend.disconnect()


class TestSingleOperationPerformance:
    """Test performance of single operations."""

    def test_set_performance_inmemory(self, inmemory_backend):
        """Test set operation performance on InMemory backend."""
        stats = measure_operation(
            lambda: inmemory_backend.set("perf_key", b"perf_value"),
            iterations=1000
        )

        print(f"\nInMemory SET performance:")
        print(f"  Mean: {stats['mean']:.4f} ms")
        print(f"  Median: {stats['median']:.4f} ms")
        print(f"  Min: {stats['min']:.4f} ms")
        print(f"  Max: {stats['max']:.4f} ms")

        # InMemory should be very fast (< 1ms mean)
        assert stats['mean'] < 1.0

    def test_set_performance_redis(self, redis_backend):
        """Test set operation performance on Redis backend."""
        stats = measure_operation(
            lambda: redis_backend.set("perf_key", b"perf_value"),
            iterations=1000
        )

        print(f"\nRedis SET performance:")
        print(f"  Mean: {stats['mean']:.4f} ms")
        print(f"  Median: {stats['median']:.4f} ms")
        print(f"  Min: {stats['min']:.4f} ms")
        print(f"  Max: {stats['max']:.4f} ms")

        # Redis should be reasonably fast (< 10ms mean)
        assert stats['mean'] < 10.0

    def test_get_performance_inmemory(self, inmemory_backend):
        """Test get operation performance on InMemory backend."""
        # Setup
        inmemory_backend.set("perf_key", b"perf_value")

        stats = measure_operation(
            lambda: inmemory_backend.get("perf_key"),
            iterations=1000
        )

        print(f"\nInMemory GET performance:")
        print(f"  Mean: {stats['mean']:.4f} ms")
        print(f"  Median: {stats['median']:.4f} ms")

        assert stats['mean'] < 1.0

    def test_get_performance_redis(self, redis_backend):
        """Test get operation performance on Redis backend."""
        # Setup
        redis_backend.set("perf_key", b"perf_value")

        stats = measure_operation(
            lambda: redis_backend.get("perf_key"),
            iterations=1000
        )

        print(f"\nRedis GET performance:")
        print(f"  Mean: {stats['mean']:.4f} ms")
        print(f"  Median: {stats['median']:.4f} ms")

        assert stats['mean'] < 10.0

    def test_delete_performance_inmemory(self, inmemory_backend):
        """Test delete operation performance on InMemory backend."""
        def operation():
            key = f"del_{time.time()}"
            inmemory_backend.set(key, b"value")
            inmemory_backend.delete(key)

        stats = measure_operation(operation, iterations=1000)

        print(f"\nInMemory DELETE performance:")
        print(f"  Mean: {stats['mean']:.4f} ms")

        assert stats['mean'] < 1.0

    def test_delete_performance_redis(self, redis_backend):
        """Test delete operation performance on Redis backend."""
        def operation():
            key = f"del_{time.time()}"
            redis_backend.set(key, b"value")
            redis_backend.delete(key)

        stats = measure_operation(operation, iterations=1000)

        print(f"\nRedis DELETE performance:")
        print(f"  Mean: {stats['mean']:.4f} ms")

        assert stats['mean'] < 10.0


class TestBatchOperationPerformance:
    """Test performance of batch operations."""

    def test_set_many_performance_inmemory(self, inmemory_backend):
        """Test set_many performance on InMemory backend."""
        items = {
            f"batch_key{i}": f"batch_value{i}".encode()
            for i in range(100)
        }

        stats = measure_operation(
            lambda: inmemory_backend.set_many(items),
            iterations=100
        )

        print(f"\nInMemory SET_MANY (100 items) performance:")
        print(f"  Mean: {stats['mean']:.4f} ms")
        print(f"  Per item: {stats['mean']/100:.4f} ms")

        # Should be fast
        assert stats['mean'] < 100.0

    def test_set_many_performance_redis(self, redis_backend):
        """Test set_many performance on Redis backend."""
        items = {
            f"batch_key{i}": f"batch_value{i}".encode()
            for i in range(100)
        }

        stats = measure_operation(
            lambda: redis_backend.set_many(items),
            iterations=100
        )

        print(f"\nRedis SET_MANY (100 items) performance:")
        print(f"  Mean: {stats['mean']:.4f} ms")
        print(f"  Per item: {stats['mean']/100:.4f} ms")

        # Should be faster than 100 individual operations
        assert stats['mean'] < 1000.0

    def test_get_many_performance_inmemory(self, inmemory_backend):
        """Test get_many performance on InMemory backend."""
        # Setup
        for i in range(100):
            inmemory_backend.set(f"batch_key{i}", f"value{i}".encode())

        keys = [f"batch_key{i}" for i in range(100)]

        stats = measure_operation(
            lambda: inmemory_backend.get_many(keys),
            iterations=100
        )

        print(f"\nInMemory GET_MANY (100 items) performance:")
        print(f"  Mean: {stats['mean']:.4f} ms")

        assert stats['mean'] < 100.0

    def test_get_many_performance_redis(self, redis_backend):
        """Test get_many performance on Redis backend."""
        # Setup
        items = {
            f"batch_key{i}": f"value{i}".encode()
            for i in range(100)
        }
        redis_backend.set_many(items)

        keys = [f"batch_key{i}" for i in range(100)]

        stats = measure_operation(
            lambda: redis_backend.get_many(keys),
            iterations=100
        )

        print(f"\nRedis GET_MANY (100 items) performance:")
        print(f"  Mean: {stats['mean']:.4f} ms")

        assert stats['mean'] < 1000.0


class TestScalabilityPerformance:
    """Test performance with varying data sizes."""

    @pytest.mark.parametrize("value_size", [100, 1000, 10000, 100000])
    def test_value_size_impact_inmemory(self, inmemory_backend, value_size):
        """Test how value size impacts InMemory performance."""
        value = b"x" * value_size

        stats = measure_operation(
            lambda: inmemory_backend.set("size_test", value),
            iterations=100
        )

        print(f"\nInMemory SET with {value_size} bytes:")
        print(f"  Mean: {stats['mean']:.4f} ms")

    @pytest.mark.parametrize("value_size", [100, 1000, 10000, 100000])
    def test_value_size_impact_redis(self, redis_backend, value_size):
        """Test how value size impacts Redis performance."""
        value = b"x" * value_size

        stats = measure_operation(
            lambda: redis_backend.set("size_test", value),
            iterations=100
        )

        print(f"\nRedis SET with {value_size} bytes:")
        print(f"  Mean: {stats['mean']:.4f} ms")

    @pytest.mark.parametrize("num_keys", [10, 100, 1000])
    def test_num_keys_impact_inmemory(self, inmemory_backend, num_keys):
        """Test how number of keys impacts InMemory performance."""
        # Setup
        for i in range(num_keys):
            inmemory_backend.set(f"key{i}", b"value")

        stats = measure_operation(
            lambda: inmemory_backend.get(f"key{num_keys//2}"),
            iterations=100
        )

        print(f"\nInMemory GET with {num_keys} total keys:")
        print(f"  Mean: {stats['mean']:.4f} ms")

    @pytest.mark.parametrize("num_keys", [10, 100, 1000])
    def test_num_keys_impact_redis(self, redis_backend, num_keys):
        """Test how number of keys impacts Redis performance."""
        # Setup
        items = {f"key{i}": b"value" for i in range(num_keys)}
        redis_backend.set_many(items)

        stats = measure_operation(
            lambda: redis_backend.get(f"key{num_keys//2}"),
            iterations=100
        )

        print(f"\nRedis GET with {num_keys} total keys:")
        print(f"  Mean: {stats['mean']:.4f} ms")


class TestConcurrencyPerformance:
    """Test performance under concurrent load."""

    def test_concurrent_reads_inmemory(self, inmemory_backend):
        """Test concurrent read performance on InMemory."""
        import threading

        # Setup
        for i in range(100):
            inmemory_backend.set(f"key{i}", f"value{i}".encode())

        results = []

        def worker():
            start = time.perf_counter()
            for i in range(100):
                inmemory_backend.get(f"key{i}")
            end = time.perf_counter()
            results.append((end - start) * 1000)

        threads = [threading.Thread(target=worker) for _ in range(10)]

        start_all = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        end_all = time.perf_counter()

        total_time = (end_all - start_all) * 1000
        mean_thread_time = statistics.mean(results)

        print(f"\nInMemory concurrent reads (10 threads, 100 reads each):")
        print(f"  Total time: {total_time:.2f} ms")
        print(f"  Mean thread time: {mean_thread_time:.2f} ms")
        print(f"  Operations/sec: {1000/(total_time/1000):.0f}")

    def test_concurrent_reads_redis(self, redis_backend):
        """Test concurrent read performance on Redis."""
        import threading

        # Setup
        items = {f"key{i}": f"value{i}".encode() for i in range(100)}
        redis_backend.set_many(items)

        results = []

        def worker():
            start = time.perf_counter()
            for i in range(100):
                redis_backend.get(f"key{i}")
            end = time.perf_counter()
            results.append((end - start) * 1000)

        threads = [threading.Thread(target=worker) for _ in range(10)]

        start_all = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        end_all = time.perf_counter()

        total_time = (end_all - start_all) * 1000
        mean_thread_time = statistics.mean(results)

        print(f"\nRedis concurrent reads (10 threads, 100 reads each):")
        print(f"  Total time: {total_time:.2f} ms")
        print(f"  Mean thread time: {mean_thread_time:.2f} ms")
        print(f"  Operations/sec: {1000/(total_time/1000):.0f}")


class TestThroughputPerformance:
    """Test throughput capabilities."""

    def test_throughput_inmemory(self, inmemory_backend):
        """Measure InMemory backend throughput."""
        num_operations = 10000

        start = time.perf_counter()
        for i in range(num_operations):
            inmemory_backend.set(f"throughput{i}", f"value{i}".encode())
        end = time.perf_counter()

        elapsed = end - start
        ops_per_sec = num_operations / elapsed

        print(f"\nInMemory throughput:")
        print(f"  {num_operations} operations in {elapsed:.2f}s")
        print(f"  {ops_per_sec:.0f} ops/sec")

        # Should be very high
        assert ops_per_sec > 10000

    def test_throughput_redis(self, redis_backend):
        """Measure Redis backend throughput."""
        num_operations = 10000

        start = time.perf_counter()
        for i in range(num_operations):
            redis_backend.set(f"throughput{i}", f"value{i}".encode())
        end = time.perf_counter()

        elapsed = end - start
        ops_per_sec = num_operations / elapsed

        print(f"\nRedis throughput:")
        print(f"  {num_operations} operations in {elapsed:.2f}s")
        print(f"  {ops_per_sec:.0f} ops/sec")

        # Should be reasonable
        assert ops_per_sec > 100

    def test_batch_throughput_comparison(self, inmemory_backend, redis_backend):
        """Compare batch operation throughput."""
        num_batches = 100
        batch_size = 100

        # InMemory
        start = time.perf_counter()
        for i in range(num_batches):
            items = {
                f"batch{i}_key{j}": f"value{j}".encode()
                for j in range(batch_size)
            }
            inmemory_backend.set_many(items)
        inmemory_time = time.perf_counter() - start

        # Redis
        start = time.perf_counter()
        for i in range(num_batches):
            items = {
                f"batch{i}_key{j}": f"value{j}".encode()
                for j in range(batch_size)
            }
            redis_backend.set_many(items)
        redis_time = time.perf_counter() - start

        total_ops = num_batches * batch_size

        print(f"\nBatch throughput comparison ({total_ops} total operations):")
        print(f"  InMemory: {total_ops/inmemory_time:.0f} ops/sec")
        print(f"  Redis: {total_ops/redis_time:.0f} ops/sec")
        print(f"  Speed ratio: {inmemory_time/redis_time:.2f}x")


class TestPerformanceRequirements:
    """Validate that performance meets requirements."""

    def test_inmemory_response_time(self, inmemory_backend):
        """Test that InMemory meets response time requirements."""
        # Requirement: 99% of operations should complete in < 1ms
        times = []

        for i in range(1000):
            start = time.perf_counter()
            inmemory_backend.set(f"key{i}", b"value")
            end = time.perf_counter()
            times.append((end - start) * 1000)

        times.sort()
        p99 = times[int(len(times) * 0.99)]

        print(f"\nInMemory 99th percentile: {p99:.4f} ms")
        assert p99 < 1.0

    def test_redis_response_time(self, redis_backend):
        """Test that Redis meets response time requirements."""
        # Requirement: 99% of operations should complete in < 10ms
        times = []

        for i in range(1000):
            start = time.perf_counter()
            redis_backend.set(f"key{i}", b"value")
            end = time.perf_counter()
            times.append((end - start) * 1000)

        times.sort()
        p99 = times[int(len(times) * 0.99)]

        print(f"\nRedis 99th percentile: {p99:.4f} ms")
        assert p99 < 10.0


if __name__ == "__main__":
    # Run with verbose output to see performance metrics
    pytest.main([__file__, "-v", "-s"])


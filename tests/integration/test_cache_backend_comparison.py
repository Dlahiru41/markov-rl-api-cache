"""
Integration tests comparing InMemory and Redis backends.

Tests that both backends implement the same interface correctly
and can be used interchangeably.
"""

import pytest
import time
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


# Fixture providing both backend types
@pytest.fixture(params=['inmemory', 'redis'])
def cache_backend(request):
    """
    Parametrized fixture that provides both InMemory and Redis backends.

    Tests using this fixture will run twice: once with each backend.
    """
    if request.param == 'inmemory':
        backend = InMemoryBackend(max_size_bytes=10 * 1024 * 1024)  # 10MB
        yield backend

    elif request.param == 'redis':
        if not check_redis_available():
            pytest.skip("Redis server not available")

        config = RedisConfig(
            key_prefix="test_comparison:",
            socket_timeout=2.0
        )
        backend = RedisBackend(config)

        if not backend.connect():
            pytest.skip("Cannot connect to Redis")

        backend.clear()
        yield backend

        backend.clear()
        backend.disconnect()


class TestBackendInterface:
    """Test that both backends implement the interface correctly."""

    def test_is_cache_backend(self, cache_backend):
        """Test that backend implements CacheBackend."""
        assert isinstance(cache_backend, CacheBackend)

    def test_has_required_methods(self, cache_backend):
        """Test that backend has all required methods."""
        required_methods = [
            'get', 'set', 'delete', 'exists', 'clear',
            'get_stats', 'get_many', 'set_many', 'delete_many', 'keys'
        ]

        for method_name in required_methods:
            assert hasattr(cache_backend, method_name)
            assert callable(getattr(cache_backend, method_name))


class TestBasicOperations:
    """Test basic operations work the same on both backends."""

    def test_set_and_get(self, cache_backend):
        """Test basic set and get."""
        assert cache_backend.set("key", b"value")
        assert cache_backend.get("key") == b"value"

    def test_get_nonexistent(self, cache_backend):
        """Test get returns None for nonexistent key."""
        assert cache_backend.get("nonexistent") is None

    def test_overwrite(self, cache_backend):
        """Test overwriting existing key."""
        cache_backend.set("key", b"value1")
        cache_backend.set("key", b"value2")
        assert cache_backend.get("key") == b"value2"

    def test_delete(self, cache_backend):
        """Test delete operation."""
        cache_backend.set("key", b"value")
        assert cache_backend.delete("key") is True
        assert cache_backend.get("key") is None
        assert cache_backend.delete("key") is False

    def test_exists(self, cache_backend):
        """Test exists operation."""
        assert not cache_backend.exists("key")
        cache_backend.set("key", b"value")
        assert cache_backend.exists("key")
        cache_backend.delete("key")
        assert not cache_backend.exists("key")

    def test_clear(self, cache_backend):
        """Test clear operation."""
        for i in range(10):
            cache_backend.set(f"key{i}", b"value")

        count = cache_backend.clear()
        assert count >= 10
        assert cache_backend.get("key0") is None


class TestTTLOperations:
    """Test TTL operations work the same on both backends."""

    def test_ttl_expiration(self, cache_backend):
        """Test that TTL expires entries."""
        cache_backend.set("temp", b"data", ttl=1)
        assert cache_backend.exists("temp")

        time.sleep(1.5)
        assert not cache_backend.exists("temp")

    def test_ttl_no_expiration(self, cache_backend):
        """Test that entries without TTL don't expire."""
        cache_backend.set("permanent", b"data")
        time.sleep(1)
        assert cache_backend.exists("permanent")


class TestBatchOperations:
    """Test batch operations work the same on both backends."""

    def test_get_many(self, cache_backend):
        """Test batch get operation."""
        cache_backend.set("key1", b"value1")
        cache_backend.set("key2", b"value2")
        cache_backend.set("key3", b"value3")

        values = cache_backend.get_many(["key1", "key2", "key3", "missing"])
        assert len(values) == 3
        assert values["key1"] == b"value1"
        assert values["key2"] == b"value2"
        assert values["key3"] == b"value3"

    def test_set_many(self, cache_backend):
        """Test batch set operation."""
        items = {
            "batch1": b"value1",
            "batch2": b"value2",
            "batch3": b"value3"
        }

        count = cache_backend.set_many(items)
        assert count == 3

        assert cache_backend.get("batch1") == b"value1"
        assert cache_backend.get("batch2") == b"value2"
        assert cache_backend.get("batch3") == b"value3"

    def test_delete_many(self, cache_backend):
        """Test batch delete operation."""
        for i in range(5):
            cache_backend.set(f"del{i}", b"value")

        count = cache_backend.delete_many(["del0", "del2", "del4", "missing"])
        assert count == 3

        assert not cache_backend.exists("del0")
        assert cache_backend.exists("del1")
        assert not cache_backend.exists("del2")


class TestKeysOperations:
    """Test keys operations work the same on both backends."""

    def test_keys_all(self, cache_backend):
        """Test listing all keys."""
        for i in range(5):
            cache_backend.set(f"list_key{i}", b"value")

        keys = cache_backend.keys()
        assert len(keys) >= 5
        for i in range(5):
            assert f"list_key{i}" in keys

    def test_keys_pattern(self, cache_backend):
        """Test listing keys with pattern."""
        cache_backend.set("user:1", b"data")
        cache_backend.set("user:2", b"data")
        cache_backend.set("post:1", b"data")

        user_keys = cache_backend.keys("user:*")
        assert len(user_keys) == 2
        assert "user:1" in user_keys
        assert "user:2" in user_keys


class TestStatistics:
    """Test statistics work the same on both backends."""

    def test_stats_structure(self, cache_backend):
        """Test that stats have required fields."""
        stats = cache_backend.get_stats()

        assert hasattr(stats, 'hits')
        assert hasattr(stats, 'misses')
        assert hasattr(stats, 'sets')
        assert hasattr(stats, 'deletes')
        assert hasattr(stats, 'hit_rate')

    def test_stats_tracking(self, cache_backend):
        """Test that stats are tracked."""
        # Clear and get initial stats
        cache_backend.clear()

        # Perform operations
        cache_backend.set("key", b"value")
        cache_backend.get("key")  # Hit
        cache_backend.get("missing")  # Miss

        stats = cache_backend.get_stats()
        assert stats.sets >= 1
        assert stats.hits >= 1
        assert stats.misses >= 1


class TestMetadata:
    """Test metadata operations."""

    def test_set_with_metadata(self, cache_backend):
        """Test setting values with metadata."""
        metadata = {"endpoint": "/api", "user": "123"}
        result = cache_backend.set("key", b"value", metadata=metadata)
        assert result is True

        # Should be able to retrieve value
        value = cache_backend.get("key")
        assert value == b"value"


class TestEdgeCases:
    """Test edge cases work the same on both backends."""

    def test_empty_value(self, cache_backend):
        """Test storing empty value."""
        cache_backend.set("empty", b"")
        assert cache_backend.get("empty") == b""

    def test_large_value(self, cache_backend):
        """Test storing large value."""
        large = b"x" * 100000  # 100KB
        cache_backend.set("large", large)
        assert cache_backend.get("large") == large

    def test_binary_data(self, cache_backend):
        """Test storing binary data."""
        binary = bytes(range(256))
        cache_backend.set("binary", binary)
        assert cache_backend.get("binary") == binary


class TestCacheService:
    """
    Test a service using cache backend abstraction.

    This demonstrates that application code can work with any backend.
    """

    class UserService:
        """Example service using cache."""

        def __init__(self, cache: CacheBackend):
            self.cache = cache
            self.db_calls = 0

        def get_user(self, user_id: str) -> dict:
            """Get user with caching."""
            key = f"user:{user_id}"
            cached = self.cache.get(key)

            if cached:
                return {
                    "id": user_id,
                    "name": cached.decode(),
                    "from_cache": True
                }

            # Simulate DB call
            self.db_calls += 1
            name = f"User_{user_id}"
            self.cache.set(key, name.encode(), ttl=300)

            return {
                "id": user_id,
                "name": name,
                "from_cache": False
            }

    def test_service_with_backend(self, cache_backend):
        """Test that service works with any backend."""
        service = self.UserService(cache_backend)

        # First call - cache miss
        user = service.get_user("123")
        assert user["from_cache"] is False
        assert service.db_calls == 1

        # Second call - cache hit
        user = service.get_user("123")
        assert user["from_cache"] is True
        assert service.db_calls == 1  # No additional DB call


class TestSwappability:
    """Test that backends can be swapped without code changes."""

    def test_swap_backends(self):
        """Test swapping backends in a service."""

        class CacheManager:
            """Manager that can use any backend."""

            def __init__(self, backend: CacheBackend):
                self.backend = backend

            def store_result(self, key: str, value: bytes):
                return self.backend.set(key, value, ttl=60)

            def fetch_result(self, key: str):
                return self.backend.get(key)

        # Test with InMemory
        inmemory = InMemoryBackend()
        manager1 = CacheManager(inmemory)

        manager1.store_result("test", b"inmemory_data")
        assert manager1.fetch_result("test") == b"inmemory_data"

        # Test with Redis (if available)
        if check_redis_available():
            redis_backend = RedisBackend(
                RedisConfig(key_prefix="test_swap:")
            )
            redis_backend.connect()

            try:
                manager2 = CacheManager(redis_backend)

                manager2.store_result("test", b"redis_data")
                assert manager2.fetch_result("test") == b"redis_data"
            finally:
                redis_backend.clear()
                redis_backend.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


"""
Integration tests for Redis backend.

These tests require an actual Redis server running.
Run Redis with: docker run -d -p 6379:6379 redis

To skip these tests if Redis is not available, run:
pytest -m "not redis"
"""

import pytest
import time
import threading
from src.cache.redis_backend import (
    RedisBackend, RedisConfig, REDIS_AVAILABLE
)


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


# Mark all tests in this module as requiring Redis
pytestmark = pytest.mark.skipif(
    not check_redis_available(),
    reason="Redis server not available"
)


@pytest.fixture
def redis_backend():
    """Fixture providing a connected Redis backend."""
    config = RedisConfig(
        key_prefix="test_integration:",
        socket_timeout=2.0
    )
    backend = RedisBackend(config)

    if not backend.connect():
        pytest.skip("Cannot connect to Redis server")

    # Clear any existing test data
    backend.clear()

    yield backend

    # Cleanup
    backend.clear()
    backend.disconnect()


class TestRedisBasicOperations:
    """Test basic Redis operations with real server."""

    def test_set_and_get(self, redis_backend):
        """Test basic set and get."""
        result = redis_backend.set("test_key", b"test_value")
        assert result is True

        value = redis_backend.get("test_key")
        assert value == b"test_value"

    def test_get_nonexistent(self, redis_backend):
        """Test get for nonexistent key."""
        value = redis_backend.get("nonexistent_key")
        assert value is None

    def test_set_overwrite(self, redis_backend):
        """Test overwriting existing key."""
        redis_backend.set("key", b"value1")
        assert redis_backend.get("key") == b"value1"

        redis_backend.set("key", b"value2")
        assert redis_backend.get("key") == b"value2"

    def test_delete_existing(self, redis_backend):
        """Test deleting existing key."""
        redis_backend.set("key", b"value")

        result = redis_backend.delete("key")
        assert result is True

        assert redis_backend.get("key") is None

    def test_delete_nonexistent(self, redis_backend):
        """Test deleting nonexistent key."""
        result = redis_backend.delete("nonexistent")
        assert result is False

    def test_exists(self, redis_backend):
        """Test exists operation."""
        assert not redis_backend.exists("key")

        redis_backend.set("key", b"value")
        assert redis_backend.exists("key")

        redis_backend.delete("key")
        assert not redis_backend.exists("key")


class TestRedisTTL:
    """Test TTL functionality."""

    def test_set_with_ttl(self, redis_backend):
        """Test setting key with TTL."""
        redis_backend.set("temp_key", b"temp_value", ttl=2)

        # Should exist immediately
        assert redis_backend.exists("temp_key")
        value = redis_backend.get("temp_key")
        assert value == b"temp_value"

        # Should expire after TTL
        time.sleep(2.5)
        assert not redis_backend.exists("temp_key")
        assert redis_backend.get("temp_key") is None

    def test_set_without_ttl(self, redis_backend):
        """Test setting key without TTL (permanent)."""
        redis_backend.set("permanent_key", b"permanent_value")

        # Should still exist after some time
        time.sleep(1)
        assert redis_backend.exists("permanent_key")
        assert redis_backend.get("permanent_key") == b"permanent_value"

    def test_overwrite_with_different_ttl(self, redis_backend):
        """Test overwriting key with different TTL."""
        redis_backend.set("key", b"value1", ttl=10)
        redis_backend.set("key", b"value2", ttl=2)

        time.sleep(2.5)
        assert redis_backend.get("key") is None


class TestRedisMetadata:
    """Test metadata storage."""

    def test_set_with_metadata(self, redis_backend):
        """Test storing metadata with value."""
        metadata = {
            "endpoint": "/api/users",
            "user_id": "123",
            "timestamp": 1234567890
        }

        result = redis_backend.set(
            "api_result",
            b"response_data",
            ttl=300,
            metadata=metadata
        )
        assert result is True

        # Value should be retrievable
        value = redis_backend.get("api_result")
        assert value == b"response_data"

    def test_metadata_expires_with_value(self, redis_backend):
        """Test that metadata expires with the value."""
        metadata = {"info": "test"}
        redis_backend.set("temp", b"data", ttl=1, metadata=metadata)

        time.sleep(1.5)

        # Both value and metadata should be gone
        assert redis_backend.get("temp") is None


class TestRedisBatchOperations:
    """Test batch operations."""

    def test_get_many(self, redis_backend):
        """Test batch get operation."""
        # Set multiple keys
        redis_backend.set("key1", b"value1")
        redis_backend.set("key2", b"value2")
        redis_backend.set("key3", b"value3")

        # Get multiple keys
        values = redis_backend.get_many(["key1", "key2", "key3", "missing"])

        assert len(values) == 3
        assert values["key1"] == b"value1"
        assert values["key2"] == b"value2"
        assert values["key3"] == b"value3"
        assert "missing" not in values

    def test_get_many_empty(self, redis_backend):
        """Test get_many with empty list."""
        values = redis_backend.get_many([])
        assert values == {}

    def test_set_many(self, redis_backend):
        """Test batch set operation."""
        items = {
            "batch1": b"value1",
            "batch2": b"value2",
            "batch3": b"value3"
        }

        count = redis_backend.set_many(items)
        assert count == 3

        # Verify all were set
        assert redis_backend.get("batch1") == b"value1"
        assert redis_backend.get("batch2") == b"value2"
        assert redis_backend.get("batch3") == b"value3"

    def test_set_many_with_ttl(self, redis_backend):
        """Test batch set with TTL."""
        items = {
            "temp1": b"data1",
            "temp2": b"data2"
        }

        redis_backend.set_many(items, ttl=1)

        assert redis_backend.exists("temp1")
        assert redis_backend.exists("temp2")

        time.sleep(1.5)

        assert not redis_backend.exists("temp1")
        assert not redis_backend.exists("temp2")

    def test_set_many_empty(self, redis_backend):
        """Test set_many with empty dict."""
        count = redis_backend.set_many({})
        assert count == 0

    def test_delete_many(self, redis_backend):
        """Test batch delete operation."""
        # Set multiple keys
        for i in range(5):
            redis_backend.set(f"del_key{i}", f"value{i}".encode())

        # Delete some keys
        count = redis_backend.delete_many([
            "del_key0", "del_key2", "del_key4", "missing"
        ])

        # Should delete 3 existing keys
        assert count == 3

        # Verify deletions
        assert not redis_backend.exists("del_key0")
        assert redis_backend.exists("del_key1")
        assert not redis_backend.exists("del_key2")
        assert redis_backend.exists("del_key3")
        assert not redis_backend.exists("del_key4")

    def test_delete_many_empty(self, redis_backend):
        """Test delete_many with empty list."""
        count = redis_backend.delete_many([])
        assert count == 0


class TestRedisKeys:
    """Test key listing operations."""

    def test_keys_all(self, redis_backend):
        """Test listing all keys."""
        # Add some keys
        for i in range(5):
            redis_backend.set(f"key{i}", b"value")

        keys = redis_backend.keys()
        assert len(keys) >= 5
        for i in range(5):
            assert f"key{i}" in keys

    def test_keys_with_pattern(self, redis_backend):
        """Test listing keys with pattern."""
        # Add different types of keys
        redis_backend.set("user:1", b"data")
        redis_backend.set("user:2", b"data")
        redis_backend.set("post:1", b"data")
        redis_backend.set("comment:1", b"data")

        # Get only user keys
        user_keys = redis_backend.keys("user:*")
        assert len(user_keys) == 2
        assert "user:1" in user_keys
        assert "user:2" in user_keys
        assert "post:1" not in user_keys

    def test_keys_empty(self, redis_backend):
        """Test keys when no keys exist."""
        redis_backend.clear()
        keys = redis_backend.keys()
        assert keys == []


class TestRedisClear:
    """Test clear operation."""

    def test_clear_all(self, redis_backend):
        """Test clearing all keys."""
        # Add multiple keys
        for i in range(10):
            redis_backend.set(f"clear_key{i}", b"value")

        count = redis_backend.clear()
        assert count >= 10

        # Verify all cleared
        keys = redis_backend.keys()
        assert len(keys) == 0

    def test_clear_empty(self, redis_backend):
        """Test clearing when no keys exist."""
        redis_backend.clear()
        count = redis_backend.clear()
        assert count == 0


class TestRedisStatistics:
    """Test statistics tracking."""

    def test_stats_tracking(self, redis_backend):
        """Test that statistics are tracked correctly."""
        # Clear stats
        stats = redis_backend.get_stats()
        initial_hits = stats.hits
        initial_misses = stats.misses

        # Perform operations
        redis_backend.set("key1", b"value1")
        redis_backend.set("key2", b"value2")
        redis_backend.get("key1")  # Hit
        redis_backend.get("missing")  # Miss
        redis_backend.delete("key1")

        stats = redis_backend.get_stats()
        assert stats.hits == initial_hits + 1
        assert stats.misses == initial_misses + 1
        assert stats.sets >= 2
        assert stats.deletes >= 1

    def test_hit_rate_calculation(self, redis_backend):
        """Test hit rate calculation."""
        redis_backend.clear()

        # Create a fresh backend to reset stats
        config = RedisConfig(key_prefix="test_hit_rate:")
        backend = RedisBackend(config)
        backend.connect()

        try:
            # Set some keys
            backend.set("hit1", b"value")
            backend.set("hit2", b"value")

            # Generate hits and misses
            backend.get("hit1")  # Hit
            backend.get("hit2")  # Hit
            backend.get("miss1")  # Miss

            stats = backend.get_stats()
            # Hit rate should be 2/3 = 0.666...
            assert 0.6 <= stats.hit_rate <= 0.7
        finally:
            backend.clear()
            backend.disconnect()

    def test_stats_include_redis_info(self, redis_backend):
        """Test that stats include Redis server info."""
        stats = redis_backend.get_stats()

        # Should have size information from Redis
        assert stats.current_size_bytes >= 0
        assert stats.max_size_bytes >= 0


class TestRedisConnectionHandling:
    """Test connection management."""

    def test_reconnect(self):
        """Test reconnecting after disconnect."""
        backend = RedisBackend()

        # Connect
        assert backend.connect()
        assert backend.is_connected

        # Disconnect
        backend.disconnect()
        assert not backend.is_connected

        # Reconnect
        assert backend.connect()
        assert backend.is_connected

        backend.disconnect()

    def test_ping(self, redis_backend):
        """Test ping operation."""
        assert redis_backend.ping()

    def test_context_manager(self):
        """Test using backend as context manager."""
        config = RedisConfig(key_prefix="test_context:")

        with RedisBackend(config) as backend:
            assert backend.is_connected
            backend.set("key", b"value")
            assert backend.get("key") == b"value"

        # Should be disconnected after exiting context
        assert not backend.is_connected


class TestRedisThreadSafety:
    """Test thread safety."""

    def test_concurrent_operations(self, redis_backend):
        """Test concurrent operations from multiple threads."""
        num_threads = 10
        ops_per_thread = 100
        errors = []

        def worker(thread_id):
            try:
                for i in range(ops_per_thread):
                    key = f"thread{thread_id}_key{i}"
                    value = f"thread{thread_id}_value{i}".encode()

                    # Set
                    redis_backend.set(key, value)

                    # Get and verify
                    retrieved = redis_backend.get(key)
                    if retrieved != value:
                        errors.append(
                            f"Thread {thread_id}: Expected {value}, got {retrieved}"
                        )

                    # Delete
                    redis_backend.delete(key)
            except Exception as e:
                errors.append(f"Thread {thread_id}: {str(e)}")

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_concurrent_batch_operations(self, redis_backend):
        """Test concurrent batch operations."""
        num_threads = 5
        errors = []

        def worker(thread_id):
            try:
                # Batch set
                items = {
                    f"batch{thread_id}_{i}": f"value{i}".encode()
                    for i in range(50)
                }
                count = redis_backend.set_many(items)
                if count != 50:
                    errors.append(f"Thread {thread_id}: set_many returned {count}")

                # Batch get
                keys = list(items.keys())
                values = redis_backend.get_many(keys)
                if len(values) != 50:
                    errors.append(f"Thread {thread_id}: get_many returned {len(values)}")

                # Batch delete
                count = redis_backend.delete_many(keys)
                if count != 50:
                    errors.append(f"Thread {thread_id}: delete_many returned {count}")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {str(e)}")

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Batch operation errors: {errors}"


class TestRedisLargeValues:
    """Test handling of large values."""

    def test_large_value(self, redis_backend):
        """Test storing and retrieving large value."""
        # 1MB value
        large_value = b"x" * (1024 * 1024)

        result = redis_backend.set("large", large_value)
        assert result is True

        retrieved = redis_backend.get("large")
        assert retrieved == large_value

    def test_many_small_values(self, redis_backend):
        """Test storing many small values."""
        # Store 1000 small values
        items = {
            f"small{i}": f"value{i}".encode()
            for i in range(1000)
        }

        count = redis_backend.set_many(items)
        assert count == 1000

        # Verify some random ones
        assert redis_backend.get("small0") == b"value0"
        assert redis_backend.get("small500") == b"value500"
        assert redis_backend.get("small999") == b"value999"


class TestRedisEdgeCases:
    """Test edge cases."""

    def test_empty_value(self, redis_backend):
        """Test storing empty value."""
        result = redis_backend.set("empty", b"")
        assert result is True

        value = redis_backend.get("empty")
        assert value == b""

    def test_special_characters_in_key(self, redis_backend):
        """Test keys with special characters."""
        special_key = "key:with:colons:and-dashes_and_underscores"
        redis_backend.set(special_key, b"value")

        value = redis_backend.get(special_key)
        assert value == b"value"

    def test_binary_value(self, redis_backend):
        """Test storing binary data."""
        binary_data = bytes(range(256))
        redis_backend.set("binary", binary_data)

        retrieved = redis_backend.get("binary")
        assert retrieved == binary_data

    def test_zero_ttl(self, redis_backend):
        """Test TTL of 0 (should be treated as no TTL)."""
        redis_backend.set("key", b"value", ttl=0)

        time.sleep(0.5)
        assert redis_backend.exists("key")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not redis"])


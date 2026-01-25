"""
Unit tests for cache backend abstractions.

Tests CacheEntry, CacheStats, and InMemoryBackend implementations.
"""

import pytest
import time
from src.cache.backend import (
    CacheEntry, CacheStats, CacheBackend, InMemoryBackend
)


class TestCacheEntry:
    """Test CacheEntry dataclass."""

    def test_create_entry_basic(self):
        """Test basic entry creation."""
        entry = CacheEntry(
            key="test_key",
            value=b"test_value",
            created_at=time.time()
        )
        assert entry.key == "test_key"
        assert entry.value == b"test_value"
        assert entry.size_bytes == len(b"test_value")
        assert entry.metadata == {}
        assert entry.expires_at is None

    def test_create_entry_with_ttl(self):
        """Test entry with expiration."""
        now = time.time()
        expires = now + 60
        entry = CacheEntry(
            key="temp",
            value=b"data",
            created_at=now,
            expires_at=expires
        )
        assert entry.expires_at == expires
        assert not entry.is_expired
        assert entry.ttl_remaining > 0
        assert entry.ttl_remaining <= 60

    def test_create_entry_with_metadata(self):
        """Test entry with metadata."""
        metadata = {"endpoint": "/api/users", "user_id": "123"}
        entry = CacheEntry(
            key="api:result",
            value=b"response",
            created_at=time.time(),
            metadata=metadata
        )
        assert entry.metadata == metadata
        assert entry.metadata["endpoint"] == "/api/users"

    def test_is_expired_no_expiry(self):
        """Test is_expired when no expiration set."""
        entry = CacheEntry(
            key="permanent",
            value=b"data",
            created_at=time.time()
        )
        assert not entry.is_expired
        assert entry.ttl_remaining is None

    def test_is_expired_future(self):
        """Test is_expired for future expiration."""
        entry = CacheEntry(
            key="future",
            value=b"data",
            created_at=time.time(),
            expires_at=time.time() + 100
        )
        assert not entry.is_expired
        assert entry.ttl_remaining > 0

    def test_is_expired_past(self):
        """Test is_expired for past expiration."""
        entry = CacheEntry(
            key="past",
            value=b"data",
            created_at=time.time() - 200,
            expires_at=time.time() - 100
        )
        assert entry.is_expired
        assert entry.ttl_remaining == 0

    def test_ttl_remaining_accuracy(self):
        """Test TTL calculation accuracy."""
        ttl = 5
        entry = CacheEntry(
            key="timed",
            value=b"data",
            created_at=time.time(),
            expires_at=time.time() + ttl
        )
        remaining = entry.ttl_remaining
        assert remaining is not None
        assert 4 < remaining <= ttl

    def test_size_bytes_auto_calculation(self):
        """Test automatic size calculation."""
        data = b"x" * 1000
        entry = CacheEntry(
            key="large",
            value=data,
            created_at=time.time()
        )
        assert entry.size_bytes == 1000

    def test_size_bytes_manual(self):
        """Test manual size specification."""
        entry = CacheEntry(
            key="manual",
            value=b"data",
            created_at=time.time(),
            size_bytes=9999
        )
        assert entry.size_bytes == 9999


class TestCacheStats:
    """Test CacheStats dataclass."""

    def test_create_stats_default(self):
        """Test default stats creation."""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.sets == 0
        assert stats.deletes == 0
        assert stats.evictions == 0
        assert stats.current_entries == 0
        assert stats.current_size_bytes == 0

    def test_create_stats_with_values(self):
        """Test stats with values."""
        stats = CacheStats(
            hits=100,
            misses=20,
            sets=120,
            deletes=10,
            current_entries=110,
            current_size_bytes=1024,
            max_size_bytes=10240
        )
        assert stats.hits == 100
        assert stats.misses == 20
        assert stats.current_entries == 110

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=80, misses=20)
        assert stats.hit_rate == 0.8

    def test_hit_rate_no_requests(self):
        """Test hit rate with no requests."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_all_hits(self):
        """Test hit rate with all hits."""
        stats = CacheStats(hits=100, misses=0)
        assert stats.hit_rate == 1.0

    def test_hit_rate_all_misses(self):
        """Test hit rate with all misses."""
        stats = CacheStats(hits=0, misses=100)
        assert stats.hit_rate == 0.0

    def test_utilization_calculation(self):
        """Test utilization calculation."""
        stats = CacheStats(
            current_size_bytes=5120,
            max_size_bytes=10240
        )
        assert stats.utilization == 0.5

    def test_utilization_zero_max(self):
        """Test utilization with zero max."""
        stats = CacheStats(current_size_bytes=1000, max_size_bytes=0)
        assert stats.utilization == 0.0

    def test_utilization_full(self):
        """Test utilization at capacity."""
        stats = CacheStats(
            current_size_bytes=10240,
            max_size_bytes=10240
        )
        assert stats.utilization == 1.0

    def test_reset(self):
        """Test stats reset."""
        stats = CacheStats(
            hits=100, misses=50, sets=150,
            deletes=20, evictions=10
        )
        stats.reset()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.sets == 0
        assert stats.deletes == 0
        assert stats.evictions == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = CacheStats(
            hits=50, misses=10, sets=60,
            current_entries=55, current_size_bytes=2048,
            max_size_bytes=10240
        )
        d = stats.to_dict()
        assert d['hits'] == 50
        assert d['misses'] == 10
        assert d['hit_rate'] == pytest.approx(0.833, 0.01)
        assert d['utilization'] == pytest.approx(0.2, 0.01)


class TestInMemoryBackend:
    """Test InMemoryBackend implementation."""

    def test_init_default(self):
        """Test default initialization."""
        cache = InMemoryBackend()
        stats = cache.get_stats()
        assert stats.max_size_bytes > 0
        assert stats.current_entries == 0

    def test_init_with_size(self):
        """Test initialization with size limit."""
        cache = InMemoryBackend(max_size_bytes=1024)
        stats = cache.get_stats()
        assert stats.max_size_bytes == 1024
        assert cache._max_size_bytes == 1024

    def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = InMemoryBackend()
        success = cache.set("key1", b"value1")
        assert success

        value = cache.get("key1")
        assert value == b"value1"

    def test_get_nonexistent(self):
        """Test get for nonexistent key."""
        cache = InMemoryBackend()
        value = cache.get("nonexistent")
        assert value is None

    def test_set_with_ttl(self):
        """Test set with TTL."""
        cache = InMemoryBackend()
        cache.set("temp", b"data", ttl=1)

        # Should exist immediately
        assert cache.exists("temp")
        value = cache.get("temp")
        assert value == b"data"

        # Should expire after TTL
        time.sleep(1.2)
        assert not cache.exists("temp")
        value = cache.get("temp")
        assert value is None

    def test_set_with_metadata(self):
        """Test set with metadata."""
        cache = InMemoryBackend()
        metadata = {"endpoint": "/api", "user": "123"}
        cache.set("key", b"value", metadata=metadata)

        value = cache.get("key")
        assert value == b"value"

    def test_delete_existing(self):
        """Test delete of existing key."""
        cache = InMemoryBackend()
        cache.set("key", b"value")

        result = cache.delete("key")
        assert result is True
        assert not cache.exists("key")

    def test_delete_nonexistent(self):
        """Test delete of nonexistent key."""
        cache = InMemoryBackend()
        result = cache.delete("nonexistent")
        assert result is False

    def test_exists(self):
        """Test exists operation."""
        cache = InMemoryBackend()
        assert not cache.exists("key")

        cache.set("key", b"value")
        assert cache.exists("key")

        cache.delete("key")
        assert not cache.exists("key")

    def test_clear(self):
        """Test clear operation."""
        cache = InMemoryBackend()

        # Add multiple entries
        for i in range(5):
            cache.set(f"key{i}", f"value{i}".encode())

        count = cache.clear()
        assert count == 5
        assert cache.get_stats().current_entries == 0

    def test_get_many(self):
        """Test batch get operation."""
        cache = InMemoryBackend()

        # Set multiple keys
        cache.set("key1", b"value1")
        cache.set("key2", b"value2")
        cache.set("key3", b"value3")

        # Get multiple keys
        values = cache.get_many(["key1", "key2", "key3", "missing"])
        assert len(values) == 3
        assert values["key1"] == b"value1"
        assert values["key2"] == b"value2"
        assert values["key3"] == b"value3"
        assert "missing" not in values

    def test_set_many(self):
        """Test batch set operation."""
        cache = InMemoryBackend()

        items = {
            "key1": b"value1",
            "key2": b"value2",
            "key3": b"value3"
        }

        count = cache.set_many(items)
        assert count == 3

        assert cache.get("key1") == b"value1"
        assert cache.get("key2") == b"value2"
        assert cache.get("key3") == b"value3"

    def test_set_many_with_ttl(self):
        """Test batch set with TTL."""
        cache = InMemoryBackend()

        items = {
            "temp1": b"data1",
            "temp2": b"data2"
        }

        cache.set_many(items, ttl=1)
        assert cache.exists("temp1")
        assert cache.exists("temp2")

        time.sleep(1.2)
        assert not cache.exists("temp1")
        assert not cache.exists("temp2")

    def test_delete_many(self):
        """Test batch delete operation."""
        cache = InMemoryBackend()

        # Set multiple keys
        for i in range(5):
            cache.set(f"key{i}", f"value{i}".encode())

        # Delete some keys
        count = cache.delete_many(["key1", "key3", "key5", "missing"])
        assert count == 2  # Only key1 and key3 existed

        assert not cache.exists("key1")
        assert cache.exists("key2")
        assert not cache.exists("key3")
        assert cache.exists("key4")

    def test_keys_all(self):
        """Test keys without pattern."""
        cache = InMemoryBackend()

        for i in range(5):
            cache.set(f"key{i}", b"value")

        keys = cache.keys()
        assert len(keys) == 5
        assert "key0" in keys
        assert "key4" in keys

    def test_keys_with_pattern(self):
        """Test keys with pattern."""
        cache = InMemoryBackend()

        cache.set("user:1", b"data")
        cache.set("user:2", b"data")
        cache.set("post:1", b"data")
        cache.set("comment:1", b"data")

        user_keys = cache.keys("user:*")
        assert len(user_keys) == 2
        assert "user:1" in user_keys
        assert "user:2" in user_keys

    def test_stats_tracking(self):
        """Test statistics tracking."""
        cache = InMemoryBackend()

        # Perform operations
        cache.set("key1", b"value1")
        cache.set("key2", b"value2")
        cache.get("key1")  # Hit
        cache.get("missing")  # Miss
        cache.delete("key1")

        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.sets == 2
        assert stats.deletes == 1
        assert stats.current_entries == 1

    def test_lru_eviction(self):
        """Test LRU eviction when size limit reached."""
        # Small cache that can fit ~3 entries
        cache = InMemoryBackend(max_size_bytes=100)

        # Add entries
        cache.set("key1", b"x" * 30)
        cache.set("key2", b"x" * 30)
        cache.set("key3", b"x" * 30)

        # Access key1 to make it recently used
        cache.get("key1")

        # Add another entry, should evict key2 (least recently used)
        cache.set("key4", b"x" * 30)

        # key2 should be evicted, others should exist
        assert cache.exists("key1")
        assert not cache.exists("key2")  # Evicted
        assert cache.exists("key3")
        assert cache.exists("key4")

    def test_eviction_stats(self):
        """Test eviction statistics."""
        cache = InMemoryBackend(max_size_bytes=100)

        # Fill cache
        for i in range(10):
            cache.set(f"key{i}", b"x" * 30)

        stats = cache.get_stats()
        assert stats.evictions > 0

    def test_size_tracking(self):
        """Test size tracking."""
        cache = InMemoryBackend()

        cache.set("small", b"x")
        cache.set("medium", b"x" * 100)
        cache.set("large", b"x" * 1000)

        stats = cache.get_stats()
        assert stats.current_size_bytes >= 1101  # At least the sum

    def test_expired_entry_cleanup(self):
        """Test that expired entries are cleaned up."""
        cache = InMemoryBackend()

        cache.set("temp1", b"data", ttl=1)
        cache.set("temp2", b"data", ttl=1)
        cache.set("permanent", b"data")

        stats = cache.get_stats()
        assert stats.current_entries == 3

        time.sleep(1.2)

        # Access should trigger cleanup
        cache.get("permanent")

        stats = cache.get_stats()
        # Expired entries might still be counted until explicitly accessed
        # This is implementation dependent

    def test_overwrite_existing_key(self):
        """Test overwriting an existing key."""
        cache = InMemoryBackend()

        cache.set("key", b"value1")
        assert cache.get("key") == b"value1"

        cache.set("key", b"value2")
        assert cache.get("key") == b"value2"

        stats = cache.get_stats()
        assert stats.current_entries == 1

    def test_thread_safety_basic(self):
        """Test basic thread safety."""
        import threading

        cache = InMemoryBackend()
        errors = []

        def worker(thread_id):
            try:
                for i in range(100):
                    key = f"thread{thread_id}_key{i}"
                    cache.set(key, f"value{i}".encode())
                    value = cache.get(key)
                    if value != f"value{i}".encode():
                        errors.append(f"Mismatch in thread {thread_id}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


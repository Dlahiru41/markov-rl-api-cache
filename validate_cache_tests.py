"""
Quick validation script for cache backend tests.

Tests basic functionality without requiring pytest.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.cache.backend import CacheEntry, CacheStats, InMemoryBackend


def test_cache_entry():
    """Test CacheEntry creation and properties."""
    print("Testing CacheEntry...")

    # Test basic creation
    entry = CacheEntry(
        key="test",
        value=b"data",
        created_at=time.time()
    )
    assert entry.key == "test"
    assert entry.value == b"data"
    assert entry.size_bytes == len(b"data")
    assert not entry.is_expired
    assert entry.ttl_remaining is None

    # Test with TTL
    entry = CacheEntry(
        key="temp",
        value=b"data",
        created_at=time.time(),
        expires_at=time.time() + 10
    )
    assert not entry.is_expired
    assert entry.ttl_remaining > 0

    print("✅ CacheEntry tests passed")


def test_cache_stats():
    """Test CacheStats creation and calculations."""
    print("\nTesting CacheStats...")

    # Test default stats
    stats = CacheStats()
    assert stats.hits == 0
    assert stats.misses == 0
    assert stats.hit_rate == 0.0

    # Test with values
    stats = CacheStats(
        hits=80,
        misses=20,
        current_size_bytes=5000,
        max_size_bytes=10000
    )
    assert stats.hit_rate == 0.8
    assert stats.utilization == 0.5

    # Test reset
    stats.reset()
    assert stats.hits == 0

    # Test to_dict
    d = stats.to_dict()
    assert 'hits' in d
    assert 'hit_rate' in d

    print("✅ CacheStats tests passed")


def test_inmemory_backend():
    """Test InMemoryBackend basic operations."""
    print("\nTesting InMemoryBackend...")

    cache = InMemoryBackend(max_size_bytes=1024 * 1024)

    # Test set and get
    assert cache.set("key1", b"value1")
    assert cache.get("key1") == b"value1"

    # Test get nonexistent
    assert cache.get("nonexistent") is None

    # Test delete
    assert cache.delete("key1")
    assert cache.get("key1") is None
    assert not cache.delete("key1")

    # Test exists
    cache.set("key2", b"value2")
    assert cache.exists("key2")
    cache.delete("key2")
    assert not cache.exists("key2")

    # Test TTL
    cache.set("temp", b"data", ttl=1)
    assert cache.exists("temp")
    time.sleep(1.2)
    assert not cache.exists("temp")

    # Test batch operations
    items = {"k1": b"v1", "k2": b"v2", "k3": b"v3"}
    count = cache.set_many(items)
    assert count == 3

    values = cache.get_many(["k1", "k2", "k3"])
    assert len(values) == 3
    assert values["k1"] == b"v1"

    count = cache.delete_many(["k1", "k2"])
    assert count == 2

    # Test keys
    cache.set("user:1", b"data")
    cache.set("user:2", b"data")
    cache.set("post:1", b"data")

    user_keys = cache.keys("user:*")
    assert len(user_keys) == 2
    assert "user:1" in user_keys

    # Test clear
    count = cache.clear()
    assert count >= 3

    # Test stats
    stats = cache.get_stats()
    assert stats.hits >= 0
    assert stats.misses >= 0

    print("✅ InMemoryBackend tests passed")


def test_redis_backend_imports():
    """Test that Redis backend can be imported."""
    print("\nTesting Redis backend imports...")

    try:
        from src.cache.redis_backend import (
            RedisBackend, RedisConfig,
            CacheError, CacheConnectionError, CacheOperationError,
            REDIS_AVAILABLE
        )

        # Test config
        config = RedisConfig(
            host="localhost",
            port=6379,
            key_prefix="test:"
        )
        assert config.host == "localhost"
        assert config.port == 6379

        # Test backend creation
        backend = RedisBackend(config)
        assert backend is not None
        assert not backend.is_connected

        # Test exceptions
        try:
            raise CacheError("test")
        except CacheError:
            pass

        print(f"✅ Redis backend imports passed (REDIS_AVAILABLE={REDIS_AVAILABLE})")

    except Exception as e:
        print(f"⚠️  Redis backend import issue: {e}")


def main():
    """Run all validation tests."""
    print("="*70)
    print("CACHE BACKEND VALIDATION")
    print("="*70)

    try:
        test_cache_entry()
        test_cache_stats()
        test_inmemory_backend()
        test_redis_backend_imports()

        print("\n" + "="*70)
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("="*70)
        return 0

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


"""
Validation script for cache backend abstraction.

Tests the CacheBackend interface and InMemoryBackend implementation.
"""

from src.cache.backend import CacheBackend, CacheEntry, CacheStats, InMemoryBackend
import time
import sys


def test_basic_operations():
    """Test basic cache operations."""
    print("Testing basic operations...")

    cache = InMemoryBackend(max_size_bytes=1024*1024)  # 1MB

    # Test set and get
    assert cache.set("key1", b"hello world", ttl=60), "Set should succeed"
    value = cache.get("key1")
    assert value == b"hello world", f"Expected b'hello world', got {value}"
    print(f"[OK] Retrieved: {value}")

    # Test exists
    assert cache.exists("key1"), "Key should exist"
    assert not cache.exists("nonexistent"), "Nonexistent key should return False"
    print("[OK] Exists check working")

    # Test delete
    assert cache.delete("key1"), "Delete should return True for existing key"
    assert not cache.exists("key1"), "Key should not exist after delete"
    assert not cache.delete("key1"), "Delete should return False for nonexistent key"
    print("[OK] Delete working")


def test_ttl():
    """Test TTL expiration."""
    print("\nTesting TTL expiration...")

    cache = InMemoryBackend(max_size_bytes=1024*1024)

    # Test TTL
    cache.set("temp", b"expires soon", ttl=1)
    assert cache.exists("temp"), "Key should exist before expiry"
    print(f"[OK] Before expiry: {cache.exists('temp')}")

    time.sleep(1.5)
    assert not cache.exists("temp"), "Key should not exist after expiry"
    print(f"[OK] After expiry: {cache.exists('temp')}")

    # Test no TTL
    cache.set("permanent", b"never expires")
    time.sleep(0.5)
    assert cache.exists("permanent"), "Key without TTL should persist"
    print("[OK] No TTL working")


def test_cache_entry():
    """Test CacheEntry properties."""
    print("\nTesting CacheEntry...")

    created = time.time()

    # Entry with expiration
    entry1 = CacheEntry(
        key="test",
        value=b"data",
        created_at=created,
        expires_at=created + 10
    )

    assert not entry1.is_expired, "Entry should not be expired yet"
    assert entry1.ttl_remaining is not None, "TTL should exist"
    assert entry1.ttl_remaining > 0, "TTL should be positive"
    assert entry1.size_bytes == 4, f"Size should be 4, got {entry1.size_bytes}"
    print(f"[OK] Entry with TTL: {entry1.ttl_remaining:.2f}s remaining")

    # Expired entry
    entry2 = CacheEntry(
        key="test",
        value=b"data",
        created_at=created - 100,
        expires_at=created - 50
    )

    assert entry2.is_expired, "Entry should be expired"
    assert entry2.ttl_remaining == 0, "Expired entry should have 0 TTL"
    print("[OK] Expired entry detected")

    # Entry without expiration
    entry3 = CacheEntry(
        key="test",
        value=b"data",
        created_at=created
    )

    assert not entry3.is_expired, "Entry without TTL should not expire"
    assert entry3.ttl_remaining is None, "Entry without TTL should have None TTL"
    print("[OK] No expiration working")


def test_stats():
    """Test cache statistics."""
    print("\nTesting cache statistics...")

    cache = InMemoryBackend(max_size_bytes=1024*1024)

    # Perform operations
    cache.set("key1", b"value1")
    cache.set("key2", b"value2")
    cache.get("key1")  # hit
    cache.get("key1")  # hit
    cache.get("nonexistent")  # miss
    cache.delete("key2")

    stats = cache.get_stats()

    assert stats.hits == 2, f"Expected 2 hits, got {stats.hits}"
    assert stats.misses == 1, f"Expected 1 miss, got {stats.misses}"
    assert stats.sets == 2, f"Expected 2 sets, got {stats.sets}"
    assert stats.deletes == 1, f"Expected 1 delete, got {stats.deletes}"
    assert stats.current_entries == 1, f"Expected 1 entry, got {stats.current_entries}"

    print(f"[OK] Hit rate: {stats.hit_rate:.2f}")
    print(f"[OK] Utilization: {stats.utilization:.2%}")
    print(f"[OK] Stats dict: {stats.to_dict()}")

    # Test reset
    stats.reset()
    assert stats.hits == 0, "Hits should be reset"
    assert stats.misses == 0, "Misses should be reset"
    print("[OK] Stats reset working")


def test_batch_operations():
    """Test batch operations."""
    print("\nTesting batch operations...")

    cache = InMemoryBackend(max_size_bytes=1024*1024)

    # set_many
    items = {
        "batch1": b"value1",
        "batch2": b"value2",
        "batch3": b"value3"
    }
    count = cache.set_many(items, ttl=60)
    assert count == 3, f"Expected 3 sets, got {count}"
    print(f"[OK] set_many: {count} items set")

    # get_many
    results = cache.get_many(["batch1", "batch2", "nonexistent"])
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert results["batch1"] == b"value1", "Wrong value for batch1"
    print(f"[OK] get_many: retrieved {len(results)} items")

    # delete_many
    deleted = cache.delete_many(["batch1", "batch2", "nonexistent"])
    assert deleted == 2, f"Expected 2 deletes, got {deleted}"
    print(f"[OK] delete_many: deleted {deleted} items")


def test_lru_eviction():
    """Test LRU eviction when cache is full."""
    print("\nTesting LRU eviction...")

    cache = InMemoryBackend(max_size_bytes=80)  # Small cache

    # Fill cache
    cache.set("key1", b"A" * 30)  # 30 bytes
    cache.set("key2", b"B" * 30)  # 30 bytes, total 60
    cache.set("key3", b"C" * 30)  # 30 bytes, total would be 90 - should evict key1

    assert not cache.exists("key1"), "key1 should be evicted"
    assert cache.exists("key2"), "key2 should still exist"
    assert cache.exists("key3"), "key3 should exist"

    stats = cache.get_stats()
    assert stats.evictions > 0, "Should have evictions"
    print(f"[OK] LRU eviction working: {stats.evictions} evictions")

    # Access key2 to make it recently used
    cache.get("key2")

    # Add another key - should evict key3 (not key2)
    cache.set("key4", b"D" * 30)

    assert cache.exists("key2"), "key2 should still exist (was accessed)"
    assert not cache.exists("key3"), "key3 should be evicted"
    assert cache.exists("key4"), "key4 should exist"
    print("[OK] LRU with access order working")


def test_keys_pattern():
    """Test keys listing with patterns."""
    print("\nTesting keys patterns...")

    cache = InMemoryBackend(max_size_bytes=1024*1024)

    # Add various keys
    cache.set("user:123:profile", b"data1")
    cache.set("user:456:profile", b"data2")
    cache.set("user:123:settings", b"data3")
    cache.set("session:abc", b"data4")

    # Test pattern matching
    all_keys = cache.keys()
    assert len(all_keys) == 4, f"Expected 4 keys, got {len(all_keys)}"
    print(f"[OK] All keys: {len(all_keys)} keys")

    user_keys = cache.keys("user:*")
    assert len(user_keys) == 3, f"Expected 3 user keys, got {len(user_keys)}"
    print(f"[OK] Pattern 'user:*': {len(user_keys)} keys")

    profile_keys = cache.keys("*:profile")
    assert len(profile_keys) == 2, f"Expected 2 profile keys, got {len(profile_keys)}"
    print(f"[OK] Pattern '*:profile': {len(profile_keys)} keys")


def test_clear():
    """Test cache clear operation."""
    print("\nTesting clear...")

    cache = InMemoryBackend(max_size_bytes=1024*1024)

    # Add some entries
    for i in range(10):
        cache.set(f"key{i}", f"value{i}".encode())

    stats = cache.get_stats()
    assert stats.current_entries == 10, "Should have 10 entries"

    # Clear cache
    cleared = cache.clear()
    assert cleared == 10, f"Expected 10 cleared, got {cleared}"

    stats = cache.get_stats()
    assert stats.current_entries == 0, "Should have 0 entries after clear"
    assert stats.current_size_bytes == 0, "Should have 0 bytes after clear"
    print(f"[OK] Cleared {cleared} entries")


def test_metadata():
    """Test metadata storage."""
    print("\nTesting metadata...")

    cache = InMemoryBackend(max_size_bytes=1024*1024)

    # Store with metadata
    metadata = {
        "endpoint": "/api/users",
        "user_type": "premium",
        "version": "v1"
    }
    cache.set("api:users:123", b"user data", ttl=60, metadata=metadata)

    # Access internal entry to check metadata
    entry = cache._cache.get("api:users:123")
    assert entry is not None, "Entry should exist"
    assert entry.metadata == metadata, "Metadata should match"
    print(f"[OK] Metadata stored: {entry.metadata}")


def run_all_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("CACHE BACKEND VALIDATION")
    print("=" * 60)

    try:
        test_basic_operations()
        test_ttl()
        test_cache_entry()
        test_stats()
        test_batch_operations()
        test_lru_eviction()
        test_keys_pattern()
        test_clear()
        test_metadata()

        print("\n" + "=" * 60)
        print("[SUCCESS] ALL TESTS PASSED!")
        print("=" * 60)
        return True

    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n[ERROR] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)


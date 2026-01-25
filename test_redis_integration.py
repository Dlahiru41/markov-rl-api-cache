"""
Integration test demonstrating Redis and InMemory backends working together.
Shows the power of the CacheBackend abstraction.
"""

import sys
from src.cache.backend import CacheBackend, InMemoryBackend
from src.cache.redis_backend import RedisBackend, RedisConfig


class CacheService:
    """Example service that works with any cache backend."""

    def __init__(self, cache: CacheBackend):
        self.cache = cache
        self._db_calls = 0

    def get_user(self, user_id: str) -> dict:
        """Get user with caching."""
        key = f"user:{user_id}"
        cached = self.cache.get(key)

        if cached:
            return {"id": user_id, "data": cached.decode(), "from_cache": True}

        # Simulate DB call
        self._db_calls += 1
        user_data = f"user_data_{user_id}"
        self.cache.set(key, user_data.encode(), ttl=300)

        return {"id": user_id, "data": user_data, "from_cache": False}

    def cache_users_batch(self, user_ids: list) -> int:
        """Cache multiple users at once."""
        items = {
            f"user:{uid}": f"user_data_{uid}".encode()
            for uid in user_ids
        }
        return self.cache.set_many(items, ttl=300)

    def get_stats(self) -> dict:
        stats = self.cache.get_stats()
        return {
            "hit_rate": f"{stats.hit_rate:.2%}",
            "hits": stats.hits,
            "misses": stats.misses,
            "db_calls": self._db_calls,
            "entries": stats.current_entries
        }


def test_with_backend(backend: CacheBackend, backend_name: str):
    """Test service with a specific backend."""
    print(f"\n{'='*60}")
    print(f"Testing with {backend_name}")
    print(f"{'='*60}")

    service = CacheService(backend)

    # Test 1: First access (cache miss)
    print("\n1. First access (cache miss):")
    user1 = service.get_user("123")
    print(f"   Result: {user1}")
    assert not user1['from_cache'], "Should be from DB"

    # Test 2: Second access (cache hit)
    print("\n2. Second access (cache hit):")
    user1_cached = service.get_user("123")
    print(f"   Result: {user1_cached}")
    assert user1_cached['from_cache'], "Should be from cache"

    # Test 3: Batch operations
    print("\n3. Batch cache multiple users:")
    user_ids = [f"{i}" for i in range(10)]
    count = service.cache_users_batch(user_ids)
    print(f"   Cached {count} users")

    # Test 4: Retrieve cached users
    print("\n4. Retrieve cached users:")
    for uid in ["0", "5", "9"]:
        user = service.get_user(uid)
        print(f"   User {uid}: from_cache={user['from_cache']}")

    # Test 5: Statistics
    print("\n5. Performance Statistics:")
    stats = service.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Test 6: Pattern operations
    if hasattr(backend, 'keys'):
        print("\n6. Pattern operations:")
        try:
            user_keys = backend.keys("user:*")
            print(f"   Found {len(user_keys)} user keys")
        except Exception as e:
            print(f"   Keys operation: {e}")


def main():
    print("="*60)
    print("CACHE BACKEND INTEGRATION TEST")
    print("Testing both InMemory and Redis backends")
    print("="*60)

    # Test 1: InMemoryBackend
    print("\n[Test 1] InMemoryBackend")
    inmemory = InMemoryBackend(max_size_bytes=1024*1024)
    test_with_backend(inmemory, "InMemoryBackend")

    # Test 2: RedisBackend (if available)
    print("\n\n[Test 2] RedisBackend")

    config = RedisConfig(
        host='localhost',
        port=6379,
        key_prefix='integration_test:',
        socket_timeout=2.0
    )
    redis_backend = RedisBackend(config)

    if redis_backend.connect():
        print("[OK] Redis available!")
        try:
            test_with_backend(redis_backend, "RedisBackend")
        finally:
            # Cleanup
            print("\nCleaning up Redis test data...")
            test_keys = redis_backend.keys()
            if test_keys:
                redis_backend.delete_many(test_keys)
                print(f"Deleted {len(test_keys)} test keys")
            redis_backend.disconnect()
    else:
        print("[SKIP] Redis not available")
        print("To test with Redis: docker run -d -p 6379:6379 redis")

    # Test 3: Demonstrate swappability
    print("\n\n[Test 3] Backend Swappability")
    print("="*60)
    print("\nThe CacheService code is IDENTICAL for both backends!")
    print("This demonstrates the power of the CacheBackend abstraction.")
    print("\nIn production:")
    print("  - Use InMemoryBackend for unit tests")
    print("  - Use RedisBackend for production")
    print("  - Switch with a single line change")
    print("  - Application code stays the same!")

    print("\n" + "="*60)
    print("[SUCCESS] Integration test completed!")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


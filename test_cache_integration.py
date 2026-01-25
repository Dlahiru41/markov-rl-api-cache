"""
Integration test demonstrating cache backend abstraction.
Shows how to swap implementations without changing application code.
"""

from src.cache.backend import CacheBackend, InMemoryBackend
from typing import Optional
import json


class UserService:
    """Example service using cache backend abstraction."""

    def __init__(self, cache: CacheBackend):
        """
        Initialize service with any cache backend.

        Args:
            cache: Any implementation of CacheBackend
        """
        self.cache = cache
        self._db_calls = 0  # Track database calls for demo

    def get_user(self, user_id: str) -> Optional[dict]:
        """Get user from cache or 'database'."""
        # Try cache first
        cache_key = f"user:{user_id}"
        cached = self.cache.get(cache_key)

        if cached:
            print(f"  [CACHE HIT] User {user_id} from cache")
            return json.loads(cached.decode())

        # Cache miss - fetch from "database"
        print(f"  [CACHE MISS] User {user_id} from database")
        self._db_calls += 1
        user = self._fetch_from_db(user_id)

        # Store in cache
        if user:
            self.cache.set(
                cache_key,
                json.dumps(user).encode(),
                ttl=300,
                metadata={"service": "user", "id": user_id}
            )

        return user

    def _fetch_from_db(self, user_id: str) -> dict:
        """Simulate database fetch."""
        return {
            "id": user_id,
            "name": f"User {user_id}",
            "email": f"user{user_id}@example.com"
        }

    def invalidate_user(self, user_id: str) -> bool:
        """Remove user from cache."""
        return self.cache.delete(f"user:{user_id}")

    def get_cache_stats(self) -> dict:
        """Get cache performance stats."""
        stats = self.cache.get_stats()
        return {
            "hit_rate": f"{stats.hit_rate:.1%}",
            "hits": stats.hits,
            "misses": stats.misses,
            "db_calls": self._db_calls,
            "entries": stats.current_entries
        }


def demo_abstraction():
    """Demonstrate cache backend abstraction."""
    print("=" * 60)
    print("CACHE BACKEND ABSTRACTION DEMO")
    print("=" * 60)

    # Create service with InMemoryBackend
    # In production, we'd pass RedisBackend instead
    cache = InMemoryBackend(max_size_bytes=1024*1024)
    service = UserService(cache)

    print("\n1. First access (cache miss):")
    user1 = service.get_user("123")
    print(f"   Result: {user1}")

    print("\n2. Second access (cache hit):")
    user1_cached = service.get_user("123")
    print(f"   Result: {user1_cached}")

    print("\n3. Access different user (cache miss):")
    user2 = service.get_user("456")
    print(f"   Result: {user2}")

    print("\n4. Access both users (both cache hits):")
    service.get_user("123")
    service.get_user("456")

    print("\n5. Invalidate user 123:")
    invalidated = service.invalidate_user("123")
    print(f"   Invalidated: {invalidated}")

    print("\n6. Access user 123 again (cache miss after invalidation):")
    user1_fresh = service.get_user("123")
    print(f"   Result: {user1_fresh}")

    print("\n7. Performance Statistics:")
    stats = service.get_cache_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n8. Pattern-based operations:")
    all_user_keys = cache.keys("user:*")
    print(f"   All user keys: {all_user_keys}")

    print("\n" + "=" * 60)
    print("KEY INSIGHT: The service code doesn't know or care")
    print("whether it's using InMemoryBackend or RedisBackend!")
    print("We can swap implementations without changing UserService.")
    print("=" * 60)


def demo_swap_backends():
    """Show how easy it is to swap backends."""
    print("\n\n" + "=" * 60)
    print("SWAPPING BACKENDS DEMO")
    print("=" * 60)

    # Start with in-memory cache
    print("\nUsing InMemoryBackend:")
    cache1 = InMemoryBackend()
    service1 = UserService(cache1)
    service1.get_user("789")
    print(f"  Backend type: {type(cache1).__name__}")

    # Could easily swap to a different backend:
    # cache2 = RedisBackend(host='localhost', port=6379)
    # service2 = UserService(cache2)
    # service2.get_user("789")

    print("\nIn production, just change one line:")
    print("  # cache = InMemoryBackend()  # Development")
    print("  cache = RedisBackend()        # Production")

    print("\nApplication code stays the same!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        demo_abstraction()
        demo_swap_backends()
        print("\n[SUCCESS] Integration test completed!")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


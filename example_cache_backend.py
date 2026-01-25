"""
Example validation from user's requirements.
This demonstrates the exact usage pattern specified in the request.
"""

from src.cache.backend import CacheBackend, CacheEntry, CacheStats, InMemoryBackend
import time

print("=" * 60)
print("CACHE BACKEND EXAMPLE - As Specified in Requirements")
print("=" * 60)

# Test in-memory backend
cache = InMemoryBackend(max_size_bytes=1024*1024)  # 1MB

# Test basic operations
print("\n1. Basic Operations:")
cache.set("key1", b"hello world", ttl=60)
value = cache.get("key1")
print(f"   Retrieved: {value}")  # b"hello world"

# Test TTL
print("\n2. TTL Testing:")
cache.set("temp", b"expires soon", ttl=1)
print(f"   Before expiry: {cache.exists('temp')}")  # True
time.sleep(1.5)
print(f"   After expiry: {cache.exists('temp')}")  # False

# Test stats
print("\n3. Statistics:")
stats = cache.get_stats()
print(f"   Hit rate: {stats.hit_rate:.2f}")
print(f"   Utilization: {stats.utilization:.2%}")

print("\n" + "=" * 60)
print("Example completed successfully!")
print("=" * 60)


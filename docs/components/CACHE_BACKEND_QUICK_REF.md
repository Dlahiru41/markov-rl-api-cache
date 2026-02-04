# Cache Backend - Quick Reference

## Import

```python
from src.cache.backend import CacheBackend, CacheEntry, CacheStats, InMemoryBackend
```

## Create Cache

```python
# In-memory cache with 1MB limit
cache = InMemoryBackend(max_size_bytes=1024*1024)

# Default 100MB cache
cache = InMemoryBackend()
```

## Basic Operations

```python
# Set with TTL (in seconds)
cache.set("key", b"value", ttl=300)

# Set without expiration
cache.set("key", b"value")

# Get value
value = cache.get("key")  # Returns bytes or None

# Check existence
if cache.exists("key"):
    print("Key exists")

# Delete key
deleted = cache.delete("key")  # Returns True if existed
```

## Batch Operations

```python
# Set multiple
items = {"key1": b"val1", "key2": b"val2", "key3": b"val3"}
count = cache.set_many(items, ttl=300)

# Get multiple
values = cache.get_many(["key1", "key2", "key3"])
# Returns: {"key1": b"val1", "key2": b"val2"}

# Delete multiple
deleted = cache.delete_many(["key1", "key2"])
```

## Metadata

```python
cache.set(
    key="api:users:123",
    value=b"user data",
    ttl=600,
    metadata={
        "endpoint": "/api/users",
        "user_type": "premium",
        "version": "v1"
    }
)
```

## Pattern Matching

```python
# List all keys
all_keys = cache.keys()

# Glob patterns
user_keys = cache.keys("user:*")
profile_keys = cache.keys("*:profile")
specific = cache.keys("user:123:*")
```

## Statistics

```python
stats = cache.get_stats()

# Access metrics
print(f"Hit rate: {stats.hit_rate:.2%}")
print(f"Hits: {stats.hits}")
print(f"Misses: {stats.misses}")
print(f"Entries: {stats.current_entries}")
print(f"Size: {stats.current_size_bytes}")
print(f"Utilization: {stats.utilization:.2%}")
print(f"Evictions: {stats.evictions}")

# Export to dict
stats_dict = stats.to_dict()

# Reset counters
stats.reset()
```

## Clear Cache

```python
# Remove all entries
count = cache.clear()
print(f"Cleared {count} entries")
```

## CacheEntry Properties

```python
entry = CacheEntry(
    key="test",
    value=b"data",
    created_at=time.time(),
    expires_at=time.time() + 3600
)

# Check expiration
if entry.is_expired:
    print("Entry expired")

# Get TTL remaining
ttl = entry.ttl_remaining  # Seconds or None
```

## Creating Custom Backend

```python
from src.cache.backend import CacheBackend, CacheStats
from typing import Optional, Dict, List, Any

class MyBackend(CacheBackend):
    def __init__(self):
        self._stats = CacheStats()
    
    def get(self, key: str) -> Optional[bytes]:
        # Your implementation
        pass
    
    def set(self, key: str, value: bytes, ttl: Optional[int] = None,
            metadata: Optional[Dict[str, Any]] = None) -> bool:
        # Your implementation
        pass
    
    def delete(self, key: str) -> bool:
        # Your implementation
        pass
    
    def exists(self, key: str) -> bool:
        # Your implementation
        pass
    
    def clear(self) -> int:
        # Your implementation
        pass
    
    def get_stats(self) -> CacheStats:
        return self._stats
```

## Common Patterns

### Cache with Fallback

```python
def get_user(user_id: str) -> dict:
    # Try cache first
    cached = cache.get(f"user:{user_id}")
    if cached:
        return json.loads(cached)
    
    # Fetch from database
    user = db.get_user(user_id)
    
    # Store in cache
    cache.set(f"user:{user_id}", json.dumps(user).encode(), ttl=300)
    
    return user
```

### Bulk Cache Warming

```python
# Warm cache with user data
users = db.get_all_users()
items = {
    f"user:{user['id']}": json.dumps(user).encode()
    for user in users
}
cache.set_many(items, ttl=3600)
```

### Cache Invalidation

```python
# Invalidate all user caches
user_keys = cache.keys("user:*")
cache.delete_many(user_keys)

# Invalidate specific pattern
cache.delete_many(cache.keys(f"user:{user_id}:*"))
```

### Monitoring

```python
import logging

def log_cache_stats():
    stats = cache.get_stats()
    logging.info(
        f"Cache: {stats.hit_rate:.1%} hit rate, "
        f"{stats.current_entries} entries, "
        f"{stats.utilization:.1%} full"
    )
```

## Files Reference

- **Implementation**: `src/cache/backend.py`
- **Tests**: `validate_cache_backend.py`
- **Quick Test**: `quick_test_cache.py`
- **Example**: `example_cache_backend.py`
- **Full Docs**: `CACHE_BACKEND_README.md`
- **Summary**: `CACHE_BACKEND_COMPLETE.md`

## Key Points

- ✅ All values must be **bytes** (not strings)
- ✅ TTL is in **seconds**
- ✅ Expired entries return **None** on get
- ✅ Pattern matching uses **glob style** (*, ?)
- ✅ Metadata is **optional** but useful for tracking
- ✅ Statistics track **all operations**
- ✅ LRU eviction is **automatic** when full

## Performance Tips

1. Use **batch operations** for multiple keys
2. Set appropriate **TTL** to avoid stale data
3. Monitor **hit rate** and **utilization**
4. Use **patterns** for efficient key management
5. Set **max_size_bytes** based on available memory

## Validation

```bash
# Full test suite
python validate_cache_backend.py

# Quick test
python quick_test_cache.py

# Example from requirements
python example_cache_backend.py
```


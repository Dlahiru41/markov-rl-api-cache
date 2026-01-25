# Cache Backend Abstraction

## Overview

The cache backend module (`src/cache/backend.py`) provides an abstract interface for cache implementations, allowing the system to swap between different cache backends (Redis, in-memory, etc.) without changing the rest of the codebase.

## Components

### 1. CacheEntry

A dataclass representing a cached item with metadata and expiration tracking.

**Attributes:**
- `key` (str): The cache key
- `value` (bytes): The cached data
- `created_at` (float): Unix timestamp when cached
- `expires_at` (Optional[float]): When the entry expires (None if no expiration)
- `size_bytes` (int): Size of the cached value in bytes
- `metadata` (Optional[Dict]): Extra information (endpoint, user_type, etc.)

**Properties:**
- `is_expired`: Returns True if current time is past expires_at
- `ttl_remaining`: Seconds until expiration (0 if expired, None if no expiry)

**Example:**
```python
from src.cache.backend import CacheEntry
import time

entry = CacheEntry(
    key="user:123",
    value=b"user data",
    created_at=time.time(),
    expires_at=time.time() + 3600,  # 1 hour
    metadata={"endpoint": "/api/users", "user_type": "premium"}
)

print(f"Expired: {entry.is_expired}")
print(f"TTL remaining: {entry.ttl_remaining}s")
```

### 2. CacheStats

A dataclass for tracking cache performance metrics.

**Attributes:**
- `hits` (int): Number of successful gets
- `misses` (int): Number of gets that returned None
- `sets` (int): Number of set operations
- `deletes` (int): Number of delete operations
- `evictions` (int): Number of entries removed due to capacity/TTL
- `current_entries` (int): Current number of entries in cache
- `current_size_bytes` (int): Total size of cached data
- `max_size_bytes` (int): Maximum allowed size

**Properties:**
- `hit_rate`: Calculated as hits / (hits + misses), or 0 if no requests
- `utilization`: Calculated as current_size_bytes / max_size_bytes

**Methods:**
- `reset()`: Zero out the counters (hits, misses, sets, deletes, evictions)
- `to_dict()`: Convert to dictionary for logging

**Example:**
```python
from src.cache.backend import InMemoryBackend

cache = InMemoryBackend()
cache.set("key1", b"value1")
cache.get("key1")  # hit
cache.get("key2")  # miss

stats = cache.get_stats()
print(f"Hit rate: {stats.hit_rate:.2%}")
print(f"Utilization: {stats.utilization:.2%}")
print(f"Stats: {stats.to_dict()}")
```

### 3. CacheBackend (Abstract Base Class)

The abstract interface that all cache implementations must follow.

**Required Methods (must be implemented by subclasses):**

- `get(key: str) -> Optional[bytes]`
  - Retrieve value from cache
  - Returns None if not found or expired

- `set(key: str, value: bytes, ttl: Optional[int] = None, metadata: Optional[Dict] = None) -> bool`
  - Store value in cache with optional TTL (in seconds)
  - Returns True if successfully stored

- `delete(key: str) -> bool`
  - Remove a key from cache
  - Returns True if the key existed

- `exists(key: str) -> bool`
  - Check if key exists and is not expired

- `clear() -> int`
  - Remove all entries
  - Returns count of entries deleted

- `get_stats() -> CacheStats`
  - Return current statistics

**Optional Methods (with default implementations):**

- `get_many(keys: List[str]) -> Dict[str, bytes]`
  - Get multiple keys at once
  - Default: loops calling get()

- `set_many(items: Dict[str, bytes], ttl: Optional[int] = None) -> int`
  - Set multiple key-value pairs
  - Default: loops calling set()

- `delete_many(keys: List[str]) -> int`
  - Delete multiple keys
  - Default: loops calling delete()

- `keys(pattern: Optional[str] = None) -> List[str]`
  - List keys matching pattern
  - Default: raises NotImplementedError

### 4. InMemoryBackend

A simple in-memory cache implementation for testing and development.

**Features:**
- Dictionary-based storage
- TTL support with automatic expiration checking
- LRU eviction when capacity is reached
- Configurable maximum size
- Full implementation of all CacheBackend methods

**Constructor:**
```python
InMemoryBackend(max_size_bytes: int = 1024 * 1024 * 100)  # 100MB default
```

**Example:**
```python
from src.cache.backend import InMemoryBackend
import time

# Create cache with 1MB limit
cache = InMemoryBackend(max_size_bytes=1024*1024)

# Basic operations
cache.set("key1", b"hello world", ttl=60)
value = cache.get("key1")
print(f"Value: {value}")  # b"hello world"

# TTL expiration
cache.set("temp", b"expires soon", ttl=1)
print(f"Before: {cache.exists('temp')}")  # True
time.sleep(1.5)
print(f"After: {cache.exists('temp')}")   # False

# Batch operations
cache.set_many({
    "user:1": b"Alice",
    "user:2": b"Bob",
    "user:3": b"Charlie"
}, ttl=300)

users = cache.get_many(["user:1", "user:2"])
print(f"Users: {users}")

# Pattern matching
user_keys = cache.keys("user:*")
print(f"User keys: {user_keys}")

# Statistics
stats = cache.get_stats()
print(f"Hit rate: {stats.hit_rate:.2%}")
print(f"Utilization: {stats.utilization:.2%}")

# Clear all
count = cache.clear()
print(f"Cleared {count} entries")
```

## Usage Patterns

### Basic Cache Operations

```python
from src.cache.backend import InMemoryBackend

cache = InMemoryBackend()

# Store data
cache.set("api:users:123", b'{"name": "Alice"}', ttl=300)

# Retrieve data
data = cache.get("api:users:123")
if data:
    print(f"Cache hit: {data}")
else:
    print("Cache miss")

# Check existence
if cache.exists("api:users:123"):
    print("Key exists")

# Delete
cache.delete("api:users:123")
```

### With Metadata

```python
# Store with metadata for tracking
cache.set(
    key="api:search:query123",
    value=b"search results...",
    ttl=600,
    metadata={
        "endpoint": "/api/search",
        "user_type": "premium",
        "query": "machine learning"
    }
)
```

### Batch Operations

```python
# Batch set
items = {
    f"session:{i}": f"session_data_{i}".encode()
    for i in range(100)
}
count = cache.set_many(items, ttl=3600)
print(f"Stored {count} items")

# Batch get
keys = [f"session:{i}" for i in range(10)]
results = cache.get_many(keys)

# Batch delete
deleted = cache.delete_many(keys)
print(f"Deleted {deleted} items")
```

### Pattern-Based Key Management

```python
# Store various keys
cache.set("user:123:profile", b"...")
cache.set("user:123:settings", b"...")
cache.set("user:456:profile", b"...")
cache.set("session:abc", b"...")

# Find all user keys
user_keys = cache.keys("user:*")

# Find all profile keys
profile_keys = cache.keys("*:profile")

# Delete all user:123 keys
keys_to_delete = cache.keys("user:123:*")
cache.delete_many(keys_to_delete)
```

### Monitoring Cache Performance

```python
# Get statistics
stats = cache.get_stats()

print(f"Cache Performance:")
print(f"  Hit Rate: {stats.hit_rate:.2%}")
print(f"  Hits: {stats.hits}")
print(f"  Misses: {stats.misses}")
print(f"  Entries: {stats.current_entries}")
print(f"  Size: {stats.current_size_bytes / 1024:.1f} KB")
print(f"  Utilization: {stats.utilization:.2%}")
print(f"  Evictions: {stats.evictions}")

# Reset counters (keeps entries, resets metrics)
stats.reset()

# Export for logging
import json
print(json.dumps(stats.to_dict(), indent=2))
```

## Creating Custom Backends

To create a custom cache backend (e.g., RedisBackend), extend CacheBackend:

```python
from src.cache.backend import CacheBackend, CacheStats
from typing import Optional, Dict, List, Any
import redis

class RedisBackend(CacheBackend):
    def __init__(self, host='localhost', port=6379, db=0):
        self.client = redis.Redis(host=host, port=port, db=db)
        self._stats = CacheStats()
    
    def get(self, key: str) -> Optional[bytes]:
        value = self.client.get(key)
        if value:
            self._stats.hits += 1
        else:
            self._stats.misses += 1
        return value
    
    def set(self, key: str, value: bytes, ttl: Optional[int] = None,
            metadata: Optional[Dict[str, Any]] = None) -> bool:
        try:
            if ttl:
                self.client.setex(key, ttl, value)
            else:
                self.client.set(key, value)
            self._stats.sets += 1
            return True
        except Exception:
            return False
    
    def delete(self, key: str) -> bool:
        result = self.client.delete(key)
        if result:
            self._stats.deletes += 1
        return bool(result)
    
    def exists(self, key: str) -> bool:
        return self.client.exists(key) > 0
    
    def clear(self) -> int:
        count = self.client.dbsize()
        self.client.flushdb()
        return count
    
    def get_stats(self) -> CacheStats:
        info = self.client.info('stats')
        self._stats.current_entries = self.client.dbsize()
        return self._stats
    
    def keys(self, pattern: Optional[str] = None) -> List[str]:
        pattern = pattern or '*'
        return [k.decode() for k in self.client.keys(pattern)]
```

## Testing

Run the validation script to test the implementation:

```bash
python validate_cache_backend.py
```

Or use the quick test:

```bash
python quick_test_cache.py
```

## Design Benefits

1. **Abstraction**: Swap cache implementations without changing application code
2. **Testability**: Use InMemoryBackend for tests, production backend in deployment
3. **Extensibility**: Easy to add new backends (Redis, Memcached, DynamoDB, etc.)
4. **Monitoring**: Built-in statistics for performance tracking
5. **Type Safety**: Strong typing with Python type hints
6. **Flexibility**: Optional methods can be overridden for optimization

## Files

- `src/cache/backend.py` - Core implementation
- `validate_cache_backend.py` - Comprehensive test suite
- `quick_test_cache.py` - Quick validation
- `CACHE_BACKEND_README.md` - This documentation


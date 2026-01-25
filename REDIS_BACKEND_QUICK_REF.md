# Redis Backend - Quick Reference

## Import

```python
from src.cache.redis_backend import RedisBackend, RedisConfig, CacheError
```

## Create & Connect

```python
# Default config
config = RedisConfig()
backend = RedisBackend(config)

if backend.connect():
    print("Connected!")

# Custom config
config = RedisConfig(
    host='redis.example.com',
    port=6380,
    password='secret',
    key_prefix='myapp:',
    max_memory=1024*1024*200  # 200MB
)
```

## Basic Operations

```python
# Set
backend.set("key", b"value", ttl=300)

# Get
value = backend.get("key")

# Exists
if backend.exists("key"):
    print("Found")

# Delete
backend.delete("key")

# Clear (WARNING: flushes DB!)
backend.clear()
```

## Batch Operations

```python
# Set many
items = {"key1": b"val1", "key2": b"val2"}
backend.set_many(items, ttl=600)

# Get many
values = backend.get_many(["key1", "key2"])
# Returns: {"key1": b"val1", "key2": b"val2"}

# Delete many
backend.delete_many(["key1", "key2"])
```

## With Metadata

```python
backend.set(
    "api:result",
    b"data",
    ttl=300,
    metadata={"endpoint": "/api", "user": "123"}
)
```

## Pattern Matching

```python
# All keys
all_keys = backend.keys()

# Pattern
user_keys = backend.keys("user:*")
sessions = backend.keys("session:*")
```

## Statistics

```python
stats = backend.get_stats()
print(f"Hit rate: {stats.hit_rate:.2%}")
print(f"Hits: {stats.hits}")
print(f"Misses: {stats.misses}")
print(f"Entries: {stats.current_entries}")
```

## Context Manager

```python
with RedisBackend(config) as cache:
    cache.set("key", b"value")
    value = cache.get("key")
# Auto-disconnects
```

## Health Check

```python
if backend.is_connected:
    print("Connected")

if backend.ping():
    print("Redis responding")
```

## Error Handling

```python
from src.cache.redis_backend import CacheConnectionError

try:
    value = backend.get("key")
except CacheConnectionError:
    # Reconnect
    backend.connect()
```

## Configuration Options

```python
config = RedisConfig(
    host="localhost",           # Redis host
    port=6379,                  # Redis port
    db=0,                       # Database number
    password=None,              # Auth password
    max_memory=100*1024*1024,   # 100MB limit
    eviction_policy="allkeys-lru",
    key_prefix="markov_cache:",
    socket_timeout=5.0,         # Seconds
    max_connections=50,         # Pool size
    decode_responses=False      # Keep as bytes
)
```

## Common Patterns

### Cache with Fallback
```python
def get_user(user_id):
    key = f"user:{user_id}"
    cached = backend.get(key)
    
    if cached:
        return json.loads(cached)
    
    user = db.get_user(user_id)
    backend.set(key, json.dumps(user).encode(), ttl=300)
    return user
```

### Bulk Cache Warm
```python
users = db.get_all_users()
items = {
    f"user:{u['id']}": json.dumps(u).encode()
    for u in users
}
backend.set_many(items, ttl=3600)
```

### Invalidation
```python
# Invalidate all sessions
keys = backend.keys("session:*")
backend.delete_many(keys)
```

### With InMemory Fallback
```python
from src.cache.backend import InMemoryBackend

def create_cache():
    redis_backend = RedisBackend(config)
    if redis_backend.connect():
        return redis_backend
    return InMemoryBackend()
```

## Thread Safety

```python
# Safe to use from multiple threads
import threading

def worker(id):
    backend.set(f"thread:{id}", b"data")
    backend.get(f"thread:{id}")

threads = [threading.Thread(target=worker, args=(i,)) 
           for i in range(10)]
for t in threads: t.start()
for t in threads: t.join()
```

## Validation

```bash
# Quick test
python quick_test_redis.py

# Full validation
python validate_redis_backend.py

# Example from requirements
python example_redis_backend.py
```

## Start Redis

```bash
# Docker
docker run -d -p 6379:6379 redis

# With password
docker run -d -p 6379:6379 redis redis-server --requirepass mypassword

# With persistence
docker run -d -p 6379:6379 -v redis-data:/data redis redis-server --appendonly yes
```

## Performance Tips

1. Use batch operations for multiple keys
2. Set appropriate TTL on all keys
3. Use key prefixes for namespacing
4. Monitor hit rate (aim for >70%)
5. Use SCAN (already done in `keys()`)
6. Configure connection pool size

## Key Differences from InMemory

| Feature | Redis | InMemory |
|---------|-------|----------|
| Persistence | ✅ | ❌ |
| Distributed | ✅ | ❌ |
| Setup | Redis server needed | None |
| Speed | Fast | Fastest |
| Best for | Production | Testing |

## Troubleshooting

**Can't connect?**
```python
# Check config
print(config.to_dict())

# Test connection
if not backend.connect():
    print("Redis not running?")
```

**Timeouts?**
```python
config = RedisConfig(socket_timeout=30.0)
```

**Memory full?**
```python
config = RedisConfig(
    max_memory=1024*1024*500,
    eviction_policy='allkeys-lru'
)
```

## Files

- Implementation: `src/cache/redis_backend.py`
- Full docs: `REDIS_BACKEND_README.md`
- Tests: `validate_redis_backend.py`
- Quick test: `quick_test_redis.py`
- Example: `example_redis_backend.py`


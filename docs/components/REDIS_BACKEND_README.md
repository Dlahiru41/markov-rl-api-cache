# Redis Backend Implementation

## Overview

The Redis backend (`src/cache/redis_backend.py`) provides a production-ready cache implementation using Redis. It implements the `CacheBackend` interface with full error handling, connection pooling, batch operations, and metadata support.

## Features

### Core Features
- ✅ Full CacheBackend interface implementation
- ✅ Connection pooling (thread-safe)
- ✅ Automatic reconnection on transient failures
- ✅ Optimized batch operations using Redis pipelines
- ✅ Metadata storage in separate keys
- ✅ Non-blocking key scanning (SCAN instead of KEYS)
- ✅ Context manager support (`with` statement)
- ✅ Comprehensive error handling
- ✅ Thread-safe statistics tracking

### Production Features
- ✅ Configurable connection timeouts
- ✅ Key prefixing for namespace isolation
- ✅ TTL support with Redis SETEX
- ✅ Connection health checks (ping)
- ✅ Graceful error handling and logging
- ✅ Memory management configuration

## Installation

Redis backend requires the `redis` package:

```bash
pip install redis
```

Or install with optional fast parser:

```bash
pip install redis hiredis
```

## Components

### 1. RedisConfig

Configuration dataclass for Redis connection.

**Attributes:**
- `host` (str): Redis server hostname, default "localhost"
- `port` (int): Redis port, default 6379
- `db` (int): Database number, default 0
- `password` (Optional[str]): Authentication password, default None
- `max_memory` (int): Maximum memory in bytes, default 100MB
- `eviction_policy` (str): Eviction policy, default "allkeys-lru"
- `key_prefix` (str): Prefix for all keys, default "markov_cache:"
- `socket_timeout` (float): Connection timeout in seconds, default 5.0
- `max_connections` (int): Connection pool size, default 50
- `decode_responses` (bool): Decode bytes to strings, default False

**Methods:**
- `to_dict()`: Convert config to dictionary

**Example:**
```python
from src.cache.redis_backend import RedisConfig

# Default configuration
config = RedisConfig()

# Custom configuration
config = RedisConfig(
    host="redis.example.com",
    port=6380,
    db=1,
    password="secret123",
    key_prefix="myapp:",
    max_memory=1024*1024*200,  # 200MB
    socket_timeout=10.0,
    max_connections=100
)

print(config.to_dict())
```

### 2. RedisBackend

Main Redis cache backend implementation.

#### Initialization

```python
from src.cache.redis_backend import RedisBackend, RedisConfig

config = RedisConfig(host='localhost', port=6379)
backend = RedisBackend(config)
```

#### Connection Management

**`connect() -> bool`**
- Establishes connection to Redis
- Creates connection pool
- Tests connection with PING
- Returns True if successful

**`disconnect() -> None`**
- Closes connection cleanly
- Releases connection pool

**`ping() -> bool`**
- Health check for Redis connection
- Returns True if Redis responds

**`is_connected` property**
- Check if connection is alive
- Returns True if connected and responsive

**Example:**
```python
backend = RedisBackend(config)

if backend.connect():
    print("Connected!")
    
    # Check health
    if backend.ping():
        print("Redis is healthy")
    
    # ... use backend ...
    
    backend.disconnect()
else:
    print("Connection failed")
```

#### Core Operations

All operations implement the CacheBackend interface:

**`get(key: str) -> Optional[bytes]`**
```python
value = backend.get("user:123")
if value:
    print(f"Found: {value}")
```

**`set(key: str, value: bytes, ttl: Optional[int] = None, metadata: Optional[Dict] = None) -> bool`**
```python
# Set with TTL
backend.set("session:abc", b"session data", ttl=3600)

# Set with metadata
backend.set(
    "api:result",
    b"response data",
    ttl=300,
    metadata={"endpoint": "/api/search", "user": "123"}
)
```

**`delete(key: str) -> bool`**
```python
if backend.delete("user:123"):
    print("Deleted")
```

**`exists(key: str) -> bool`**
```python
if backend.exists("user:123"):
    print("Key exists")
```

**`clear() -> int`**
```python
# WARNING: Clears entire Redis database!
count = backend.clear()
print(f"Cleared {count} keys")
```

**`get_stats() -> CacheStats`**
```python
stats = backend.get_stats()
print(f"Hit rate: {stats.hit_rate:.2%}")
print(f"Entries: {stats.current_entries}")
```

#### Batch Operations

Optimized operations using Redis pipelines:

**`get_many(keys: List[str]) -> Dict[str, bytes]`**
```python
# Single round trip to Redis
values = backend.get_many(["user:1", "user:2", "user:3"])
# Returns: {"user:1": b"data1", "user:2": b"data2", ...}
```

**`set_many(items: Dict[str, bytes], ttl: Optional[int] = None) -> int`**
```python
items = {
    "session:1": b"data1",
    "session:2": b"data2",
    "session:3": b"data3"
}
count = backend.set_many(items, ttl=3600)
print(f"Set {count} items")
```

**`delete_many(keys: List[str]) -> int`**
```python
deleted = backend.delete_many(["old:1", "old:2", "old:3"])
print(f"Deleted {deleted} keys")
```

**`keys(pattern: Optional[str] = None) -> List[str]`**
```python
# List all keys (uses SCAN, non-blocking)
all_keys = backend.keys()

# Pattern matching
user_keys = backend.keys("user:*")
session_keys = backend.keys("session:*")
```

#### Context Manager

Use with `with` statement for automatic connection management:

```python
from src.cache.redis_backend import RedisBackend, RedisConfig

config = RedisConfig(host='localhost', port=6379)

with RedisBackend(config) as cache:
    # Automatically connected
    cache.set("key", b"value")
    value = cache.get("key")
    print(value)
# Automatically disconnected
```

## Error Handling

The backend provides custom exceptions:

- `CacheError`: Base exception for cache operations
- `CacheConnectionError`: Connection-related errors
- `CacheOperationError`: Operation failures

**Example:**
```python
from src.cache.redis_backend import (
    RedisBackend, RedisConfig,
    CacheError, CacheConnectionError
)

backend = RedisBackend(config)

try:
    value = backend.get("key")
except CacheConnectionError as e:
    print(f"Connection error: {e}")
    # Handle reconnection
except CacheError as e:
    print(f"Cache error: {e}")
```

## Usage Patterns

### Basic Usage

```python
from src.cache.redis_backend import RedisBackend, RedisConfig

# Create and connect
config = RedisConfig(host='localhost', port=6379, key_prefix='myapp:')
backend = RedisBackend(config)

if backend.connect():
    # Set values
    backend.set("user:123", b'{"name": "Alice"}', ttl=300)
    
    # Get values
    data = backend.get("user:123")
    
    # Check existence
    if backend.exists("user:123"):
        print("User exists")
    
    # Delete
    backend.delete("user:123")
    
    backend.disconnect()
```

### With Metadata

```python
# Store with tracking metadata
backend.set(
    key="api:search:query123",
    value=b"search results...",
    ttl=600,
    metadata={
        "endpoint": "/api/search",
        "user_id": "456",
        "query": "machine learning",
        "timestamp": time.time()
    }
)
```

### Batch Operations

```python
# Efficient bulk operations
users = {
    f"user:{i}": f"user_data_{i}".encode()
    for i in range(1000)
}

# Single pipeline operation
count = backend.set_many(users, ttl=3600)
print(f"Cached {count} users")

# Bulk retrieval
user_ids = [f"user:{i}" for i in range(100)]
cached_users = backend.get_many(user_ids)
```

### Pattern-Based Management

```python
# Find all session keys
session_keys = backend.keys("session:*")

# Delete all expired sessions
backend.delete_many(session_keys)

# Find specific patterns
profile_keys = backend.keys("user:*:profile")
```

### Health Monitoring

```python
import logging

def check_redis_health(backend):
    if not backend.is_connected:
        logging.warning("Redis not connected")
        if backend.connect():
            logging.info("Reconnected to Redis")
        else:
            logging.error("Failed to reconnect")
            return False
    
    if not backend.ping():
        logging.error("Redis ping failed")
        return False
    
    stats = backend.get_stats()
    logging.info(f"Redis stats: {stats.to_dict()}")
    
    if stats.hit_rate < 0.5:
        logging.warning(f"Low hit rate: {stats.hit_rate:.2%}")
    
    return True
```

### With Fallback to InMemory

```python
from src.cache.backend import InMemoryBackend
from src.cache.redis_backend import RedisBackend, RedisConfig

def create_cache():
    """Create cache with Redis, fallback to in-memory."""
    config = RedisConfig()
    redis_cache = RedisBackend(config)
    
    if redis_cache.connect():
        print("Using Redis cache")
        return redis_cache
    else:
        print("Redis unavailable, using in-memory cache")
        return InMemoryBackend()

# Use in application
cache = create_cache()
cache.set("key", b"value")
```

## Thread Safety

The Redis backend is thread-safe:

1. **Connection Pooling**: Uses redis-py's connection pool (thread-safe by default)
2. **Statistics**: Protected with threading.Lock
3. **Pipeline Operations**: Each pipeline is independent

**Example:**
```python
import threading

backend = RedisBackend(config)
backend.connect()

def worker(thread_id):
    for i in range(100):
        key = f"thread:{thread_id}:item:{i}"
        backend.set(key, f"data_{i}".encode())
        backend.get(key)

# Multiple threads can safely use the same backend
threads = [
    threading.Thread(target=worker, args=(i,))
    for i in range(10)
]

for t in threads:
    t.start()
for t in threads:
    t.join()

stats = backend.get_stats()
print(f"Total operations: {stats.sets + stats.hits + stats.misses}")
```

## Configuration Best Practices

### Development
```python
config = RedisConfig(
    host='localhost',
    port=6379,
    key_prefix='dev:',
    max_memory=1024*1024*50,  # 50MB
    socket_timeout=5.0
)
```

### Production
```python
config = RedisConfig(
    host='redis.prod.example.com',
    port=6379,
    db=0,
    password=os.getenv('REDIS_PASSWORD'),
    key_prefix='prod:',
    max_memory=1024*1024*1024,  # 1GB
    eviction_policy='allkeys-lru',
    socket_timeout=10.0,
    max_connections=200
)
```

### Testing
```python
config = RedisConfig(
    host='localhost',
    port=6379,
    db=15,  # Use separate database for tests
    key_prefix='test:',
    socket_timeout=1.0
)
```

## Performance Tips

1. **Use Batch Operations**: Prefer `set_many` and `get_many` for multiple keys
2. **Set Appropriate TTL**: Prevents memory bloat
3. **Use Key Prefixes**: Namespace isolation and easier management
4. **Monitor Hit Rate**: Aim for >70% hit rate
5. **Use SCAN not KEYS**: Already implemented in `keys()` method
6. **Connection Pooling**: Configured by default
7. **Pipeline Operations**: Used internally for batch operations

## Comparison: Redis vs InMemory

| Feature | RedisBackend | InMemoryBackend |
|---------|--------------|-----------------|
| **Persistence** | ✅ Yes (configurable) | ❌ No |
| **Distributed** | ✅ Yes | ❌ No |
| **Memory Limit** | ✅ Configurable | ✅ Configurable |
| **Performance** | ⚡ Fast (network) | ⚡⚡ Fastest (local) |
| **Thread-Safe** | ✅ Yes | ✅ Yes |
| **Setup** | Requires Redis server | No setup |
| **Best For** | Production, distributed | Testing, single-process |

## Files

- **Implementation**: `src/cache/redis_backend.py` (670 lines)
- **Validation**: `validate_redis_backend.py`
- **Quick Test**: `quick_test_redis.py`
- **Example**: `example_redis_backend.py`
- **Documentation**: `REDIS_BACKEND_README.md` (this file)

## Testing

### Without Redis Server
```bash
python quick_test_redis.py
# Tests imports and config (no Redis needed)
```

### With Redis Server
```bash
# Start Redis
docker run -d -p 6379:6379 redis

# Run tests
python validate_redis_backend.py
python example_redis_backend.py
```

## Troubleshooting

### Redis Not Connecting
```python
backend = RedisBackend(config)
if not backend.connect():
    print("Check:")
    print("1. Is Redis running?")
    print("2. Correct host/port?")
    print("3. Firewall rules?")
    print("4. Authentication (password)?")
```

### Connection Timeouts
```python
# Increase timeout
config = RedisConfig(socket_timeout=30.0)
```

### Memory Issues
```python
# Set memory limit in Redis
config = RedisConfig(
    max_memory=1024*1024*500,  # 500MB
    eviction_policy='allkeys-lru'
)
```

## Next Steps

1. **Deploy Redis**: Set up Redis server for production
2. **Configure**: Adjust settings for your use case
3. **Monitor**: Track hit rate and performance
4. **Scale**: Add Redis Cluster for high availability
5. **Backup**: Configure Redis persistence (RDB/AOF)

---

**Status**: ✅ Production Ready

**Requirements**: Python 3.8+, redis >= 5.0

**Thread-Safe**: Yes

**Production Features**: Yes


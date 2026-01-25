# Cache Manager - Quick Reference

## 🚀 Quick Start

```python
from src.cache.cache_manager import CacheManager, CacheManagerConfig

# Create and start
config = CacheManagerConfig(backend_type='memory')
manager = CacheManager(config)
manager.start()

# Use cache
manager.set('/api/users/123', {'name': 'John'}, ttl=300)
user = manager.get('/api/users/123')

# Stop
manager.stop()
```

## 📋 Configuration

```python
CacheManagerConfig(
    backend_type='memory',            # 'memory' or 'redis'
    backend_config=None,              # Backend-specific config
    default_ttl=300,                  # Default TTL (seconds)
    max_entry_size=1024*1024,         # Max entry size (1MB)
    compression_enabled=True,         # Enable compression
    compression_threshold=1024,       # Compress if > 1KB
    compression_level=6,              # zlib level (1-9)
    serialization_format='pickle'     # 'pickle' or 'json'
)
```

## 🎯 Core Operations

### Get
```python
# Simple get
value = manager.get(key)

# With default
value = manager.get(key, default={'error': 'not found'})
```

### Set
```python
# Simple set
manager.set(key, value)

# With TTL
manager.set(key, value, ttl=600)

# With metadata
manager.set(key, value, ttl=300, metadata={'user': 'admin'})
```

### Delete
```python
success = manager.delete(key)
```

### Get or Set
```python
def factory():
    return expensive_computation()

value = manager.get_or_set(key, factory, ttl=300)
```

## 📡 Prefetch

### Single Prefetch
```python
manager.prefetch('/api/endpoint', priority=0.8)
manager.prefetch('/api/endpoint', params={'id': 123}, priority=0.9)
```

### Batch Prefetch
```python
items = [
    ('/api/users/1', None, 0.7),
    ('/api/users/2', None, 0.6),
    ('/api/products/42', {'details': 'full'}, 0.9),
]
manager.prefetch_many(items)
```

### Queue Inspection
```python
queue = manager.get_prefetch_queue()
for item in queue:
    print(f"{item['endpoint']}: priority={item['priority']}")
```

## 🗑️ Eviction

### LRU Eviction
```python
evicted = manager.evict_lru(count=10)
```

### Probability-Based Eviction
```python
predictions = {
    'key1': 0.1,   # Low probability
    'key2': 0.8,   # High probability
    'key3': 0.05,  # Very low
}
evicted = manager.evict_low_probability(predictions, count=10)
```

## 📊 Metrics

```python
metrics = manager.get_metrics()

# Key metrics
metrics['hit_rate']              # Cache hit rate (0-1)
metrics['hits']                  # Number of hits
metrics['misses']                # Number of misses
metrics['current_entries']       # Entries in cache
metrics['compression_ratio']     # Compression effectiveness
metrics['prefetch_queue_size']   # Prefetch queue size
```

## 🔑 Cache Keys

```python
from src.cache.cache_manager import generate_cache_key

# Simple key
key = generate_cache_key('/api/users/123')

# With parameters (sorted automatically)
key = generate_cache_key('/api/users', {'id': 123, 'filter': 'active'})

# Long keys are hashed
key = generate_cache_key('/api/endpoint', large_params_dict)
```

## 🎨 Common Patterns

### API Response Caching
```python
def get_user(user_id):
    key = f'/api/users/{user_id}'
    
    def fetch():
        return database.query('SELECT * FROM users WHERE id=?', user_id)
    
    return manager.get_or_set(key, fetch, ttl=300)
```

### Conditional Caching
```python
if should_cache:
    manager.set(key, value, ttl=300)
else:
    # Just return value without caching
    pass
```

### Bulk Operations
```python
# Cache multiple items
for item in items:
    manager.set(f'/api/items/{item.id}', item.to_dict(), ttl=600)

# Retrieve multiple items
results = [manager.get(f'/api/items/{id}') for id in item_ids]
```

## 🔧 Backend Selection

### In-Memory (Development)
```python
config = CacheManagerConfig(backend_type='memory')
```

### Redis (Production)
```python
from src.cache.redis_backend import RedisConfig

redis_config = RedisConfig(
    host='localhost',
    port=6379,
    key_prefix='myapp:'
)

config = CacheManagerConfig(
    backend_type='redis',
    backend_config=redis_config
)
```

## ⚙️ Compression

```python
# Enable compression (default)
config = CacheManagerConfig(compression_enabled=True)

# Disable compression
config = CacheManagerConfig(compression_enabled=False)

# Custom threshold (compress only if > 10KB)
config = CacheManagerConfig(
    compression_enabled=True,
    compression_threshold=10240
)

# Faster compression (level 1-3)
config = CacheManagerConfig(
    compression_enabled=True,
    compression_level=3
)
```

## 🔄 Serialization

### Pickle (Default)
```python
config = CacheManagerConfig(serialization_format='pickle')
# Pros: Fast, supports all Python objects
# Cons: Python-only, security concerns
```

### JSON
```python
config = CacheManagerConfig(serialization_format='json')
# Pros: Language-agnostic, safe, readable
# Cons: Limited types, slower
```

## 🐛 Error Handling

```python
try:
    manager.start()
except Exception as e:
    logger.error(f"Failed to start: {e}")
    # Fallback to no-cache mode

try:
    manager.set(key, value)
except Exception as e:
    logger.error(f"Cache set failed: {e}")
    # Continue without caching

# Metrics track errors
metrics = manager.get_metrics()
if metrics['serialization_errors'] > 0:
    logger.warning("Serialization errors detected")
```

## 📝 Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Cache manager logs:
# - Start/stop events
# - Cache hits/misses
# - Prefetch operations
# - Evictions
# - Errors
```

## 🎯 RL Integration

### State for RL Agent
```python
metrics = manager.get_metrics()
state = {
    'hit_rate': metrics['hit_rate'],
    'cache_size': metrics['current_entries'],
    'prefetch_queue': metrics['prefetch_queue_size'],
}
```

### RL Actions
```python
# Evict action
if action == 'evict':
    predictions = markov_model.predict()
    manager.evict_low_probability(predictions, count=10)

# Prefetch action
if action == 'prefetch':
    items = markov_model.predict_next()
    manager.prefetch_many(items)
```

## 🚦 Lifecycle

```python
# Create
manager = CacheManager(config)

# Start (required)
if not manager.start():
    raise RuntimeError("Failed to start cache")

# Check status
if manager.is_running:
    # Use cache
    pass

# Stop (cleanup)
manager.stop()
```

## 📊 Performance Tips

1. **Use appropriate TTL**: Short for dynamic data, long for static
2. **Enable compression for large objects**: >1KB
3. **Use pickle for performance**: Faster than JSON
4. **Batch operations**: Use prefetch_many() for multiple items
5. **Monitor metrics**: Track hit rate and adjust strategy

## 🔍 Debugging

```python
# Get detailed metrics
metrics = manager.get_metrics()
for key, value in metrics.items():
    print(f"{key}: {value}")

# Inspect prefetch queue
queue = manager.get_prefetch_queue()
print(f"Queue size: {len(queue)}")
for item in queue:
    print(f"  {item['endpoint']} (priority={item['priority']})")

# Check compression effectiveness
ratio = metrics['compression_ratio']
print(f"Compression ratio: {ratio:.1%}")
```

## ✅ Validation

```bash
# Run validation script
python validate_cache_manager.py

# Should see:
# ✅ All tests passed
# ✅ Hit rate > 75%
# ✅ Compression works
# ✅ Prefetch functional
```

## 📚 Examples

See `validate_cache_manager.py` for comprehensive examples of all features.

---

**Quick Reference Version**: 1.0  
**Last Updated**: January 25, 2026  
**Status**: ✅ Production Ready


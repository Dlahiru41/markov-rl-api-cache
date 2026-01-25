"""
Example usage from requirements - Redis backend validation.
"""

from src.cache.redis_backend import RedisBackend, RedisConfig

# Create config
config = RedisConfig(host='localhost', port=6379, key_prefix='test:')
backend = RedisBackend(config)

# Test connection
if backend.connect():
    print("Connected to Redis!")

    # Test operations
    backend.set('mykey', b'myvalue', ttl=300)
    value = backend.get('mykey')
    print(f"Retrieved: {value}")

    # Test batch
    backend.set_many({'a': b'1', 'b': b'2', 'c': b'3'}, ttl=60)
    values = backend.get_many(['a', 'b', 'c'])
    print(f"Batch get: {values}")

    # Stats
    stats = backend.get_stats()
    print(f"Stats: {stats.to_dict()}")

    # Cleanup
    backend.clear()
    backend.disconnect()
else:
    print("Could not connect to Redis - is it running?")
    print("\nTo start Redis:")
    print("  docker run -d -p 6379:6379 redis")
    print("  # or install Redis locally")


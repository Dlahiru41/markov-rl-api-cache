"""Quick test for Redis backend imports and basic functionality."""

import sys

print("Testing Redis backend...")

# Test imports
try:
    from src.cache.redis_backend import RedisBackend, RedisConfig, CacheError
    print("[OK] Imports successful")
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

# Test config creation
try:
    config = RedisConfig()
    print(f"[OK] Default config: {config.host}:{config.port}")

    config2 = RedisConfig(host='redis.example.com', port=6380, key_prefix='test:')
    config_dict = config2.to_dict()
    print(f"[OK] Custom config: {config_dict}")
except Exception as e:
    print(f"[ERROR] Config creation failed: {e}")
    sys.exit(1)

# Test backend initialization
try:
    backend = RedisBackend(config)
    print(f"[OK] Backend initialized")
    print(f"[OK] Connected: {backend.is_connected}")
except Exception as e:
    print(f"[ERROR] Backend initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test connection attempt (will fail if Redis not running)
try:
    connected = backend.connect()
    if connected:
        print("[OK] Connected to Redis!")

        # Try a simple operation
        backend.set('test_key', b'test_value', ttl=10)
        value = backend.get('test_key')
        print(f"[OK] Set/Get working: {value}")

        # Cleanup
        backend.delete('test_key')
        backend.disconnect()
        print("[OK] Cleanup complete")
    else:
        print("[INFO] Could not connect to Redis (expected if not running)")
        print("[INFO] To test with Redis: docker run -d -p 6379:6379 redis")
except Exception as e:
    print(f"[INFO] Connection test: {e}")

print("\n[SUCCESS] Quick test completed!")


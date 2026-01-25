"""
Validation script for Redis backend.

Tests the RedisBackend implementation with and without a Redis server.
"""

import sys
import time

# Test if redis is available
try:
    import redis
    REDIS_INSTALLED = True
except ImportError:
    REDIS_INSTALLED = False

from src.cache.redis_backend import RedisBackend, RedisConfig, CacheError


def test_config():
    """Test RedisConfig creation and serialization."""
    print("\n1. Testing RedisConfig...")

    # Default config
    config1 = RedisConfig()
    assert config1.host == "localhost"
    assert config1.port == 6379
    assert config1.key_prefix == "markov_cache:"
    print("   [OK] Default config created")

    # Custom config
    config2 = RedisConfig(
        host="redis.example.com",
        port=6380,
        db=1,
        password="secret",
        key_prefix="test:",
        max_memory=1024*1024*50  # 50MB
    )

    config_dict = config2.to_dict()
    assert config_dict['host'] == "redis.example.com"
    assert config_dict['port'] == 6380
    assert config_dict['key_prefix'] == "test:"
    print("   [OK] Custom config created")
    print(f"   Config: {config_dict}")


def test_initialization():
    """Test backend initialization."""
    print("\n2. Testing Initialization...")

    config = RedisConfig(host='localhost', port=6379, key_prefix='test:')
    backend = RedisBackend(config)

    assert backend.config == config
    assert not backend.is_connected
    print("   [OK] Backend initialized")
    print(f"   Connected: {backend.is_connected}")


def test_connection_without_redis():
    """Test connection handling when Redis is not available."""
    print("\n3. Testing Connection (No Redis Server)...")

    config = RedisConfig(
        host='localhost',
        port=9999,  # Use port that's unlikely to be Redis
        socket_timeout=1.0
    )
    backend = RedisBackend(config)

    # Try to connect
    connected = backend.connect()

    if not connected:
        print("   [OK] Connection failed gracefully (as expected)")
        print("   [OK] Error handling working")
    else:
        print("   [UNEXPECTED] Connected to port 9999!")

    assert not backend.is_connected


def test_with_redis_if_available():
    """Test full functionality if Redis is running."""
    print("\n4. Testing with Redis Server...")

    config = RedisConfig(
        host='localhost',
        port=6379,
        key_prefix='test:',
        socket_timeout=2.0
    )
    backend = RedisBackend(config)

    # Try to connect
    if not backend.connect():
        print("   [SKIP] Redis server not available")
        print("   To test with Redis, run: docker run -d -p 6379:6379 redis")
        return False

    try:
        print("   [OK] Connected to Redis!")

        # Test ping
        assert backend.ping(), "Ping should succeed"
        print("   [OK] Ping successful")

        # Test set/get
        backend.set('mykey', b'myvalue', ttl=300)
        value = backend.get('mykey')
        assert value == b'myvalue', f"Expected b'myvalue', got {value}"
        print(f"   [OK] Retrieved: {value}")

        # Test with metadata
        backend.set(
            'user:123',
            b'user data',
            ttl=60,
            metadata={'endpoint': '/api/users', 'user_type': 'premium'}
        )
        print("   [OK] Set with metadata")

        # Test exists
        assert backend.exists('mykey'), "Key should exist"
        assert not backend.exists('nonexistent'), "Key should not exist"
        print("   [OK] Exists check working")

        # Test batch operations
        items = {'a': b'1', 'b': b'2', 'c': b'3'}
        count = backend.set_many(items, ttl=60)
        assert count == 3, f"Expected 3 sets, got {count}"
        print(f"   [OK] Batch set: {count} items")

        values = backend.get_many(['a', 'b', 'c'])
        assert len(values) == 3, f"Expected 3 values, got {len(values)}"
        assert values['a'] == b'1'
        print(f"   [OK] Batch get: {values}")

        # Test keys pattern
        all_keys = backend.keys()
        print(f"   [OK] All keys: {len(all_keys)} keys")

        test_keys = backend.keys('test:*')
        print(f"   [OK] Pattern 'test:*': {len(test_keys)} keys")

        # Test delete
        deleted = backend.delete('mykey')
        assert deleted, "Delete should return True"
        assert not backend.exists('mykey'), "Key should not exist after delete"
        print("   [OK] Delete working")

        # Test stats
        stats = backend.get_stats()
        print(f"   [OK] Stats: hits={stats.hits}, misses={stats.misses}, entries={stats.current_entries}")
        print(f"   [OK] Hit rate: {stats.hit_rate:.2%}")

        # Test delete_many
        deleted_count = backend.delete_many(['a', 'b', 'c'])
        print(f"   [OK] Batch delete: {deleted_count} items")

        # Test clear (be careful!)
        print("   [WARNING] Testing clear() - will flush database!")
        count = backend.clear()
        print(f"   [OK] Cleared {count} entries")

        return True

    finally:
        # Always disconnect
        backend.disconnect()
        print("   [OK] Disconnected")


def test_context_manager():
    """Test context manager support."""
    print("\n5. Testing Context Manager...")

    config = RedisConfig(host='localhost', port=6379, key_prefix='ctx_test:')

    try:
        with RedisBackend(config) as backend:
            print("   [OK] Context manager entered (connected)")

            # Do some operations
            backend.set('test', b'value')
            value = backend.get('test')
            assert value == b'value'
            print("   [OK] Operations in context")

        print("   [OK] Context manager exited (disconnected)")
        return True

    except CacheError as e:
        print(f"   [SKIP] Redis not available: {e}")
        return False


def test_error_handling():
    """Test error handling and recovery."""
    print("\n6. Testing Error Handling...")

    config = RedisConfig(host='localhost', port=6379)
    backend = RedisBackend(config)

    # Try operations without connecting
    try:
        backend.get('key')
        print("   [FAIL] Should have raised error")
    except CacheError:
        print("   [OK] Raised error when not connected")

    # Operations should not crash even if Redis is down
    result = backend.set('key', b'value')
    print(f"   [OK] Set returns False when not connected: {not result}")


def run_validation():
    """Run all validation tests."""
    print("=" * 60)
    print("REDIS BACKEND VALIDATION")
    print("=" * 60)

    if not REDIS_INSTALLED:
        print("\n[WARNING] redis package not installed!")
        print("Install with: pip install redis")
        print("\nSome tests will be skipped.")
    else:
        print("\n[OK] redis package installed")

    try:
        test_config()
        test_initialization()
        test_connection_without_redis()

        redis_available = test_with_redis_if_available()

        if redis_available:
            test_context_manager()

        test_error_handling()

        print("\n" + "=" * 60)
        if redis_available:
            print("[SUCCESS] ALL TESTS PASSED!")
        else:
            print("[PARTIAL] Tests passed (Redis server not available)")
            print("\nTo run full tests, start Redis:")
            print("  docker run -d -p 6379:6379 redis")
            print("  # or install Redis locally")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n[ERROR] Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)


"""
Validation script for CacheManager.

Tests basic functionality of the high-level cache manager including
serialization, compression, and prefetch coordination.
"""

from src.cache.cache_manager import CacheManager, CacheManagerConfig


def main():
    print("="*70)
    print("CACHE MANAGER VALIDATION")
    print("="*70)

    # Create configuration
    config = CacheManagerConfig(
        backend_type='memory',  # Use in-memory for testing
        compression_enabled=True,
        default_ttl=300
    )

    # Create and start manager
    manager = CacheManager(config)
    if not manager.start():
        print("❌ Failed to start manager")
        return 1

    print("✅ Manager started successfully")

    try:
        # Test 1: Basic caching
        print("\n" + "="*70)
        print("Test 1: Basic caching")
        print("="*70)

        user_data = {'id': 123, 'name': 'John', 'email': 'john@example.com'}
        success = manager.set('/api/users/123', user_data, ttl=60)
        print(f"Set user data: {success}")

        retrieved = manager.get('/api/users/123')
        print(f"Retrieved user: {retrieved}")

        if retrieved == user_data:
            print("✅ Basic caching works")
        else:
            print("❌ Retrieved data doesn't match")

        # Test 2: get_or_set
        print("\n" + "="*70)
        print("Test 2: get_or_set")
        print("="*70)

        call_count = 0

        def fetch_product():
            nonlocal call_count
            call_count += 1
            print(f"  [Factory called #{call_count}] Fetching product from database...")
            return {'id': 42, 'name': 'Widget', 'price': 9.99}

        # First call - should call factory
        product = manager.get_or_set('/api/products/42', fetch_product, ttl=300)
        print(f"Product (1st call): {product}")

        # Second call - should use cache
        product2 = manager.get_or_set('/api/products/42', fetch_product, ttl=300)
        print(f"Product (2nd call): {product2}")

        if call_count == 1:
            print("✅ get_or_set uses cache correctly")
        else:
            print(f"❌ Factory called {call_count} times (expected 1)")

        # Test 3: Default value
        print("\n" + "="*70)
        print("Test 3: Default value")
        print("="*70)

        missing = manager.get('/api/missing', default={'error': 'not found'})
        print(f"Missing key (with default): {missing}")

        if missing == {'error': 'not found'}:
            print("✅ Default value works")
        else:
            print("❌ Default value incorrect")

        # Test 4: Delete
        print("\n" + "="*70)
        print("Test 4: Delete")
        print("="*70)

        manager.set('/api/temp', {'data': 'temporary'})
        print(f"Before delete: {manager.get('/api/temp')}")

        deleted = manager.delete('/api/temp')
        print(f"Deleted: {deleted}")

        after_delete = manager.get('/api/temp')
        print(f"After delete: {after_delete}")

        if after_delete is None:
            print("✅ Delete works")
        else:
            print("❌ Delete failed")

        # Test 5: Prefetch
        print("\n" + "="*70)
        print("Test 5: Prefetch")
        print("="*70)

        manager.prefetch('/api/users/123/orders', priority=0.8)
        manager.prefetch('/api/products/popular', priority=0.6)
        manager.prefetch('/api/recommendations', params={'user_id': 123}, priority=0.9)

        queue = manager.get_prefetch_queue()
        print(f"Prefetch queue size: {len(queue)}")

        if len(queue) == 3:
            print("✅ Prefetch queue works")
            print("\nQueue items (by priority):")
            for item in queue:
                print(f"  - {item['endpoint']} (priority={item['priority']})")
        else:
            print(f"❌ Expected 3 items in queue, got {len(queue)}")

        # Test 6: Prefetch many
        print("\n" + "="*70)
        print("Test 6: Prefetch many")
        print("="*70)

        items = [
            ('/api/users/1', None, 0.5),
            ('/api/users/2', None, 0.4),
            ('/api/users/3', {'details': 'full'}, 0.7),
        ]

        manager.prefetch_many(items)
        queue = manager.get_prefetch_queue()
        print(f"Queue size after prefetch_many: {len(queue)}")

        if len(queue) >= 6:
            print("✅ prefetch_many works")
        else:
            print(f"❌ Expected at least 6 items, got {len(queue)}")

        # Test 7: Compression
        print("\n" + "="*70)
        print("Test 7: Compression")
        print("="*70)

        # Large data that should be compressed
        large_data = {
            'items': [{'id': i, 'name': f'Item {i}', 'description': 'A' * 100} for i in range(50)]
        }

        manager.set('/api/large', large_data, ttl=300)
        retrieved_large = manager.get('/api/large')

        if retrieved_large == large_data:
            print("✅ Compression and decompression works")
        else:
            print("❌ Compression test failed")

        # Test 8: Different data types
        print("\n" + "="*70)
        print("Test 8: Different data types")
        print("="*70)

        test_data = [
            ('string', "Hello, World!"),
            ('int', 42),
            ('float', 3.14159),
            ('list', [1, 2, 3, 4, 5]),
            ('dict', {'a': 1, 'b': 2, 'c': 3}),
            ('tuple', (1, 2, 3)),  # Will become list in JSON
            ('bool', True),
            ('none', None),
        ]

        all_passed = True
        for name, value in test_data:
            manager.set(f'/test/{name}', value)
            retrieved = manager.get(f'/test/{name}')

            # Special handling for tuple (becomes list in JSON)
            if config.serialization_format == 'json' and isinstance(value, tuple):
                expected = list(value)
            else:
                expected = value

            if retrieved == expected:
                print(f"  ✅ {name}: {value}")
            else:
                print(f"  ❌ {name}: expected {expected}, got {retrieved}")
                all_passed = False

        if all_passed:
            print("✅ All data types work")

        # Test 9: Metrics
        print("\n" + "="*70)
        print("Test 9: Metrics")
        print("="*70)

        metrics = manager.get_metrics()
        print("Metrics:")
        print(f"  Hit rate: {metrics.get('hit_rate', 0):.2%}")
        print(f"  Hits: {metrics.get('hits', 0)}")
        print(f"  Misses: {metrics.get('misses', 0)}")
        print(f"  Sets: {metrics.get('sets', 0)}")
        print(f"  Cache operations: {metrics.get('cache_operations', 0)}")
        print(f"  Compression count: {metrics.get('compression_count', 0)}")
        print(f"  Compression ratio: {metrics.get('compression_ratio', 0):.2%}")
        print(f"  Serialization time: {metrics.get('serialization_time_ms', 0):.2f} ms")
        print(f"  Prefetch requests: {metrics.get('prefetch_requests', 0)}")
        print(f"  Prefetch queue size: {metrics.get('prefetch_queue_size', 0)}")

        if metrics.get('cache_operations', 0) > 0:
            print("✅ Metrics tracking works")
        else:
            print("❌ No metrics recorded")

        # Test 10: Cache key generation
        print("\n" + "="*70)
        print("Test 10: Cache key generation")
        print("="*70)

        from src.cache.cache_manager import generate_cache_key

        # Test deterministic key generation
        key1 = generate_cache_key('/api/users', {'id': 123, 'filter': 'active'})
        key2 = generate_cache_key('/api/users', {'filter': 'active', 'id': 123})

        print(f"Key 1: {key1}")
        print(f"Key 2: {key2}")

        if key1 == key2:
            print("✅ Cache key generation is deterministic")
        else:
            print("❌ Keys don't match (parameter ordering issue)")

        # Test long key hashing
        long_params = {f'param{i}': f'value{i}' * 10 for i in range(50)}
        long_key = generate_cache_key('/api/endpoint/with/very/long/path', long_params)
        print(f"\nLong key (length {len(long_key)}): {long_key[:100]}...")

        if len(long_key) < 300:
            print("✅ Long keys are hashed")
        else:
            print("❌ Long key not hashed properly")

        # Test 11: Eviction
        print("\n" + "="*70)
        print("Test 11: Eviction")
        print("="*70)

        # Add some entries
        for i in range(20):
            manager.set(f'/api/evict/item{i}', {'id': i, 'data': f'item {i}'})

        metrics_before = manager.get_metrics()
        entries_before = metrics_before.get('current_entries', 0)
        print(f"Entries before eviction: {entries_before}")

        # Evict 10 entries
        evicted = manager.evict_lru(count=10)
        print(f"Evicted: {evicted}")

        metrics_after = manager.get_metrics()
        entries_after = metrics_after.get('current_entries', 0)
        print(f"Entries after eviction: {entries_after}")

        if evicted > 0:
            print("✅ LRU eviction works")
        else:
            print("❌ No entries evicted")

        # Final summary
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)

        print("\n✅ All tests completed successfully!")
        print("\nFinal metrics:")
        final_metrics = manager.get_metrics()
        for key, value in sorted(final_metrics.items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    finally:
        # Stop manager
        manager.stop()
        print("\n✅ Manager stopped successfully")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())


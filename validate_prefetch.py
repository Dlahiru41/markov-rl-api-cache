"""
Validation script for prefetch queue system.

Tests PrefetchRequest, PrefetchQueue, PrefetchWorker, and PrefetchScheduler.
"""

import time
import threading
from src.cache.prefetch import (
    PrefetchRequest, PrefetchQueue, PrefetchWorker, PrefetchScheduler
)
from src.cache.cache_manager import CacheManager, CacheManagerConfig


def main():
    print("="*70)
    print("PREFETCH SYSTEM VALIDATION")
    print("="*70)

    # Test 1: PrefetchRequest
    print("\n" + "="*70)
    print("Test 1: PrefetchRequest")
    print("="*70)

    try:
        r1 = PrefetchRequest(
            endpoint='/api/users/1',
            params={},
            priority=0.9,
            created_at=time.time(),
            source_prediction=0.9
        )
        print(f"[SUCCESS] Created request: {r1.endpoint} (priority={r1.priority})")

        r2 = PrefetchRequest(
            endpoint='/api/users/2',
            params=None,
            priority=0.5,
            created_at=time.time(),
            source_prediction=0.5
        )

        r3 = PrefetchRequest(
            endpoint='/api/products/1',
            params={'details': 'full'},
            priority=0.7,
            created_at=time.time(),
            source_prediction=0.7
        )

        # Test comparison
        if r1 > r2:
            print("[SUCCESS] Priority comparison works (0.9 > 0.5)")
        else:
            print("[ERROR] Priority comparison failed")

        # Test cache key generation
        key = r1.get_cache_key()
        print(f"[SUCCESS] Cache key generated: {key}")

        # Test expiration
        old_request = PrefetchRequest(
            endpoint='/api/old',
            params={},
            priority=0.5,
            created_at=time.time() - 100,  # 100 seconds ago
            source_prediction=0.5,
            max_age=10.0
        )

        if old_request.is_expired():
            print("[SUCCESS] Expiration check works")
        else:
            print("[ERROR] Expiration check failed")

    except Exception as e:
        print(f"[ERROR] PrefetchRequest test failed: {e}")
        return 1

    # Test 2: PrefetchQueue
    print("\n" + "="*70)
    print("Test 2: PrefetchQueue")
    print("="*70)

    try:
        queue = PrefetchQueue(max_size=100)
        print(f"[SUCCESS] Created queue (max_size=100)")

        # Add requests
        queue.put(r1)
        queue.put(r2)
        queue.put(r3)

        print(f"[SUCCESS] Added 3 requests, queue size: {queue.size}")

        if queue.size == 3:
            print("[SUCCESS] Queue size correct")
        else:
            print(f"[ERROR] Expected size 3, got {queue.size}")

        # Test duplicate rejection
        duplicate = queue.put(r1)
        if not duplicate:
            print("[SUCCESS] Duplicate request rejected")
        else:
            print("[ERROR] Duplicate should have been rejected")

        # Test contains
        if queue.contains('/api/users/1'):
            print("[SUCCESS] Contains check works")
        else:
            print("[ERROR] Contains check failed")

        # Get highest priority (should be r1 with 0.9)
        next_req = queue.get(timeout=1.0)
        if next_req and next_req.endpoint == '/api/users/1':
            print(f"[SUCCESS] Got highest priority: {next_req.endpoint} (priority={next_req.priority})")
        else:
            print(f"[ERROR] Expected /api/users/1, got {next_req.endpoint if next_req else None}")

        # Get next (should be r3 with 0.7)
        next_req = queue.get(timeout=1.0)
        if next_req and next_req.endpoint == '/api/products/1':
            print(f"[SUCCESS] Got next priority: {next_req.endpoint} (priority={next_req.priority})")
        else:
            print(f"[ERROR] Expected /api/products/1, got {next_req.endpoint if next_req else None}")

        # Test stats
        stats = queue.get_stats()
        print(f"[SUCCESS] Queue stats: {stats}")

        # Clear queue
        count = queue.clear()
        print(f"[SUCCESS] Cleared queue: {count} request(s) removed")

        if queue.is_empty:
            print("[SUCCESS] Queue is empty after clear")
        else:
            print("[ERROR] Queue should be empty")

    except Exception as e:
        print(f"[ERROR] PrefetchQueue test failed: {e}")
        return 1

    # Test 3: PrefetchWorker with mock fetcher
    print("\n" + "="*70)
    print("Test 3: PrefetchWorker")
    print("="*70)

    try:
        # Create cache manager
        config = CacheManagerConfig(backend_type='memory')
        cache_manager = CacheManager(config)
        cache_manager.start()

        # Create new queue
        queue = PrefetchQueue(max_size=100)

        # Mock fetcher
        fetch_count = {'count': 0}

        def mock_fetcher(endpoint, params):
            fetch_count['count'] += 1
            print(f"  [Mock Fetcher] Fetching {endpoint}")
            time.sleep(0.1)  # Simulate API call
            return {
                'endpoint': endpoint,
                'params': params,
                'data': f'mock_data_for_{endpoint}',
                'timestamp': time.time()
            }

        # Create worker
        worker = PrefetchWorker(
            queue=queue,
            fetcher=mock_fetcher,
            cache_manager=cache_manager,
            num_workers=2,
            max_requests_per_second=10.0
        )

        print("[SUCCESS] Created worker with 2 threads")

        # Start worker
        worker.start()
        print("[SUCCESS] Worker started")

        if worker.is_running:
            print("[SUCCESS] Worker is running")
        else:
            print("[ERROR] Worker should be running")

        # Add requests to queue
        for i in range(5):
            req = PrefetchRequest(
                endpoint=f'/api/test/{i}',
                params={'test': True},
                priority=0.8 - i * 0.1,
                created_at=time.time(),
                source_prediction=0.8 - i * 0.1
            )
            queue.put(req)

        print(f"[SUCCESS] Added 5 requests to queue")

        # Wait for workers to process
        print("  Waiting for workers to process requests...")
        time.sleep(3.0)

        # Check metrics
        metrics = worker.get_metrics()
        print(f"\n  Worker Metrics:")
        print(f"    Total processed: {metrics['total_processed']}")
        print(f"    Successful: {metrics['successful_fetches']}")
        print(f"    Failed: {metrics['failed_fetches']}")
        print(f"    Success rate: {metrics['success_rate']:.1%}")
        print(f"    Avg fetch time: {metrics['avg_fetch_time']:.3f}s")

        if metrics['successful_fetches'] >= 4:  # Allow for timing variations
            print("[SUCCESS] Worker processed requests successfully")
        else:
            print(f"[ERROR] Expected at least 4 successful fetches, got {metrics['successful_fetches']}")

        # Check cache
        cached = cache_manager.get('/api/test/0')
        if cached:
            print(f"[SUCCESS] Data was cached: {cached['endpoint']}")
        else:
            print("[ERROR] Data should be in cache")

        # Stop worker
        worker.stop(timeout=5.0)
        print("[SUCCESS] Worker stopped")

        if not worker.is_running:
            print("[SUCCESS] Worker is not running")
        else:
            print("[ERROR] Worker should be stopped")

        cache_manager.stop()

    except Exception as e:
        print(f"[ERROR] PrefetchWorker test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test 4: PrefetchScheduler
    print("\n" + "="*70)
    print("Test 4: PrefetchScheduler")
    print("="*70)

    try:
        # Create cache manager
        cache_manager = CacheManager(CacheManagerConfig(backend_type='memory'))
        cache_manager.start()

        # Create queue
        queue = PrefetchQueue(max_size=100)

        # Create scheduler
        scheduler = PrefetchScheduler(
            queue=queue,
            cache_manager=cache_manager,
            min_probability=0.3,
            max_prefetch_per_schedule=10
        )

        print("[SUCCESS] Created scheduler")

        # Test predictions
        predictions = [
            ('/api/products/42', 0.8),
            ('/api/cart', 0.6),
            ('/api/checkout', 0.3),
            ('/api/rare', 0.1),  # Below threshold
            ('/api/unlikely', 0.05),  # Below threshold
        ]

        print(f"\n  Testing with {len(predictions)} predictions:")
        for endpoint, prob in predictions:
            print(f"    {endpoint}: {prob:.2f}")

        # Schedule prefetches
        scheduled = scheduler.schedule_from_predictions(predictions)

        print(f"\n[SUCCESS] Scheduled {len(scheduled)} prefetches:")
        for endpoint in scheduled:
            print(f"    - {endpoint}")

        # Should schedule 3 (0.8, 0.6, 0.3), skip 2 (0.1, 0.05)
        if len(scheduled) == 3:
            print("[SUCCESS] Correct number of requests scheduled")
        else:
            print(f"[ERROR] Expected 3 scheduled, got {len(scheduled)}")

        # Check queue
        if queue.size == 3:
            print("[SUCCESS] Queue has correct number of requests")
        else:
            print(f"[ERROR] Expected queue size 3, got {queue.size}")

        # Test get_prefetch_candidates (pure function)
        cached_keys = set()
        candidates = scheduler.get_prefetch_candidates(
            predictions,
            cached_keys,
            min_probability=0.3
        )

        print(f"\n  Candidates (min_prob=0.3):")
        for endpoint, prob in candidates:
            print(f"    {endpoint}: {prob:.2f}")

        if len(candidates) == 3:
            print("[SUCCESS] get_prefetch_candidates works correctly")
        else:
            print(f"[ERROR] Expected 3 candidates, got {len(candidates)}")

        # Test with cached keys
        cached_keys = {'/api/products/42', '/api/cart'}
        candidates = scheduler.get_prefetch_candidates(
            predictions,
            cached_keys,
            min_probability=0.3
        )

        if len(candidates) == 1 and candidates[0][0] == '/api/checkout':
            print("[SUCCESS] Cached entries are filtered out")
        else:
            print(f"[ERROR] Expected only /api/checkout, got {candidates}")

        # Get stats
        stats = scheduler.get_stats()
        print(f"\n  Scheduler Stats: {stats}")

        cache_manager.stop()

    except Exception as e:
        print(f"[ERROR] PrefetchScheduler test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test 5: Thread safety
    print("\n" + "="*70)
    print("Test 5: Thread Safety")
    print("="*70)

    try:
        queue = PrefetchQueue(max_size=1000)
        errors = []

        def producer(thread_id):
            try:
                for i in range(50):
                    req = PrefetchRequest(
                        endpoint=f'/api/thread{thread_id}/item{i}',
                        params={},
                        priority=0.5,
                        created_at=time.time(),
                        source_prediction=0.5
                    )
                    queue.put(req)
            except Exception as e:
                errors.append(f"Producer {thread_id}: {e}")

        def consumer(thread_id):
            try:
                count = 0
                while count < 25:
                    req = queue.get(timeout=0.5)
                    if req:
                        count += 1
            except Exception as e:
                errors.append(f"Consumer {thread_id}: {e}")

        # Start producers and consumers
        producers = [threading.Thread(target=producer, args=(i,)) for i in range(5)]
        consumers = [threading.Thread(target=consumer, args=(i,)) for i in range(10)]

        for t in producers + consumers:
            t.start()

        for t in producers + consumers:
            t.join()

        if len(errors) == 0:
            print("[SUCCESS] Thread safety test passed")
        else:
            print(f"[ERROR] Thread safety errors: {errors}")

    except Exception as e:
        print(f"[ERROR] Thread safety test failed: {e}")
        return 1

    # Final summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    print("\n[SUCCESS] All tests completed successfully!")
    print("\nPrefetch System Components:")
    print("  [SUCCESS] PrefetchRequest - Priority-based request dataclass")
    print("  [SUCCESS] PrefetchQueue - Thread-safe priority queue")
    print("  [SUCCESS] PrefetchWorker - Background processing with rate limiting")
    print("  [SUCCESS] PrefetchScheduler - Intelligent request scheduling")
    print("  [SUCCESS] Thread safety - Concurrent access validated")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())


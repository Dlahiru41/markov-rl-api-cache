"""
Prefetch System - Complete Documentation

The prefetch system manages proactive caching of predicted API responses based on
Markov chain predictions. It consists of four main components that work together
to intelligently prefetch and cache likely-to-be-requested resources.
"""

# ============================================================================
# COMPONENT 1: PrefetchRequest
# ============================================================================

from src.cache.prefetch import PrefetchRequest
import time

# Create a prefetch request
request = PrefetchRequest(
    endpoint='/api/users/123',
    params={'details': 'full'},
    priority=0.85,  # 0-1, higher = more urgent
    created_at=time.time(),
    source_prediction=0.85,  # Markov probability
    max_age=30.0  # Discard if waiting > 30s
)

# Properties
print(f"Priority: {request.priority}")
print(f"Age: {request.age_seconds():.1f}s")
print(f"Expired: {request.is_expired()}")
print(f"Cache key: {request.get_cache_key()}")

# Requests are comparable (higher priority first)
r1 = PrefetchRequest('/api/high', {}, 0.9, time.time(), 0.9)
r2 = PrefetchRequest('/api/low', {}, 0.3, time.time(), 0.3)
assert r1 > r2  # Higher priority comes first


# ============================================================================
# COMPONENT 2: PrefetchQueue
# ============================================================================

from src.cache.prefetch import PrefetchQueue

# Create queue
queue = PrefetchQueue(max_size=1000)

# Add requests
request1 = PrefetchRequest('/api/users/1', {}, 0.9, time.time(), 0.9)
request2 = PrefetchRequest('/api/users/2', {}, 0.5, time.time(), 0.5)
request3 = PrefetchRequest('/api/products/1', {}, 0.7, time.time(), 0.7)

queue.put(request1)  # Returns True if added
queue.put(request2)
queue.put(request3)

# Duplicate rejection
duplicate = queue.put(request1)  # Returns False - already in queue

# Check queue state
print(f"Queue size: {queue.size}")
print(f"Is empty: {queue.is_empty}")
print(f"Is full: {queue.is_full}")
print(f"Contains endpoint: {queue.contains('/api/users/1')}")

# Get highest priority request
next_request = queue.get(timeout=1.0)
print(f"Next: {next_request.endpoint} (priority={next_request.priority})")
# Returns request1 (priority 0.9)

# Get statistics
stats = queue.get_stats()
print(f"Total added: {stats['total_added']}")
print(f"Duplicates rejected: {stats['duplicates_rejected']}")

# Clear queue
count = queue.clear()
print(f"Cleared {count} requests")


# ============================================================================
# COMPONENT 3: PrefetchWorker
# ============================================================================

from src.cache.prefetch import PrefetchWorker
from src.cache.cache_manager import CacheManager, CacheManagerConfig

# Create cache manager
config = CacheManagerConfig(backend_type='memory')
cache_manager = CacheManager(config)
cache_manager.start()

# Create queue
queue = PrefetchQueue(max_size=100)

# Define fetcher function (your API client)
def fetch_api_response(endpoint, params):
    """Fetch data from API."""
    # Your actual API call here
    import requests
    url = f"https://api.example.com{endpoint}"
    response = requests.get(url, params=params)
    return response.json()

# Create worker
worker = PrefetchWorker(
    queue=queue,
    fetcher=fetch_api_response,
    cache_manager=cache_manager,
    num_workers=2,  # Number of parallel workers
    max_requests_per_second=10.0  # Rate limit
)

# Start workers
worker.start()
print(f"Worker running: {worker.is_running}")

# Workers will process requests from queue automatically

# Get metrics
metrics = worker.get_metrics()
print(f"Processed: {metrics['total_processed']}")
print(f"Successful: {metrics['successful_fetches']}")
print(f"Failed: {metrics['failed_fetches']}")
print(f"Success rate: {metrics['success_rate']:.1%}")
print(f"Avg fetch time: {metrics['avg_fetch_time']:.3f}s")

# Stop workers when done
worker.stop(timeout=10.0)


# ============================================================================
# COMPONENT 4: PrefetchScheduler
# ============================================================================

from src.cache.prefetch import PrefetchScheduler

# Create scheduler
scheduler = PrefetchScheduler(
    queue=queue,
    cache_manager=cache_manager,
    min_probability=0.3,  # Only prefetch if prob > 30%
    max_prefetch_per_schedule=10,  # Limit batch size
    cache_space_threshold=0.8  # Don't prefetch if cache > 80% full
)

# Get predictions from Markov model
predictions = [
    ('/api/products/42', 0.85),  # High probability
    ('/api/cart', 0.60),         # Medium probability
    ('/api/checkout', 0.35),     # Above threshold
    ('/api/rare', 0.10),         # Below threshold - skipped
]

# Schedule prefetches
scheduled = scheduler.schedule_from_predictions(predictions)
print(f"Scheduled: {scheduled}")
# Returns: ['/api/products/42', '/api/cart', '/api/checkout']

# Get candidates (pure function for testing)
cached_keys = {'/api/products/42'}  # Already cached
candidates = scheduler.get_prefetch_candidates(
    predictions,
    cached_keys,
    min_probability=0.3
)
print(f"Candidates: {candidates}")
# Returns: [('/api/cart', 0.60), ('/api/checkout', 0.35)]
# Excludes /api/products/42 (cached) and /api/rare (too low)

# Get scheduler stats
stats = scheduler.get_stats()
print(f"Schedule rate: {stats['schedule_rate']:.1%}")


# ============================================================================
# COMPLETE EXAMPLE: Integration with Markov Chain
# ============================================================================

from src.markov.predictor import MarkovPredictor
from src.cache.prefetch import (
    PrefetchQueue, PrefetchWorker, PrefetchScheduler
)
from src.cache.cache_manager import CacheManager, CacheManagerConfig

# 1. Initialize components
cache_manager = CacheManager(CacheManagerConfig(backend_type='memory'))
cache_manager.start()

queue = PrefetchQueue(max_size=1000)

# 2. Create Markov predictor
predictor = MarkovPredictor(order=1, context_aware=False)
# Train predictor with your data
# predictor.fit(sequences)

# 3. Create scheduler
scheduler = PrefetchScheduler(
    queue=queue,
    cache_manager=cache_manager,
    min_probability=0.3
)

# 4. Create and start worker
def api_fetcher(endpoint, params):
    # Your API fetching logic
    return {'data': f'mock_data_for_{endpoint}'}

worker = PrefetchWorker(
    queue=queue,
    fetcher=api_fetcher,
    cache_manager=cache_manager,
    num_workers=3
)
worker.start()

# 5. Main application loop
def handle_api_request(current_endpoint):
    # Observe current request
    predictor.observe(current_endpoint)
    
    # Get predictions for next requests
    predictions = predictor.predict(k=5)
    
    # Convert to (endpoint, probability) format
    prediction_list = [(pred['endpoint'], pred['probability']) 
                      for pred in predictions]
    
    # Schedule prefetches
    scheduled = scheduler.schedule_from_predictions(prediction_list)
    print(f"Scheduled {len(scheduled)} prefetches")
    
    # Workers will process them in background

# Usage
handle_api_request('/api/users/123')
handle_api_request('/api/users/123/profile')
# Markov chain predicts likely next calls
# Scheduler queues them for prefetch
# Workers fetch and cache them

# 6. Cleanup
worker.stop()
cache_manager.stop()


# ============================================================================
# ADVANCED USAGE
# ============================================================================

# Custom priority calculation
def calculate_priority(markov_prob, recency, user_type):
    """Calculate prefetch priority from multiple factors."""
    base_priority = markov_prob
    
    # Boost for recent requests
    if recency < 60:  # Last minute
        base_priority *= 1.2
    
    # Boost for premium users
    if user_type == 'premium':
        base_priority *= 1.1
    
    return min(base_priority, 1.0)

# Use custom priority
priority = calculate_priority(0.7, 30, 'premium')
request = PrefetchRequest(
    endpoint='/api/premium/feature',
    params={},
    priority=priority,
    created_at=time.time(),
    source_prediction=0.7
)


# Conditional scheduling
def smart_schedule(predictions, current_time_of_day, cache_state):
    """Only schedule during off-peak hours."""
    # Don't prefetch during peak hours
    if 9 <= current_time_of_day <= 17:
        print("Peak hours - skipping prefetch")
        return []
    
    # Check cache capacity
    if cache_state['utilization'] > 0.9:
        print("Cache nearly full - skipping prefetch")
        return []
    
    # Schedule normally
    return scheduler.schedule_from_predictions(predictions, cache_state)


# Dynamic worker scaling
def scale_workers(queue_size, worker):
    """Adjust worker count based on queue size."""
    if queue_size > 100 and worker._num_workers < 5:
        # Add more workers
        print("Scaling up workers")
    elif queue_size < 20 and worker._num_workers > 1:
        # Reduce workers
        print("Scaling down workers")


# Monitoring and metrics
def monitor_prefetch_system():
    """Log prefetch system health."""
    queue_stats = queue.get_stats()
    worker_metrics = worker.get_metrics()
    scheduler_stats = scheduler.get_stats()
    
    print("Prefetch System Status:")
    print(f"  Queue size: {queue_stats['size']}/{queue_stats['max_size']}")
    print(f"  Success rate: {worker_metrics['success_rate']:.1%}")
    print(f"  Schedule rate: {scheduler_stats['schedule_rate']:.1%}")
    
    # Alert if performance degrades
    if worker_metrics['success_rate'] < 0.8:
        print("WARNING: High failure rate")
    
    if queue_stats['full_rejections'] > 100:
        print("WARNING: Queue frequently full")


# ============================================================================
# ERROR HANDLING
# ============================================================================

# Worker automatically handles errors
def robust_fetcher(endpoint, params):
    """Fetcher with error handling."""
    try:
        # Your API call
        response = fetch_from_api(endpoint, params)
        return response
    except ConnectionError:
        # Network error - worker will track failure
        raise
    except Timeout:
        # Timeout - worker will track failure
        raise
    except Exception as e:
        # Unexpected error - log and re-raise
        logger.error(f"Unexpected error fetching {endpoint}: {e}")
        raise

# Worker features automatic error handling:
# - Tracks consecutive errors
# - Backs off after 5 consecutive failures
# - Continues with next request
# - Reports errors in metrics


# ============================================================================
# TESTING
# ============================================================================

# Mock fetcher for testing
def mock_fetcher(endpoint, params):
    """Mock fetcher that doesn't make real API calls."""
    return {
        'endpoint': endpoint,
        'params': params,
        'data': f'mock_data_for_{endpoint}',
        'timestamp': time.time()
    }

# Test with mock
worker = PrefetchWorker(
    queue=queue,
    fetcher=mock_fetcher,  # Use mock instead of real API
    cache_manager=cache_manager,
    num_workers=1
)


# ============================================================================
# PERFORMANCE TUNING
# ============================================================================

# For high-throughput systems
high_perf_config = {
    'queue_max_size': 5000,
    'num_workers': 10,
    'max_requests_per_second': 100.0,
    'min_probability': 0.5,  # Higher threshold
    'max_prefetch_per_schedule': 50
}

# For resource-constrained systems
low_resource_config = {
    'queue_max_size': 100,
    'num_workers': 1,
    'max_requests_per_second': 5.0,
    'min_probability': 0.7,  # Only high-confidence predictions
    'max_prefetch_per_schedule': 5
}

# For development/testing
dev_config = {
    'queue_max_size': 50,
    'num_workers': 1,
    'max_requests_per_second': 1.0,
    'min_probability': 0.3,
    'max_prefetch_per_schedule': 10
}


# ============================================================================
# BEST PRACTICES
# ============================================================================

# 1. Always start/stop worker properly
worker.start()
try:
    # Your application logic
    pass
finally:
    worker.stop(timeout=10.0)

# 2. Monitor queue size
if queue.size > queue._max_size * 0.9:
    print("Queue nearly full - consider increasing size or workers")

# 3. Adjust min_probability based on hit rate
cache_metrics = cache_manager.get_metrics()
if cache_metrics['hit_rate'] < 0.6:
    # Lower threshold to prefetch more
    scheduler._min_probability = 0.2
elif cache_metrics['hit_rate'] > 0.9:
    # Raise threshold to prefetch less
    scheduler._min_probability = 0.5

# 4. Set appropriate TTLs based on prediction confidence
# Higher confidence = longer TTL
# This is done automatically by the worker

# 5. Use rate limiting to avoid overwhelming backend
# Already built into PrefetchWorker

# 6. Monitor and log
import logging
logging.basicConfig(level=logging.INFO)
# All components log important events


print("""
===============================================================================
PREFETCH SYSTEM OVERVIEW
===============================================================================

Components:
  1. PrefetchRequest - Represents a single prefetch job
  2. PrefetchQueue - Thread-safe priority queue
  3. PrefetchWorker - Background processing with rate limiting
  4. PrefetchScheduler - Intelligent request scheduling

Features:
  ✓ Priority-based queue (high probability first)
  ✓ Duplicate detection (avoid redundant fetches)
  ✓ Expiration (discard old requests)
  ✓ Thread-safe (concurrent access)
  ✓ Rate limiting (don't overwhelm backend)
  ✓ Error handling (automatic retry/backoff)
  ✓ Comprehensive metrics (track everything)
  ✓ Filtering (skip cached, low probability)

Integration:
  - Markov chain predicts next requests
  - Scheduler filters and queues them
  - Workers fetch and cache them
  - Cache manager serves them instantly

Benefits:
  - Reduced latency (prefetched = instant)
  - Better cache hit rate (proactive caching)
  - Efficient resource use (intelligent filtering)
  - Scalable (add more workers as needed)

===============================================================================
""")


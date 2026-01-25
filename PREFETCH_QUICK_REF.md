# Prefetch System - Quick Reference

## 🚀 Quick Start

```python
from src.cache.prefetch import (
    PrefetchRequest, PrefetchQueue, PrefetchWorker, PrefetchScheduler
)
from src.cache.cache_manager import CacheManager, CacheManagerConfig
import time

# 1. Setup cache
cache_manager = CacheManager(CacheManagerConfig(backend_type='memory'))
cache_manager.start()

# 2. Create queue
queue = PrefetchQueue(max_size=1000)

# 3. Define fetcher
def fetch_api(endpoint, params):
    # Your API call here
    return {'data': f'result_for_{endpoint}'}

# 4. Create and start worker
worker = PrefetchWorker(
    queue=queue,
    fetcher=fetch_api,
    cache_manager=cache_manager,
    num_workers=2
)
worker.start()

# 5. Create scheduler
scheduler = PrefetchScheduler(
    queue=queue,
    cache_manager=cache_manager,
    min_probability=0.3
)

# 6. Schedule prefetches from predictions
predictions = [
    ('/api/users/1', 0.8),
    ('/api/users/2', 0.6),
]
scheduled = scheduler.schedule_from_predictions(predictions)

# 7. Cleanup
worker.stop()
cache_manager.stop()
```

## 📋 Components

### PrefetchRequest

```python
request = PrefetchRequest(
    endpoint='/api/users/123',    # API endpoint
    params={'details': 'full'},   # Optional parameters
    priority=0.85,                # 0-1, higher = more urgent
    created_at=time.time(),       # Creation timestamp
    source_prediction=0.85,       # Markov probability
    max_age=30.0                  # Max seconds in queue
)

# Properties
request.is_expired()      # Check if too old
request.age_seconds()     # Get age
request.get_cache_key()   # Generate cache key

# Comparison (higher priority first)
r1 > r2  # True if r1 has higher priority
```

### PrefetchQueue

```python
queue = PrefetchQueue(max_size=1000)

# Add request
queue.put(request)  # Returns True if added

# Get highest priority
next_req = queue.get(timeout=1.0)

# Check state
queue.size           # Current size
queue.is_empty       # True if empty
queue.is_full        # True if at max_size

# Check contains
queue.contains('/api/users/1')

# Clear
count = queue.clear()

# Stats
stats = queue.get_stats()
```

### PrefetchWorker

```python
worker = PrefetchWorker(
    queue=queue,
    fetcher=fetch_function,
    cache_manager=cache_manager,
    num_workers=2,
    max_requests_per_second=10.0
)

# Lifecycle
worker.start()
worker.is_running  # Check status
worker.stop(timeout=10.0)

# Metrics
metrics = worker.get_metrics()
# Returns: total_processed, successful_fetches, failed_fetches,
#          success_rate, avg_wait_time, avg_fetch_time
```

### PrefetchScheduler

```python
scheduler = PrefetchScheduler(
    queue=queue,
    cache_manager=cache_manager,
    min_probability=0.3,
    max_prefetch_per_schedule=10,
    cache_space_threshold=0.8
)

# Schedule from predictions
predictions = [('/api/endpoint', 0.8), ...]
scheduled = scheduler.schedule_from_predictions(predictions)

# Get candidates (pure function)
candidates = scheduler.get_prefetch_candidates(
    predictions,
    cached_keys,
    min_probability=0.3
)

# Stats
stats = scheduler.get_stats()
```

## 🎯 Common Patterns

### Pattern 1: Basic Prefetching

```python
# After each request, predict and prefetch next
def handle_request(current_endpoint):
    # Get predictions from Markov
    predictions = markov_predictor.predict(k=5)
    
    # Convert to list format
    pred_list = [(p['endpoint'], p['probability']) for p in predictions]
    
    # Schedule prefetches
    scheduler.schedule_from_predictions(pred_list)
```

### Pattern 2: Conditional Prefetching

```python
# Only prefetch during off-peak hours
import datetime

def smart_prefetch(predictions):
    hour = datetime.datetime.now().hour
    
    if 9 <= hour <= 17:  # Peak hours
        return []
    
    return scheduler.schedule_from_predictions(predictions)
```

### Pattern 3: Custom Priority

```python
def custom_priority(markov_prob, user_type):
    priority = markov_prob
    
    if user_type == 'premium':
        priority *= 1.2  # Boost for premium users
    
    return min(priority, 1.0)

# Create request with custom priority
priority = custom_priority(0.7, 'premium')
request = PrefetchRequest(..., priority=priority, ...)
```

### Pattern 4: Monitoring

```python
def monitor_system():
    queue_stats = queue.get_stats()
    worker_metrics = worker.get_metrics()
    
    print(f"Queue: {queue_stats['size']}/{queue_stats['max_size']}")
    print(f"Success rate: {worker_metrics['success_rate']:.1%}")
    
    if worker_metrics['success_rate'] < 0.8:
        print("WARNING: High failure rate!")
```

## 🔧 Configuration

### High Performance

```python
queue = PrefetchQueue(max_size=5000)
worker = PrefetchWorker(
    queue, fetcher, cache_manager,
    num_workers=10,
    max_requests_per_second=100.0
)
scheduler = PrefetchScheduler(
    queue, cache_manager,
    min_probability=0.5,
    max_prefetch_per_schedule=50
)
```

### Low Resource

```python
queue = PrefetchQueue(max_size=100)
worker = PrefetchWorker(
    queue, fetcher, cache_manager,
    num_workers=1,
    max_requests_per_second=5.0
)
scheduler = PrefetchScheduler(
    queue, cache_manager,
    min_probability=0.7,
    max_prefetch_per_schedule=5
)
```

### Development/Testing

```python
queue = PrefetchQueue(max_size=50)
worker = PrefetchWorker(
    queue, mock_fetcher, cache_manager,
    num_workers=1,
    max_requests_per_second=1.0
)
scheduler = PrefetchScheduler(
    queue, cache_manager,
    min_probability=0.3,
    max_prefetch_per_schedule=10
)
```

## 📊 Metrics

### Queue Metrics

```python
stats = queue.get_stats()
# {
#     'size': 42,
#     'max_size': 1000,
#     'is_empty': False,
#     'is_full': False,
#     'total_added': 150,
#     'total_removed': 108,
#     'duplicates_rejected': 12,
#     'full_rejections': 0
# }
```

### Worker Metrics

```python
metrics = worker.get_metrics()
# {
#     'total_processed': 100,
#     'successful_fetches': 95,
#     'failed_fetches': 5,
#     'expired_requests': 3,
#     'success_rate': 0.95,
#     'avg_wait_time': 2.5,  # seconds
#     'avg_fetch_time': 0.3,  # seconds
#     'consecutive_errors': 0,
#     'is_running': True,
#     'num_workers': 2
# }
```

### Scheduler Metrics

```python
stats = scheduler.get_stats()
# {
#     'total_predictions': 200,
#     'scheduled': 85,
#     'filtered_low_prob': 60,
#     'filtered_cached': 40,
#     'filtered_space': 15,
#     'schedule_rate': 0.425
# }
```

## 🐛 Error Handling

### Worker Errors

```python
# Worker automatically handles errors
# - Logs errors
# - Tracks failure count
# - Backs off after 5 consecutive errors
# - Continues with next request

# Check for issues
metrics = worker.get_metrics()
if metrics['consecutive_errors'] >= 5:
    print("Worker experiencing issues!")
```

### Queue Full

```python
# Queue rejects when full
success = queue.put(request)
if not success:
    # Either duplicate or queue full
    if queue.is_full:
        print("Queue is full!")
```

### Expired Requests

```python
# Worker discards expired requests
request = PrefetchRequest(..., max_age=30.0)
# If request waits > 30s, worker discards it
# Check metrics for expired count
```

## 🎓 Best Practices

### 1. Always Start/Stop Properly

```python
worker.start()
try:
    # Your code
    pass
finally:
    worker.stop(timeout=10.0)
```

### 2. Monitor Queue Size

```python
if queue.size > queue._max_size * 0.9:
    print("Queue nearly full - scale up!")
```

### 3. Adjust Thresholds Dynamically

```python
cache_metrics = cache_manager.get_metrics()
if cache_metrics['hit_rate'] < 0.6:
    scheduler._min_probability = 0.2  # Prefetch more
elif cache_metrics['hit_rate'] > 0.9:
    scheduler._min_probability = 0.5  # Prefetch less
```

### 4. Use Rate Limiting

```python
# Already built in - just configure
worker = PrefetchWorker(
    ...,
    max_requests_per_second=10.0  # Limit to 10 req/s
)
```

### 5. Test with Mock Fetcher

```python
def mock_fetcher(endpoint, params):
    return {'data': f'mock_for_{endpoint}'}

worker = PrefetchWorker(queue, mock_fetcher, cache_manager)
```

## 🔍 Debugging

### Enable Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# All components log:
# - Request additions
# - Fetches
# - Errors
# - State changes
```

### Check Queue Contents

```python
# Queue size
print(f"Queue has {queue.size} requests")

# Check if specific endpoint is queued
if queue.contains('/api/users/1'):
    print("Already queued")
```

### Monitor Worker Health

```python
metrics = worker.get_metrics()
print(f"Success rate: {metrics['success_rate']:.1%}")
print(f"Errors: {metrics['consecutive_errors']}")

if not worker.is_running:
    print("Worker stopped!")
```

## 📚 Integration

### With Markov Predictor

```python
from src.markov.predictor import MarkovPredictor

predictor = MarkovPredictor(order=1)
predictor.fit(sequences)

# After each request
predictor.observe(current_endpoint)
predictions = predictor.predict(k=5)

# Convert and schedule
pred_list = [(p['endpoint'], p['probability']) for p in predictions]
scheduler.schedule_from_predictions(pred_list)
```

### With RL Agent

```python
# RL agent observes prefetch metrics
state = {
    'queue_size': queue.size / queue._max_size,
    'success_rate': worker.get_metrics()['success_rate'],
    'cache_hit_rate': cache_manager.get_metrics()['hit_rate']
}

# RL agent decides prefetch threshold
action = rl_agent.select_action(state)
scheduler._min_probability = action['threshold']
```

## ⚡ Performance Tips

1. **Tune worker count** - More workers = higher throughput
2. **Adjust queue size** - Larger queue = handle bursts better
3. **Set appropriate thresholds** - Balance prefetch vs. accuracy
4. **Use rate limiting** - Protect backend from overload
5. **Monitor metrics** - Track and optimize continuously

---

**Quick Reference Version**: 1.0  
**Last Updated**: January 25, 2026  
**Status**: ✅ Production Ready


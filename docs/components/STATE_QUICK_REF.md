# State Representation Quick Reference

## Import
```python
from src.rl.state import StateBuilder, StateConfig
```

## Quick Start
```python
# 1. Configure
config = StateConfig(markov_top_k=5)

# 2. Create and fit
builder = StateBuilder(config)
builder.fit(['api1', 'api2', 'api3', ...])

# 3. Build state
state = builder.build_state(
    markov_predictions=[('api1', 0.8), ('api2', 0.15)],
    cache_metrics={'utilization': 0.6, 'hit_rate': 0.75},
    system_metrics={'cpu': 0.3, 'memory': 0.5, 'p95_latency': 150},
    context={'user_type': 'premium', 'hour': 14, 'day': 2}
)
```

## StateConfig Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `markov_top_k` | int | 5 | Number of top predictions |
| `include_probabilities` | bool | True | Include prediction probabilities |
| `include_confidence` | bool | True | Include confidence score |
| `include_cache_metrics` | bool | True | Include cache stats |
| `include_system_metrics` | bool | True | Include system stats |
| `include_user_context` | bool | True | Include user type |
| `include_temporal_context` | bool | True | Include time features |
| `include_session_context` | bool | True | Include session info |
| `vocab_size` | int | 1000 | API vocabulary size |

## Input Format

### markov_predictions
```python
[('api_name', probability), ...]
```

### cache_metrics
```python
{
    'utilization': 0.0-1.0,      # Cache capacity used
    'hit_rate': 0.0-1.0,          # Cache hit rate
    'entries': int,               # Number of cached items
    'eviction_rate': float        # Items evicted/time
}
```

### system_metrics
```python
{
    'cpu': 0.0-1.0,              # CPU usage
    'memory': 0.0-1.0,           # Memory usage
    'request_rate': float,        # Requests/second
    'p50_latency': float,         # Median latency (ms)
    'p95_latency': float,         # 95th percentile (ms)
    'p99_latency': float,         # 99th percentile (ms)
    'error_rate': 0.0-1.0,       # Error rate
    'connections': int,           # Active connections
    'queue_depth': int            # Queue depth
}
```

### context
```python
{
    'user_type': 'premium'|'free'|'guest',  # User type
    'hour': 0-23,                            # Hour of day
    'day': 0-6,                              # Day of week
    'session_position': int,                 # Position in session
    'session_duration': float,               # Duration (seconds)
    'call_count': int                        # API calls in session
}
```

## State Vector Structure

### Default Configuration (state_dim = 36)

1. **Markov API indices** (5): Normalized API indices [0-1]
2. **Markov probabilities** (5): Prediction probabilities [0-1]
3. **Markov confidence** (1): Max probability [0-1]
4. **Cache utilization** (1): [0-1]
5. **Cache hit rate** (1): [0-1]
6. **Cache entries** (1): Normalized [0-1]
7. **Cache eviction rate** (1): Normalized [0-1]
8. **CPU usage** (1): [0-1]
9. **Memory usage** (1): [0-1]
10. **Request rate** (1): Normalized [0-1]
11. **P50 latency** (1): Normalized [0-1]
12. **P95 latency** (1): Normalized [0-1]
13. **P99 latency** (1): Normalized [0-1]
14. **Error rate** (1): [0-1]
15. **Connections** (1): Normalized [0-1]
16. **Queue depth** (1): Normalized [0-1]
17. **User: is_premium** (1): Binary [0 or 1]
18. **User: is_free** (1): Binary [0 or 1]
19. **User: is_guest** (1): Binary [0 or 1]
20. **Time: hour_sin** (1): [-1, 1]
21. **Time: hour_cos** (1): [-1, 1]
22. **Time: day_sin** (1): [-1, 1]
23. **Time: day_cos** (1): [-1, 1]
24. **Time: is_weekend** (1): Binary [0 or 1]
25. **Time: is_peak** (1): Binary [0 or 1]
26. **Session position** (1): Normalized [0-1]
27. **Session duration** (1): Normalized [0-1]
28. **Session call count** (1): Normalized [0-1]

## Key Methods

### StateBuilder.fit(vocabulary)
Must be called before build_state(). Maps API names to indices.

### StateBuilder.build_state(...)
Creates state vector. All parameters optional (defaults to zeros).

### StateBuilder.get_feature_names()
Returns list of feature names in order.

## Feature Ranges

- Most features: [0, 1]
- Cyclical time features: [-1, 1]
- All features suitable for neural networks

## Normalization Constants

```python
MAX_CACHE_ENTRIES = 10000
MAX_EVICTION_RATE = 1000
MAX_REQUEST_RATE = 5000
MAX_LATENCY_MS = 1000
MAX_CONNECTIONS = 1000
MAX_QUEUE_DEPTH = 500
MAX_SESSION_POSITION = 100
MAX_SESSION_DURATION = 3600
MAX_CALL_COUNT = 500
```

## Common Patterns

### Minimal State (Markov only)
```python
config = StateConfig(
    markov_top_k=10,
    include_cache_metrics=False,
    include_system_metrics=False,
    include_user_context=False,
    include_temporal_context=False,
    include_session_context=False
)
```

### No Markov (System only)
```python
config = StateConfig(
    markov_top_k=0,
    include_probabilities=False,
    include_confidence=False
)
```

### Feature Analysis
```python
names = builder.get_feature_names()
for name, value in zip(names, state):
    print(f"{name}: {value:.3f}")
```

## Error Handling

```python
# Must fit before building
builder = StateBuilder(config)
try:
    state = builder.build_state()  # Raises ValueError
except ValueError as e:
    print(e)  # "StateBuilder must be fit()..."

# Correct usage
builder.fit(vocabulary)
state = builder.build_state()  # Works!
```

## Testing

```bash
# Quick validation
python test_state_validation.py

# Comprehensive tests
python test_state_comprehensive.py
```

## Tips

1. **Always fit first:** Call `fit(vocabulary)` before `build_state()`
2. **Missing inputs OK:** All parameters optional, default to zeros
3. **Check dimension:** Use `config.state_dim` to verify expected size
4. **Feature names:** Use `get_feature_names()` for debugging
5. **Batch processing:** Can build multiple states in a loop
6. **Reuse builder:** Create once, use many times
7. **Custom config:** Disable unused features to reduce dimension

## Example: Full RL Loop

```python
from src.rl.state import StateBuilder, StateConfig

# Setup
config = StateConfig(markov_top_k=5)
builder = StateBuilder(config)
builder.fit(api_vocabulary)

# Training loop
for episode in range(1000):
    state = builder.build_state(
        markov_predictions=predictor.predict(history),
        cache_metrics=cache.get_metrics(),
        system_metrics=monitor.get_metrics(),
        context=get_context()
    )
    
    action = agent.select_action(state)
    reward = execute(action)
    next_state = builder.build_state(...)
    
    agent.learn(state, action, reward, next_state)
```

## Debugging

```python
# Check state properties
print(f"Shape: {state.shape}")
print(f"Dtype: {state.dtype}")
print(f"Range: [{state.min():.2f}, {state.max():.2f}]")
print(f"NaNs: {np.isnan(state).sum()}")
print(f"Infs: {np.isinf(state).sum()}")

# Inspect features
for i, (name, value) in enumerate(zip(builder.get_feature_names(), state)):
    print(f"{i:2d}. {name:30s}: {value:7.3f}")
```


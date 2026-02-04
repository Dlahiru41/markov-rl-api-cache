# State Representation Module Documentation

## Overview

The `src.rl.state` module provides a robust state representation system for Reinforcement Learning agents making caching decisions. It converts heterogeneous system information into fixed-size numerical observation vectors suitable for neural networks.

## Architecture

### StateConfig

A dataclass that defines what information to include in state vectors and their dimensions.

**Configuration Options:**
- `markov_top_k` (int, default=5): Number of top Markov predictions to include
- `include_probabilities` (bool, default=True): Include prediction probabilities
- `include_confidence` (bool, default=True): Include overall prediction confidence
- `include_cache_metrics` (bool, default=True): Include cache performance metrics
- `include_system_metrics` (bool, default=True): Include system resource metrics
- `include_user_context` (bool, default=True): Include user type encoding
- `include_temporal_context` (bool, default=True): Include time-based features
- `include_session_context` (bool, default=True): Include session information
- `vocab_size` (int, default=1000): Size of API vocabulary for normalization

**Property:**
- `state_dim`: Calculated total state vector dimension based on configuration

### StateBuilder

Main class for constructing state vectors.

**Methods:**

#### `__init__(config: StateConfig)`
Initialize builder with configuration.

#### `fit(vocabulary: List[str]) -> StateBuilder`
Fit the builder on API vocabulary to create API-to-index mapping.
- **Parameters:** List of API names
- **Returns:** Self (for method chaining)

#### `build_state(...) -> np.ndarray`
Construct fixed-size state vector from inputs.

**Parameters:**
- `markov_predictions`: List of (api_name, probability) tuples
- `cache_metrics`: Dict with keys: utilization, hit_rate, entries, eviction_rate
- `system_metrics`: Dict with keys: cpu, memory, request_rate, p50_latency, p95_latency, p99_latency, error_rate, connections, queue_depth
- `context`: Dict with keys: user_type, hour, day, session_position, session_duration, call_count

**Returns:** numpy array of shape (state_dim,) with dtype float32

#### `get_feature_names() -> List[str]`
Get ordered list of feature names for interpretability.

## State Vector Components

### 1. Markov Predictions (markov_top_k features)
- **API Indices:** Normalized to [0, 1] by dividing by vocab_size
- **Probabilities:** Already in [0, 1] range
- **Confidence:** Maximum probability among predictions
- **Padding:** Zeros if fewer than k predictions

### 2. Cache Metrics (4 features)
- `utilization`: Cache capacity usage [0, 1]
- `hit_rate`: Cache hit rate [0, 1]
- `entries`: Number of cached items (normalized by 10,000)
- `eviction_rate`: Items evicted per time unit (normalized by 1,000)

### 3. System Metrics (9 features)
- `cpu`: CPU usage [0, 1]
- `memory`: Memory usage [0, 1]
- `request_rate`: Requests per second (normalized by 5,000)
- `p50_latency`: 50th percentile latency (normalized by 1,000ms)
- `p95_latency`: 95th percentile latency (normalized by 1,000ms)
- `p99_latency`: 99th percentile latency (normalized by 1,000ms)
- `error_rate`: Error rate [0, 1]
- `connections`: Active connections (normalized by 1,000)
- `queue_depth`: Request queue depth (normalized by 500)

### 4. User Context (3 features)
One-hot encoding of user type:
- `is_premium`: 1 if premium user, 0 otherwise
- `is_free`: 1 if free user, 0 otherwise
- `is_guest`: 1 if guest user, 0 otherwise

### 5. Temporal Context (6 features)
**Cyclical Encoding** (preserves temporal proximity):
- `hour_sin`: sin(2π × hour / 24) ∈ [-1, 1]
- `hour_cos`: cos(2π × hour / 24) ∈ [-1, 1]
- `day_sin`: sin(2π × day / 7) ∈ [-1, 1]
- `day_cos`: cos(2π × day / 7) ∈ [-1, 1]

**Binary Flags:**
- `is_weekend`: 1 if day ≥ 5, 0 otherwise
- `is_peak_hour`: 1 if 9 ≤ hour ≤ 17, 0 otherwise

### 6. Session Context (3 features)
- `position`: Request position in session (normalized by 100)
- `duration`: Session duration in seconds (normalized by 3,600)
- `call_count`: Number of API calls in session (normalized by 500)

## Design Principles

### 1. Fixed-Size Vectors
All state vectors have the same dimension determined by `state_dim`, regardless of input variability. This is required for neural network compatibility.

### 2. Normalized Values
All features are normalized to [-1, 1] or [0, 1] ranges:
- Prevents gradient explosion/vanishing
- Ensures features have comparable scales
- Improves neural network training stability

### 3. Cyclical Temporal Encoding
Uses sine/cosine transformations for hours and days:
- Hour 23 and Hour 1 are close in representation
- Preserves circular nature of time
- Better than linear encoding for temporal patterns

### 4. Graceful Degradation
Missing inputs default to zero:
- System continues functioning with partial information
- No crashes or exceptions for missing metrics
- Robust to monitoring failures

### 5. Interpretability
`get_feature_names()` provides human-readable labels for each dimension, critical for:
- Debugging model behavior
- Feature importance analysis
- Understanding learned policies

## Usage Examples

### Basic Usage
```python
from src.rl.state import StateBuilder, StateConfig

# Configure state representation
config = StateConfig(markov_top_k=5)
print(f"State dimension: {config.state_dim}")

# Create and fit builder
builder = StateBuilder(config)
builder.fit(['login', 'profile', 'browse', 'product', 'cart', 'checkout'])

# Build state vector
state = builder.build_state(
    markov_predictions=[('profile', 0.8), ('browse', 0.15)],
    cache_metrics={'utilization': 0.6, 'hit_rate': 0.75},
    system_metrics={'cpu': 0.3, 'memory': 0.5, 'p95_latency': 150},
    context={'user_type': 'premium', 'hour': 14, 'day': 2}
)

print(f"State shape: {state.shape}")  # (36,)
print(f"State range: [{state.min():.2f}, {state.max():.2f}]")
```

### Custom Configuration
```python
# Minimal configuration (only Markov predictions)
config = StateConfig(
    markov_top_k=10,
    include_cache_metrics=False,
    include_system_metrics=False,
    include_user_context=False,
    include_temporal_context=False,
    include_session_context=False
)
```

### Integration with RL Agent
```python
# In RL training loop
for episode in range(num_episodes):
    # Get current situation
    predictions = markov_predictor.predict(history)
    cache_state = cache_system.get_metrics()
    sys_state = monitoring.get_metrics()
    ctx = get_current_context()
    
    # Build state vector
    state = builder.build_state(predictions, cache_state, sys_state, ctx)
    
    # Agent makes decision
    action = agent.select_action(state)
    
    # Execute and observe reward
    reward = execute_action(action)
    next_state = builder.build_state(...)
    
    # Train agent
    agent.train(state, action, reward, next_state)
```

### Feature Analysis
```python
# Analyze feature importance
feature_names = builder.get_feature_names()
feature_values = state

for name, value in zip(feature_names, feature_values):
    print(f"{name:30s}: {value:7.3f}")
```

## Implementation Details

### Normalization Strategies

**API Indices:**
```python
normalized_idx = api_index / vocab_size  # [0, 1]
```

**Latency Metrics:**
```python
normalized_latency = latency_ms / 1000.0  # Assume max 1000ms
```

**Request Rate:**
```python
normalized_rate = rate / 5000.0  # Assume max 5000 req/s
```

**Cyclical Time:**
```python
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```

### Performance Considerations

- **Memory:** State vectors are float32 (4 bytes per feature)
- **Computation:** O(state_dim) time complexity
- **Storage:** Typical state_dim ranges from 20-50 features
- **Batch-friendly:** Can be vectorized for batch processing

## Validation

Run comprehensive tests:
```bash
python test_state_comprehensive.py
```

Tests cover:
1. Basic functionality
2. Missing input handling
3. Feature name generation
4. Cyclical encoding
5. Value range validation
6. User type encoding
7. Error handling

## Future Enhancements

Potential improvements:
1. **Adaptive normalization:** Learn normalization bounds from data
2. **Feature selection:** Automatically disable unused features
3. **Custom encoders:** Plugin architecture for domain-specific features
4. **Batch processing:** Vectorized state building for multiple states
5. **State caching:** Cache frequently-requested states
6. **Feature engineering:** Derived features (e.g., cache_hit_rate × utilization)

## References

- Cyclical encoding: [Learning to Represent Time](https://arxiv.org/abs/1811.09964)
- Feature normalization: Deep Learning (Goodfellow et al., 2016)
- RL state design: Sutton & Barto, Reinforcement Learning: An Introduction


# MarkovPredictor - Unified Interface for RL Integration

## Overview

`MarkovPredictor` provides a unified interface for all Markov chain variants (first-order, second-order, context-aware) with seamless RL integration. The RL agent doesn't need to know which type of Markov chain is being used internally.

## Key Innovation

**Single Interface, Multiple Models:**
- First-order Markov chains
- Second-order Markov chains  
- Context-aware Markov chains
- All through one consistent API

**RL-Ready:**
- Fixed-size state vectors
- Automatic history management
- Real-time metrics tracking
- Online learning support

## Quick Start

```python
from src.markov import MarkovPredictor

# Create predictor
predictor = MarkovPredictor(
    order=1,
    context_aware=True,
    context_features=['user_type', 'time_of_day'],
    history_size=10
)

# Train
predictor.fit(sequences, contexts)

# Use in RL loop
predictor.reset_history()
predictor.observe('login')
state = predictor.get_state_vector(k=5)  # For RL agent
predictions = predictor.predict(k=5)      # For interpretation
```

## Constructor

```python
MarkovPredictor(
    order: int = 1,
    context_aware: bool = False,
    context_features: Optional[List[str]] = None,
    smoothing: float = 0.001,
    history_size: int = 10,
    fallback_strategy: str = 'global'
)
```

**Parameters:**
- `order`: Markov chain order (1 or 2)
- `context_aware`: Whether to use context-aware predictions
- `context_features`: Context feature names (required if context_aware=True)
- `smoothing`: Laplace smoothing parameter
- `history_size`: Maximum call history window size
- `fallback_strategy`: For context-aware: 'global', 'similar', or 'none'

**Examples:**

```python
# First-order, no context
predictor = MarkovPredictor(order=1)

# Second-order, no context
predictor = MarkovPredictor(order=2)

# First-order, context-aware
predictor = MarkovPredictor(
    order=1,
    context_aware=True,
    context_features=['user_type', 'time_of_day']
)

# Second-order, context-aware
predictor = MarkovPredictor(
    order=2,
    context_aware=True,
    context_features=['user_type']
)
```

## API Reference

### Training

#### `fit(sequences, contexts=None)`
Train the predictor:
```python
predictor.fit(sequences, contexts)
```

**Parameters:**
- `sequences`: List of API call sequences
- `contexts`: List of context dicts (required if context_aware=True)

### History Management

#### `observe(api, context=None, update=False)`
Record a new API call:
```python
predictor.observe('login')
predictor.observe('profile', context={'user_type': 'premium', 'hour': 10})
predictor.observe('orders', update=True)  # Online learning
```

**Parameters:**
- `api`: API endpoint observed
- `context`: Context dict (if context_aware)
- `update`: If True, update model (online learning)

#### `reset_history()`
Clear history (e.g., new session):
```python
predictor.reset_history()
```

### Predictions

#### `predict(k=5, context=None)`
Get top-k predictions:
```python
predictions = predictor.predict(k=5, context={'user_type': 'premium', 'hour': 10})
# [('orders', 0.8), ('settings', 0.15), ...]
```

#### `predict_sequence(length=5, context=None)`
Look-ahead predictions:
```python
seq_predictions = predictor.predict_sequence(length=5)
# [[('next1', 0.8), ...], [('next2', 0.7), ...], ...]
```

Returns predictions for each of the next `length` positions. Useful for prefetch planning.

### RL Integration

#### `get_state_vector(k=5, context=None, include_history=True)`
Get fixed-size state vector for RL:
```python
state = predictor.get_state_vector(k=5)
# np.ndarray of shape (21,) or larger
```

**State Vector Contents:**
- Top-k predicted API indices (normalized to 0-1)
- Top-k prediction probabilities
- Confidence score
- Recent history encoding (if include_history=True)
- Context encoding (if context_aware=True)

**Example shapes:**
```python
# order=1, no context, history_size=10, k=5
state.shape  # (21,) = 5 + 5 + 1 + 10

# order=1, context_aware, history_size=10, k=5
state.shape  # (28,) = 5 + 5 + 1 + 10 + 7 (context encoding)
```

The vector is **always the same size** for a given configuration, crucial for RL!

### Metrics Tracking

#### `record_outcome(actual_next)`
Record what actually happened:
```python
predictions = predictor.predict(k=5)
# ... actual API call happens ...
predictor.record_outcome('profile')
```

#### `get_metrics()`
Get accuracy metrics:
```python
metrics = predictor.get_metrics()
# {
#     'prediction_count': 100,
#     'avg_confidence': 0.85,
#     'top_1_accuracy': 0.72,
#     'top_3_accuracy': 0.89,
#     'top_5_accuracy': 0.94,
#     'coverage': 1.0
# }
```

### Persistence

```python
# Save
predictor.save('models/predictor.json')

# Load
predictor = MarkovPredictor.load('models/predictor.json')
```

### Factory Function

```python
from src.markov import create_predictor

predictor = create_predictor(config)
```

Creates predictor from project config object with attributes:
- `markov_order`
- `context_aware`
- `context_features`
- `smoothing`
- `history_size`
- `fallback_strategy`

## RL Integration Workflow

### Typical RL Episode

```python
# Setup
predictor = MarkovPredictor(order=1, history_size=10)
predictor.fit(training_sequences)

# Episode loop
predictor.reset_history()

for step in range(max_steps):
    # 1. Get state for RL agent
    state = predictor.get_state_vector(k=5)
    
    # 2. RL agent chooses action based on state
    action = rl_agent.select_action(state)
    
    # 3. Execute action (e.g., prefetch decision)
    reward = environment.step(action)
    
    # 4. Observe actual API call
    actual_api = environment.get_next_api()
    predictor.observe(actual_api)
    
    # 5. Record outcome for metrics
    predictor.record_outcome(actual_api)
    
    # 6. Train RL agent
    next_state = predictor.get_state_vector(k=5)
    rl_agent.update(state, action, reward, next_state)
```

### State Vector Usage

```python
state = predictor.get_state_vector(k=5, include_history=True)

# Shape: (21,) for order=1, no context
# [0:5]   - Predicted API indices (normalized)
# [5:10]  - Prediction probabilities
# [10]    - Confidence score
# [11:21] - History encoding
```

The RL agent treats this as a standard observation vector:
```python
# In RL agent (e.g., DQN)
q_values = self.q_network(torch.FloatTensor(state))
action = torch.argmax(q_values).item()
```

## Usage Patterns

### Pattern 1: Simple First-Order

```python
predictor = MarkovPredictor(order=1)
predictor.fit(sequences)

predictor.reset_history()
predictor.observe('login')

# Get predictions
predictions = predictor.predict(k=5)

# Get state for RL
state = predictor.get_state_vector(k=5)
```

### Pattern 2: Context-Aware

```python
predictor = MarkovPredictor(
    order=1,
    context_aware=True,
    context_features=['user_type', 'time_of_day']
)
predictor.fit(sequences, contexts)

context = {'user_type': 'premium', 'hour': 10}

predictor.reset_history()
predictor.observe('login', context=context)

# Context-aware predictions
predictions = predictor.predict(k=5, context=context)
state = predictor.get_state_vector(k=5, context=context)
```

### Pattern 3: Online Learning

```python
predictor = MarkovPredictor(order=1)
predictor.fit(initial_sequences)

# During deployment
for api in observed_apis:
    predictor.observe(api, update=True)  # Online learning
```

### Pattern 4: Prefetch Planning

```python
# Look ahead 5 steps
seq_predictions = predictor.predict_sequence(length=5)

# Prefetch based on high-confidence predictions
for i, preds in enumerate(seq_predictions):
    if preds and preds[0][1] > 0.7:  # High confidence
        prefetch_api(preds[0][0])
```

## Examples

### Example 1: Basic RL Integration

```python
import numpy as np
from src.markov import MarkovPredictor

# Setup
predictor = MarkovPredictor(order=1, history_size=10)
sequences = [
    ['login', 'profile', 'orders'],
    ['login', 'browse', 'product'],
    ['browse', 'product', 'cart']
]
predictor.fit(sequences)

# Simulate episode
predictor.reset_history()
session = ['login', 'profile', 'orders']

for api in session:
    # Before action: get state
    state = predictor.get_state_vector(k=3)
    print(f"State shape: {state.shape}")
    
    # RL agent would use this state to decide action
    # action = rl_agent.select_action(state)
    
    # Observe API
    predictor.observe(api)
    
    # After action: get reward based on prediction accuracy
    # (implementation depends on caching system)
```

### Example 2: Context-Aware RL

```python
predictor = MarkovPredictor(
    order=1,
    context_aware=True,
    context_features=['user_type', 'time_of_day']
)

sequences = [...]
contexts = [...]
predictor.fit(sequences, contexts)

# Episode with context
predictor.reset_history()
context = {'user_type': 'premium', 'hour': 10}

predictor.observe('login', context=context)
state = predictor.get_state_vector(k=5, context=context)

# State now includes context encoding
print(f"State shape with context: {state.shape}")
```

### Example 3: Metrics Tracking

```python
predictor = MarkovPredictor(order=1)
predictor.fit(sequences)

# Track predictions
predictor.reset_history()
predictor.observe('login')

for _ in range(10):
    predictions = predictor.predict(k=5)
    predictor.all_predictions.append(predictions)
    
    # Actual API happens
    actual = get_actual_api()
    predictor.record_outcome(actual)
    predictor.observe(actual)

# Get metrics
metrics = predictor.get_metrics()
print(f"Top-1 accuracy: {metrics['top_1_accuracy']:.1%}")
print(f"Top-5 accuracy: {metrics['top_5_accuracy']:.1%}")
```

## State Vector Details

### Components

```python
state = predictor.get_state_vector(k=5, include_history=True)
```

**For order=1, no context:**
```
[0:5]   Predicted indices (normalized)
[5:10]  Prediction probabilities
[10]    Confidence (max probability)
[11:21] History (last 10 APIs, normalized)
Total: 21 dimensions
```

**For order=1, with context (user_type, time_of_day):**
```
[0:5]   Predicted indices
[5:10]  Probabilities
[10]    Confidence
[11:21] History
[21:28] Context encoding (3 for user_type + 4 for time_of_day)
Total: 28 dimensions
```

### Normalization

- **API indices**: Divided by vocab_size (range: 0-1)
- **Probabilities**: Already 0-1
- **Confidence**: 0-1 (max probability)
- **History**: API indices divided by vocab_size
- **Context**: One-hot or normalized numeric

### Why Fixed Size Matters

RL algorithms (DQN, PPO, etc.) require fixed-size observations:
```python
# Neural network expects consistent input size
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)  # Must be fixed!
        # ...
```

MarkovPredictor guarantees this by:
- Padding predictions to k
- Fixed history window
- Fixed context encoding

## Configuration

### Via Constructor

```python
predictor = MarkovPredictor(
    order=1,
    context_aware=True,
    context_features=['user_type'],
    smoothing=0.001,
    history_size=10,
    fallback_strategy='global'
)
```

### Via Config Object

```python
from src.markov import create_predictor

class Config:
    markov_order = 1
    context_aware = True
    context_features = ['user_type', 'time_of_day']
    smoothing = 0.001
    history_size = 10
    fallback_strategy = 'global'

config = Config()
predictor = create_predictor(config)
```

## Performance Tips

### 1. Choose Appropriate History Size

```python
# Short history for fast-changing patterns
predictor = MarkovPredictor(history_size=5)

# Longer history for capturing more context
predictor = MarkovPredictor(history_size=20)
```

### 2. Balance k with Computation

```python
# Smaller k for faster state vectors
state = predictor.get_state_vector(k=3)  # Faster

# Larger k for more information
state = predictor.get_state_vector(k=10)  # More info, slower
```

### 3. Use Online Learning Sparingly

```python
# Batch updates are more efficient
predictor.fit(all_sequences)

# Online learning only when needed
if should_update:
    predictor.observe(api, update=True)
```

### 4. Context Features

```python
# Fewer features = less sparse data
predictor = MarkovPredictor(
    context_aware=True,
    context_features=['user_type']  # Just one feature
)

# More features = richer but sparser
predictor = MarkovPredictor(
    context_aware=True,
    context_features=['user_type', 'time_of_day', 'day_type']
)
```

## Common Issues

### Issue: State vector size changes

**Cause:** Changing k or configuration between calls

**Solution:**
```python
# Always use same k
k = 5
state1 = predictor.get_state_vector(k=k)
state2 = predictor.get_state_vector(k=k)
assert state1.shape == state2.shape  # ✓
```

### Issue: No predictions available

**Cause:** Empty history or unfitted model

**Solution:**
```python
if not predictor.is_fitted:
    predictor.fit(sequences)

if len(predictor.history) == 0:
    predictor.observe('initial_api')
```

### Issue: Context required but not provided

**Cause:** context_aware=True but context=None

**Solution:**
```python
if predictor.context_aware:
    state = predictor.get_state_vector(k=5, context=context)
else:
    state = predictor.get_state_vector(k=5)
```

## Files

- **Implementation:** `src/markov/predictor.py`
- **Demo:** `demo_predictor.py`
- **Validation:** `validate_predictor.py`
- **Quick Ref:** `PREDICTOR_QUICK_REF.md` (this file)

## Related Components

- **FirstOrderMarkovChain:** Used when order=1, context_aware=False
- **SecondOrderMarkovChain:** Used when order=2, context_aware=False
- **ContextAwareMarkovChain:** Used when context_aware=True

---

**Status:** ✅ Complete and Validated  
**Date:** January 17, 2026


# Feature Engineer Guide

## Overview

The `FeatureEngineer` module extracts numerical features from API calls for reinforcement learning state representation. It converts API calls and their context into fixed-size feature vectors that RL agents can use for decision-making.

## Key Concept

**Problem**: RL agents need numerical vectors as input, but API calls are complex objects with strings, timestamps, and varied structure.

**Solution**: Extract and encode relevant features into a consistent numerical format, normalized for neural network compatibility.

## Architecture

### Sklearn-Style Pattern

The FeatureEngineer follows the sklearn fit/transform pattern:

```python
# Training phase
fe = FeatureEngineer()
fe.fit(training_sessions)  # Learn vocabularies and statistics

# Inference phase  
features = fe.transform(call, session, history)  # Use learned parameters
```

This ensures consistent encoding between training and deployment.

## Feature Groups

### 1. Temporal Features (6 features)

Capture time-based patterns:

| Feature | Description | Range | Encoding |
|---------|-------------|-------|----------|
| hour_sin | Hour of day (sine) | [-1, 1] | Cyclic |
| hour_cos | Hour of day (cosine) | [-1, 1] | Cyclic |
| day_sin | Day of week (sine) | [-1, 1] | Cyclic |
| day_cos | Day of week (cosine) | [-1, 1] | Cyclic |
| is_weekend | Weekend flag | {0, 1} | Binary |
| is_peak_hour | Peak hour flag | {0, 1} | Binary |

**Why Cyclic Encoding?**
- Hour 23 and Hour 0 are adjacent but numerically far apart
- Cyclic encoding (sin/cos) makes them close in feature space
- Essential for capturing daily patterns

**Peak Hours**: Defined as 10-12 and 14-16 (customizable)

### 2. User Features (5 features)

Capture user context and session state:

| Feature | Description | Range | Type |
|---------|-------------|-------|------|
| user_premium | Is premium user | {0, 1} | One-hot |
| user_free | Is free user | {0, 1} | One-hot |
| user_guest | Is guest user | {0, 1} | One-hot |
| session_progress | Position in session | [0, 1] | Normalized |
| session_duration_normalized | Time elapsed | [0, 1] | Normalized |

**Session Progress**: `current_call_index / total_calls_in_session`
**Session Duration**: Capped at 30 minutes (configurable)

### 3. Request Features (N features)

Capture request characteristics:

| Feature | Description | Type |
|---------|-------------|------|
| method_GET | HTTP method | One-hot (7 methods) |
| method_POST | HTTP method | One-hot |
| method_PUT | HTTP method | One-hot |
| method_DELETE | HTTP method | One-hot |
| method_PATCH | HTTP method | One-hot |
| method_HEAD | HTTP method | One-hot |
| method_OPTIONS | HTTP method | One-hot |
| category_* | Endpoint category | One-hot (learned) |
| num_params_normalized | Parameter count | [0, 1] normalized |

**Endpoint Categories**: Extracted from path (e.g., `/api/users/123` → `users`)
- Vocabulary learned during `fit()`
- Unknown categories handled gracefully

### 4. History Features (3 features)

Capture session history:

| Feature | Description | Range |
|---------|-------------|-------|
| num_previous_calls_normalized | Calls so far | [0, 1] |
| time_since_start_normalized | Time elapsed | [0, 1] |
| avg_response_time_normalized | Avg latency | [0, 1] |

## Usage

### Basic Usage

```python
from preprocessing.feature_engineer import FeatureEngineer
from preprocessing.models import Session

# Initialize
fe = FeatureEngineer(
    temporal_features=True,
    user_features=True,
    request_features=True,
    history_features=True
)

# Fit on training data
fe.fit(training_sessions)

# Transform single call
features = fe.transform(call, session, history=[previous_calls])
print(f"Feature vector: {features.shape}")  # (N,) where N is feature count
```

### Configuration Options

```python
fe = FeatureEngineer(
    temporal_features=True,      # Include time-based features
    user_features=True,          # Include user context
    request_features=True,       # Include request details
    history_features=True,       # Include session history
    normalize_endpoints=True     # Normalize endpoint paths (recommended)
)
```

### Get Feature Information

```python
# After fitting
feature_names = fe.get_feature_names()
feature_dim = fe.get_feature_dim()
info = fe.get_feature_info()

print(f"Feature dimension: {feature_dim}")
print(f"Categories learned: {info['categories']}")
```

### Batch Processing

```python
# Fit and transform all calls
all_features = fe.fit_transform(sessions)

# Result: List of feature vectors
for i, features in enumerate(all_features):
    print(f"Call {i}: {features.shape}")
```

## Complete Example

```python
from preprocessing.feature_engineer import FeatureEngineer
from preprocessing.models import APICall, Session, Dataset
from datetime import datetime, timedelta

# Load data
dataset = Dataset.load_from_parquet('sessions.parquet')

# Split train/test
train, test = dataset.split(train_ratio=0.8)

# Initialize and fit
fe = FeatureEngineer()
fe.fit(train.sessions)

print(f"Feature dimension: {fe.get_feature_dim()}")
print(f"Feature names: {fe.get_feature_names()}")

# Transform test data
for session in test.sessions:
    for i, call in enumerate(session.calls):
        history = session.calls[:i]
        features = fe.transform(call, session, history)
        
        # Use features for RL training/inference
        state = features
        # action = rl_agent.select_action(state)
```

## Cyclic Encoding Detail

### The Problem

Linear encoding of cyclic values is problematic:
- Hour 0 encoded as 0
- Hour 23 encoded as 23
- Distance: 23 (far apart numerically)
- But hour 0 and 23 are only 1 hour apart!

### The Solution

Encode on a circle using sine and cosine:

```python
hour = 23
hour_sin = sin(2π × 23/24) ≈ -0.26
hour_cos = cos(2π × 23/24) ≈ 0.97

hour = 0
hour_sin = sin(2π × 0/24) = 0.00
hour_cos = cos(2π × 0/24) = 1.00

# Euclidean distance
distance = sqrt((0.00 - (-0.26))² + (1.00 - 0.97)²) ≈ 0.26
```

Now hour 0 and 23 are close in feature space!

### Implementation

```python
@staticmethod
def cyclic_encode(value: float, max_value: float) -> Tuple[float, float]:
    """Encode cyclic value as (sin, cos) pair."""
    normalized = 2 * math.pi * value / max_value
    return math.sin(normalized), math.cos(normalized)

# Usage
hour_sin, hour_cos = FeatureEngineer.cyclic_encode(14, 24)  # 2 PM
```

## Edge Cases

### 1. Unknown Endpoints

Endpoints not seen during training get a default category:

```python
# Training: sees /api/users, /api/products
fe.fit(training_sessions)

# Inference: encounters /api/unknown
features = fe.transform(unknown_call, None, None)
# category_unknown will be activated (or default encoding)
```

### 2. Missing Context

The module handles missing session/history gracefully:

```python
# No session context
features = fe.transform(call, session=None, history=None)
# Uses default values: session_progress=0.5, duration=0, etc.

# No history
features = fe.transform(call, session, history=None)
# Uses default: num_previous_calls=0, avg_response_time=0
```

### 3. First Call in Session

```python
# First call has no history
first_call = session.calls[0]
features = fe.transform(first_call, session, history=[])
# history_features will be zeros
```

## Feature Interpretability

### Why It Matters

Understanding what the RL agent sees helps with:
- Debugging poor decisions
- Understanding learned policies
- Identifying important features

### How to Use

```python
fe = FeatureEngineer()
fe.fit(sessions)

features = fe.transform(call, session, history)
feature_names = fe.get_feature_names()

# Print features with names
for name, value in zip(feature_names, features):
    print(f"{name:30s} = {value:7.4f}")
```

### Example Output

```
hour_sin                       =  0.5000
hour_cos                       = -0.8660
day_sin                        = -0.7818
day_cos                        =  0.6235
is_weekend                     =  1.0000
is_peak_hour                   =  1.0000
user_premium                   =  1.0000
user_free                      =  0.0000
user_guest                     =  0.0000
session_progress               =  0.2500
...
```

## Normalization Strategy

All features are normalized to roughly [-1, 1] or [0, 1]:

| Feature Type | Range | Method |
|--------------|-------|--------|
| Cyclic (sin/cos) | [-1, 1] | Natural from trigonometry |
| Binary flags | {0, 1} | Boolean to float |
| One-hot | {0, 1} | Binary per category |
| Counts | [0, 1] | Divide by max (capped) |
| Durations | [0, 1] | Divide by max (capped) |

This ensures:
- Neural networks train efficiently
- No feature dominates due to scale
- Gradients flow properly

## Integration with RL

### State Representation

```python
class RLEnvironment:
    def __init__(self, feature_engineer):
        self.fe = feature_engineer
    
    def get_state(self, call, session, history):
        # Feature vector becomes the state
        state = self.fe.transform(call, session, history)
        return state
    
    def reset(self):
        # Start new episode
        self.history = []
        return self.get_state(first_call, session, [])
    
    def step(self, action):
        # Execute action, get next state
        next_call = ...
        self.history.append(current_call)
        next_state = self.get_state(next_call, session, self.history)
        reward = ...
        return next_state, reward, done
```

### Policy Network Input

```python
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, features):
        # features come from FeatureEngineer.transform()
        return self.network(features)

# Usage
fe = FeatureEngineer()
fe.fit(training_sessions)

policy = PolicyNetwork(
    feature_dim=fe.get_feature_dim(),
    action_dim=num_actions
)

# Training loop
for call in training_calls:
    features = fe.transform(call, session, history)
    state_tensor = torch.tensor(features)
    action_probs = policy(state_tensor)
    # ... RL training logic
```

## Performance Considerations

### Memory

- **Vocabulary Size**: O(unique_endpoints + unique_categories)
- **Feature Vector**: Fixed size regardless of input
- **Typical Size**: 20-50 features

### Computation

- **Transform Speed**: ~0.1ms per call (very fast)
- **Bottleneck**: Usually not the feature extraction
- **Optimization**: Pre-compute feature vectors for training

### Caching

```python
# Pre-compute features for training
training_features = []
training_labels = []

for session in training_sessions:
    for i, call in enumerate(session.calls):
        history = session.calls[:i]
        features = fe.transform(call, session, history)
        training_features.append(features)
        training_labels.append(label)

# Cache to disk
np.save('training_features.npy', training_features)
np.save('training_labels.npy', training_labels)
```

## Best Practices

1. **Always fit before transform**: Learn vocabularies from training data first
2. **Use consistent configuration**: Same feature groups for train and test
3. **Normalize endpoints**: Ensures consistent representation
4. **Include temporal features**: Captures usage patterns
5. **Include history features**: Gives agent memory of session
6. **Monitor feature distributions**: Check for outliers or skewed features
7. **Version your feature engineer**: Save fitted models for reproducibility

## Troubleshooting

### Problem: Features all zeros
**Solution**: Make sure you called `fit()` before `transform()`

### Problem: Unknown category warnings
**Solution**: Normal for test data. The module handles this gracefully.

### Problem: Feature dimension mismatch
**Solution**: Ensure same configuration (feature groups) for train and test

### Problem: Poor RL performance
**Solution**: Check feature interpretability. Are important signals captured?

## Related Modules

- **preprocessing.models**: APICall, Session, Dataset classes
- **preprocessing.sequence_builder**: Sequence extraction for Markov chains
- **src.rl**: RL training using these features
- **src.markov**: Markov chain training (uses different features)

## References

- Cyclic encoding: [Fourier Features](https://arxiv.org/abs/2006.10739)
- Feature engineering for RL: [Deep RL Book Chapter 3](http://incompleteideas.net/book/)
- Normalization: [Neural Network Best Practices](https://arxiv.org/abs/1502.03167)


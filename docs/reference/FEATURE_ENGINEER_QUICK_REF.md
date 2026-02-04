# FeatureEngineer Quick Reference

## Import
```python
from preprocessing.feature_engineer import FeatureEngineer
```

## Initialize
```python
# Default (all features enabled)
fe = FeatureEngineer()

# Custom configuration
fe = FeatureEngineer(
    temporal_features=True,
    user_features=True,
    request_features=True,
    history_features=True,
    normalize_endpoints=True
)
```

## Fit/Transform Pattern

```python
# Training: Learn from data
fe.fit(training_sessions)

# Inference: Transform calls
features = fe.transform(call, session, history)
```

## Main Methods

### fit()
```python
fe.fit(sessions: List[Session]) -> FeatureEngineer
```
Learn vocabularies and statistics from training data.

### transform()
```python
features = fe.transform(
    call: APICall,
    session: Optional[Session] = None,
    history: Optional[List[APICall]] = None
) -> np.ndarray
```
Convert API call to feature vector.

### fit_transform()
```python
all_features = fe.fit_transform(sessions: List[Session]) -> List[np.ndarray]
```
Fit and transform all calls in one step.

### get_feature_names()
```python
names = fe.get_feature_names() -> List[str]
```
Get ordered list of feature names.

### get_feature_dim()
```python
dim = fe.get_feature_dim() -> int
```
Get feature vector dimensionality.

### get_feature_info()
```python
info = fe.get_feature_info() -> Dict
```
Get statistics about fitted feature engineer.

## Feature Groups

### Temporal (6 features)
- `hour_sin`, `hour_cos` - Hour of day (cyclic)
- `day_sin`, `day_cos` - Day of week (cyclic)
- `is_weekend` - Weekend flag
- `is_peak_hour` - Peak hour flag (10-12, 14-16)

### User (5 features)
- `user_premium`, `user_free`, `user_guest` - User type (one-hot)
- `session_progress` - Position in session (0-1)
- `session_duration_normalized` - Time elapsed (0-1)

### Request (N features)
- `method_GET`, `method_POST`, ... - HTTP method (one-hot)
- `category_*` - Endpoint category (one-hot, learned)
- `num_params_normalized` - Parameter count (0-1)

### History (3 features)
- `num_previous_calls_normalized` - Calls so far (0-1)
- `time_since_start_normalized` - Time elapsed (0-1)
- `avg_response_time_normalized` - Average latency (0-1)

## Cyclic Encoding

```python
# Static method
sin_val, cos_val = FeatureEngineer.cyclic_encode(value, max_value)

# Examples
hour_sin, hour_cos = FeatureEngineer.cyclic_encode(14, 24)  # 2 PM
day_sin, day_cos = FeatureEngineer.cyclic_encode(5, 7)      # Saturday
```

## Complete Workflow

```python
from preprocessing.feature_engineer import FeatureEngineer
from preprocessing.models import Dataset

# 1. Load data
dataset = Dataset.load_from_parquet('sessions.parquet')
train, test = dataset.split(0.8)

# 2. Fit feature engineer
fe = FeatureEngineer()
fe.fit(train.sessions)

print(f"Feature dim: {fe.get_feature_dim()}")
print(f"Categories: {fe.get_feature_info()['categories']}")

# 3. Transform calls
for session in test.sessions:
    for i, call in enumerate(session.calls):
        history = session.calls[:i]
        features = fe.transform(call, session, history)
        # Use features for RL/ML
```

## RL Integration

```python
class Environment:
    def __init__(self, feature_engineer):
        self.fe = feature_engineer
        
    def get_state(self, call, session, history):
        return self.fe.transform(call, session, history)
    
    def step(self, action):
        # Execute action
        next_call = ...
        next_state = self.get_state(next_call, session, history)
        reward = ...
        return next_state, reward, done

# Create environment
fe = FeatureEngineer()
fe.fit(training_sessions)
env = Environment(fe)

# RL training loop
state = env.reset()
for t in range(max_steps):
    action = agent.select_action(state)
    next_state, reward, done = env.step(action)
    agent.update(state, action, reward, next_state)
    state = next_state
    if done:
        break
```

## Common Patterns

### Debug Features
```python
features = fe.transform(call, session, history)
names = fe.get_feature_names()

print("Feature vector:")
for name, value in zip(names, features):
    print(f"  {name:30s} = {value:7.4f}")
```

### Batch Processing
```python
# Process all calls
feature_vectors = []
for session in sessions:
    for i, call in enumerate(session.calls):
        history = session.calls[:i]
        features = fe.transform(call, session, history)
        feature_vectors.append(features)

# Convert to numpy array
X = np.array(feature_vectors)
print(f"Dataset shape: {X.shape}")
```

### Save/Load Fitted Model
```python
import pickle

# Save
with open('feature_engineer.pkl', 'wb') as f:
    pickle.dump(fe, f)

# Load
with open('feature_engineer.pkl', 'rb') as f:
    fe = pickle.load(f)
```

### Feature Analysis
```python
# Get all features for a dataset
all_features = fe.fit_transform(sessions)

# Compute statistics
features_array = np.array(all_features)
means = features_array.mean(axis=0)
stds = features_array.std(axis=0)

print("Feature statistics:")
for name, mean, std in zip(fe.get_feature_names(), means, stds):
    print(f"  {name:30s}: μ={mean:7.4f}, σ={std:7.4f}")
```

## Edge Cases

### No Session Context
```python
# Will use defaults
features = fe.transform(call, session=None, history=None)
```

### First Call
```python
# Empty history
features = fe.transform(first_call, session, history=[])
```

### Unknown Endpoint
```python
# Not seen during fit - uses default category
features = fe.transform(unknown_call, session, history)
```

## Configuration Examples

### Minimal Features (Fast)
```python
fe = FeatureEngineer(
    temporal_features=True,
    user_features=True,
    request_features=False,  # Skip request features
    history_features=False   # Skip history features
)
# Result: ~11 features
```

### Maximum Features (Expressive)
```python
fe = FeatureEngineer(
    temporal_features=True,
    user_features=True,
    request_features=True,
    history_features=True
)
# Result: ~20-50 features depending on vocabulary
```

### Only Temporal
```python
fe = FeatureEngineer(
    temporal_features=True,
    user_features=False,
    request_features=False,
    history_features=False
)
# Result: 6 features
```

## Performance Tips

1. **Pre-compute for training**: Call `fit_transform()` once, cache results
2. **Batch inference**: Process multiple calls at once
3. **Feature selection**: Disable unused feature groups
4. **Endpoint normalization**: Always enable for consistency

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Features all zeros | Call `fit()` before `transform()` |
| Dimension mismatch | Use same config for train/test |
| Unknown categories | Normal - handled automatically |
| Poor RL performance | Check feature interpretability |

## Feature Value Ranges

| Feature Type | Expected Range |
|--------------|----------------|
| Cyclic (sin/cos) | [-1, 1] |
| Binary flags | {0, 1} |
| One-hot encoding | {0, 1} |
| Normalized counts | [0, 1] |
| Normalized durations | [0, 1] |

## Constants

```python
# In FeatureEngineer class
PEAK_HOURS = {10, 11, 14, 15}
HTTP_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']
USER_TYPES = ['premium', 'free', 'guest']
MAX_SESSION_DURATION_SECONDS = 1800  # 30 minutes
MAX_PARAMS = 20
MAX_HISTORY_LENGTH = 50
```

## See Also

- `preprocessing/FEATURE_ENGINEER_GUIDE.md` - Comprehensive documentation
- `test_feature_engineer.py` - Test suite with examples
- `preprocessing/models.py` - Data model definitions
- `src/rl/` - RL training using features


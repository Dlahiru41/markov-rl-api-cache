# Action Space Quick Reference

## Import
```python
from src.rl.actions import CacheAction, ActionSpace, ActionConfig, ActionHistory
```

## CacheAction Enum (7 Actions)

| Value | Name | Description |
|-------|------|-------------|
| 0 | DO_NOTHING | Let normal LRU behavior happen |
| 1 | CACHE_CURRENT | Explicitly cache current response |
| 2 | PREFETCH_CONSERVATIVE | Prefetch top-1 if prob > 70% |
| 3 | PREFETCH_MODERATE | Prefetch top-3 if prob > 50% |
| 4 | PREFETCH_AGGRESSIVE | Prefetch top-5 if prob > 30% |
| 5 | EVICT_LRU | Evict least-recently-used entries |
| 6 | EVICT_LOW_PROB | Evict low-probability entries |

### Methods
```python
CacheAction.num_actions()              # → 7
CacheAction.get_name(2)                # → "PREFETCH_CONSERVATIVE"
CacheAction.get_description(2)         # → "Prefetch only the top-1..."
```

## ActionConfig (Default Values)

```python
config = ActionConfig(
    conservative_threshold=0.7,   # Min probability for conservative
    moderate_threshold=0.5,        # Min probability for moderate
    aggressive_threshold=0.3,      # Min probability for aggressive
    conservative_count=1,          # Number to prefetch (conservative)
    moderate_count=3,              # Number to prefetch (moderate)
    aggressive_count=5,            # Number to prefetch (aggressive)
    eviction_batch_size=10         # Number to evict at once
)
```

## ActionSpace

### Create
```python
space = ActionSpace()                    # Default config
space = ActionSpace(config=my_config)    # Custom config
```

### Properties
```python
space.n                                  # → 7 (number of actions)
```

### Methods

#### Sample Random Action
```python
action = space.sample()                  # → Random int in [0, 6]
```

#### Get Valid Actions
```python
valid = space.get_valid_actions(
    cache_utilization=0.6,    # Cache fullness (0.0-1.0)
    has_predictions=True,     # Are Markov predictions available?
    cache_size=100           # Number of entries in cache
)
# Returns: List of valid action indices
```

#### Get Action Mask
```python
mask = space.get_action_mask(
    cache_utilization=0.6,
    has_predictions=True,
    cache_size=100
)
# Returns: Boolean numpy array (shape: 7)
# Use to mask invalid actions in policy network
```

#### Decode Action
```python
predictions = [('api1', 0.85), ('api2', 0.55), ('api3', 0.35)]
decoded = space.decode_action(
    action=CacheAction.PREFETCH_MODERATE,
    predictions=predictions
)
# Returns: Dictionary with execution instructions
```

### Decoded Action Format

```python
{
    'action_type': 'none' | 'cache' | 'prefetch' | 'evict',
    'cache_current': bool,
    'apis_to_prefetch': List[str],
    'eviction_strategy': 'lru' | 'low_prob' | None,
    'eviction_count': int
}
```

## ActionHistory

### Create and Record
```python
history = ActionHistory()

history.record(
    action=CacheAction.PREFETCH_MODERATE,
    state=state_vector,      # numpy array
    reward=0.75,             # float
    context={'step': 100}    # optional dict
)
```

### Analyze
```python
# Action distribution (frequencies)
dist = history.get_action_distribution()
# → {'DO_NOTHING': 0.3, 'CACHE_CURRENT': 0.2, ...}

# Average reward per action
rewards = history.get_reward_by_action()
# → {'DO_NOTHING': 0.5, 'CACHE_CURRENT': 0.8, ...}

# Full statistics
stats = history.get_statistics()
# → {'total_actions': 1000, 'total_reward': 650, ...}

# Recent actions
recent = history.get_recent_actions(n=10)
# → List of last 10 action records

# Clear history
history.clear()
```

## Valid Action Logic

| Condition | Valid Actions |
|-----------|--------------|
| Always | DO_NOTHING, CACHE_CURRENT |
| has_predictions=True | + All PREFETCH actions |
| cache_size > 0 | + EVICT_LRU |
| has_predictions=True AND cache_size > 0 | + EVICT_LOW_PROB |

## Prefetch Filtering Examples

Given predictions: `[('api1', 0.85), ('api2', 0.60), ('api3', 0.40), ('api4', 0.15)]`

| Action | Threshold | Count | Result |
|--------|-----------|-------|--------|
| PREFETCH_CONSERVATIVE | >0.7 | 1 | ['api1'] |
| PREFETCH_MODERATE | >0.5 | 3 | ['api1', 'api2'] |
| PREFETCH_AGGRESSIVE | >0.3 | 5 | ['api1', 'api2', 'api3'] |

## Common Use Cases

### RL Agent Action Selection
```python
# Get state
state = state_builder.build_state(...)

# Get valid actions
valid = space.get_valid_actions(
    cache_utilization=cache.utilization,
    has_predictions=len(predictions) > 0,
    cache_size=cache.size
)

# Agent selects from valid actions
action = agent.select_action(state, valid_actions=valid)

# Decode and execute
decoded = space.decode_action(action, predictions)
execute_cache_action(decoded)
```

### Action Masking in Neural Network
```python
def forward(self, state, cache_info):
    # Get action logits
    logits = self.policy_network(state)
    
    # Get valid action mask
    mask = self.action_space.get_action_mask(
        cache_info['utilization'],
        cache_info['has_predictions'],
        cache_info['size']
    )
    
    # Mask invalid actions
    logits[~mask] = float('-inf')
    
    # Compute probabilities
    probs = torch.softmax(logits, dim=0)
    return probs
```

### Analyzing Agent Behavior
```python
# After training
dist = history.get_action_distribution()
rewards = history.get_reward_by_action()

print("Most used action:", max(dist, key=dist.get))
print("Best rewarded action:", max(rewards, key=rewards.get))

# Check for issues
if dist['DO_NOTHING'] > 0.8:
    print("⚠️ Agent is too passive")
if dist['PREFETCH_AGGRESSIVE'] > 0.5:
    print("⚠️ Agent may be over-prefetching")
```

## Custom Configuration Example

```python
# More aggressive for low-latency requirements
aggressive_config = ActionConfig(
    conservative_threshold=0.6,
    moderate_threshold=0.4,
    aggressive_threshold=0.2,
    conservative_count=2,
    moderate_count=5,
    aggressive_count=10,
    eviction_batch_size=20
)

# More conservative for resource-constrained environments
conservative_config = ActionConfig(
    conservative_threshold=0.85,
    moderate_threshold=0.70,
    aggressive_threshold=0.55,
    conservative_count=1,
    moderate_count=2,
    aggressive_count=3,
    eviction_batch_size=5
)
```

## Decoding All Actions

### DO_NOTHING
```python
{'action_type': 'none', 'cache_current': False, 
 'apis_to_prefetch': [], 'eviction_strategy': None, 'eviction_count': 0}
```

### CACHE_CURRENT
```python
{'action_type': 'cache', 'cache_current': True,
 'apis_to_prefetch': [], 'eviction_strategy': None, 'eviction_count': 0}
```

### PREFETCH_* (any)
```python
{'action_type': 'prefetch', 'cache_current': False,
 'apis_to_prefetch': ['api1', 'api2'], 'eviction_strategy': None, 'eviction_count': 0}
```

### EVICT_LRU
```python
{'action_type': 'evict', 'cache_current': False,
 'apis_to_prefetch': [], 'eviction_strategy': 'lru', 'eviction_count': 10}
```

### EVICT_LOW_PROB
```python
{'action_type': 'evict', 'cache_current': False,
 'apis_to_prefetch': [], 'eviction_strategy': 'low_prob', 'eviction_count': 10}
```

## Debugging Tips

### Check Action Distribution
```python
dist = history.get_action_distribution()
for action, freq in sorted(dist.items(), key=lambda x: -x[1]):
    print(f"{action:25s}: {freq*100:5.1f}%")
```

### Compare Rewards
```python
rewards = history.get_reward_by_action()
for action, reward in sorted(rewards.items(), key=lambda x: -x[1]):
    print(f"{action:25s}: {reward:6.3f}")
```

### Validate Actions
```python
# Test all actions are decodable
for i in range(CacheAction.num_actions()):
    name = CacheAction.get_name(i)
    decoded = space.decode_action(i, [])
    print(f"{i}: {name:25s} → {decoded['action_type']}")
```

## Integration Checklist

- [ ] Create ActionSpace with appropriate config
- [ ] Implement get_valid_actions() in your environment
- [ ] Use action masking in policy network
- [ ] Decode actions before execution
- [ ] Track actions with ActionHistory
- [ ] Monitor action distribution during training
- [ ] Analyze reward by action type
- [ ] Adjust config based on performance

## Performance Notes

- Action space size: 7 (small, fast to learn)
- Decoding: O(k) where k = number of predictions
- Sampling: O(1)
- Masking: O(1)
- History: O(1) per record, O(n) for statistics


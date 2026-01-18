# Action Space Module - Complete Guide

## Overview

The `src.rl.actions` module defines the action space for the caching RL agent - the set of decisions it can make to optimize cache performance. This includes prefetching strategies, eviction policies, and explicit caching control.

## Architecture

### 1. CacheAction Enum

An IntEnum defining 7 possible actions the RL agent can take:

| Action | Value | Description |
|--------|-------|-------------|
| **DO_NOTHING** | 0 | Take no action, let normal LRU behavior happen |
| **CACHE_CURRENT** | 1 | Explicitly cache the current API response |
| **PREFETCH_CONSERVATIVE** | 2 | Prefetch top-1 prediction if probability > 70% |
| **PREFETCH_MODERATE** | 3 | Prefetch top-3 predictions if probability > 50% |
| **PREFETCH_AGGRESSIVE** | 4 | Prefetch top-5 predictions if probability > 30% |
| **EVICT_LRU** | 5 | Proactively evict least-recently-used entries |
| **EVICT_LOW_PROB** | 6 | Evict entries with lowest predicted future access probability |

**Why IntEnum?**
- Direct use as array indices
- Neural network compatibility (outputs 0-6)
- Efficient storage and comparison

**Methods:**
- `num_actions()` → Returns 7
- `get_name(action)` → Human-readable name (e.g., "PREFETCH_MODERATE")
- `get_description(action)` → Detailed explanation of what action does

### 2. ActionConfig Dataclass

Tunable configuration for action execution:

```python
@dataclass
class ActionConfig:
    # Probability thresholds for prefetching
    conservative_threshold: float = 0.7  # Only high-confidence predictions
    moderate_threshold: float = 0.5      # Medium confidence
    aggressive_threshold: float = 0.3    # Lower confidence acceptable
    
    # Number of APIs to prefetch for each strategy
    conservative_count: int = 1   # Prefetch only top prediction
    moderate_count: int = 3       # Prefetch top 3
    aggressive_count: int = 5     # Prefetch top 5
    
    # Eviction parameters
    eviction_batch_size: int = 10  # How many entries to evict at once
```

**Design Philosophy:**
- **Conservative**: High precision, low recall - only prefetch when very confident
- **Moderate**: Balanced approach - reasonable confidence required
- **Aggressive**: High recall, lower precision - prefetch more speculatively

### 3. ActionSpace Class

Main interface for working with actions:

#### Properties
- `n` → Number of actions (7)

#### Methods

**`sample() → int`**
- Returns random action for exploration
- Uses internal RNG for reproducibility

**`get_valid_actions(cache_utilization, has_predictions, cache_size) → List[int]`**
- Returns list of valid action indices given current system state
- Logic:
  - DO_NOTHING and CACHE_CURRENT: Always valid
  - Prefetch actions: Valid only if `has_predictions=True`
  - EVICT_LRU: Valid only if `cache_size > 0`
  - EVICT_LOW_PROB: Valid only if `cache_size > 0` AND `has_predictions=True`

**`get_action_mask(cache_utilization, has_predictions, cache_size) → np.ndarray`**
- Returns boolean array (shape: 7) indicating valid actions
- Use case: Mask invalid actions in policy network output
- True = valid, False = invalid

**`decode_action(action, predictions) → Dict[str, Any]`**
- Converts action index into concrete execution instructions
- Returns dictionary with:
  - `action_type`: 'none', 'cache', 'prefetch', or 'evict'
  - `cache_current`: bool - whether to cache current response
  - `apis_to_prefetch`: list of API endpoints to prefetch
  - `eviction_strategy`: 'lru', 'low_prob', or None
  - `eviction_count`: number of entries to evict

### 4. ActionHistory Class

Tracks and analyzes agent behavior over time:

**Methods:**
- `record(action, state, reward, context)` - Store an action taken
- `get_action_distribution()` - Return frequency of each action (0.0-1.0)
- `get_reward_by_action()` - Return average reward per action type
- `get_statistics()` - Comprehensive statistics
- `get_recent_actions(n)` - Get last n actions
- `clear()` - Reset history

**Use Cases:**
- Debugging: "Is the agent learning sensible behavior?"
- Analysis: "Which actions lead to highest rewards?"
- Monitoring: "Is the agent stuck in local optima?"

## Action Decoding Examples

### DO_NOTHING (0)
```python
{
    'action_type': 'none',
    'cache_current': False,
    'apis_to_prefetch': [],
    'eviction_strategy': None,
    'eviction_count': 0
}
```

### CACHE_CURRENT (1)
```python
{
    'action_type': 'cache',
    'cache_current': True,
    'apis_to_prefetch': [],
    'eviction_strategy': None,
    'eviction_count': 0
}
```

### PREFETCH_MODERATE (3)
Given predictions: `[('profile', 0.8), ('browse', 0.55), ('cart', 0.35)]`

```python
{
    'action_type': 'prefetch',
    'cache_current': False,
    'apis_to_prefetch': ['profile', 'browse'],  # Both > 0.5 threshold
    'eviction_strategy': None,
    'eviction_count': 0
}
```

### EVICT_LRU (5)
```python
{
    'action_type': 'evict',
    'cache_current': False,
    'apis_to_prefetch': [],
    'eviction_strategy': 'lru',
    'eviction_count': 10
}
```

## Usage Examples

### Basic Usage
```python
from src.rl.actions import CacheAction, ActionSpace, ActionConfig

# Create action space with default config
space = ActionSpace()
print(f"Number of actions: {space.n}")  # 7

# Sample random action
action = space.sample()
print(f"Action: {CacheAction.get_name(action)}")

# Get valid actions
valid = space.get_valid_actions(
    cache_utilization=0.6,
    has_predictions=True,
    cache_size=100
)
print(f"Valid actions: {[CacheAction.get_name(a) for a in valid]}")

# Decode action
predictions = [('api1', 0.85), ('api2', 0.55), ('api3', 0.35)]
decoded = space.decode_action(CacheAction.PREFETCH_MODERATE, predictions)
print(f"Prefetch: {decoded['apis_to_prefetch']}")  # ['api1', 'api2']
```

### Custom Configuration
```python
# Create more aggressive configuration
config = ActionConfig(
    conservative_threshold=0.6,  # Lower threshold
    moderate_threshold=0.4,
    aggressive_threshold=0.2,
    conservative_count=2,
    moderate_count=5,
    aggressive_count=10,
    eviction_batch_size=20
)

space = ActionSpace(config=config)

# Now PREFETCH_MODERATE is more aggressive
predictions = [('api1', 0.85), ('api2', 0.55), ('api3', 0.45), ('api4', 0.35)]
decoded = space.decode_action(CacheAction.PREFETCH_MODERATE, predictions)
print(decoded['apis_to_prefetch'])  # ['api1', 'api2', 'api3'] (all > 0.4)
```

### Action Masking in RL
```python
# In policy network forward pass
def select_action(self, state, cache_state):
    # Get action logits from network
    logits = self.policy_network(state)  # Shape: (7,)
    
    # Get valid action mask
    mask = self.action_space.get_action_mask(
        cache_utilization=cache_state['utilization'],
        has_predictions=cache_state['has_predictions'],
        cache_size=cache_state['size']
    )
    
    # Mask invalid actions (set logits to -inf)
    logits[~mask] = float('-inf')
    
    # Sample from valid actions
    probs = torch.softmax(logits, dim=0)
    action = torch.multinomial(probs, 1).item()
    
    return action
```

### Tracking Agent Behavior
```python
from src.rl.actions import ActionHistory
import numpy as np

history = ActionHistory()

# During training
for step in range(1000):
    state = get_current_state()
    action = agent.select_action(state)
    reward = execute_action(action)
    
    history.record(action, state, reward, context={'step': step})

# Analyze behavior
print("Action distribution:")
for action_name, freq in history.get_action_distribution().items():
    print(f"  {action_name}: {freq*100:.1f}%")

print("\nAverage reward by action:")
for action_name, avg_reward in history.get_reward_by_action().items():
    print(f"  {action_name}: {avg_reward:.3f}")

# Check if agent is stuck
dist = history.get_action_distribution()
if dist['DO_NOTHING'] > 0.8:
    print("⚠️ Agent taking DO_NOTHING 80% of the time - may need exploration")
```

### Integration with Cache System
```python
def execute_cache_action(action_idx, predictions, cache_system):
    """Execute the action in the real cache system."""
    
    # Decode action
    decoded = action_space.decode_action(action_idx, predictions)
    
    if decoded['action_type'] == 'none':
        pass  # Do nothing
    
    elif decoded['action_type'] == 'cache':
        cache_system.cache_current_response()
    
    elif decoded['action_type'] == 'prefetch':
        for api in decoded['apis_to_prefetch']:
            cache_system.prefetch(api)
    
    elif decoded['action_type'] == 'evict':
        if decoded['eviction_strategy'] == 'lru':
            cache_system.evict_lru(count=decoded['eviction_count'])
        elif decoded['eviction_strategy'] == 'low_prob':
            cache_system.evict_low_probability(
                count=decoded['eviction_count'],
                predictions=predictions
            )
```

## Design Decisions

### Why 7 Actions?

**Sufficiently Expressive:**
- Covers key caching decisions: do nothing, cache, prefetch, evict
- Multiple prefetch strategies for different confidence levels
- Multiple eviction strategies for different scenarios

**Not Too Large:**
- Smaller action space → faster learning
- Easier to explore all actions
- Simpler policy network

**Expandable:**
- Easy to add more actions if needed (e.g., PREFETCH_CUSTOM)
- IntEnum supports arbitrary values

### Why Three Prefetch Levels?

**Conservative (prob > 0.7):**
- Use when: Cache is nearly full, eviction is expensive
- Trade-off: High precision, low coverage
- Example: Premium users with strict SLA

**Moderate (prob > 0.5):**
- Use when: Balanced performance/resource trade-off
- Trade-off: Reasonable precision and coverage
- Example: Normal operating conditions

**Aggressive (prob > 0.3):**
- Use when: Cache is empty, prefetch is cheap, latency critical
- Trade-off: High coverage, more wasted prefetches
- Example: Cold start, off-peak hours with spare capacity

### Why Action Masking?

**Prevents Invalid Actions:**
- Can't evict from empty cache
- Can't prefetch without predictions
- Forces agent to learn valid behaviors

**Improves Learning:**
- No wasted exploration on impossible actions
- Clearer signal about state-action validity
- Faster convergence

**Safety:**
- Prevents runtime errors
- Guarantees system stability

## Validation Logic

### Valid Action Matrix

| Action | Always Valid | Requires Predictions | Requires Cache Entries |
|--------|-------------|---------------------|----------------------|
| DO_NOTHING | ✓ | ✗ | ✗ |
| CACHE_CURRENT | ✓ | ✗ | ✗ |
| PREFETCH_CONSERVATIVE | ✗ | ✓ | ✗ |
| PREFETCH_MODERATE | ✗ | ✓ | ✗ |
| PREFETCH_AGGRESSIVE | ✗ | ✓ | ✗ |
| EVICT_LRU | ✗ | ✗ | ✓ |
| EVICT_LOW_PROB | ✗ | ✓ | ✓ |

## Performance Considerations

**Action Space Size:**
- Small (7 actions) → fast neural network inference
- Discrete → simpler than continuous control
- IntEnum → efficient storage and comparison

**Decoding Speed:**
- O(k) where k = number of predictions
- Single pass through predictions
- No expensive operations

**Memory:**
- ActionHistory stores full states → can grow large
- Consider periodic clearing or fixed-size buffer
- Each record: ~100 bytes + state size

## Common Patterns

### Epsilon-Greedy Exploration
```python
def select_action(state, epsilon=0.1):
    if random.random() < epsilon:
        return space.sample()  # Explore
    else:
        return agent.best_action(state)  # Exploit
```

### Softmax Action Selection
```python
def select_action(state, temperature=1.0):
    logits = agent.get_action_logits(state)
    probs = softmax(logits / temperature)
    return np.random.choice(space.n, p=probs)
```

### Deterministic Policy
```python
def select_action(state):
    return agent.get_best_action(state)  # Greedy
```

## Future Enhancements

Potential additions:
1. **Dynamic thresholds**: Learn optimal thresholds
2. **Custom prefetch counts**: Specify exact number per state
3. **Partial eviction**: Evict specific entries, not batch
4. **Hybrid strategies**: Combine multiple actions
5. **Action costs**: Penalize expensive actions
6. **Context-aware actions**: Different actions for different user types

## References

- Discrete action spaces: Sutton & Barto, RL: An Introduction
- Action masking: [Huang et al., 2020 - "Action Masking in Deep RL"]
- Prefetching strategies: [Tian et al., 2017 - "Learned Prefetching"]


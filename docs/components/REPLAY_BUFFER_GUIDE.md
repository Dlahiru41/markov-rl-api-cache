# Replay Buffer Implementation - Complete Guide

## Overview

Experience replay buffers have been successfully implemented for DQN agent training. The implementation includes:

1. **ReplayBuffer** - Uniform random sampling
2. **PrioritizedReplayBuffer** - Priority-based sampling
3. **SumTree** - Efficient data structure for prioritized sampling
4. **Experience** - Named tuple for storing transitions

## File Location

`src/rl/replay_buffer.py`

## Features Implemented

### 1. Experience Named Tuple
```python
Experience = namedtuple('Experience', 
    field_names=['state', 'action', 'reward', 'next_state', 'done'])
```

### 2. ReplayBuffer Class (Uniform Sampling)

**Key Features:**
- Fixed maximum capacity with FIFO eviction
- O(1) insertion, O(batch_size) sampling
- Reproducible sampling with optional seed
- Automatic dtype conversion (float32 for states, int64 for actions)
- Save/load functionality for checkpointing

**Methods:**
- `__init__(capacity, seed=None)` - Initialize buffer
- `push(state, action, reward, next_state, done)` - Add experience
- `sample(batch_size)` - Sample random batch
- `__len__()` - Current buffer size
- `is_ready(batch_size)` - Check if enough samples available
- `clear()` - Remove all experiences
- `save(path)` - Save buffer to disk
- `load(path)` - Load buffer from disk

**Example Usage:**
```python
from src.rl.replay_buffer import ReplayBuffer
import numpy as np

# Create buffer
buffer = ReplayBuffer(capacity=1000, seed=42)

# Add experiences
for i in range(100):
    state = np.random.randn(60).astype(np.float32)
    next_state = np.random.randn(60).astype(np.float32)
    buffer.push(state, action=i % 7, reward=np.random.randn(),
                next_state=next_state, done=False)

# Check readiness
print(f"Buffer size: {len(buffer)}")
print(f"Ready for batch of 32: {buffer.is_ready(32)}")

# Sample batch
states, actions, rewards, next_states, dones = buffer.sample(32)
print(f"Batch shapes: states={states.shape}, actions={actions.shape}")
```

### 3. PrioritizedReplayBuffer Class

**Key Features:**
- Samples experiences proportional to their TD-error (priority)
- Implements importance sampling with bias correction
- Beta annealing from beta_start to beta_end over beta_frames
- SumTree data structure for O(log n) sampling and updates
- Configurable alpha (prioritization strength)

**Methods:**
- `__init__(capacity, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_frames=100000, epsilon=1e-6, seed=None)`
- `push(state, action, reward, next_state, done, priority=None)` - Add experience with priority
- `sample(batch_size)` - Sample batch with priorities
- `update_priorities(indices, priorities)` - Update priorities based on TD-errors
- `is_ready(batch_size)` - Check if enough samples
- `save(path)`, `load(path)` - Persistence

**Sampling Returns:**
- `states` - (batch_size, state_dim) float32
- `actions` - (batch_size,) int64
- `rewards` - (batch_size,) float32
- `next_states` - (batch_size, state_dim) float32
- `dones` - (batch_size,) float32
- `weights` - (batch_size,) float32 (importance sampling weights)
- `indices` - list of tree indices for priority updates

**Example Usage:**
```python
from src.rl.replay_buffer import PrioritizedReplayBuffer
import numpy as np

# Create prioritized buffer
pbuffer = PrioritizedReplayBuffer(
    capacity=1000,
    alpha=0.6,        # Prioritization strength
    beta_start=0.4,   # Initial IS correction
    beta_end=1.0,     # Final IS correction
    beta_frames=100000  # Annealing schedule
)

# Add experiences
for i in range(100):
    state = np.random.randn(60).astype(np.float32)
    next_state = np.random.randn(60).astype(np.float32)
    pbuffer.push(state, action=i % 7, reward=np.random.randn(),
                 next_state=next_state, done=False)

# Sample batch
states, actions, rewards, next_states, dones, weights, indices = pbuffer.sample(32)
print(f"Weights shape: {weights.shape}")
print(f"Indices: {indices[:5]}")

# After computing TD-errors in training loop
td_errors = np.abs(target_q - predicted_q)  # From DQN
new_priorities = td_errors + 0.01  # Add epsilon
pbuffer.update_priorities(indices, new_priorities)
```

### 4. SumTree Class

**Purpose:** Efficient prioritized sampling data structure

**Features:**
- Binary tree where each node = sum of children
- O(log n) sampling proportional to priorities
- O(log n) priority updates
- O(1) total priority query

**Methods:**
- `add(priority, data)` - Add new experience
- `update(tree_idx, priority)` - Update priority
- `get(priority_sum)` - Sample by cumulative priority
- `total_priority` - Sum of all priorities (property)
- `max_priority` - Maximum priority (property)
- `min_priority` - Minimum non-zero priority (property)

## Theory and Design

### Why Experience Replay?

1. **Breaks Temporal Correlations**: RL agents learn from sequences of highly correlated states. Sampling randomly from past experiences breaks these correlations, leading to more stable learning.

2. **Sample Efficiency**: Each experience can be used multiple times for training, improving data efficiency.

3. **Stabilizes Training**: Random sampling prevents the agent from overfitting to recent experiences.

### Prioritized Experience Replay

**Key Idea:** Not all experiences are equally important. Experiences with high TD-error (where the agent's prediction was wrong) are more informative.

**Sampling Probability:**
```
P(i) = p_i^alpha / Σ(p_j^alpha)
```
where p_i = |TD-error_i| + epsilon

**Importance Sampling Correction:**
```
w_i = (N * P(i))^(-beta) / max(w_j)
```

This corrects for the bias introduced by non-uniform sampling.

**Parameters:**
- **alpha**: Controls prioritization strength
  - 0 = uniform sampling (no prioritization)
  - 1 = full prioritization
  - Default: 0.6 (balanced)

- **beta**: Controls importance sampling correction
  - Anneals from beta_start to 1.0
  - Starts low (more bias, faster learning)
  - Ends at 1.0 (unbiased, convergence guarantees)
  - Default: 0.4 → 1.0 over 100k frames

## Memory Efficiency

- States stored as numpy arrays with float32 dtype
- Automatic dtype conversion on push
- Actions stored as int64
- Rewards and dones as float32
- No unnecessary copying or conversions

## Integration with DQN

```python
from src.rl.replay_buffer import PrioritizedReplayBuffer
import torch
import torch.nn as nn

# Initialize buffer
replay_buffer = PrioritizedReplayBuffer(
    capacity=100000,
    alpha=0.6,
    beta_start=0.4,
    beta_end=1.0,
    beta_frames=1000000
)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    
    for step in range(max_steps):
        # Select action
        action = agent.select_action(state)
        
        # Take action in environment
        next_state, reward, done, _ = env.step(action)
        
        # Store transition
        replay_buffer.push(state, action, reward, next_state, done)
        
        # Train if buffer is ready
        if replay_buffer.is_ready(batch_size):
            # Sample batch
            states, actions, rewards, next_states, dones, weights, indices = \
                replay_buffer.sample(batch_size)
            
            # Convert to PyTorch tensors
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)
            weights = torch.FloatTensor(weights)
            
            # Compute Q values
            current_q = q_network(states).gather(1, actions.unsqueeze(1))
            next_q = target_network(next_states).max(1)[0].detach()
            target_q = rewards + (1 - dones) * gamma * next_q
            
            # Compute TD errors for priority update
            td_errors = torch.abs(target_q - current_q.squeeze())
            
            # Weighted loss (importance sampling)
            loss = (weights * F.mse_loss(current_q.squeeze(), target_q, reduction='none')).mean()
            
            # Update network
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update priorities in buffer
            new_priorities = td_errors.detach().cpu().numpy()
            replay_buffer.update_priorities(indices, new_priorities)
        
        state = next_state
        if done:
            break
```

## Validation Results

All tests passed successfully! ✓

### Test Coverage:
1. ✓ Basic buffer operations (push, sample, len)
2. ✓ FIFO behavior when buffer is full
3. ✓ Save/load functionality
4. ✓ Prioritized sampling with weights and indices
5. ✓ Priority updates and beta annealing
6. ✓ Edge cases (invalid capacity, over-sampling)
7. ✓ Memory efficiency (dtype conversions)
8. ✓ Prioritized sampling behavior (high-priority experiences sampled more)

### Output Example:
```
Buffer size: 100
Ready for batch of 32: True
Batch shapes: states=(32, 60), actions=(32,)

Weights shape: (32,)
Indices: [1024, 1027, 1032, 1032, 1035]

✓ ALL TESTS PASSED
```

## Performance Characteristics

### ReplayBuffer:
- **Insertion:** O(1)
- **Sampling:** O(batch_size)
- **Space:** O(capacity × state_size)

### PrioritizedReplayBuffer:
- **Insertion:** O(log capacity)
- **Sampling:** O(batch_size × log capacity)
- **Priority Update:** O(batch_size × log capacity)
- **Space:** O(capacity × state_size)

## Best Practices

1. **Buffer Size:** Use 10k-1M depending on problem complexity
   - Simple problems: 10k-50k
   - Complex problems: 100k-1M

2. **Batch Size:** 32-128 is typical
   - Larger batches = more stable gradients
   - Smaller batches = faster iteration

3. **Alpha (prioritization):** 
   - Start with 0.6
   - Increase for harder problems (0.7-0.8)
   - Decrease for simpler problems (0.4-0.5)

4. **Beta Annealing:**
   - Start low (0.4) to learn quickly
   - End at 1.0 for unbiased convergence
   - Anneal over 50-100% of training

5. **Warm-up Period:**
   - Fill buffer with random experiences before training
   - Typical: 1000-10000 experiences

6. **Checkpointing:**
   - Save buffer periodically during long training runs
   - Allows resuming training without losing experience

## Exports

The following are exported from `src.rl`:
```python
from src.rl import (
    Experience,
    ReplayBuffer,
    PrioritizedReplayBuffer,
    SumTree
)
```

## References

1. Mnih et al. (2015) - Human-level control through deep reinforcement learning
2. Schaul et al. (2016) - Prioritized Experience Replay
3. van Hasselt et al. (2016) - Deep Reinforcement Learning with Double Q-learning

## Next Steps

The replay buffers are ready for integration with:
1. DQN agent implementation
2. Neural network architectures (Q-network, target network)
3. Training loop and optimization
4. Evaluation and monitoring

The implementation is production-ready with comprehensive error handling, type safety, and performance optimization.


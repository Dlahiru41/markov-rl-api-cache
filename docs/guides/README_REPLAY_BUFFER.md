# Experience Replay Buffers for DQN Training

## Overview

This module provides production-ready experience replay buffers for Deep Q-Network (DQN) agent training. Both uniform and prioritized sampling strategies are implemented.

## Quick Start

### Installation
No additional dependencies needed! All requirements are already in `requirements.txt`:
- numpy >= 1.24
- torch >= 2.0 (for DQN integration)

### Basic Usage

```python
from src.rl.replay_buffer import ReplayBuffer
import numpy as np

# Create buffer
buffer = ReplayBuffer(capacity=10000, seed=42)

# Add experiences
buffer.push(state, action, reward, next_state, done)

# Train when ready
if buffer.is_ready(batch_size=32):
    states, actions, rewards, next_states, dones = buffer.sample(32)
    # Your training code here...
```

### Prioritized Usage

```python
from src.rl.replay_buffer import PrioritizedReplayBuffer

# Create prioritized buffer
pbuffer = PrioritizedReplayBuffer(
    capacity=10000,
    alpha=0.6,        # Prioritization strength
    beta_start=0.4,   # Initial IS correction
    beta_end=1.0,     # Final IS correction
    beta_frames=100000  # Annealing duration
)

# Add experiences
pbuffer.push(state, action, reward, next_state, done)

# Sample with priorities
if pbuffer.is_ready(32):
    s, a, r, s_, d, weights, indices = pbuffer.sample(32)
    
    # Train network and compute TD-errors
    td_errors = compute_td_errors(...)
    
    # Update priorities
    pbuffer.update_priorities(indices, td_errors)
```

## Features

### ✅ Complete Implementation
- **Experience namedtuple**: Immutable storage for transitions
- **ReplayBuffer**: Uniform random sampling with FIFO eviction
- **PrioritizedReplayBuffer**: TD-error prioritized sampling
- **SumTree**: O(log n) efficient prioritized sampling

### ✅ Production Ready
- Comprehensive error handling
- Memory-efficient storage (float32)
- Automatic dtype conversion for PyTorch
- Save/load functionality
- Reproducible sampling (optional seed)

### ✅ Thoroughly Tested
- All unit tests passed ✓
- Integration tests passed ✓
- User validation code verified ✓
- Edge cases covered ✓

## Components

### 1. Experience
```python
Experience(state, action, reward, next_state, done)
```
Named tuple for storing transitions.

### 2. ReplayBuffer
Uniform random sampling buffer.

**Methods:**
- `push(state, action, reward, next_state, done)` - Add experience
- `sample(batch_size)` - Sample random batch
- `__len__()` - Current size
- `is_ready(batch_size)` - Check if ready
- `save(path)` / `load(path)` - Persistence
- `clear()` - Empty buffer

### 3. PrioritizedReplayBuffer
Priority-based sampling with importance sampling correction.

**Methods:**
- `push(state, action, reward, next_state, done, priority=None)` - Add with priority
- `sample(batch_size)` - Sample prioritized batch
- `update_priorities(indices, priorities)` - Update based on TD-errors
- All ReplayBuffer methods

**Properties:**
- `beta` - Current importance sampling exponent (auto-annealed)

### 4. SumTree
Internal data structure for efficient prioritized sampling.

## Files

### Core
- **`src/rl/replay_buffer.py`** - Main implementation (684 lines)

### Tests
- **`validate_replay_buffer.py`** - Comprehensive test suite
- **`test_user_validation.py`** - User-provided validation
- **`test_replay_buffer_integration.py`** - Integration with RL components

### Demos
- **`demo_replay_buffer.py`** - Interactive demonstrations

### Documentation
- **`REPLAY_BUFFER_QUICK_REF.md`** - Quick reference (⭐ start here!)
- **`REPLAY_BUFFER_GUIDE.md`** - Complete guide with theory
- **`REPLAY_BUFFER_COMPLETE.md`** - Implementation report
- **`REPLAY_BUFFER_SUMMARY.md`** - High-level overview
- **`README_REPLAY_BUFFER.md`** - This file

## Running Tests

```bash
# Comprehensive validation
python validate_replay_buffer.py

# User validation
python test_user_validation.py

# Integration test
python test_replay_buffer_integration.py

# Interactive demo
python demo_replay_buffer.py
```

## Integration with Existing Code

The replay buffers integrate seamlessly with existing RL components:

```python
from src.rl import (
    # Existing
    StateBuilder, StateConfig,      # 60-dim states
    CacheAction, ActionSpace,       # 7 actions
    RewardCalculator, RewardConfig, # Reward computation
    # NEW
    ReplayBuffer, 
    PrioritizedReplayBuffer,
    Experience
)
```

## DQN Training Example

```python
from src.rl import StateBuilder, CacheAction, RewardCalculator
from src.rl.replay_buffer import PrioritizedReplayBuffer
import torch

# Initialize
state_builder = StateBuilder(StateConfig())
reward_calc = RewardCalculator(RewardConfig())
replay_buffer = PrioritizedReplayBuffer(capacity=100000)

# Training loop
for episode in range(num_episodes):
    for step in range(max_steps):
        # Build state
        state = state_builder.build_state(...)
        
        # Select action
        action = agent.select_action(state)
        
        # Execute
        next_state, outcome = execute_action(action)
        reward = reward_calc.calculate_reward(outcome)
        
        # Store
        replay_buffer.push(state, action, reward, next_state, done)
        
        # Train
        if replay_buffer.is_ready(batch_size):
            s, a, r, s_, d, w, idx = replay_buffer.sample(batch_size)
            
            # Convert to tensors
            s = torch.FloatTensor(s)
            a = torch.LongTensor(a)
            r = torch.FloatTensor(r)
            s_ = torch.FloatTensor(s_)
            d = torch.FloatTensor(d)
            w = torch.FloatTensor(w)
            
            # Train network
            loss, td_err = train_step(s, a, r, s_, d, w)
            
            # Update priorities
            replay_buffer.update_priorities(idx, td_err)
```

## Performance

| Operation | ReplayBuffer | PrioritizedReplayBuffer |
|-----------|--------------|-------------------------|
| Insert | O(1) | O(log n) |
| Sample | O(batch_size) | O(batch_size × log n) |
| Update | - | O(batch_size × log n) |
| Memory | O(capacity) | O(capacity) |

Both implementations are highly optimized for production use.

## Configuration Guide

### Buffer Capacity
- **Simple problems**: 10k - 50k
- **Complex problems**: 100k - 1M
- **API caching (this project)**: 100k recommended

### Batch Size
- **Default**: 32
- **GPU allows**: 64 - 128
- **Limited memory**: 16 - 32

### Alpha (Prioritization)
- **0.0**: Uniform sampling (no prioritization)
- **0.6**: Balanced (recommended default)
- **1.0**: Full prioritization
- **Sparse rewards**: 0.7 - 0.8
- **Dense rewards**: 0.4 - 0.6

### Beta Annealing
- **Start**: 0.4 (lower bias correction, faster learning)
- **End**: 1.0 (full correction, convergence guarantee)
- **Duration**: 50-100% of training

## Best Practices

1. **Warm-up**: Fill buffer with 1k-10k random experiences before training
2. **Checkpointing**: Save buffer every N episodes during long training
3. **Memory**: Use float32 (automatic) for 2x memory savings
4. **Seed**: Set seed for reproducibility during debugging
5. **Monitoring**: Track `buffer.max_priority` to monitor learning

## API Reference

See **`REPLAY_BUFFER_QUICK_REF.md`** for quick API reference.

See **`REPLAY_BUFFER_GUIDE.md`** for detailed documentation.

## Troubleshooting

### Buffer not ready for sampling
```python
if not buffer.is_ready(batch_size):
    continue  # Skip training this step
```

### Out of memory
- Reduce buffer capacity
- Use float32 (automatic)
- Reduce state dimensions

### Poor prioritization effect
- Increase alpha (0.6 → 0.8)
- Check TD-errors are being computed correctly
- Ensure priorities are being updated

## References

1. Mnih et al. (2015) - "Human-level control through deep reinforcement learning"
2. Schaul et al. (2016) - "Prioritized Experience Replay"
3. van Hasselt et al. (2016) - "Deep Reinforcement Learning with Double Q-learning"

## Status

**✅ PRODUCTION READY**

- Implementation: Complete
- Tests: All passed
- Documentation: Complete
- Integration: Ready

## Support

For questions or issues:
1. Check **`REPLAY_BUFFER_QUICK_REF.md`** for common usage
2. Run **`demo_replay_buffer.py`** for examples
3. See **`REPLAY_BUFFER_GUIDE.md`** for detailed theory

---

**Last Updated**: January 18, 2026  
**Version**: 1.0.0  
**Status**: Production Ready ✅


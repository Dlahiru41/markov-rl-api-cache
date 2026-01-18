# Replay Buffer Quick Reference

## Import
```python
from src.rl.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
```

## ReplayBuffer (Uniform Sampling)

### Create
```python
buffer = ReplayBuffer(capacity=100000, seed=42)
```

### Add Experience
```python
buffer.push(state, action, reward, next_state, done)
```

### Sample Batch
```python
states, actions, rewards, next_states, dones = buffer.sample(32)
# Returns numpy arrays ready for PyTorch
```

### Check Status
```python
len(buffer)              # Current size
buffer.is_ready(32)      # Has enough samples?
buffer.clear()           # Remove all
```

### Save/Load
```python
buffer.save("buffer.pkl")
buffer.load("buffer.pkl")
```

## PrioritizedReplayBuffer

### Create
```python
pbuffer = PrioritizedReplayBuffer(
    capacity=100000,
    alpha=0.6,          # Prioritization strength (0-1)
    beta_start=0.4,     # IS correction start
    beta_end=1.0,       # IS correction end
    beta_frames=100000  # Annealing duration
)
```

### Add Experience
```python
pbuffer.push(state, action, reward, next_state, done)
# Optional: pbuffer.push(..., priority=10.0)
```

### Sample Batch
```python
states, actions, rewards, next_states, dones, weights, indices = pbuffer.sample(32)
# weights: importance sampling weights for loss
# indices: for updating priorities after computing TD-error
```

### Update Priorities
```python
# After training step, compute TD errors
td_errors = np.abs(target_q - predicted_q)
pbuffer.update_priorities(indices, td_errors)
```

## Training Loop Pattern

```python
# 1. Initialize
buffer = PrioritizedReplayBuffer(capacity=100000)

# 2. Collect experience
buffer.push(state, action, reward, next_state, done)

# 3. Train when ready
if buffer.is_ready(batch_size):
    # Sample
    s, a, r, s_, d, w, idx = buffer.sample(batch_size)
    
    # Convert to tensors
    s = torch.FloatTensor(s)
    a = torch.LongTensor(a)
    r = torch.FloatTensor(r)
    s_ = torch.FloatTensor(s_)
    d = torch.FloatTensor(d)
    w = torch.FloatTensor(w)
    
    # Compute loss with importance sampling weights
    td_error = target_q - current_q
    loss = (w * td_error.pow(2)).mean()
    
    # Update priorities
    buffer.update_priorities(idx, td_error.abs().cpu().numpy())
```

## Data Types (Automatic)
- **states/next_states**: float32
- **actions**: int64
- **rewards**: float32
- **dones**: float32 (0.0 or 1.0)
- **weights**: float32

## Common Parameters

| Parameter | Typical Range | Description |
|-----------|---------------|-------------|
| capacity | 10k - 1M | Buffer size |
| batch_size | 32 - 128 | Training batch |
| alpha | 0.4 - 0.8 | Prioritization (0=uniform, 1=full) |
| beta_start | 0.4 | Initial IS correction |
| beta_end | 1.0 | Final IS correction |
| beta_frames | 50k - 1M | Annealing duration |

## Tips

✓ **Warm-up**: Fill buffer with 1k-10k samples before training  
✓ **Batch size**: Start with 32, increase if GPU allows  
✓ **Alpha**: Use 0.6 as default, tune if needed  
✓ **Beta**: Always end at 1.0 for convergence  
✓ **Checkpointing**: Save buffer every N episodes  

## Error Handling

```python
# Buffer too small
if not buffer.is_ready(batch_size):
    continue  # Skip training this step

# Invalid capacity
try:
    buffer = ReplayBuffer(capacity=0)
except ValueError:
    pass  # Capacity must be positive
```

## Memory Management

```python
# Clear buffer to free memory
buffer.clear()

# Check current size
print(f"Buffer: {len(buffer)}/{buffer.capacity}")

# Buffer uses float32 for efficiency
# States automatically converted on push()
```

## Complete Example

```python
from src.rl.replay_buffer import PrioritizedReplayBuffer
import numpy as np

# Setup
buffer = PrioritizedReplayBuffer(capacity=10000, alpha=0.6)
batch_size = 32

# Training loop
for episode in range(1000):
    state = env.reset()
    
    for step in range(max_steps):
        # Act
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        # Store
        buffer.push(state, action, reward, next_state, done)
        
        # Train
        if buffer.is_ready(batch_size):
            s, a, r, s_, d, w, idx = buffer.sample(batch_size)
            loss, td_err = agent.train_step(s, a, r, s_, d, w)
            buffer.update_priorities(idx, td_err)
        
        state = next_state
        if done:
            break
```

## Validation

Run: `python validate_replay_buffer.py`

Expected output:
```
✓ ALL TESTS PASSED
Replay buffers are ready for DQN training!
```


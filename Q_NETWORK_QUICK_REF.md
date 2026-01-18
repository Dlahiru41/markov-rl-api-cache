# Q-Network Quick Reference

## Import

```python
from src.rl.networks import (
    QNetwork, DuelingQNetwork, QNetworkConfig, create_network
)
import torch
```

## Quick Start

### Standard QNetwork

```python
# Create config
config = QNetworkConfig(
    state_dim=60,
    action_dim=7,
    hidden_dims=[256, 128, 64],
    activation='relu',
    dropout=0.1
)

# Create network
net = QNetwork(config)

# Forward pass
state = torch.randn(32, 60)  # Batch of 32
q_values = net(state)  # (32, 7)

# Get greedy actions
actions = net.get_action(state)  # (32,)
```

### Dueling QNetwork

```python
# Create dueling config
config = QNetworkConfig(
    state_dim=60,
    action_dim=7,
    hidden_dims=[256, 128, 64],
    dueling=True  # Enable dueling
)

# Create network
net = DuelingQNetwork(config)

# Forward pass
q_values = net(state)  # (32, 7)

# Get components
values = net.get_value(state)  # (32, 1)
advantages = net.get_advantage(state)  # (32, 7)

# Verify: Q = V + (A - mean(A))
assert torch.allclose(
    q_values,
    values + (advantages - advantages.mean(dim=-1, keepdim=True))
)
```

### Factory Function

```python
# Automatically create correct type
config = QNetworkConfig(state_dim=60, action_dim=7, dueling=True)
net = create_network(config)  # Returns DuelingQNetwork
```

## Configuration Options

```python
QNetworkConfig(
    state_dim=60,              # Required: input size
    action_dim=7,              # Required: output size
    hidden_dims=[256, 128, 64],  # Hidden layer sizes
    activation='relu',         # 'relu', 'leaky_relu', 'elu', 'tanh'
    dropout=0.1,               # 0.0-0.99, dropout rate
    use_layer_norm=False,      # Layer normalization
    dueling=False              # Dueling architecture
)
```

## Training Loop

```python
import torch.nn.functional as F

# Initialize
q_network = QNetwork(config)
target_network = QNetwork(config)
target_network.load_state_dict(q_network.state_dict())
optimizer = torch.optim.Adam(q_network.parameters(), lr=1e-4)

# Training step
def train_step(states, actions, rewards, next_states, dones):
    # Current Q-values
    current_q = q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
    
    # Target Q-values
    with torch.no_grad():
        next_q = target_network(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * gamma * next_q
    
    # Loss
    loss = F.mse_loss(current_q, target_q)
    
    # Update
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_network.parameters(), 10.0)
    optimizer.step()
    
    return loss.item()

# Update target network periodically
if step % target_update_freq == 0:
    target_network.load_state_dict(q_network.state_dict())
```

## Double DQN

```python
# Use online network for action selection
with torch.no_grad():
    next_actions = q_network(next_states).argmax(1)
    next_q = target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
    target_q = rewards + (1 - dones) * gamma * next_q
```

## Action Selection

### Greedy

```python
action = net.get_action(state)  # Single state: scalar
actions = net.get_action(states)  # Batch: (batch_size,)
```

### Epsilon-Greedy

```python
def select_action(state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(action_dim)  # Random
    else:
        state_t = torch.FloatTensor(state).unsqueeze(0)
        return net.get_action(state_t).item()  # Greedy
```

## Helper Functions

### Count Parameters

```python
from src.rl.networks import count_parameters

total = count_parameters(net, trainable_only=False)
trainable = count_parameters(net, trainable_only=True)
print(f"Parameters: {trainable:,} / {total:,}")
```

### Model Summary

```python
from src.rl.networks import get_model_summary

print(get_model_summary(net))
```

### Initialize Weights

```python
from src.rl.networks import initialize_weights

net.apply(lambda m: initialize_weights(m, method='xavier'))
# Methods: 'xavier', 'he', 'orthogonal'
```

## Common Configurations

### Small (Fast, Simple Tasks)

```python
config = QNetworkConfig(
    state_dim=60,
    action_dim=7,
    hidden_dims=[64, 32],
    activation='relu'
)
```

### Medium (Balanced)

```python
config = QNetworkConfig(
    state_dim=60,
    action_dim=7,
    hidden_dims=[128, 64],
    activation='relu',
    dropout=0.1
)
```

### Large (Complex Tasks)

```python
config = QNetworkConfig(
    state_dim=60,
    action_dim=7,
    hidden_dims=[256, 128, 64],
    activation='elu',
    dropout=0.2,
    use_layer_norm=True,
    dueling=True
)
```

## Integration with Replay Buffer

```python
from src.rl.replay_buffer import PrioritizedReplayBuffer

replay_buffer = PrioritizedReplayBuffer(capacity=100000)

# Training loop
if replay_buffer.is_ready(batch_size):
    s, a, r, s_, d, w, idx = replay_buffer.sample(batch_size)
    
    # Convert to tensors
    s = torch.FloatTensor(s)
    a = torch.LongTensor(a)
    r = torch.FloatTensor(r)
    s_ = torch.FloatTensor(s_)
    d = torch.FloatTensor(d)
    w = torch.FloatTensor(w)
    
    # Train
    current_q = q_network(s).gather(1, a.unsqueeze(1)).squeeze()
    with torch.no_grad():
        next_q = target_network(s_).max(1)[0]
        target_q = r + (1 - d) * gamma * next_q
    
    td_error = (target_q - current_q).abs()
    loss = (w * F.mse_loss(current_q, target_q, reduction='none')).mean()
    
    # Update network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Update priorities
    replay_buffer.update_priorities(idx, td_error.detach().cpu().numpy())
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Q-values exploding | Gradient clipping, lower LR, layer norm |
| Not learning | Check buffer size, increase LR |
| Overfitting | Add dropout (0.2), smaller network |
| Unstable training | Layer norm, gradient clipping |

## Model Sizes

| Config | Parameters | Speed | Use Case |
|--------|------------|-------|----------|
| [64, 32] | ~5K | Fast | Simple tasks |
| [128, 64] | ~15K | Medium | Baseline |
| [256, 128, 64] | ~50K | Slower | Complex tasks |
| [512, 256, 128] | ~200K | Slow | High capacity |

## Validation

```bash
# Run comprehensive tests
python validate_q_network.py

# Run user validation
python test_user_q_network.py
```

Expected output:
```
✓ ALL TESTS PASSED!
Q-Networks are ready for DQN training!
```

## Tips

✓ **Start simple**: Begin with small network, add complexity as needed  
✓ **Use dueling**: Often better performance for same parameters  
✓ **Gradient clipping**: Always use for stability  
✓ **Target network**: Update every 1000-10000 steps  
✓ **Learning rate**: Start with 1e-4, tune as needed  
✓ **Batch size**: 32-128 is typical  

## References

- Read: `Q_NETWORK_GUIDE.md` for full documentation
- See: `validate_q_network.py` for comprehensive examples


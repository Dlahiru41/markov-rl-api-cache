# Q-Network Implementation - Complete Guide

## Overview

This module provides neural network architectures for approximating Q-values in Deep Q-Learning. Two architectures are implemented:

1. **QNetwork**: Standard DQN architecture
2. **DuelingQNetwork**: Dueling DQN with separate value and advantage streams

Both support configurable hidden layers, activations, dropout, and layer normalization for flexible experimentation.

## File Location

`src/rl/networks/q_network.py`

## Core Components

### 1. QNetworkConfig

Configuration dataclass for network architecture:

```python
@dataclass
class QNetworkConfig:
    state_dim: int                              # Input dimension (e.g., 60)
    action_dim: int                             # Output dimension (e.g., 7)
    hidden_dims: List[int] = [256, 128, 64]    # Hidden layer sizes
    activation: str = 'relu'                    # Activation function
    dropout: float = 0.1                        # Dropout rate
    use_layer_norm: bool = False                # Layer normalization
    dueling: bool = False                       # Dueling architecture
```

**Supported activations**: `'relu'`, `'leaky_relu'`, `'elu'`, `'tanh'`

**Example:**
```python
config = QNetworkConfig(
    state_dim=60,
    action_dim=7,
    hidden_dims=[256, 128, 64],
    activation='relu',
    dropout=0.1,
    use_layer_norm=False,
    dueling=False
)
```

### 2. QNetwork

Standard Deep Q-Network architecture.

**Architecture:**
```
Input (state_dim)
  ↓
Hidden Layer 1 → [LayerNorm] → Activation → [Dropout]
  ↓
Hidden Layer 2 → [LayerNorm] → Activation → [Dropout]
  ↓
...
  ↓
Output (action_dim, no activation)
```

**Methods:**

#### `forward(state: torch.Tensor) -> torch.Tensor`
Forward pass through network.

- **Input**: `(batch_size, state_dim)` or `(state_dim,)`
- **Output**: `(batch_size, action_dim)` or `(action_dim,)`
- **Returns**: Q-values for all actions

#### `get_action(state: torch.Tensor) -> torch.Tensor`
Get greedy action (argmax of Q-values).

- **Input**: `(batch_size, state_dim)` or `(state_dim,)`
- **Output**: `(batch_size,)` or scalar
- **Returns**: Action indices

**Example:**
```python
import torch
from src.rl.networks import QNetwork, QNetworkConfig

# Create network
config = QNetworkConfig(state_dim=60, action_dim=7)
net = QNetwork(config)

# Forward pass
state = torch.randn(32, 60)  # Batch of 32 states
q_values = net(state)  # (32, 7)

# Get greedy actions
actions = net.get_action(state)  # (32,)
```

### 3. DuelingQNetwork

Dueling DQN architecture with separate value and advantage streams.

**Key Insight:**

Q-values can be decomposed as:
```
Q(s, a) = V(s) + A(s, a) - mean(A(s, :))
```

Where:
- `V(s)`: Value of being in state s
- `A(s, a)`: Advantage of taking action a over average

This decomposition helps the network learn state values without needing to see every action, accelerating learning.

**Architecture:**
```
Input (state_dim)
  ↓
Shared Feature Extraction
  ├─────────────────────────┐
  ↓                         ↓
Value Stream            Advantage Stream
  ↓                         ↓
V(s) (scalar)          A(s,a) (action_dim)
  └─────────────────────────┘
              ↓
    Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
```

**Methods:**

#### `forward(state: torch.Tensor) -> torch.Tensor`
Forward pass returning Q-values.

- **Input**: `(batch_size, state_dim)` or `(state_dim,)`
- **Output**: `(batch_size, action_dim)` or `(action_dim,)`
- **Returns**: Q-values (combined from value and advantage)

#### `get_value(state: torch.Tensor) -> torch.Tensor`
Get state value V(s).

- **Input**: `(batch_size, state_dim)` or `(state_dim,)`
- **Output**: `(batch_size, 1)` or `(1,)`
- **Returns**: State values

#### `get_advantage(state: torch.Tensor) -> torch.Tensor`
Get action advantages A(s, a).

- **Input**: `(batch_size, state_dim)` or `(state_dim,)`
- **Output**: `(batch_size, action_dim)` or `(action_dim,)`
- **Returns**: Action advantages

#### `get_action(state: torch.Tensor) -> torch.Tensor`
Get greedy action (argmax of Q-values).

**Example:**
```python
from src.rl.networks import DuelingQNetwork, QNetworkConfig

# Create dueling network
config = QNetworkConfig(state_dim=60, action_dim=7, dueling=True)
net = DuelingQNetwork(config)

# Forward pass
state = torch.randn(32, 60)
q_values = net(state)  # (32, 7)

# Get value and advantage separately
values = net.get_value(state)  # (32, 1)
advantages = net.get_advantage(state)  # (32, 7)

# Verify decomposition
reconstructed = values + (advantages - advantages.mean(dim=-1, keepdim=True))
assert torch.allclose(q_values, reconstructed)
```

### 4. Helper Functions

#### `create_network(config: QNetworkConfig) -> nn.Module`
Factory function to create appropriate network type.

```python
from src.rl.networks import create_network, QNetworkConfig

config = QNetworkConfig(state_dim=60, action_dim=7, dueling=True)
net = create_network(config)  # Returns DuelingQNetwork

config.dueling = False
net = create_network(config)  # Returns QNetwork
```

#### `initialize_weights(module: nn.Module, method: str = 'xavier') -> None`
Initialize network weights.

**Methods:**
- `'xavier'`: Xavier/Glorot initialization (good for tanh/sigmoid)
- `'he'`: He initialization (good for ReLU)
- `'orthogonal'`: Orthogonal initialization (good for RNNs)

```python
from src.rl.networks import initialize_weights

# Apply to model
model.apply(lambda m: initialize_weights(m, method='xavier'))
```

#### `count_parameters(model: nn.Module, trainable_only: bool = True) -> int`
Count model parameters.

```python
from src.rl.networks import count_parameters

total = count_parameters(model, trainable_only=False)
trainable = count_parameters(model, trainable_only=True)
print(f"Total: {total:,}, Trainable: {trainable:,}")
```

#### `get_model_summary(model: nn.Module) -> str`
Get detailed model summary.

```python
from src.rl.networks import get_model_summary

summary = get_model_summary(model)
print(summary)
```

## Configuration Options

### Hidden Layers

Control network capacity and expressiveness:

```python
# Small network (faster, less capacity)
hidden_dims=[64, 32]

# Medium network (balanced)
hidden_dims=[128, 64]

# Large network (slower, more capacity)
hidden_dims=[512, 256, 128]

# Very deep network
hidden_dims=[256, 256, 128, 128, 64, 64]
```

### Activations

Different activation functions for different behaviors:

```python
# ReLU - standard, fast
activation='relu'

# Leaky ReLU - prevents dying ReLU problem
activation='leaky_relu'

# ELU - smooth, can be negative
activation='elu'

# Tanh - bounded [-1, 1]
activation='tanh'
```

### Dropout

Regularization to prevent overfitting:

```python
# No dropout
dropout=0.0

# Light regularization
dropout=0.1

# Heavy regularization
dropout=0.3
```

### Layer Normalization

Stabilizes training, especially with deep networks:

```python
# Standard network
use_layer_norm=False

# Normalized network (more stable)
use_layer_norm=True
```

### Dueling Architecture

Better for environments where actions have similar values:

```python
# Standard DQN
dueling=False

# Dueling DQN (often better performance)
dueling=True
```

## Integration with DQN

### Basic Training Loop

```python
import torch
import torch.nn.functional as F
from src.rl.networks import QNetwork, QNetworkConfig
from src.rl.replay_buffer import PrioritizedReplayBuffer

# Initialize network and target network
config = QNetworkConfig(state_dim=60, action_dim=7)
q_network = QNetwork(config)
target_network = QNetwork(config)
target_network.load_state_dict(q_network.state_dict())

# Optimizer
optimizer = torch.optim.Adam(q_network.parameters(), lr=1e-4)

# Replay buffer
replay_buffer = PrioritizedReplayBuffer(capacity=100000)

# Training loop
for step in range(num_steps):
    # Sample batch
    if replay_buffer.is_ready(batch_size):
        s, a, r, s_, d, w, idx = replay_buffer.sample(batch_size)
        
        # Convert to tensors
        s = torch.FloatTensor(s)
        a = torch.LongTensor(a)
        r = torch.FloatTensor(r)
        s_ = torch.FloatTensor(s_)
        d = torch.FloatTensor(d)
        w = torch.FloatTensor(w)
        
        # Compute current Q-values
        current_q = q_network(s).gather(1, a.unsqueeze(1)).squeeze()
        
        # Compute target Q-values
        with torch.no_grad():
            next_q = target_network(s_).max(1)[0]
            target_q = r + (1 - d) * gamma * next_q
        
        # TD error
        td_error = (target_q - current_q).abs()
        
        # Weighted loss (importance sampling)
        loss = (w * F.mse_loss(current_q, target_q, reduction='none')).mean()
        
        # Update network
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10.0)
        optimizer.step()
        
        # Update priorities
        replay_buffer.update_priorities(idx, td_error.detach().cpu().numpy())
    
    # Update target network periodically
    if step % target_update_freq == 0:
        target_network.load_state_dict(q_network.state_dict())
```

### Double DQN

Reduces overestimation bias:

```python
# Use online network for action selection, target network for evaluation
with torch.no_grad():
    next_actions = q_network(s_).argmax(1)
    next_q = target_network(s_).gather(1, next_actions.unsqueeze(1)).squeeze()
    target_q = r + (1 - d) * gamma * next_q
```

### Epsilon-Greedy Exploration

```python
def select_action(state, epsilon):
    if np.random.random() < epsilon:
        # Random action
        return np.random.randint(action_dim)
    else:
        # Greedy action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            return q_network.get_action(state_tensor).item()
```

## Performance Considerations

### Model Size

| Hidden Dims | Parameters | Training Speed | Performance |
|-------------|------------|----------------|-------------|
| [64, 32] | ~5K | Fast | Good for simple tasks |
| [128, 64] | ~15K | Medium | Good baseline |
| [256, 128, 64] | ~50K | Slower | Good for complex tasks |
| [512, 256, 128] | ~200K | Slow | High capacity |

### Memory Usage

- **Standard DQN**: O(state_dim × hidden_dims + hidden_dims × action_dim)
- **Dueling DQN**: Slightly higher due to two streams
- **Batch processing**: Memory scales linearly with batch size

### Training Speed

Factors affecting speed:
1. **Network size**: Larger networks = slower forward/backward passes
2. **Batch size**: Larger batches = better GPU utilization
3. **Dropout**: Adds minimal overhead
4. **Layer norm**: Adds small overhead

## Best Practices

### 1. Start Simple

```python
# Begin with small network
config = QNetworkConfig(
    state_dim=60,
    action_dim=7,
    hidden_dims=[128, 64],
    activation='relu',
    dropout=0.0
)
```

### 2. Add Complexity Gradually

```python
# Add dropout if overfitting
config.dropout = 0.1

# Add layer norm for stability
config.use_layer_norm = True

# Try dueling for better performance
config.dueling = True
```

### 3. Tune Hyperparameters

- **Learning rate**: Start with 1e-4, tune between 1e-5 and 1e-3
- **Hidden dims**: Scale with problem complexity
- **Dropout**: Use 0.1-0.2 for regularization
- **Activation**: ReLU is good default, ELU for smoother gradients

### 4. Monitor Training

```python
# Log metrics
print(f"Loss: {loss.item():.4f}")
print(f"Avg Q-value: {current_q.mean().item():.4f}")
print(f"Max Q-value: {current_q.max().item():.4f}")

# Check gradient norms
total_norm = 0
for p in q_network.parameters():
    if p.grad is not None:
        total_norm += p.grad.norm().item() ** 2
total_norm = total_norm ** 0.5
print(f"Gradient norm: {total_norm:.4f}")
```

## Troubleshooting

### Q-values exploding

**Solutions:**
- Use gradient clipping: `torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)`
- Reduce learning rate
- Add layer normalization
- Use smaller network

### Q-values not learning

**Solutions:**
- Check replay buffer has enough samples
- Increase learning rate
- Verify target network updates
- Check reward scaling

### Overfitting

**Solutions:**
- Add dropout: `dropout=0.2`
- Use smaller network
- Increase replay buffer size
- Add L2 regularization

### Unstable training

**Solutions:**
- Add layer normalization: `use_layer_norm=True`
- Use gradient clipping
- Reduce learning rate
- Update target network more frequently

## Examples

### Minimal Example

```python
from src.rl.networks import QNetwork, QNetworkConfig
import torch

config = QNetworkConfig(state_dim=60, action_dim=7)
net = QNetwork(config)

state = torch.randn(32, 60)
q_values = net(state)
actions = net.get_action(state)
```

### With Custom Configuration

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

net = create_network(config)
print(get_model_summary(net))
```

### Training Step

```python
# Forward
q_values = q_network(states)
current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze()

# Target
with torch.no_grad():
    next_q = target_network(next_states).max(1)[0]
    target_q = rewards + (1 - dones) * gamma * next_q

# Loss and update
loss = F.mse_loss(current_q, target_q)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## References

1. Mnih et al. (2015) - "Human-level control through deep reinforcement learning"
2. Wang et al. (2016) - "Dueling Network Architectures for Deep Reinforcement Learning"
3. van Hasselt et al. (2016) - "Deep Reinforcement Learning with Double Q-learning"

## Status

**✅ PRODUCTION READY**

- All features implemented
- Comprehensive testing
- Full documentation
- Integration ready

---

**Last Updated**: January 18, 2026  
**Version**: 1.0.0  
**Status**: Complete ✅


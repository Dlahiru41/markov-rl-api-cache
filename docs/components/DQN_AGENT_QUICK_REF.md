# DQN Agent - Quick Reference

## ✅ STATUS: FULLY IMPLEMENTED & TESTED

## Import
```python
from src.rl.agents.dqn_agent import DQNAgent, DoubleDQNAgent, DQNConfig
```

## Basic Usage
```python
# 1. Configure
config = DQNConfig(
    state_dim=60,
    action_dim=7,
    hidden_dims=[128, 64],
    epsilon_start=1.0,
    epsilon_end=0.1,
    seed=42
)

# 2. Initialize
agent = DQNAgent(config, seed=42)

# 3. Training loop
for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        metrics = agent.train_step()
        state = next_state

# 4. Save
agent.save("trained_agent.pt")

# 5. Load and evaluate
agent.load("trained_agent.pt")
action = agent.select_action(state, evaluate=True)  # Greedy
```

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `select_action(state, evaluate=False)` | Choose action (ε-greedy or greedy) | int (action index) |
| `store_transition(s, a, r, s', done)` | Add experience to buffer | None |
| `train_step()` | Single gradient update | dict or None |
| `save(path)` | Save agent checkpoint | None |
| `load(path)` | Load agent checkpoint | None |
| `get_metrics()` | Get current stats | dict |

## Configuration Cheat Sheet

### Essential
```python
state_dim: int          # Input size (from state representation)
action_dim: int         # Number of actions (7 for caching)
hidden_dims: [128, 64]  # Network architecture
```

### Optimization
```python
learning_rate: 0.001    # Adam LR (0.0003 for stability)
gamma: 0.99            # Discount factor (0.9-0.999)
```

### Exploration
```python
epsilon_start: 1.0     # Initial exploration (100%)
epsilon_end: 0.05      # Final exploration (5%)
epsilon_decay: 0.995   # Decay rate per step
```

### Buffer
```python
buffer_size: 100000           # Max experiences
batch_size: 64                # Training batch
prioritized_replay: False     # Use PER (True/False)
```

### Stability
```python
target_update_freq: 1000  # Steps between target updates
max_grad_norm: 10.0       # Gradient clipping threshold
```

## Training Metrics

```python
metrics = agent.train_step()
if metrics:
    loss = metrics['loss']        # TD loss
    q_mean = metrics['q_mean']    # Average Q-value
    epsilon = metrics['epsilon']  # Current exploration rate
```

## DoubleDQN

```python
agent = DoubleDQNAgent(config, seed=42)
# Same API as DQNAgent, reduces overestimation bias
```

## Common Patterns

### Evaluation Mode
```python
# Training: explore
action = agent.select_action(state)

# Testing: greedy
action = agent.select_action(state, evaluate=True)
```

### Checkpointing
```python
# Save every 100 episodes
if episode % 100 == 0:
    agent.save(f"checkpoint_ep{episode}.pt")

# Resume training
agent.load("checkpoint_ep900.pt")
```

### Monitoring
```python
metrics = agent.get_metrics()
print(f"Steps: {metrics['steps']}")
print(f"Epsilon: {metrics['epsilon']:.3f}")
print(f"Buffer: {metrics['buffer_size']}")
```

## Testing

```bash
# Comprehensive test (all features)
python demo_dqn_agent.py

# Quick test (basic functionality)
python quick_test_dqn.py

# User validation (exact requirements)
python test_user_validation.py
```

## Hyperparameter Quick Fixes

| Problem | Solution |
|---------|----------|
| Loss oscillating | Lower LR to 0.0003 |
| Not learning | Slower epsilon decay (0.999) |
| Q-values exploding | Lower max_grad_norm to 5.0 |
| Too slow | Larger batch_size (128) |
| Overestimating | Use DoubleDQNAgent |

## File Locations

```
src/rl/agents/
├── __init__.py       # Exports
└── dqn_agent.py      # Implementation (267 lines)

Tests:
├── demo_dqn_agent.py         # Full validation
├── quick_test_dqn.py         # Quick test
└── test_user_validation.py   # User's code

Docs:
└── DQN_AGENT_COMPLETE.md     # Full guide
```

## Example Output

```
Testing DQN Agent import and basic functionality...
------------------------------------------------------------
✓ Imports successful
✓ Config created
✓ Agent created (device: cpu)
✓ Selected action: 3 (epsilon=1.00)

Collecting experiences...
✓ Stored 200 experiences

Training agent...
  Step 0: loss=2.3792, q_mean=0.84, eps=0.995
  Step 1: loss=2.0999, q_mean=0.98, eps=0.990
  Step 2: loss=1.7251, q_mean=1.03, eps=0.985
  ...
  Step 9: loss=0.8633, q_mean=1.22, eps=0.951

Testing save/load...
✓ Agent saved
✓ Agent loaded (epsilon: 0.951)

============================================================
ALL TESTS PASSED! ✓
============================================================
```

## Integration with Caching

```python
from src.rl.agents.dqn_agent import DQNAgent, DQNConfig
from src.rl.state import StateRepresentation
from src.rl.reward import RewardFunction
from src.rl.actions import CacheAction

# Match dimensions
config = DQNConfig(
    state_dim=60,  # StateRepresentation.state_dim
    action_dim=7,  # CacheAction.num_actions()
    seed=42
)

agent = DQNAgent(config, seed=42)
state_builder = StateRepresentation(...)
reward_fn = RewardFunction(...)

# Training
state_vector = state_builder.build_state(...)
action_idx = agent.select_action(state_vector)
cache_action = CacheAction(action_idx)
# ... execute action, get reward ...
agent.store_transition(state_vector, action_idx, reward, next_state, done)
agent.train_step()
```

## Key Features ✓

- [x] Epsilon-greedy exploration
- [x] Experience replay (uniform + prioritized)
- [x] Target network for stability
- [x] Gradient clipping
- [x] DoubleDQN variant
- [x] GPU support (auto-detect)
- [x] Save/load checkpoints
- [x] Comprehensive metrics
- [x] Deterministic evaluation mode
- [x] Reproducible (seed support)

## Ready to Use!

The DQN agent is fully functional and tested. Start training with your caching environment!


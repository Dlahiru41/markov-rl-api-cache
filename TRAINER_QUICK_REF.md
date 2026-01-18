# Training Orchestration - Quick Reference

## ✅ FULLY IMPLEMENTED

## Quick Start

### Python API
```python
from src.rl.agents.dqn_agent import DQNAgent, DQNConfig
from src.rl.training.trainer import Trainer, TrainingConfig

# Setup
agent = DQNAgent(DQNConfig(state_dim=60, action_dim=7, seed=42))
config = TrainingConfig(max_episodes=10000, eval_frequency=100)
trainer = Trainer(agent, env, config, output_dir="results/run1")

# Train
stats = trainer.train()

# Evaluate
eval_metrics = trainer.evaluate_only("results/run1/checkpoints/best.pt")
```

### CLI
```bash
# Train
python scripts/train.py --episodes 1000 --output results/run1

# Resume
python scripts/train.py --resume results/run1/checkpoints/latest.pt --episodes 2000

# Evaluate
python scripts/train.py --eval-only --checkpoint results/run1/checkpoints/best.pt
```

## TrainingConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_episodes` | 50000 | Maximum training episodes |
| `max_steps_per_episode` | 1000 | Truncate episodes after this |
| `eval_frequency` | 100 | Evaluate every N episodes |
| `eval_episodes` | 10 | Average over N eval episodes |
| `checkpoint_frequency` | 500 | Save every N episodes |
| `early_stopping` | True | Stop if no improvement |
| `patience` | 50 | Episodes without improvement |
| `min_episodes` | 1000 | Minimum before early stop |
| `log_frequency` | 10 | Log every N episodes |
| `use_wandb` | False | Weights & Biases logging |

## Trainer Methods

```python
trainer = Trainer(agent, env, config, output_dir)

# Main training
stats = trainer.train()

# Evaluation only
metrics = trainer.evaluate_only(checkpoint_path, num_episodes=20)

# Resume training
episode = trainer.load_checkpoint(checkpoint_path)

# Plot curves
trainer.plot_training_curves("curves.png")

# Get metrics
metrics = trainer.get_metrics()
```

## CLI Options

```bash
--config, -c PATH          # Config file
--episodes, -e INT         # Max episodes
--resume, -r PATH          # Resume checkpoint
--eval-only                # Evaluation mode
--checkpoint PATH          # Model to evaluate
--output, -o PATH          # Output directory
--seed INT                 # Random seed
--wandb                    # Enable W&B
--agent-type CHOICE        # dqn or double_dqn
--eval-episodes INT        # Eval episode count
```

## Output Structure

```
results/run1/
├── checkpoints/
│   ├── best.pt            # Best model
│   ├── latest.pt          # Latest checkpoint
│   └── checkpoint_ep*.pt  # Regular saves
├── logs/
│   └── training.log       # Detailed log
├── training_curves.png    # Plots
└── metrics.json           # All metrics
```

## Common Workflows

### Basic Training
```bash
python scripts/train.py --episodes 10000 --output results/experiment1
```

### With Config
```bash
python scripts/train.py --config configs/default.yaml --episodes 5000
```

### Resume
```bash
python scripts/train.py --resume results/experiment1/checkpoints/latest.pt --episodes 15000
```

### Evaluate
```bash
python scripts/train.py --eval-only --checkpoint results/experiment1/checkpoints/best.pt
```

### With W&B
```bash
python scripts/train.py --wandb --episodes 10000 --output results/wandb_run
```

## Features

✅ **Training loop** - Automatic episode execution and learning  
✅ **Evaluation** - Periodic testing with greedy policy  
✅ **Checkpointing** - Best, latest, and regular saves  
✅ **Early stopping** - Stop when not improving  
✅ **Logging** - Console + file + W&B  
✅ **Visualization** - Training curves  
✅ **Resumable** - Continue from any checkpoint  
✅ **Robust** - Error handling + emergency saves  

## Metrics Tracked

- Training rewards (per episode)
- Evaluation rewards (mean over episodes)
- Training losses (TD error)
- Epsilon decay (exploration rate)
- Q-values (mean estimates)
- Episode lengths
- Training time

## Example Output

```
Episode 100/10000 | Reward: 125.34+/-15.23 | Length: 98 | eps: 0.905 | Loss: 2.4531 | Time: 125s | ETA: 11375s

--------------------------------------------------------------------------------
EVALUATION (Episode 100)
  Mean Reward: 145.67 +/- 12.34
  Min/Max: 118.23 / 171.89
  Mean Length: 95
  Best So Far: 145.67
--------------------------------------------------------------------------------
```

## Configuration Tips

### Fast Training
```python
TrainingConfig(
    max_episodes=5000,
    eval_frequency=200,        # Evaluate less often
    checkpoint_frequency=1000, # Save less often
    early_stopping=False       # Train to completion
)
```

### Careful Training
```python
TrainingConfig(
    max_episodes=50000,
    eval_frequency=50,         # Evaluate more often
    checkpoint_frequency=250,  # Save more often
    early_stopping=True,
    patience=100              # Wait longer
)
```

### Hyperparameter Search
```python
TrainingConfig(
    max_episodes=10000,
    eval_frequency=100,
    early_stopping=True,
    patience=50,              # Stop bad runs early
    min_episodes=1000         # But give a fair chance
)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Not improving | Lower learning rate, check environment |
| Too slow | Reduce eval_frequency, increase batch_size |
| Out of memory | Reduce buffer_size, smaller network |
| Loss NaN | Lower learning rate, check rewards |
| Crashes | Check logs/, emergency checkpoint saved |

## Testing

```bash
# Quick test
python quick_test_trainer.py

# Full validation
python demo_trainer.py
```

## Files

```
src/rl/training/
├── trainer.py              # Main implementation
└── __init__.py             # Exports

scripts/
└── train.py                # CLI interface

tests/
├── quick_test_trainer.py   # Quick validation
└── demo_trainer.py         # Full tests

docs/
├── TRAINER_COMPLETE.md     # Complete guide
└── TRAINER_QUICK_REF.md    # This file
```

## Python API Example

```python
from src.rl.agents.dqn_agent import DQNAgent, DQNConfig
from src.rl.training.trainer import Trainer, TrainingConfig

# Agent
agent_config = DQNConfig(
    state_dim=60,
    action_dim=7,
    hidden_dims=[128, 64],
    learning_rate=0.001,
    seed=42
)
agent = DQNAgent(agent_config)

# Training
training_config = TrainingConfig(
    max_episodes=10000,
    eval_frequency=100,
    checkpoint_frequency=500,
    early_stopping=True,
    patience=50,
    use_wandb=False
)

# Train
trainer = Trainer(agent, env, training_config, output_dir="results/my_run")
stats = trainer.train()

print(f"Best reward: {stats['best_eval_reward']:.2f}")
print(f"Total episodes: {stats['total_episodes']}")
```

## CLI Example

```bash
# Simple
python scripts/train.py --episodes 1000

# Full control
python scripts/train.py \
    --config configs/default.yaml \
    --episodes 10000 \
    --output results/experiment1 \
    --seed 42 \
    --wandb

# Resume
python scripts/train.py \
    --resume results/experiment1/checkpoints/latest.pt \
    --episodes 20000

# Evaluate
python scripts/train.py \
    --eval-only \
    --checkpoint results/experiment1/checkpoints/best.pt \
    --eval-episodes 50
```

## Status

**✅ PRODUCTION READY**

- Complete implementation
- Fully tested
- Comprehensive documentation
- CLI + Python API
- Ready for use

---

**Quick Links**:
- Full Guide: `TRAINER_COMPLETE.md`
- Source: `src/rl/training/trainer.py`
- CLI: `scripts/train.py`
- Tests: `quick_test_trainer.py`


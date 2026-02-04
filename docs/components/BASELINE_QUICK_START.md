# Baseline Policies - Quick Reference Guide

## 🚀 Quick Start (5 minutes)

### 1. Test Individual Policy
```python
from baselines import LRUPolicy
import numpy as np

# Create policy
policy = LRUPolicy()

# Create dummy state and predictions
state = np.random.rand(60)
predictions = [('/api/products', 0.8), ('/api/cart', 0.6)]

# Select action
action = policy.select_action(state, predictions)
print(f"Action: {action}")  # Returns: 0-6 (CacheAction enum)
```

### 2. Compare Multiple Policies
```python
from baselines import BaselineComparator, LRUPolicy, LFUPolicy, RandomPolicy
from src.integration.gym_environment import CachingEnv, CacheEnvConfig

# Create environment
env = CachingEnv(CacheEnvConfig())

# Create comparator
comparator = BaselineComparator()
comparator.add_policy('LRU', LRUPolicy())
comparator.add_policy('LFU', LFUPolicy())
comparator.add_policy('Random', RandomPolicy())

# Run comparison
results = comparator.run_comparison(env, num_episodes=50)
print(results)
```

### 3. CLI Comparison
```bash
python scripts/compare_baselines.py --baselines lru,lfu,random --episodes 50
```

## 📋 Available Baselines

| Baseline | Class | Best For |
|----------|-------|----------|
| LRU | `LRUPolicy()` | Industry standard, temporal locality |
| LFU | `LFUPolicy()` | Stable workloads, hot items |
| Static Markov | `StaticMarkovPolicy()` | Fixed rules with predictions |
| Random | `RandomPolicy()` | Lower bound sanity check |
| Adaptive | `AdaptivePolicy()` | Dynamic workloads |
| Oracle | `OraclePolicy()` | Upper bound (cheating) |

## 🎯 Common Use Cases

### Compare Against RL Agent
```python
from baselines import BaselineComparator, LRUPolicy
from stable_baselines3 import DQN

# Load trained RL agent
agent = DQN.load('results/best_model.zip')

# Compare
comparator = BaselineComparator()
comparator.add_policy('LRU', LRUPolicy())
comparator.add_trained_agent('DQN', agent)
results = comparator.run_comparison(env, num_episodes=100)
```

### Generate Thesis Plots
```python
# Run comparison
results = comparator.run_comparison(env, num_episodes=200)

# Generate plots
comparator.plot_comparison('reward', 'thesis/figures/reward_comparison.png')
comparator.plot_comparison('cache_hit_rate', 'thesis/figures/hitrate.png')

# Generate report
comparator.generate_report(results, 'thesis/evaluation/baselines')
```

### Test Custom Policy
```python
from baselines import CachingPolicy

class MyPolicy(CachingPolicy):
    def select_action(self, state, predictions):
        return 1  # Always CACHE_CURRENT
    
    def get_name(self):
        return "My Custom Policy"
    
    def reset(self):
        pass
    
    def get_statistics(self):
        return {}

# Use it
comparator.add_policy('Custom', MyPolicy())
```

## 📊 Metrics Explained

| Metric | Range | Meaning |
|--------|-------|---------|
| `mean_reward` | -100 to 100 | Average episode reward |
| `cache_hit_rate` | 0 to 1 | % of requests served from cache |
| `cascade_rate` | 0 to 1 | % of episodes with cascades |
| `prefetch_efficiency` | 0 to 1 | Useful prefetches / total |

## 🔧 Configuration

### Environment Config
```python
from src.integration.gym_environment import CacheEnvConfig, SimulatorConfig

config = CacheEnvConfig(
    simulator_config=SimulatorConfig(
        num_apis=20,
        session_length_range=(10, 100)
    ),
    max_steps_per_episode=500,
    seed=42
)
env = CachingEnv(config)
```

### Policy Config
```python
# LRU with custom threshold
lru = LRUPolicy(eviction_threshold=0.85)

# Static Markov with custom thresholds
markov = StaticMarkovPolicy(
    conservative_threshold=0.7,
    moderate_threshold=0.5
)

# Adaptive with custom window
adaptive = AdaptivePolicy(window_size=100, aggression_step=0.05)
```

## 🐛 Troubleshooting

### Import Error
```python
# Add project to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```

### Environment Not Found
```bash
# Make sure you're in project root
cd /path/to/markov-rl-api-cache
python scripts/compare_baselines.py
```

### Missing Dependencies
```bash
pip install numpy pandas matplotlib seaborn scipy pyyaml
```

## 📈 Expected Results

Typical baseline performance (approximate):

| Baseline | Mean Reward | Hit Rate | Cascade Rate |
|----------|-------------|----------|--------------|
| Random | -50 to 0 | 30-40% | 20-30% |
| LRU | 50 to 100 | 60-70% | 5-10% |
| LFU | 60 to 110 | 65-75% | 5-10% |
| Static Markov | 80 to 130 | 70-80% | 3-8% |
| Adaptive | 90 to 140 | 75-85% | 2-5% |
| **RL Agent** | **100 to 160** | **80-90%** | **1-3%** |
| Oracle | 150 to 200 | 90-95% | 0-1% |

## 💡 Tips

1. **Start Small**: Test with 10 episodes first, then scale up
2. **Use Seed**: Set `seed=42` for reproducible results
3. **Compare Incrementally**: Add baselines one at a time
4. **Save Results**: Use `--save-json` to keep detailed data
5. **Check Significance**: Look for p < 0.05 in statistical tests

## 🎓 For Thesis

### Essential Comparisons
1. **RL vs Random**: Shows learning (expect >100% improvement)
2. **RL vs LRU**: Shows beating standard (expect 20-40% improvement)
3. **RL vs Static Markov**: Shows value of learning (expect 10-30% improvement)
4. **RL vs Oracle**: Shows improvement potential (expect 30-50% of gap)

### Key Plots for Thesis
```bash
python scripts/compare_baselines.py \
  --baselines lru,lfu,static_markov,random \
  --agent results/best_agent.zip \
  --episodes 200 \
  --output thesis/evaluation/final
```

This generates:
- `reward_comparison.png` - Box plot of rewards
- `hitrate_comparison.png` - Cache hit rates
- `cascade_comparison.png` - Cascade prevention
- `baseline_comparison_report.md` - Full report

## 📞 Need Help?

- Documentation: `baselines/README.md`
- Examples: `demo_baselines.py`
- Validation: `validate_baselines.py`
- Full implementation: `BASELINE_IMPLEMENTATION_COMPLETE.md`

## ⚡ One-Liners

```bash
# Quick test
python -c "from baselines import LRUPolicy; print(LRUPolicy().get_name())"

# Run all demos
python demo_baselines.py

# Validate everything
python validate_baselines.py

# Full comparison (5 min)
python scripts/compare_baselines.py --baselines all --episodes 50

# Thesis evaluation (30 min)
python scripts/compare_baselines.py \
  --baselines lru,lfu,static_markov,adaptive,random \
  --agent results/best.zip \
  --episodes 200 \
  --output thesis/final_evaluation
```

---

**That's it!** You're ready to compare baselines and demonstrate the value of your RL approach. 🎉


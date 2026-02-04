# Baseline Caching Policies

Comprehensive baseline implementations for comparing against RL-based caching strategies.

## Overview

This module provides multiple baseline caching policies representing different approaches to intelligent caching. These baselines are essential for demonstrating the value of the RL approach in thesis evaluation.

## Available Baselines

### 1. **LRU (Least Recently Used)**
Classic baseline - the industry standard.

```python
from baselines import LRUPolicy

policy = LRUPolicy(eviction_threshold=0.9)
action = policy.select_action(state, predictions)
```

**Strategy:**
- Always cache current response
- Evict least recently used items when cache > 90% full
- Never proactively prefetch

**Variants:**
- `LRUPolicy`: Standard LRU
- `AdaptiveLRUPolicy`: Adjusts eviction threshold based on hit rate

---

### 2. **LFU (Least Frequently Used)**
Frequency-based eviction - often better for stable workloads.

```python
from baselines import LFUPolicy

policy = LFUPolicy(eviction_threshold=0.85, decay_rate=0.01)
action = policy.select_action(state, predictions)
```

**Strategy:**
- Track access frequency per endpoint
- Evict lowest frequency items when full
- Apply time-based decay to prevent stale items

**Variants:**
- `LFUPolicy`: Standard LFU with decay
- `WindowedLFUPolicy`: Only consider recent window

---

### 3. **Static Markov Policy**
Uses Markov predictions with hand-crafted rules.

```python
from baselines import StaticMarkovPolicy

policy = StaticMarkovPolicy(
    conservative_threshold=0.7,
    moderate_threshold=0.5,
    aggressive_threshold=0.3
)
action = policy.select_action(state, predictions)
```

**Strategy:**
- High confidence (>0.7): Prefetch top-1 (conservative)
- Medium confidence (>0.5): Prefetch top-3 (moderate)
- Low confidence (>0.3): Prefetch top-5 (aggressive)
- Very low: Just cache current

**Variants:**
- `StaticMarkovPolicy`: Standard confidence-based
- `InverseStaticMarkovPolicy`: Prefetch MORE when uncertain
- `BalancedStaticMarkovPolicy`: Consider both confidence and cache state

---

### 4. **Random Policy**
Lower bound baseline - any intelligent policy should beat this.

```python
from baselines import RandomPolicy

# Uniform random
policy = RandomPolicy()

# Weighted random (favor certain actions)
policy = RandomPolicy(action_weights=[0.3, 0.4, 0.1, 0.1, 0.05, 0.03, 0.02])
```

**Strategy:**
- Select actions uniformly at random
- Optional: weighted random for non-uniform distribution

**Variants:**
- `RandomPolicy`: Pure random
- `EpsilonRandomPolicy`: ε-greedy with base policy
- `BiasedRandomPolicy`: Exclude certain actions

---

### 5. **Adaptive Heuristic**
Sophisticated hand-crafted policy that adapts to performance.

```python
from baselines import AdaptivePolicy

policy = AdaptivePolicy(
    window_size=100,
    aggression_step=0.05,
    min_hit_rate=0.5,
    max_hit_rate=0.8
)
action = policy.select_action(state, predictions)
```

**Strategy:**
- Track recent performance (hit rate, CPU, latency)
- Adapt thresholds based on metrics
- Become aggressive if hit rate dropping
- Become conservative if hit rate high and stable

**Variants:**
- `AdaptivePolicy`: Single-objective (hit rate focused)
- `MultiObjectiveAdaptivePolicy`: Balance multiple objectives

---

### 6. **Oracle Policy**
Perfect knowledge upper bound (cheating baseline).

```python
from baselines import OraclePolicy

policy = OraclePolicy(lookahead_window=10)

# In evaluation loop:
policy.set_future_calls(['api1', 'api2', 'api3'])  # Cheat!
action = policy.select_action(state, predictions)
```

**Strategy:**
- Has access to future API calls
- Perfectly prefetches items that will be accessed
- Never wastes cache space
- Represents theoretical best performance

**Variants:**
- `OraclePolicy`: Perfect knowledge
- `PartialOraclePolicy`: Probabilistic knowledge
- `NoisyOraclePolicy`: Correct but noisy predictions

---

## Comparison Framework

The `BaselineComparator` provides comprehensive comparison tools.

### Basic Usage

```python
from baselines import BaselineComparator, LRUPolicy, LFUPolicy, RandomPolicy
from src.integration.gym_environment import CachingEnv, CacheEnvConfig

# Create environment
config = CacheEnvConfig(max_steps_per_episode=200)
env = CachingEnv(config)

# Create comparator
comparator = BaselineComparator()

# Register policies
comparator.add_policy('LRU', LRUPolicy())
comparator.add_policy('LFU', LFUPolicy())
comparator.add_policy('Random', RandomPolicy())

# Add trained RL agent
# comparator.add_trained_agent('DQN', trained_agent)

# Run comparison
results = comparator.run_comparison(env, num_episodes=100)
print(results)

# Generate report
report_path = comparator.generate_report(
    results, 
    output_path='results/comparison',
    include_plots=True
)

# Create specific plots
comparator.plot_comparison(metric='reward', output_path='results/reward.png')
comparator.plot_comparison(metric='cache_hit_rate', output_path='results/hitrate.png')
```

### Metrics Tracked

- **Episode Reward**: Total reward per episode
- **Cache Hit Rate**: Percentage of cache hits
- **Prediction Accuracy**: How often predictions were correct
- **Cascade Rate**: Percentage of episodes with cascade failures
- **Prefetch Efficiency**: Useful prefetches / total prefetches
- **Average Latency**: Mean response latency
- **Bandwidth Used**: Data transferred for prefetching

### Statistical Analysis

The comparator performs:
- **t-tests** for pairwise significance
- **Confidence intervals** for all metrics
- **Distribution analysis** (box plots)
- **Summary statistics** (mean, std, min, max)

---

## CLI Tool

Use the command-line interface for easy comparisons:

```bash
# Compare all baselines
python scripts/compare_baselines.py --baselines all --episodes 100 --output results/

# Compare specific baselines
python scripts/compare_baselines.py \
  --baselines lru,lfu,static_markov,random \
  --episodes 50 \
  --output results/baseline_comparison

# Include trained RL agent
python scripts/compare_baselines.py \
  --baselines all \
  --agent results/trained_agent/best.zip \
  --agent-name "DQN" \
  --episodes 100 \
  --output results/rl_vs_baselines

# Use custom environment config
python scripts/compare_baselines.py \
  --env-config configs/evaluation.yaml \
  --baselines lru,adaptive,static_markov \
  --episodes 200 \
  --output results/custom_eval
```

### CLI Options

- `--baselines`: Comma-separated list or "all"
- `--episodes`: Episodes per policy (default: 100)
- `--agent`: Path to trained RL agent
- `--agent-type`: 'sb3' or 'torch'
- `--output`: Output directory
- `--seed`: Random seed
- `--no-plots`: Disable plot generation
- `--save-json`: Save detailed JSON results

---

## Integration with RL Agents

### Stable-Baselines3 Agent

```python
from stable_baselines3 import DQN
from baselines import BaselineComparator, RLAgentAdapter

# Load trained agent
agent = DQN.load('results/best_model.zip')

# Wrap in adapter
rl_policy = RLAgentAdapter(agent, name='DQN')

# Compare
comparator = BaselineComparator()
comparator.add_policy('LRU', LRUPolicy())
comparator.add_policy('DQN', rl_policy)
results = comparator.run_comparison(env, num_episodes=100)
```

### Custom PyTorch Agent

```python
from baselines import TorchAgentAdapter
from src.rl.agents.dqn_agent import DQNAgent

# Load agent
agent = DQNAgent(state_dim=60, action_dim=7)
agent.load('checkpoint.pt')

# Wrap in adapter
rl_policy = TorchAgentAdapter(agent, name='Custom DQN')

# Compare
comparator.add_policy('Custom DQN', rl_policy)
```

---

## Validation

Test all baselines:

```bash
python validate_baselines.py
```

This validates:
- All policy implementations
- Action selection logic
- Statistics tracking
- Comparison framework
- Integration with environment

---

## For Thesis Evaluation

The baselines are designed to support comprehensive thesis evaluation:

1. **Demonstrate RL Value**: Compare RL agent vs all baselines
2. **Statistical Significance**: Automated significance tests
3. **Multiple Metrics**: Not just reward, but hit rate, cascades, etc.
4. **Visualizations**: Publication-ready plots
5. **Ablation Studies**: Compare policy variants
6. **Upper Bound**: Oracle shows improvement potential

### Example Evaluation Section

```python
# Complete evaluation for thesis
comparator = BaselineComparator()

# Add all baselines
comparator.add_policy('LRU', LRUPolicy())
comparator.add_policy('LFU', LFUPolicy())
comparator.add_policy('Static Markov', StaticMarkovPolicy())
comparator.add_policy('Adaptive', AdaptivePolicy())
comparator.add_policy('Random', RandomPolicy())
comparator.add_policy('Oracle', OraclePolicy())

# Add YOUR trained RL agent
comparator.add_trained_agent('DQN (Ours)', your_trained_agent)

# Run comprehensive evaluation
results = comparator.run_comparison(env, num_episodes=200)

# Generate thesis-ready report
report_path = comparator.generate_report(
    results,
    output_path='thesis/evaluation/baseline_comparison',
    include_plots=True
)

# Results include:
# - Summary table (LaTeX-ready)
# - Statistical significance tests
# - Box plots for all metrics
# - Detailed per-policy analysis
```

---

## Architecture

```
baselines/
├── base_policy.py          # Abstract CachingPolicy interface
├── lru_policy.py           # LRU variants
├── lfu_policy.py           # LFU variants
├── static_markov_policy.py # Static Markov variants
├── random_policy.py        # Random variants
├── adaptive_policy.py      # Adaptive heuristics
├── oracle_policy.py        # Oracle variants
├── comparison.py           # Comparison framework
├── agent_adapter.py        # RL agent adapters
└── README.md              # This file
```

---

## Contributing

To add a new baseline:

1. Inherit from `CachingPolicy`
2. Implement required methods:
   - `select_action(state, predictions) -> int`
   - `get_name() -> str`
   - `reset()`
   - `get_statistics() -> Dict`
3. Add to `__init__.py`
4. Add tests to `validate_baselines.py`

Example:

```python
from baselines import CachingPolicy

class MyNewPolicy(CachingPolicy):
    def select_action(self, state, predictions):
        # Your logic here
        return action
    
    def get_name(self):
        return "My New Policy"
    
    def reset(self):
        pass
    
    def get_statistics(self):
        return {'custom_metric': 0.0}
```

---

## References

- **LRU/LFU**: Classic caching algorithms
- **Markov Predictions**: See `src/markov/`
- **RL Environment**: See `src/integration/gym_environment.py`
- **Evaluation Framework**: See `EVALUATION_QUICK_REF.md`


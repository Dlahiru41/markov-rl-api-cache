# Reward Function Module - Complete Guide

## Overview

The `src.rl.reward` module implements a carefully designed multi-objective reward function for the caching RL agent. This is the most critical component of the RL system - it defines exactly what behavior the agent should learn.

## Design Philosophy

### 1. CASCADE PREVENTION IS PARAMOUNT

The reward magnitudes are intentionally designed so that **cascade prevention dominates all other objectives**:

- Cascade prevented: **+50.0** (5x cache hit)
- Cascade occurred: **-100.0** (100x cache miss)

**Rationale**: A single cascade failure can affect thousands of requests and potentially bring down the entire system. The agent must learn that preventing cascades is more important than optimizing individual cache hits.

### 2. Multi-Objective Balancing

The reward function balances six competing objectives:

| Objective | Weight | Priority |
|-----------|--------|----------|
| Cascade prevention | ±50-100 | **HIGHEST** |
| Cache performance | ±1-10 | High |
| Prefetch efficiency | ±3-5 | Medium |
| Latency optimization | ±0.1-0.2/ms | Medium |
| Resource management | -0.01-5 | Low |
| Reward shaping | +0.1-1.0 | Auxiliary |

### 3. Asymmetric Penalties

Losses hurt more than equivalent gains help:
- Latency degradation: **-0.2/ms**
- Latency improvement: **+0.1/ms**

This encourages conservative behavior - don't make things worse.

## Architecture

### 1. RewardConfig Dataclass

Configures all reward weights and thresholds:

```python
@dataclass
class RewardConfig:
    # Cache performance
    cache_hit_reward: float = 10.0
    cache_miss_penalty: float = -1.0
    
    # Cascade prevention (MOST IMPORTANT)
    cascade_prevented_reward: float = 50.0
    cascade_occurred_penalty: float = -100.0
    
    # Prefetch efficiency
    prefetch_used_reward: float = 5.0
    prefetch_wasted_penalty: float = -3.0
    
    # Latency optimization (asymmetric)
    latency_improvement_weight: float = 0.1
    latency_degradation_weight: float = -0.2
    
    # Resource management
    bandwidth_penalty_weight: float = -0.01  # Per KB
    cache_full_penalty: float = -5.0
    
    # Reward shaping
    enable_shaping: bool = True
    correct_prediction_bonus: float = 1.0
    exploration_bonus: float = 0.1
    
    # Stability bounds
    clip_min: float = -100.0
    clip_max: float = 100.0
```

### 2. ActionOutcome Dataclass

Captures everything that happened after taking an action:

```python
@dataclass
class ActionOutcome:
    # Cache performance
    cache_hit: bool = False
    cache_miss: bool = False
    
    # Prefetch metrics
    prefetch_attempted: int = 0
    prefetch_successful: int = 0
    prefetch_used: int = 0
    prefetch_wasted: int = 0
    prefetch_bytes: int = 0
    
    # Cascade detection/prevention
    cascade_risk_detected: bool = False
    cascade_prevented: bool = False
    cascade_occurred: bool = False
    
    # Latency metrics
    actual_latency_ms: float = 0.0
    baseline_latency_ms: float = 0.0
    
    # Cache state
    cache_utilization: float = 0.0
    evictions_triggered: int = 0
    
    # Prediction quality
    prediction_was_correct: bool = False
    prediction_confidence: float = 0.0
```

### 3. RewardCalculator Class

Computes rewards from outcomes:

**Methods:**
- `calculate(outcome)` → Returns single float reward
- `calculate_detailed(outcome)` → Returns breakdown dict

**Reward Components:**
1. **Cache**: Hit/miss rewards
2. **Cascade**: Prevention/occurrence (dominates)
3. **Prefetch**: Used vs wasted efficiency
4. **Latency**: Improvement/degradation (asymmetric)
5. **Bandwidth**: Resource usage penalties
6. **Shaping**: Auxiliary learning signals

### 4. RewardNormalizer Class

Normalizes rewards for training stability:

```python
normalizer = RewardNormalizer()

# Update with new rewards
normalizer.update(reward)

# Normalize for training
normalized = normalizer.normalize(reward)  # (reward - mean) / std

# Denormalize for analysis
original = normalizer.denormalize(normalized)
```

Uses Welford's online algorithm for numerical stability.

### 5. RewardTracker Class

Monitors and analyzes reward history:

```python
tracker = RewardTracker(window_size=1000)

# Record rewards
tracker.record(reward, breakdown)

# Get statistics
stats = tracker.get_statistics()  # mean, std, min, max
comp_stats = tracker.get_component_statistics()  # per component
contributions = tracker.get_component_contributions()  # percentages
```

## Reward Breakdown Examples

### Example 1: Simple Cache Hit
```python
outcome = ActionOutcome(cache_hit=True)
breakdown = calc.calculate_detailed(outcome)

{
    'cache': +10.0,
    'cascade': 0.0,
    'prefetch': 0.0,
    'latency': 0.0,
    'bandwidth': 0.0,
    'shaping': +0.1,  # Exploration bonus
    'total': +10.1
}
```

### Example 2: Cascade Prevented (BEST OUTCOME)
```python
outcome = ActionOutcome(
    cascade_prevented=True,
    cache_hit=True
)
breakdown = calc.calculate_detailed(outcome)

{
    'cache': +10.0,
    'cascade': +50.0,  # ← DOMINANT
    'prefetch': 0.0,
    'latency': 0.0,
    'bandwidth': 0.0,
    'shaping': +0.1,
    'total': +60.1
}
```

### Example 3: Cascade Occurred (WORST OUTCOME)
```python
outcome = ActionOutcome(cascade_occurred=True)
breakdown = calc.calculate_detailed(outcome)

{
    'cache': 0.0,
    'cascade': -100.0,  # ← DOMINANT PENALTY
    'prefetch': 0.0,
    'latency': 0.0,
    'bandwidth': 0.0,
    'shaping': +0.1,
    'total': -99.9
}
```

### Example 4: Efficient Prefetch with Latency Improvement
```python
outcome = ActionOutcome(
    cache_hit=True,
    prefetch_used=3,
    prefetch_wasted=1,
    actual_latency_ms=50,
    baseline_latency_ms=200,
    prediction_was_correct=True
)
breakdown = calc.calculate_detailed(outcome)

{
    'cache': +10.0,
    'cascade': 0.0,
    'prefetch': +12.0,  # 3×5 - 1×3 = 15 - 3 = 12
    'latency': +15.0,   # 150ms × 0.1 = 15
    'bandwidth': 0.0,
    'shaping': +1.1,    # 1.0 correct + 0.1 exploration
    'total': +38.1
}
```

### Example 5: Wasteful Prefetch
```python
outcome = ActionOutcome(
    prefetch_wasted=5,
    prefetch_bytes=500 * 1024,  # 500 KB
    cache_utilization=0.96
)
breakdown = calc.calculate_detailed(outcome)

{
    'cache': 0.0,
    'cascade': 0.0,
    'prefetch': -15.0,  # 5 × -3
    'latency': 0.0,
    'bandwidth': -10.0, # -5 KB penalty + -5 cache full
    'shaping': +0.1,
    'total': -24.9
}
```

## Usage Examples

### Basic Usage
```python
from src.rl.reward import RewardCalculator, ActionOutcome

calc = RewardCalculator()

# After agent takes action
outcome = ActionOutcome(
    cache_hit=True,
    prefetch_used=2,
    actual_latency_ms=80,
    baseline_latency_ms=150
)

reward = calc.calculate(outcome)
print(f"Reward: {reward}")  # ~27.2
```

### Detailed Analysis
```python
# Get breakdown for debugging
breakdown = calc.calculate_detailed(outcome)

print("Reward breakdown:")
for component, value in breakdown.items():
    if value != 0 or component == 'total':
        print(f"  {component:12s}: {value:+7.2f}")

# Output:
# cache       :  +10.00
# prefetch    :  +10.00
# latency     :   +7.00
# shaping     :   +0.10
# total       :  +27.10
```

### Custom Configuration
```python
# More aggressive cascade prevention
config = RewardConfig(
    cascade_prevented_reward=100.0,
    cascade_occurred_penalty=-200.0,
    cache_hit_reward=5.0  # Reduce relative importance
)

calc = RewardCalculator(config=config)
```

### Training with Normalization
```python
normalizer = RewardNormalizer()
tracker = RewardTracker()

for episode in range(num_episodes):
    outcome = get_outcome()
    reward = calc.calculate(outcome)
    breakdown = calc.calculate_detailed(outcome)
    
    # Track for monitoring
    tracker.record(reward, breakdown)
    
    # Normalize for stable training
    normalizer.update(reward)
    normalized_reward = normalizer.normalize(reward)
    
    # Train agent with normalized reward
    agent.train(state, action, normalized_reward, next_state)

# Analyze after training
print(tracker.get_statistics())
print(tracker.get_component_contributions())
```

### Monitoring During Training
```python
tracker = RewardTracker()

# ... training loop ...

# Check reward trends
stats = tracker.get_statistics()
print(f"Average reward: {stats['mean']:.2f}")
print(f"Reward std: {stats['std']:.2f}")

# Check component balance
contributions = tracker.get_component_contributions()
print("\nComponent contributions:")
for comp, pct in sorted(contributions.items(), key=lambda x: -x[1]):
    print(f"  {comp:12s}: {pct:5.1f}%")

# If cascade component is too low, agent may not be learning
# to prevent cascades
if contributions.get('cascade', 0) < 5:
    print("⚠️ Warning: Cascade prevention not being triggered enough")
```

## Reward Magnitude Rationale

### Why These Specific Values?

**Cache Hit (+10.0):**
- Baseline "good" signal
- Reference point for other rewards
- 10:1 ratio with cache miss encourages hitting cache

**Cache Miss (-1.0):**
- Small penalty (misses are normal)
- Shouldn't dominate learning
- Agent should learn when it's OK to miss

**Cascade Prevented (+50.0):**
- **5x cache hit** - strongly incentivizes prevention
- One cascade prevention = 5 cache hits worth of value
- Should be agent's secondary objective

**Cascade Occurred (-100.0):**
- **100x cache miss** - maximum deterrent
- One cascade failure = 10 cache hits lost
- Should be agent's primary avoidance objective
- Clipped to -100 to prevent destabilizing training

**Prefetch Used (+5.0):**
- Half a cache hit (reasonable value)
- Encourages useful prefetching
- Not as valuable as actual cache hit (speculative)

**Prefetch Wasted (-3.0):**
- 60% of prefetch used reward
- Discourages wasteful prefetching
- But not so harsh it prevents all prefetching

**Latency Weights (±0.1, -0.2):**
- Per-millisecond granularity
- 100ms improvement = +10 (one cache hit)
- 100ms degradation = -20 (2x penalty, asymmetric)
- Encourages latency-aware decisions

**Bandwidth (-0.01/KB):**
- Small penalty (resources matter but don't dominate)
- 100 KB prefetch = -1.0 (one cache miss)
- Prevents excessive prefetching

**Cache Full (-5.0):**
- Half a cache hit penalty
- Warns agent when approaching capacity
- Encourages eviction before crisis

**Reward Shaping (+0.1 - +1.0):**
- Small auxiliary signals
- Help early learning
- Don't dominate final behavior

## Design Decisions

### Why Clip Rewards?

Clipping to [-100, +100] prevents:
1. Training instability from extreme values
2. Gradient explosion in neural networks
3. One bad episode from dominating learning

### Why Asymmetric Latency Weights?

Degrading service is worse than improving it:
- Users notice slowdowns more than speedups
- SLA violations have contractual penalties
- Conservative policy preferred

### Why Enable/Disable Shaping?

- **Early training**: Shaping helps exploration
- **Late training**: May want pure objectives
- **Evaluation**: Disable to measure true performance

### Why Track Component Contributions?

Helps identify:
- Is cascade prevention actually being triggered?
- Is one component dominating too much?
- Is the reward function balanced for this workload?

## Common Patterns

### Pattern 1: Basic RL Loop
```python
calc = RewardCalculator()

for step in range(num_steps):
    action = agent.select_action(state)
    outcome = environment.step(action)
    reward = calc.calculate(outcome)
    agent.learn(state, action, reward, next_state)
```

### Pattern 2: With Detailed Logging
```python
tracker = RewardTracker()

for step in range(num_steps):
    action = agent.select_action(state)
    outcome = environment.step(action)
    
    reward = calc.calculate(outcome)
    breakdown = calc.calculate_detailed(outcome)
    
    tracker.record(reward, breakdown)
    agent.learn(state, action, reward, next_state)
    
    if step % 100 == 0:
        stats = tracker.get_statistics()
        print(f"Step {step}: mean reward = {stats['mean']:.2f}")
```

### Pattern 3: With Normalization
```python
normalizer = RewardNormalizer()

for step in range(num_steps):
    action = agent.select_action(state)
    outcome = environment.step(action)
    
    raw_reward = calc.calculate(outcome)
    
    # Normalize for training
    normalizer.update(raw_reward)
    normalized_reward = normalizer.normalize(raw_reward)
    
    agent.learn(state, action, normalized_reward, next_state)
```

### Pattern 4: Evaluation Mode
```python
# Disable shaping for evaluation
eval_config = RewardConfig(enable_shaping=False)
eval_calc = RewardCalculator(config=eval_config)

for episode in range(eval_episodes):
    total_reward = 0
    for step in range(max_steps):
        outcome = environment.step(agent.select_action(state))
        reward = eval_calc.calculate(outcome)
        total_reward += reward
    print(f"Episode {episode} return: {total_reward}")
```

## Tuning Guidelines

### When to Increase Cascade Weights

If agent isn't learning cascade prevention:
- Increase `cascade_prevented_reward` (e.g., 50 → 100)
- Increase `cascade_occurred_penalty` (e.g., -100 → -200)
- Check that cascades are actually being detected

### When to Adjust Prefetch Weights

If agent over-prefetches:
- Increase `prefetch_wasted_penalty` (e.g., -3 → -5)
- Decrease `prefetch_used_reward` (e.g., 5 → 3)

If agent under-prefetches:
- Decrease `prefetch_wasted_penalty` (e.g., -3 → -2)
- Increase `prefetch_used_reward` (e.g., 5 → 7)

### When to Adjust Latency Weights

For latency-critical applications:
- Increase `latency_improvement_weight` (e.g., 0.1 → 0.2)
- Increase `latency_degradation_weight` magnitude (e.g., -0.2 → -0.3)

### When to Disable Shaping

- Final evaluation
- Comparing to baselines
- After convergence (optional)

## Validation Checklist

- [ ] Cascade prevention gives highest reward
- [ ] Cascade occurrence gives highest penalty
- [ ] One cascade failure offsets ~10 cache hits
- [ ] Latency degradation hurts 2x improvement
- [ ] Rewards clip to [-100, +100]
- [ ] Prefetch efficiency properly incentivized
- [ ] Bandwidth penalties don't dominate
- [ ] Reward shaping can be disabled
- [ ] Component contributions sum to 100%
- [ ] Normalizer maintains running statistics

## References

- Multi-objective RL: [Vamplew et al., 2011]
- Reward shaping: [Ng et al., 1999]
- Asymmetric rewards: [Kahneman & Tversky - Prospect Theory]
- Cascade prevention: System reliability literature


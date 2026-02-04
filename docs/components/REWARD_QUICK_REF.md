# Reward Function Quick Reference

## Import
```python
from src.rl.reward import (
    RewardConfig, ActionOutcome, RewardCalculator,
    RewardNormalizer, RewardTracker
)
```

## Quick Start
```python
# 1. Create calculator
calc = RewardCalculator()

# 2. Create outcome
outcome = ActionOutcome(
    cache_hit=True,
    prefetch_used=2,
    actual_latency_ms=50,
    baseline_latency_ms=150
)

# 3. Calculate reward
reward = calc.calculate(outcome)  # Single float

# 4. Get breakdown
breakdown = calc.calculate_detailed(outcome)  # Dict with components
```

## Reward Weights (Default)

| Component | Weight | When |
|-----------|--------|------|
| **CASCADE PREVENTION** | **+50.0** | Cascade prevented |
| **CASCADE OCCURRED** | **-100.0** | Cascade happened |
| Cache hit | +10.0 | Request served from cache |
| Cache miss | -1.0 | Request went to backend |
| Prefetch used | +5.0 | Per prefetched item used |
| Prefetch wasted | -3.0 | Per prefetched item evicted |
| Latency improved | +0.1/ms | Faster than baseline |
| Latency degraded | -0.2/ms | Slower than baseline |
| Bandwidth | -0.01/KB | Prefetch bandwidth used |
| Cache full | -5.0 | Utilization > 95% |
| Prediction correct | +1.0 | Markov was right |
| Exploration | +0.1 | Always |

## RewardConfig

```python
config = RewardConfig(
    # Most important
    cascade_prevented_reward=50.0,
    cascade_occurred_penalty=-100.0,
    
    # Cache performance
    cache_hit_reward=10.0,
    cache_miss_penalty=-1.0,
    
    # Prefetch
    prefetch_used_reward=5.0,
    prefetch_wasted_penalty=-3.0,
    
    # Latency (asymmetric)
    latency_improvement_weight=0.1,
    latency_degradation_weight=-0.2,
    
    # Resources
    bandwidth_penalty_weight=-0.01,
    cache_full_penalty=-5.0,
    
    # Shaping
    enable_shaping=True,
    correct_prediction_bonus=1.0,
    exploration_bonus=0.1,
    
    # Bounds
    clip_min=-100.0,
    clip_max=100.0
)
```

## ActionOutcome

```python
outcome = ActionOutcome(
    # Cache
    cache_hit=True,
    cache_miss=False,
    
    # Prefetch
    prefetch_attempted=3,
    prefetch_successful=3,
    prefetch_used=2,
    prefetch_wasted=1,
    prefetch_bytes=150_000,  # bytes
    
    # Cascade
    cascade_risk_detected=False,
    cascade_prevented=False,
    cascade_occurred=False,
    
    # Latency
    actual_latency_ms=80.0,
    baseline_latency_ms=150.0,
    
    # Cache state
    cache_utilization=0.75,
    evictions_triggered=0,
    
    # Prediction
    prediction_was_correct=True,
    prediction_confidence=0.85
)
```

## RewardCalculator

### Calculate Reward
```python
calc = RewardCalculator()

# Simple
reward = calc.calculate(outcome)  # → float

# Detailed
breakdown = calc.calculate_detailed(outcome)
# → {
#     'cache': 10.0,
#     'cascade': 0.0,
#     'prefetch': 7.0,
#     'latency': 7.0,
#     'bandwidth': -1.5,
#     'shaping': 1.1,
#     'total': 23.6
#   }
```

### Custom Config
```python
config = RewardConfig(cascade_prevented_reward=100.0)
calc = RewardCalculator(config=config)
```

## RewardNormalizer

```python
normalizer = RewardNormalizer()

# Update running stats
normalizer.update(reward)

# Normalize
normalized = normalizer.normalize(reward)  # (reward - mean) / std

# Denormalize
original = normalizer.denormalize(normalized)

# Properties
normalizer.mean
normalizer.std
normalizer.variance
normalizer.count

# Reset
normalizer.reset()
```

## RewardTracker

```python
tracker = RewardTracker(window_size=1000)

# Record
tracker.record(reward, breakdown)

# Statistics
stats = tracker.get_statistics()
# → {'mean': 5.2, 'std': 10.5, 'min': -50, 'max': 60, 'count': 1000}

# Component stats
comp_stats = tracker.get_component_statistics()
# → {'cache': {'mean': 5.0, 'std': 5.5, ...}, 'cascade': {...}, ...}

# Contributions
contributions = tracker.get_component_contributions()
# → {'cache': 45.2, 'cascade': 30.1, 'prefetch': 15.3, ...}  # percentages

# Recent
recent = tracker.get_recent_rewards(n=10)
breakdowns = tracker.get_recent_breakdowns(n=10)

# Clear
tracker.clear()
```

## Common Scenarios

### 1. Simple Cache Hit
```python
outcome = ActionOutcome(cache_hit=True)
reward = calc.calculate(outcome)
# → ~10.1 (hit + exploration)
```

### 2. Cache Miss
```python
outcome = ActionOutcome(cache_miss=True)
reward = calc.calculate(outcome)
# → ~-0.9 (miss + exploration)
```

### 3. Cascade Prevented (BEST)
```python
outcome = ActionOutcome(
    cascade_prevented=True,
    cache_hit=True
)
reward = calc.calculate(outcome)
# → ~60.1 (50 cascade + 10 hit + 0.1 explore)
```

### 4. Cascade Occurred (WORST)
```python
outcome = ActionOutcome(cascade_occurred=True)
reward = calc.calculate(outcome)
# → ~-99.9 (clipped to min)
```

### 5. Good Prefetch
```python
outcome = ActionOutcome(
    prefetch_used=3,
    actual_latency_ms=50,
    baseline_latency_ms=150
)
reward = calc.calculate(outcome)
# → ~25.1 (15 prefetch + 10 latency + 0.1 explore)
```

### 6. Wasteful Prefetch
```python
outcome = ActionOutcome(
    prefetch_wasted=5,
    prefetch_bytes=500_000,
    cache_utilization=0.96
)
reward = calc.calculate(outcome)
# → ~-24.9 (-15 wasted -5 bandwidth -5 cache full +0.1 explore)
```

## Reward Hierarchy

```
Cascade Prevented (+50)  ═════════════════════════════════
                                             ↓ 5x
Cache Hit (+10)          ══════════
                                   ↓ 2x
Prefetch Used (+5)       ═════
                               ↓ 5x
Exploration (+0.1)       ═
                               ↑ 10x
Cache Miss (-1)          ═════
                               ↓ 3x
Prefetch Wasted (-3)     ═════════════════
                                         ↓ 2x
Cache Full (-5)          ══════════════════════════════════
                                                        ↓ 10x
Cascade Occurred (-100)  ═══════════════════════════════════════════════════════════════════
```

## Training Patterns

### Basic Loop
```python
for episode in range(num_episodes):
    state = env.reset()
    for step in range(max_steps):
        action = agent.select_action(state)
        outcome = env.step(action)
        
        reward = calc.calculate(outcome)
        agent.learn(state, action, reward, next_state)
```

### With Tracking
```python
tracker = RewardTracker()

for episode in range(num_episodes):
    for step in range(max_steps):
        outcome = env.step(action)
        reward = calc.calculate(outcome)
        breakdown = calc.calculate_detailed(outcome)
        
        tracker.record(reward, breakdown)
        agent.learn(state, action, reward, next_state)
    
    # Log every episode
    stats = tracker.get_statistics()
    print(f"Episode {episode}: mean reward = {stats['mean']:.2f}")
```

### With Normalization
```python
normalizer = RewardNormalizer()

for episode in range(num_episodes):
    for step in range(max_steps):
        outcome = env.step(action)
        raw_reward = calc.calculate(outcome)
        
        normalizer.update(raw_reward)
        normalized = normalizer.normalize(raw_reward)
        
        agent.learn(state, action, normalized, next_state)
```

### Evaluation Mode
```python
eval_config = RewardConfig(enable_shaping=False)
eval_calc = RewardCalculator(config=eval_config)

for episode in range(eval_episodes):
    total_reward = 0
    for step in range(max_steps):
        outcome = env.step(agent.select_action(state))
        reward = eval_calc.calculate(outcome)
        total_reward += reward
```

## Debugging

### Check Reward Breakdown
```python
breakdown = calc.calculate_detailed(outcome)
for comp, value in breakdown.items():
    if value != 0 or comp == 'total':
        print(f"{comp:12s}: {value:+7.2f}")
```

### Check Component Balance
```python
contributions = tracker.get_component_contributions()
print("Component contributions:")
for comp, pct in sorted(contributions.items(), key=lambda x: -x[1]):
    print(f"  {comp:12s}: {pct:5.1f}%")

if contributions.get('cascade', 0) < 5:
    print("⚠️ Cascade not being triggered")
```

### Monitor Statistics
```python
stats = tracker.get_statistics()
print(f"Mean: {stats['mean']:+.2f}")
print(f"Std:  {stats['std']:.2f}")
print(f"Min:  {stats['min']:+.2f}")
print(f"Max:  {stats['max']:+.2f}")
```

## Tuning Guidelines

### Increase Cascade Importance
```python
config = RewardConfig(
    cascade_prevented_reward=100.0,  # 50 → 100
    cascade_occurred_penalty=-200.0  # -100 → -200
)
```

### Reduce Prefetch Waste
```python
config = RewardConfig(
    prefetch_wasted_penalty=-5.0  # -3 → -5
)
```

### Emphasize Latency
```python
config = RewardConfig(
    latency_improvement_weight=0.2,  # 0.1 → 0.2
    latency_degradation_weight=-0.4  # -0.2 → -0.4
)
```

### Disable Shaping
```python
config = RewardConfig(enable_shaping=False)
```

## Key Ratios

```
Cascade prevented : Cache hit = 5:1
Cascade occurred : Cache miss = 100:1
One cascade failure = 10 cache hits lost
Latency degradation : improvement = 2:1
Prefetch used : wasted = 5:3
```

## Validation

```bash
# Basic validation
python test_reward_validation.py

# Comprehensive tests
python test_reward_comprehensive.py
```

## Common Issues

### "Agent not learning cascade prevention"
→ Check cascade detection is working
→ Increase cascade weights
→ Check component contributions

### "Agent too conservative"
→ Decrease penalties
→ Increase positive rewards
→ Check latency asymmetry

### "Agent too aggressive"
→ Increase waste penalties
→ Decrease prefetch rewards
→ Add bandwidth penalties

### "Training unstable"
→ Use RewardNormalizer
→ Check clip bounds
→ Reduce learning rate

## Best Practices

1. **Always use detailed breakdown** for debugging
2. **Track component contributions** during training
3. **Use normalization** for stable learning
4. **Disable shaping** for evaluation
5. **Monitor cascade component** - should be 10-30% of total
6. **Tune on representative workload**
7. **Compare multiple configs** side-by-side
8. **Log reward statistics** every N episodes
9. **Check for degenerate policies** (all same action)
10. **Validate reward ranges** match expectations


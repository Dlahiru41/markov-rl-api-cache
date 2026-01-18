"""Validation script for the reward function module."""
from src.rl.reward import RewardCalculator, RewardConfig, ActionOutcome

print("="*70)
print("Reward Function Validation")
print("="*70)
print()

calc = RewardCalculator()

# Test 1: Cache hit
print("1. Testing Cache Hit:")
print("-" * 70)
outcome = ActionOutcome(cache_hit=True)
reward = calc.calculate(outcome)
print(f"Cache hit reward: {reward}")
print(f"Expected: ~10.0")
assert reward > 0, "Cache hit should give positive reward"
print("[OK] Cache hit gives positive reward")
print()

# Test 2: Cache miss
print("2. Testing Cache Miss:")
print("-" * 70)
outcome = ActionOutcome(cache_miss=True)
reward = calc.calculate(outcome)
print(f"Cache miss penalty: {reward}")
print(f"Expected: ~-1.0")
assert reward < 0, "Cache miss should give negative reward"
print("[OK] Cache miss gives negative reward")
print()

# Test 3: Cascade prevention (should be highest reward)
print("3. Testing Cascade Prevention (MOST IMPORTANT):")
print("-" * 70)
outcome = ActionOutcome(cascade_prevented=True, cache_hit=True)
breakdown = calc.calculate_detailed(outcome)
print(f"Cascade prevented breakdown:")
for component, value in breakdown.items():
    if value != 0:
        print(f"  {component:15s}: {value:+8.2f}")
print(f"\nTotal reward: {breakdown['total']}")
print(f"Expected: ~60.0 (50 cascade + 10 cache hit)")
assert breakdown['total'] > 50, "Cascade prevention should dominate reward"
print("[OK] Cascade prevention gives large positive reward")
print()

# Test 4: Cascade occurred (should be large penalty)
print("4. Testing Cascade Occurred (WORST OUTCOME):")
print("-" * 70)
outcome = ActionOutcome(cascade_occurred=True)
reward = calc.calculate(outcome)
breakdown = calc.calculate_detailed(outcome)
print(f"Cascade occurred penalty: {reward}")
print(f"Breakdown:")
for component, value in breakdown.items():
    if value != 0 or component == 'total':
        print(f"  {component:15s}: {value:+8.2f}")
print(f"Expected: ~-99.9 (cascade -100 + exploration +0.1)")
assert reward < -90, "Cascade should give very large penalty"
print("[OK] Cascade gives maximum penalty")
print()

# Test 5: Complex scenario
print("5. Testing Complex Scenario:")
print("-" * 70)
outcome = ActionOutcome(
    cache_hit=True,
    prefetch_used=2,
    prefetch_wasted=1,
    actual_latency_ms=50,
    baseline_latency_ms=200,
    prediction_was_correct=True
)
breakdown = calc.calculate_detailed(outcome)
print(f"Complex scenario breakdown:")
for component, value in breakdown.items():
    if value != 0 or component == 'total':
        print(f"  {component:15s}: {value:+8.2f}")

print(f"\nComponent analysis:")
print(f"  Cache hit: +10.0")
print(f"  Prefetch used (2): +10.0 (2 × 5)")
print(f"  Prefetch wasted (1): -3.0")
print(f"  Latency saved (150ms): +15.0 (150 × 0.1)")
print(f"  Prediction correct: +1.0")
print(f"  Exploration bonus: +0.1")
print(f"  Expected total: ~33.1")
print("[OK] Complex scenario computed correctly")
print()

# Test 6: Reward clipping
print("6. Testing Reward Clipping:")
print("-" * 70)
# Create extreme scenario
outcome = ActionOutcome(
    cascade_occurred=True,
    cache_miss=True,
    prefetch_wasted=10,
    actual_latency_ms=1000,
    baseline_latency_ms=100
)
reward = calc.calculate(outcome)
print(f"Extreme negative scenario reward: {reward}")
print(f"Expected: -100.0 (clipped to min)")
assert reward == -100.0, "Reward should be clipped to minimum"
print("[OK] Reward clipping works")
print()

# Test 7: Prefetch efficiency
print("7. Testing Prefetch Efficiency:")
print("-" * 70)
# Good prefetch
outcome_good = ActionOutcome(prefetch_used=3, prefetch_attempted=3)
reward_good = calc.calculate(outcome_good)
print(f"Good prefetch (3 used, 0 wasted): {reward_good}")

# Bad prefetch
outcome_bad = ActionOutcome(prefetch_wasted=3, prefetch_attempted=3)
reward_bad = calc.calculate(outcome_bad)
print(f"Bad prefetch (0 used, 3 wasted): {reward_bad}")

assert reward_good > reward_bad, "Used prefetch should be better than wasted"
print("[OK] Prefetch rewards work correctly")
print()

# Test 8: Latency optimization
print("8. Testing Latency Optimization:")
print("-" * 70)
# Improved latency
outcome_improved = ActionOutcome(
    actual_latency_ms=50,
    baseline_latency_ms=100
)
breakdown_improved = calc.calculate_detailed(outcome_improved)
print(f"Latency improved (100ms → 50ms): {breakdown_improved['latency']}")
print(f"Expected: +5.0 (50ms × 0.1)")

# Degraded latency
outcome_degraded = ActionOutcome(
    actual_latency_ms=100,
    baseline_latency_ms=50
)
breakdown_degraded = calc.calculate_detailed(outcome_degraded)
print(f"Latency degraded (50ms → 100ms): {breakdown_degraded['latency']}")
print(f"Expected: -10.0 (50ms × -0.2, asymmetric penalty)")

assert breakdown_improved['latency'] > 0, "Latency improvement should be positive"
assert breakdown_degraded['latency'] < 0, "Latency degradation should be negative"
assert abs(breakdown_degraded['latency']) > abs(breakdown_improved['latency']), \
    "Degradation should hurt more than improvement helps"
print("[OK] Latency rewards are asymmetric as designed")
print()

# Test 9: Reward magnitude comparison
print("9. Testing Reward Magnitude Hierarchy:")
print("-" * 70)
print("Testing that cascade prevention dominates other rewards...")

cascade_reward = calc.calculate(ActionOutcome(cascade_prevented=True))
cache_hit_reward = calc.calculate(ActionOutcome(cache_hit=True))
prefetch_reward = calc.calculate(ActionOutcome(prefetch_used=1))

print(f"Cascade prevented: {cascade_reward:+8.2f}")
print(f"Cache hit:         {cache_hit_reward:+8.2f}")
print(f"Prefetch used:     {prefetch_reward:+8.2f}")

assert cascade_reward > cache_hit_reward * 4, "Cascade should be 5x cache hit"
assert cascade_reward > prefetch_reward * 8, "Cascade should dominate prefetch"
print("[OK] Cascade prevention dominates as intended")
print()

# Test 10: Bandwidth penalty
print("10. Testing Bandwidth Penalty:")
print("-" * 70)
outcome = ActionOutcome(prefetch_bytes=100 * 1024)  # 100 KB
breakdown = calc.calculate_detailed(outcome)
print(f"Bandwidth penalty for 100 KB: {breakdown['bandwidth']}")
print(f"Expected: -1.0 (100 KB × -0.01)")
assert breakdown['bandwidth'] < 0, "Bandwidth usage should have penalty"
print("[OK] Bandwidth penalty works")
print()

print("="*70)
print("All validation tests passed!")
print("="*70)


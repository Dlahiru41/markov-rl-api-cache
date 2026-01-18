"""Comprehensive test suite for the reward function module."""
import numpy as np
from src.rl.reward import (
    RewardCalculator, RewardConfig, ActionOutcome,
    RewardNormalizer, RewardTracker
)


def test_reward_config():
    """Test RewardConfig dataclass."""
    print("="*70)
    print("Test 1: RewardConfig")
    print("="*70)

    # Default config
    config = RewardConfig()
    assert config.cache_hit_reward == 10.0
    assert config.cascade_prevented_reward == 50.0
    assert config.cascade_occurred_penalty == -100.0
    print("[OK] Default config values correct")

    # Custom config
    custom = RewardConfig(
        cache_hit_reward=20.0,
        cascade_prevented_reward=100.0,
        enable_shaping=False
    )
    assert custom.cache_hit_reward == 20.0
    assert custom.cascade_prevented_reward == 100.0
    assert custom.enable_shaping == False
    print("[OK] Custom config works")
    print()


def test_action_outcome():
    """Test ActionOutcome dataclass."""
    print("="*70)
    print("Test 2: ActionOutcome")
    print("="*70)

    # Default outcome
    outcome = ActionOutcome()
    assert outcome.cache_hit == False
    assert outcome.cascade_occurred == False
    print("[OK] Default outcome initialized correctly")

    # Custom outcome
    outcome = ActionOutcome(
        cache_hit=True,
        prefetch_used=3,
        cascade_prevented=True
    )
    assert outcome.cache_hit == True
    assert outcome.prefetch_used == 3
    assert outcome.cascade_prevented == True
    print("[OK] Custom outcome works")
    print()


def test_basic_rewards():
    """Test basic reward calculations."""
    print("="*70)
    print("Test 3: Basic Reward Calculations")
    print("="*70)

    calc = RewardCalculator()

    # Cache hit
    reward = calc.calculate(ActionOutcome(cache_hit=True))
    assert reward > 9, f"Cache hit should be ~10, got {reward}"
    print(f"[OK] Cache hit: {reward:.2f}")

    # Cache miss
    reward = calc.calculate(ActionOutcome(cache_miss=True))
    assert -2 < reward < 0, f"Cache miss should be ~-1, got {reward}"
    print(f"[OK] Cache miss: {reward:.2f}")

    # Cascade prevented
    reward = calc.calculate(ActionOutcome(cascade_prevented=True))
    assert reward > 49, f"Cascade prevented should be ~50, got {reward}"
    print(f"[OK] Cascade prevented: {reward:.2f}")

    # Cascade occurred
    reward = calc.calculate(ActionOutcome(cascade_occurred=True))
    assert reward < -90, f"Cascade occurred should be ~-100, got {reward}"
    print(f"[OK] Cascade occurred: {reward:.2f}")
    print()


def test_detailed_breakdown():
    """Test detailed reward breakdown."""
    print("="*70)
    print("Test 4: Detailed Breakdown")
    print("="*70)

    calc = RewardCalculator()

    outcome = ActionOutcome(
        cache_hit=True,
        prefetch_used=2,
        prefetch_wasted=1,
        actual_latency_ms=50,
        baseline_latency_ms=200,
        prediction_was_correct=True
    )

    breakdown = calc.calculate_detailed(outcome)

    # Check all components exist
    assert 'cache' in breakdown
    assert 'cascade' in breakdown
    assert 'prefetch' in breakdown
    assert 'latency' in breakdown
    assert 'bandwidth' in breakdown
    assert 'shaping' in breakdown
    assert 'total' in breakdown
    print("[OK] All breakdown components present")

    # Check values
    assert breakdown['cache'] == 10.0, f"Cache component should be 10, got {breakdown['cache']}"
    assert breakdown['prefetch'] == 7.0, f"Prefetch should be 7 (2*5 - 1*3), got {breakdown['prefetch']}"
    assert breakdown['latency'] == 15.0, f"Latency should be 15 (150*0.1), got {breakdown['latency']}"
    assert breakdown['shaping'] == 1.1, f"Shaping should be 1.1 (1.0 + 0.1), got {breakdown['shaping']}"
    print("[OK] Breakdown values correct")

    # Check total
    expected_total = 10.0 + 7.0 + 15.0 + 1.1
    assert abs(breakdown['total'] - expected_total) < 0.01, \
        f"Total should be {expected_total}, got {breakdown['total']}"
    print(f"[OK] Total: {breakdown['total']:.2f}")
    print()


def test_reward_clipping():
    """Test reward clipping to bounds."""
    print("="*70)
    print("Test 5: Reward Clipping")
    print("="*70)

    calc = RewardCalculator()

    # Extreme positive (should clip to +100)
    outcome = ActionOutcome(
        cascade_prevented=True,
        cache_hit=True,
        prefetch_used=10,
        actual_latency_ms=0,
        baseline_latency_ms=1000
    )
    reward = calc.calculate(outcome)
    assert reward == 100.0, f"Should clip to +100, got {reward}"
    print(f"[OK] Clips to maximum: {reward}")

    # Extreme negative (should clip to -100)
    outcome = ActionOutcome(
        cascade_occurred=True,
        cache_miss=True,
        prefetch_wasted=10,
        actual_latency_ms=1000,
        baseline_latency_ms=0
    )
    reward = calc.calculate(outcome)
    assert reward == -100.0, f"Should clip to -100, got {reward}"
    print(f"[OK] Clips to minimum: {reward}")
    print()


def test_cascade_dominance():
    """Test that cascade prevention dominates other rewards."""
    print("="*70)
    print("Test 6: Cascade Dominance")
    print("="*70)

    calc = RewardCalculator()

    cascade_reward = calc.calculate(ActionOutcome(cascade_prevented=True))
    cache_hits_reward = calc.calculate(ActionOutcome(cache_hit=True))

    # Cascade should be worth 5x cache hit
    assert cascade_reward > cache_hits_reward * 4, \
        f"Cascade ({cascade_reward}) should be >4x cache hit ({cache_hits_reward})"
    print(f"[OK] Cascade prevented ({cascade_reward:.1f}) >> Cache hit ({cache_hits_reward:.1f})")

    # One cascade failure should offset many cache hits
    cascade_penalty = calc.calculate(ActionOutcome(cascade_occurred=True))
    hits_needed = abs(cascade_penalty) / cache_hits_reward
    print(f"[OK] One cascade failure offsets ~{hits_needed:.0f} cache hits")
    print()


def test_latency_asymmetry():
    """Test that latency degradation hurts more than improvement helps."""
    print("="*70)
    print("Test 7: Latency Asymmetry")
    print("="*70)

    calc = RewardCalculator()

    # 100ms improvement
    improved = calc.calculate_detailed(ActionOutcome(
        actual_latency_ms=50,
        baseline_latency_ms=150
    ))

    # 100ms degradation
    degraded = calc.calculate_detailed(ActionOutcome(
        actual_latency_ms=150,
        baseline_latency_ms=50
    ))

    print(f"100ms improvement: {improved['latency']:+.1f}")
    print(f"100ms degradation: {degraded['latency']:+.1f}")

    assert abs(degraded['latency']) > abs(improved['latency']), \
        "Degradation should hurt more than improvement helps"
    print("[OK] Latency rewards are asymmetric (2:1 ratio)")
    print()


def test_reward_normalizer():
    """Test RewardNormalizer class."""
    print("="*70)
    print("Test 8: RewardNormalizer")
    print("="*70)

    normalizer = RewardNormalizer()

    # Feed some rewards
    rewards = [10.0, 20.0, 30.0, 40.0, 50.0]
    for reward in rewards:
        normalizer.update(reward)

    assert normalizer.count == 5
    assert abs(normalizer.mean - 30.0) < 0.01
    print(f"[OK] Mean: {normalizer.mean:.2f} (expected 30.0)")
    print(f"[OK] Std: {normalizer.std:.2f}")

    # Normalize a value
    normalized = normalizer.normalize(30.0)
    assert abs(normalized) < 0.01, "Mean value should normalize to ~0"
    print(f"[OK] Normalized mean to: {normalized:.3f}")

    # Denormalize
    denormalized = normalizer.denormalize(normalized)
    assert abs(denormalized - 30.0) < 0.01
    print(f"[OK] Denormalized back to: {denormalized:.2f}")

    # Reset
    normalizer.reset()
    assert normalizer.count == 0
    print("[OK] Reset works")
    print()


def test_reward_tracker():
    """Test RewardTracker class."""
    print("="*70)
    print("Test 9: RewardTracker")
    print("="*70)

    tracker = RewardTracker(window_size=100)
    calc = RewardCalculator()

    # Record some rewards
    for i in range(50):
        outcome = ActionOutcome(
            cache_hit=(i % 2 == 0),
            cache_miss=(i % 2 == 1)
        )
        reward = calc.calculate(outcome)
        breakdown = calc.calculate_detailed(outcome)
        tracker.record(reward, breakdown)

    # Get statistics
    stats = tracker.get_statistics()
    assert stats['count'] == 50
    assert stats['mean'] > 0  # More hits than misses
    print(f"[OK] Tracked {stats['count']} rewards")
    print(f"[OK] Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")

    # Component statistics
    comp_stats = tracker.get_component_statistics()
    assert 'cache' in comp_stats
    assert 'prefetch' in comp_stats
    print(f"[OK] Component statistics available")

    # Recent rewards
    recent = tracker.get_recent_rewards(n=5)
    assert len(recent) == 5
    print(f"[OK] Recent rewards: {recent}")

    # Component contributions
    contributions = tracker.get_component_contributions()
    print(f"[OK] Component contributions:")
    for comp, pct in sorted(contributions.items(), key=lambda x: -x[1]):
        if pct > 0:
            print(f"    {comp:15s}: {pct:5.1f}%")
    print()


def test_custom_config_effects():
    """Test that custom config changes rewards."""
    print("="*70)
    print("Test 10: Custom Config Effects")
    print("="*70)

    # Default config
    default_calc = RewardCalculator()
    default_reward = default_calc.calculate(ActionOutcome(cache_hit=True))
    print(f"Default cache hit reward: {default_reward:.2f}")

    # Double the cache hit reward
    custom_config = RewardConfig(cache_hit_reward=20.0)
    custom_calc = RewardCalculator(config=custom_config)
    custom_reward = custom_calc.calculate(ActionOutcome(cache_hit=True))
    print(f"Custom cache hit reward: {custom_reward:.2f}")

    assert custom_reward > default_reward, "Custom config should increase reward"
    print("[OK] Custom config affects rewards")
    print()


def test_no_reward_shaping():
    """Test disabling reward shaping."""
    print("="*70)
    print("Test 11: Disable Reward Shaping")
    print("="*70)

    # With shaping
    with_shaping = RewardCalculator(RewardConfig(enable_shaping=True))
    reward_with = with_shaping.calculate(ActionOutcome(
        cache_hit=True,
        prediction_was_correct=True
    ))

    # Without shaping
    without_shaping = RewardCalculator(RewardConfig(enable_shaping=False))
    reward_without = without_shaping.calculate(ActionOutcome(
        cache_hit=True,
        prediction_was_correct=True
    ))

    print(f"With shaping: {reward_with:.2f}")
    print(f"Without shaping: {reward_without:.2f}")

    assert reward_with > reward_without, "Shaping should add positive signals"
    print("[OK] Reward shaping can be disabled")
    print()


def test_bandwidth_and_cache_pressure():
    """Test bandwidth and cache pressure penalties."""
    print("="*70)
    print("Test 12: Bandwidth and Cache Pressure")
    print("="*70)

    calc = RewardCalculator()

    # Bandwidth penalty
    outcome = ActionOutcome(prefetch_bytes=500 * 1024)  # 500 KB
    breakdown = calc.calculate_detailed(outcome)
    print(f"Bandwidth penalty for 500 KB: {breakdown['bandwidth']:.2f}")
    assert breakdown['bandwidth'] < 0, "Bandwidth should have penalty"

    # Cache full penalty
    outcome = ActionOutcome(cache_utilization=0.96)
    breakdown = calc.calculate_detailed(outcome)
    print(f"Cache full penalty (96%): {breakdown['bandwidth']:.2f}")
    assert breakdown['bandwidth'] < 0, "High cache util should have penalty"

    # Combined
    outcome = ActionOutcome(
        prefetch_bytes=100 * 1024,
        cache_utilization=0.97
    )
    breakdown = calc.calculate_detailed(outcome)
    print(f"Combined penalties: {breakdown['bandwidth']:.2f}")
    assert breakdown['bandwidth'] < -5, "Should have both penalties"
    print("[OK] Bandwidth and cache pressure penalties work")
    print()


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("="*70)
    print("Test 13: Edge Cases")
    print("="*70)

    calc = RewardCalculator()

    # Empty outcome
    outcome = ActionOutcome()
    reward = calc.calculate(outcome)
    assert reward >= 0, "Empty outcome should have exploration bonus"
    print(f"[OK] Empty outcome: {reward:.2f}")

    # Zero latency delta
    outcome = ActionOutcome(
        actual_latency_ms=100,
        baseline_latency_ms=100
    )
    breakdown = calc.calculate_detailed(outcome)
    assert breakdown['latency'] == 0, "Same latency should give 0 reward"
    print("[OK] Zero latency delta handled")

    # Multiple conflicting signals
    outcome = ActionOutcome(
        cache_hit=True,
        cache_miss=True,  # Shouldn't happen, but handled
        cascade_prevented=True,
        cascade_occurred=True  # Shouldn't happen, but handled
    )
    reward = calc.calculate(outcome)
    print(f"[OK] Conflicting signals handled: {reward:.2f}")
    print()


if __name__ == "__main__":
    test_reward_config()
    test_action_outcome()
    test_basic_rewards()
    test_detailed_breakdown()
    test_reward_clipping()
    test_cascade_dominance()
    test_latency_asymmetry()
    test_reward_normalizer()
    test_reward_tracker()
    test_custom_config_effects()
    test_no_reward_shaping()
    test_bandwidth_and_cache_pressure()
    test_edge_cases()

    print("="*70)
    print("ALL TESTS PASSED!")
    print("="*70)


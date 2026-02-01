"""
Validation script for baseline policies.

Tests all baseline implementations to ensure they work correctly.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from baselines import (
    LRUPolicy,
    AdaptiveLRUPolicy,
    LFUPolicy,
    WindowedLFUPolicy,
    StaticMarkovPolicy,
    InverseStaticMarkovPolicy,
    BalancedStaticMarkovPolicy,
    RandomPolicy,
    EpsilonRandomPolicy,
    BiasedRandomPolicy,
    AdaptivePolicy,
    MultiObjectiveAdaptivePolicy,
    OraclePolicy,
    PartialOraclePolicy,
    NoisyOraclePolicy,
    PolicyWrapper,
    BaselineComparator
)
from src.integration.gym_environment import CachingEnv, CacheEnvConfig, SimulatorConfig


def test_policy(policy, name, num_steps=50):
    """Test a single policy."""
    print(f"\nTesting {name}...")

    # Create dummy state and predictions
    state = np.random.rand(60)  # 60-dimensional state
    predictions = [
        ('/api/products/1', 0.8),
        ('/api/cart', 0.6),
        ('/api/checkout', 0.4),
        ('/api/profile', 0.3),
        ('/api/search', 0.2)
    ]

    # Test basic functionality
    try:
        # Reset
        policy.reset()

        # Select actions
        for _ in range(num_steps):
            action = policy.select_action(state, predictions)

            # Validate action
            assert isinstance(action, int), f"Action must be int, got {type(action)}"
            assert 0 <= action < 7, f"Action must be in [0, 6], got {action}"

        # Get name
        policy_name = policy.get_name()
        assert isinstance(policy_name, str), f"Name must be string, got {type(policy_name)}"

        # Get statistics
        stats = policy.get_statistics()
        assert isinstance(stats, dict), f"Statistics must be dict, got {type(stats)}"

        print(f"  ✓ {name} passed all tests")
        print(f"    Name: {policy_name}")
        print(f"    Stats keys: {list(stats.keys())}")
        return True

    except Exception as e:
        print(f"  ✗ {name} failed: {e}")
        return False


def test_policy_wrapper():
    """Test PolicyWrapper functionality."""
    print("\nTesting PolicyWrapper...")

    try:
        # Create base policy
        base_policy = LRUPolicy()
        wrapped = PolicyWrapper(base_policy)

        # Test action selection
        state = np.random.rand(60)
        predictions = [('/api/test', 0.5)]

        action = wrapped.select_action(state, predictions)
        assert isinstance(action, int)

        # Record reward
        wrapped.record_reward(action, 10.0)

        # Get statistics
        stats = wrapped.get_statistics()
        assert 'total_decisions' in stats
        assert 'total_reward' in stats

        print("  ✓ PolicyWrapper passed all tests")
        return True

    except Exception as e:
        print(f"  ✗ PolicyWrapper failed: {e}")
        return False


def test_comparator():
    """Test BaselineComparator."""
    print("\nTesting BaselineComparator...")

    try:
        # Create environment
        config = CacheEnvConfig(
            simulator_config=SimulatorConfig(
                num_apis=10,
                session_length_range=(5, 20)
            ),
            max_steps_per_episode=50,
            log_episode_metrics=False
        )
        env = CachingEnv(config)

        # Create comparator
        comparator = BaselineComparator()

        # Add policies
        comparator.add_policy('LRU', LRUPolicy())
        comparator.add_policy('Random', RandomPolicy())

        # Run short comparison
        print("  Running comparison (5 episodes per policy)...")
        results = comparator.run_comparison(env, num_episodes=5, verbose=False)

        # Check results
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        assert 'policy_name' in results.columns
        assert 'mean_reward' in results.columns

        env.close()

        print("  ✓ BaselineComparator passed all tests")
        print(f"\n{results.to_string(index=False)}")
        return True

    except Exception as e:
        print(f"  ✗ BaselineComparator failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("="*80)
    print("BASELINE POLICIES VALIDATION")
    print("="*80)

    results = []

    # Test all policies
    policies = [
        (LRUPolicy(), "LRUPolicy"),
        (AdaptiveLRUPolicy(), "AdaptiveLRUPolicy"),
        (LFUPolicy(), "LFUPolicy"),
        (WindowedLFUPolicy(), "WindowedLFUPolicy"),
        (StaticMarkovPolicy(), "StaticMarkovPolicy"),
        (InverseStaticMarkovPolicy(), "InverseStaticMarkovPolicy"),
        (BalancedStaticMarkovPolicy(), "BalancedStaticMarkovPolicy"),
        (RandomPolicy(), "RandomPolicy"),
        (EpsilonRandomPolicy(LRUPolicy()), "EpsilonRandomPolicy"),
        (BiasedRandomPolicy(excluded_actions=[4, 5]), "BiasedRandomPolicy"),
        (AdaptivePolicy(), "AdaptivePolicy"),
        (MultiObjectiveAdaptivePolicy(), "MultiObjectiveAdaptivePolicy"),
        (OraclePolicy(), "OraclePolicy"),
        (PartialOraclePolicy(), "PartialOraclePolicy"),
        (NoisyOraclePolicy(), "NoisyOraclePolicy"),
    ]

    for policy, name in policies:
        results.append(test_policy(policy, name))

    # Test wrapper
    results.append(test_policy_wrapper())

    # Test comparator
    results.append(test_comparator())

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    passed = sum(results)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n✓ All validation tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())


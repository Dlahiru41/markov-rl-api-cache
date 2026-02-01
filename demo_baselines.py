"""
Demo script showing how to use baseline caching policies.

This demonstrates:
1. Individual policy usage
2. Comparison framework
3. Integration with environment
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from baselines import (
    LRUPolicy,
    LFUPolicy,
    StaticMarkovPolicy,
    RandomPolicy,
    AdaptivePolicy,
    BaselineComparator
)
from src.integration.gym_environment import CachingEnv, CacheEnvConfig, SimulatorConfig


def demo_individual_policy():
    """Demonstrate using an individual policy."""
    print("\n" + "="*80)
    print("DEMO 1: Individual Policy Usage")
    print("="*80)

    # Create policy
    policy = LRUPolicy(eviction_threshold=0.9)
    print(f"\nPolicy: {policy.get_name()}")

    # Create dummy state and predictions
    state = np.random.rand(60)  # 60-dimensional state vector
    predictions = [
        ('/api/products/1', 0.8),
        ('/api/cart', 0.6),
        ('/api/checkout', 0.4),
        ('/api/profile', 0.3),
        ('/api/search', 0.2)
    ]

    print(f"\nState shape: {state.shape}")
    print(f"Top prediction: {predictions[0][0]} (p={predictions[0][1]:.2f})")

    # Reset policy
    policy.reset()

    # Select action
    action = policy.select_action(state, predictions)
    print(f"\nSelected action: {action}")

    # Get statistics
    stats = policy.get_statistics()
    print(f"\nPolicy statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def demo_policy_comparison():
    """Demonstrate comparing multiple policies."""
    print("\n" + "="*80)
    print("DEMO 2: Policy Comparison")
    print("="*80)

    # Create environment
    print("\nCreating environment...")
    config = CacheEnvConfig(
        simulator_config=SimulatorConfig(
            num_apis=15,
            session_length_range=(10, 30),
            cascade_threshold=0.75
        ),
        max_steps_per_episode=100,
        use_real_services=False,
        episode_end_on_cascade=True,
        log_episode_metrics=False,
        seed=42
    )
    env = CachingEnv(config)
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # Create comparator
    print("\nSetting up comparison...")
    comparator = BaselineComparator()

    # Add policies
    policies = [
        ('LRU', LRUPolicy()),
        ('LFU', LFUPolicy()),
        ('Static Markov', StaticMarkovPolicy()),
        ('Random', RandomPolicy()),
        ('Adaptive', AdaptivePolicy())
    ]

    for name, policy in policies:
        comparator.add_policy(name, policy)
        print(f"  Added: {name}")

    # Run comparison (short for demo)
    print("\nRunning comparison (10 episodes per policy)...")
    results = comparator.run_comparison(
        env,
        num_episodes=10,
        seed=42,
        verbose=True
    )

    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print()
    print(results.to_string(index=False))

    # Best policy
    best = results.iloc[0]
    print(f"\n🏆 Best Policy: {best['policy_name']}")
    print(f"   Mean Reward: {best['mean_reward']:.2f} ± {best['std_reward']:.2f}")
    print(f"   Cache Hit Rate: {best['mean_hit_rate']:.2%}")
    print(f"   Cascade Rate: {best['cascade_rate']:.2%}")

    env.close()


def demo_environment_integration():
    """Demonstrate policy integration with gym environment."""
    print("\n" + "="*80)
    print("DEMO 3: Environment Integration")
    print("="*80)

    # Create environment
    config = CacheEnvConfig(
        simulator_config=SimulatorConfig(num_apis=10),
        max_steps_per_episode=50,
        log_episode_metrics=False
    )
    env = CachingEnv(config)

    # Create policy
    policy = StaticMarkovPolicy()
    print(f"\nUsing policy: {policy.get_name()}")

    # Run single episode
    print("\nRunning episode...")
    obs, info = env.reset(seed=42)
    policy.reset()

    episode_reward = 0
    step_count = 0
    done = False

    while not done and step_count < 50:
        # Get predictions from environment (simplified)
        predictions = []
        if hasattr(env, 'get_current_predictions'):
            predictions = env.get_current_predictions()

        # Select action
        action = policy.select_action(obs, predictions)

        # Take step
        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        step_count += 1
        done = terminated or truncated

    print(f"\nEpisode complete:")
    print(f"  Steps: {step_count}")
    print(f"  Total reward: {episode_reward:.2f}")
    print(f"  Average reward: {episode_reward/step_count:.2f}")

    # Get episode metrics
    metrics = env.get_episode_metrics()
    print(f"\nEpisode metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    env.close()


def demo_policy_variants():
    """Show different policy variants."""
    print("\n" + "="*80)
    print("DEMO 4: Policy Variants")
    print("="*80)

    # Create dummy data
    state = np.random.rand(60)
    high_conf_predictions = [('/api/test', 0.9)]
    low_conf_predictions = [('/api/test', 0.2)]

    policies = [
        StaticMarkovPolicy(conservative_threshold=0.7),
        StaticMarkovPolicy(conservative_threshold=0.5),
        StaticMarkovPolicy(conservative_threshold=0.9),
    ]

    print("\nStatic Markov with different thresholds:")
    print("\nHigh confidence predictions (p=0.9):")
    for i, policy in enumerate(policies):
        action = policy.select_action(state, high_conf_predictions)
        print(f"  Threshold {policy.conservative_threshold:.1f}: action={action}")

    print("\nLow confidence predictions (p=0.2):")
    for i, policy in enumerate(policies):
        action = policy.select_action(state, low_conf_predictions)
        print(f"  Threshold {policy.conservative_threshold:.1f}: action={action}")


def main():
    """Run all demos."""
    print("\n" + "#"*80)
    print("# BASELINE CACHING POLICIES - DEMO")
    print("#"*80)

    try:
        # Demo 1: Individual policy
        demo_individual_policy()

        # Demo 2: Comparison framework
        demo_policy_comparison()

        # Demo 3: Environment integration
        demo_environment_integration()

        # Demo 4: Policy variants
        demo_policy_variants()

        print("\n" + "#"*80)
        print("# ALL DEMOS COMPLETE")
        print("#"*80)
        print("\nNext steps:")
        print("  1. Run full comparison: python scripts/compare_baselines.py")
        print("  2. Train RL agent: python scripts/train.py")
        print("  3. Compare RL vs baselines: python scripts/compare_baselines.py --agent results/best.zip")
        print()

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())


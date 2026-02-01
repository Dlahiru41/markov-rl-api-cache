"""
Baseline comparison script: RL agents vs. simple heuristic policies.

This demonstrates the value of RL by comparing trained agents against:
1. Random policy (baseline)
2. Always cache (aggressive)
3. Always prefetch conservatively (moderate)
4. Do nothing (passive LRU)
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from src.integration.gym_environment import CachingEnv, CacheEnvConfig, SimulatorConfig
from src.rl.actions import CacheAction


class BaselinePolicy:
    """Base class for baseline policies."""

    def __init__(self, name: str):
        self.name = name

    def predict(self, obs):
        """Predict action given observation."""
        raise NotImplementedError


class RandomPolicy(BaselinePolicy):
    """Random action selection."""

    def __init__(self):
        super().__init__("Random")
        self.rng = np.random.RandomState(42)

    def predict(self, obs):
        return self.rng.randint(0, 7), None


class AlwaysCachePolicy(BaselinePolicy):
    """Always cache current response."""

    def __init__(self):
        super().__init__("Always Cache")

    def predict(self, obs):
        return CacheAction.CACHE_CURRENT, None


class ConservativePrefetchPolicy(BaselinePolicy):
    """Always prefetch conservatively."""

    def __init__(self):
        super().__init__("Conservative Prefetch")

    def predict(self, obs):
        return CacheAction.PREFETCH_CONSERVATIVE, None


class ModeratePrefetchPolicy(BaselinePolicy):
    """Always prefetch moderately."""

    def __init__(self):
        super().__init__("Moderate Prefetch")

    def predict(self, obs):
        return CacheAction.PREFETCH_MODERATE, None


class DoNothingPolicy(BaselinePolicy):
    """Passive LRU - never intervene."""

    def __init__(self):
        super().__init__("Do Nothing (LRU)")

    def predict(self, obs):
        return CacheAction.DO_NOTHING, None


class AdaptiveHeuristicPolicy(BaselinePolicy):
    """
    Heuristic policy that adapts based on state.

    Rules:
    - If cache hit rate is low (<50%), cache aggressively
    - If system load is high (>70%), prefetch conservatively
    - If predictions are confident (>0.7), prefetch moderately
    - Otherwise, do nothing
    """

    def __init__(self):
        super().__init__("Adaptive Heuristic")
        self.step_count = 0

    def predict(self, obs):
        self.step_count += 1

        # Extract features from observation (rough approximation)
        # This assumes the state structure from StateConfig

        # Get cache hit rate (around index 6-7 in state)
        # Get max prediction probability (confidence, around index 5)
        # Get system CPU (around index 10)

        if len(obs) < 20:
            return CacheAction.DO_NOTHING, None

        # Rough feature extraction (this is approximate)
        prediction_confidence = obs[10] if len(obs) > 10 else 0.0
        cache_hit_rate = obs[11] if len(obs) > 11 else 0.5
        system_cpu = obs[15] if len(obs) > 15 else 0.3

        # Decision logic
        if cache_hit_rate < 0.5:
            # Low hit rate - cache more aggressively
            return CacheAction.CACHE_CURRENT, None
        elif system_cpu > 0.7:
            # High load - prefetch conservatively to reduce future load
            return CacheAction.PREFETCH_CONSERVATIVE, None
        elif prediction_confidence > 0.7:
            # Confident predictions - prefetch moderately
            return CacheAction.PREFETCH_MODERATE, None
        else:
            # Default - do nothing
            return CacheAction.DO_NOTHING, None


def evaluate_policy(policy, n_episodes: int = 20, seed: int = 42):
    """Evaluate a policy over multiple episodes."""
    # Create evaluation environment
    config = CacheEnvConfig(
        simulator_config=SimulatorConfig(
            num_apis=15,
            session_length_range=(10, 50),
            cascade_threshold=0.75
        ),
        max_steps_per_episode=200,
        episode_end_on_cascade=True,
        log_episode_metrics=False,
        seed=seed
    )
    env = CachingEnv(config)

    episode_rewards = []
    cache_hit_rates = []
    prediction_accuracies = []
    cascade_counts = 0
    total_steps = []
    action_counts = {i: 0 for i in range(7)}

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        episode_reward = 0
        done = False

        while not done:
            action, _ = policy.predict(obs)
            action_counts[action] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        episode_rewards.append(episode_reward)

        # Get episode metrics
        metrics = env.get_episode_metrics()
        cache_hit_rates.append(metrics['cache_hit_rate'])
        prediction_accuracies.append(metrics['prediction_accuracy'])
        if metrics['cascade_occurred']:
            cascade_counts += 1
        total_steps.append(metrics['total_steps'])

    env.close()

    # Calculate statistics
    results = {
        'policy_name': policy.name,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_cache_hit_rate': np.mean(cache_hit_rates),
        'std_cache_hit_rate': np.std(cache_hit_rates),
        'mean_pred_accuracy': np.mean(prediction_accuracies),
        'cascade_count': cascade_counts,
        'cascade_rate': cascade_counts / n_episodes,
        'mean_steps': np.mean(total_steps),
        'action_distribution': action_counts,
        'n_episodes': n_episodes
    }

    return results


def print_results_table(all_results):
    """Print formatted comparison table."""
    print("\n" + "="*80)
    print("BASELINE POLICY COMPARISON")
    print("="*80)

    # Header
    print(f"{'Policy':<25} {'Mean Reward':<18} {'Hit Rate':<12} {'Cascades':<10}")
    print("-"*80)

    # Sort by mean reward
    sorted_results = sorted(all_results, key=lambda x: x['mean_reward'], reverse=True)

    for result in sorted_results:
        print(
            f"{result['policy_name']:<25} "
            f"{result['mean_reward']:>7.2f} ± {result['std_reward']:<7.2f} "
            f"{result['mean_cache_hit_rate']:>6.1%}      "
            f"{result['cascade_count']:>3}/{result['n_episodes']}"
        )

    print("-"*80)

    # Identify best
    best = sorted_results[0]
    print(f"\n🏆 Best Policy: {best['policy_name']}")
    print(f"   Mean Reward: {best['mean_reward']:.2f}")
    print(f"   Cache Hit Rate: {best['mean_cache_hit_rate']:.1%}")
    print(f"   Cascade Rate: {best['cascade_rate']:.1%}")


def print_detailed_analysis(result):
    """Print detailed analysis for a policy."""
    print(f"\n{'='*60}")
    print(f"Detailed Analysis: {result['policy_name']}")
    print(f"{'='*60}")

    print(f"\nReward Statistics:")
    print(f"  Mean:   {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
    print(f"  Range:  [{result['min_reward']:.2f}, {result['max_reward']:.2f}]")

    print(f"\nCache Performance:")
    print(f"  Hit Rate:  {result['mean_cache_hit_rate']:.2%} ± {result['std_cache_hit_rate']:.2%}")

    print(f"\nPrediction Accuracy:")
    print(f"  Accuracy:  {result['mean_pred_accuracy']:.2%}")

    print(f"\nSystem Stability:")
    print(f"  Cascade Rate:  {result['cascade_rate']:.1%} ({result['cascade_count']}/{result['n_episodes']} episodes)")
    print(f"  Mean Steps:    {result['mean_steps']:.0f}")

    print(f"\nAction Distribution:")
    total_actions = sum(result['action_distribution'].values())
    for action_id, count in result['action_distribution'].items():
        action_name = CacheAction.get_name(action_id)
        pct = 100 * count / total_actions if total_actions > 0 else 0
        print(f"  {action_name:<25} {count:>6} ({pct:>5.1f}%)")


def main():
    """Run baseline comparison."""
    print("\n" + "#"*80)
    print("# BASELINE POLICY COMPARISON FOR INTELLIGENT CACHING")
    print("#"*80)

    # Define baseline policies
    policies = [
        RandomPolicy(),
        DoNothingPolicy(),
        AlwaysCachePolicy(),
        ConservativePrefetchPolicy(),
        ModeratePrefetchPolicy(),
        AdaptiveHeuristicPolicy()
    ]

    # Evaluate each policy
    all_results = []

    for policy in policies:
        print(f"\nEvaluating {policy.name}...")
        results = evaluate_policy(policy, n_episodes=20, seed=42)
        all_results.append(results)
        print(f"  Mean reward: {results['mean_reward']:.2f}")

    # Print comparison table
    print_results_table(all_results)

    # Print detailed analysis for best policy
    best_result = max(all_results, key=lambda x: x['mean_reward'])
    print_detailed_analysis(best_result)

    # Print insights
    print("\n" + "="*80)
    print("INSIGHTS")
    print("="*80)

    print("\n1. Random Policy:")
    random_result = next(r for r in all_results if r['policy_name'] == 'Random')
    print(f"   This is our baseline. Mean reward: {random_result['mean_reward']:.2f}")

    print("\n2. Performance Improvement:")
    for result in all_results:
        if result['policy_name'] != 'Random':
            improvement = (result['mean_reward'] - random_result['mean_reward']) / abs(random_result['mean_reward']) * 100
            print(f"   {result['policy_name']:<25} {improvement:+.1f}% vs Random")

    print("\n3. Cascade Prevention:")
    cascade_rates = [(r['policy_name'], r['cascade_rate']) for r in all_results]
    best_cascade = min(cascade_rates, key=lambda x: x[1])
    print(f"   Best at preventing cascades: {best_cascade[0]} ({best_cascade[1]:.1%})")

    print("\n4. Cache Efficiency:")
    hit_rates = [(r['policy_name'], r['mean_cache_hit_rate']) for r in all_results]
    best_hit_rate = max(hit_rates, key=lambda x: x[1])
    print(f"   Highest cache hit rate: {best_hit_rate[0]} ({best_hit_rate[1]:.1%})")

    print("\n" + "#"*80)
    print("# NEXT STEPS")
    print("#"*80)
    print("\nTrain an RL agent to beat these baselines:")
    print("  python train_rl_agents.py")
    print("\nThe RL agent should learn to:")
    print("  - Adapt to different user types and session patterns")
    print("  - Balance caching vs. prefetching based on predictions")
    print("  - Prevent cascade failures through proactive action")
    print("  - Achieve higher rewards than any fixed heuristic")


if __name__ == "__main__":
    main()


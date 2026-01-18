"""
Integration example showing reward function in complete RL training loop.

Demonstrates how the reward function works with state representation,
action space, and a simple RL agent.
"""

import numpy as np
from src.rl.state import StateBuilder, StateConfig
from src.rl.actions import CacheAction, ActionSpace, ActionConfig
from src.rl.reward import (
    RewardCalculator, RewardConfig, ActionOutcome,
    RewardNormalizer, RewardTracker
)


class MockEnvironment:
    """Simulates a caching environment for demonstration."""

    def __init__(self):
        self.cache_hits = 0
        self.cache_misses = 0
        self.cascades_prevented = 0
        self.cascades_occurred = 0
        self.total_requests = 0

    def step(self, action, predictions):
        """
        Simulate taking an action and observe outcome.

        Args:
            action: Action index
            predictions: Markov predictions

        Returns:
            ActionOutcome describing what happened
        """
        self.total_requests += 1

        # Simulate outcomes based on action
        outcome = ActionOutcome()

        if action == CacheAction.DO_NOTHING:
            # Random cache hit/miss
            if np.random.random() < 0.3:
                outcome.cache_hit = True
                self.cache_hits += 1
            else:
                outcome.cache_miss = True
                self.cache_misses += 1

        elif action == CacheAction.CACHE_CURRENT:
            # Higher hit rate when explicitly caching
            outcome.cache_hit = True
            self.cache_hits += 1

        elif action in [CacheAction.PREFETCH_CONSERVATIVE,
                        CacheAction.PREFETCH_MODERATE,
                        CacheAction.PREFETCH_AGGRESSIVE]:
            # Prefetch actions
            if action == CacheAction.PREFETCH_CONSERVATIVE:
                threshold = 0.7
                max_prefetch = 1
            elif action == CacheAction.PREFETCH_MODERATE:
                threshold = 0.5
                max_prefetch = 3
            else:
                threshold = 0.3
                max_prefetch = 5

            # Count how many predictions meet threshold
            valid_predictions = [p for api, p in predictions if p > threshold]
            outcome.prefetch_attempted = min(len(valid_predictions), max_prefetch)
            outcome.prefetch_successful = outcome.prefetch_attempted

            # Simulate some being used
            outcome.prefetch_used = int(outcome.prefetch_attempted * 0.6)
            outcome.prefetch_wasted = outcome.prefetch_attempted - outcome.prefetch_used
            outcome.prefetch_bytes = outcome.prefetch_attempted * 50_000  # 50 KB each

            # Cache hit from prefetch
            if outcome.prefetch_used > 0:
                outcome.cache_hit = True
                self.cache_hits += 1
            else:
                outcome.cache_miss = True
                self.cache_misses += 1

        elif action in [CacheAction.EVICT_LRU, CacheAction.EVICT_LOW_PROB]:
            # Eviction actions - might prevent cascade
            if np.random.random() < 0.1:  # 10% chance cascade was brewing
                outcome.cascade_prevented = True
                self.cascades_prevented += 1

            outcome.cache_miss = True  # Evicting means next request misses
            self.cache_misses += 1

        # Random cascade risk
        if np.random.random() < 0.05:  # 5% chance of cascade
            if action in [CacheAction.EVICT_LRU, CacheAction.EVICT_LOW_PROB]:
                # Eviction helped prevent it
                outcome.cascade_prevented = True
                self.cascades_prevented += 1
            else:
                # Cascade occurred
                outcome.cascade_occurred = True
                self.cascades_occurred += 1

        # Latency simulation
        if outcome.cache_hit:
            outcome.actual_latency_ms = 20 + np.random.random() * 30  # 20-50ms
        else:
            outcome.actual_latency_ms = 100 + np.random.random() * 100  # 100-200ms
        outcome.baseline_latency_ms = 150  # Baseline

        # Cache utilization
        outcome.cache_utilization = 0.5 + np.random.random() * 0.3  # 50-80%

        # Prediction quality
        if predictions and len(predictions) > 0:
            outcome.prediction_was_correct = (np.random.random() < predictions[0][1])
            outcome.prediction_confidence = predictions[0][1]

        return outcome

    def get_stats(self):
        """Get environment statistics."""
        hit_rate = self.cache_hits / max(self.total_requests, 1)
        return {
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cascades_prevented': self.cascades_prevented,
            'cascades_occurred': self.cascades_occurred
        }


def main():
    """Run integration demonstration."""
    print("="*70)
    print("Reward Function Integration Example")
    print("="*70)
    print()

    # Setup components
    print("1. Initializing components...")

    # State builder
    api_vocab = ['api1', 'api2', 'api3', 'api4', 'api5']
    state_builder = StateBuilder(StateConfig(markov_top_k=3))
    state_builder.fit(api_vocab)

    # Action space
    action_space = ActionSpace(ActionConfig())

    # Reward calculator
    reward_calc = RewardCalculator(RewardConfig())

    # Tracking
    reward_normalizer = RewardNormalizer()
    reward_tracker = RewardTracker(window_size=100)

    # Environment
    env = MockEnvironment()

    print(f"   - State dimension: {state_builder.config.state_dim}")
    print(f"   - Action space: {action_space.n}")
    print(f"   - Reward components: cache, cascade, prefetch, latency, bandwidth, shaping")
    print()

    # Run episodes
    print("2. Running episodes...")
    print()

    num_episodes = 10
    steps_per_episode = 20

    for episode in range(num_episodes):
        episode_reward = 0
        episode_outcomes = []

        for step in range(steps_per_episode):
            # Mock predictions
            predictions = [
                ('api1', 0.7 + np.random.random() * 0.2),
                ('api2', 0.4 + np.random.random() * 0.3),
                ('api3', 0.2 + np.random.random() * 0.2)
            ]

            # Select action (random for demo)
            valid_actions = action_space.get_valid_actions(
                cache_utilization=0.6,
                has_predictions=True,
                cache_size=50
            )
            action = np.random.choice(valid_actions)

            # Take action
            outcome = env.step(action, predictions)

            # Calculate reward
            reward = reward_calc.calculate(outcome)
            breakdown = reward_calc.calculate_detailed(outcome)

            # Track
            reward_tracker.record(reward, breakdown)
            reward_normalizer.update(reward)

            episode_reward += reward
            episode_outcomes.append({
                'action': CacheAction.get_name(action),
                'reward': reward,
                'outcome': outcome
            })

        # Episode summary
        print(f"Episode {episode + 1}:")
        print(f"  Total reward: {episode_reward:+.1f}")
        print(f"  Avg reward: {episode_reward / steps_per_episode:+.2f}")

        # Show interesting outcomes
        best = max(episode_outcomes, key=lambda x: x['reward'])
        worst = min(episode_outcomes, key=lambda x: x['reward'])

        print(f"  Best: {best['action']:25s} → {best['reward']:+6.1f}")
        if best['outcome'].cascade_prevented:
            print(f"        (Cascade prevented!)")

        print(f"  Worst: {worst['action']:25s} → {worst['reward']:+6.1f}")
        if worst['outcome'].cascade_occurred:
            print(f"         (Cascade occurred!)")
        print()

    # Final analysis
    print("3. Final Analysis:")
    print("-" * 70)

    # Reward statistics
    stats = reward_tracker.get_statistics()
    print("\nReward Statistics:")
    print(f"  Mean:  {stats['mean']:+.2f}")
    print(f"  Std:   {stats['std']:.2f}")
    print(f"  Min:   {stats['min']:+.2f}")
    print(f"  Max:   {stats['max']:+.2f}")
    print(f"  Count: {stats['count']}")

    # Normalizer stats
    print("\nNormalizer Statistics:")
    print(f"  Mean:  {reward_normalizer.mean:+.2f}")
    print(f"  Std:   {reward_normalizer.std:.2f}")
    print(f"  Count: {reward_normalizer.count}")

    # Component contributions
    contributions = reward_tracker.get_component_contributions()
    print("\nComponent Contributions:")
    for comp, pct in sorted(contributions.items(), key=lambda x: -x[1]):
        if pct > 0.1:
            print(f"  {comp:12s}: {pct:5.1f}%")

    # Component statistics
    comp_stats = reward_tracker.get_component_statistics()
    print("\nComponent Statistics:")
    for comp in ['cache', 'cascade', 'prefetch', 'latency']:
        if comp in comp_stats:
            s = comp_stats[comp]
            print(f"  {comp:12s}: mean={s['mean']:+6.2f}, std={s['std']:5.2f}, sum={s['sum']:+7.1f}")

    # Environment stats
    env_stats = env.get_stats()
    print("\nEnvironment Statistics:")
    print(f"  Total requests: {env_stats['total_requests']}")
    print(f"  Cache hits: {env_stats['cache_hits']}")
    print(f"  Cache misses: {env_stats['cache_misses']}")
    print(f"  Hit rate: {env_stats['hit_rate']:.1%}")
    print(f"  Cascades prevented: {env_stats['cascades_prevented']}")
    print(f"  Cascades occurred: {env_stats['cascades_occurred']}")

    # Reward design validation
    print("\n4. Reward Design Validation:")
    print("-" * 70)

    # Check cascade dominance
    if 'cascade' in contributions and contributions['cascade'] > 5:
        print("[OK] Cascade prevention is being triggered")
    else:
        print("[!] Cascade prevention not prominent - check detection logic")

    # Check balance
    if contributions.get('cache', 0) > 80:
        print("[!] Cache component dominates - may need to adjust other weights")
    else:
        print("[OK] Reward components reasonably balanced")

    # Check reward range
    if stats['min'] >= -100 and stats['max'] <= 100:
        print("[OK] Rewards within expected bounds [-100, +100]")
    else:
        print("[!] Rewards outside expected bounds")

    # Check cascade impact
    if env_stats['cascades_prevented'] > 0:
        cascade_reward = reward_calc.config.cascade_prevented_reward
        cache_hit_reward = reward_calc.config.cache_hit_reward
        ratio = cascade_reward / cache_hit_reward
        print(f"[OK] Cascade worth {ratio:.0f}x cache hit - appropriately weighted")

    print()
    print("="*70)
    print("Integration example completed!")
    print("="*70)


if __name__ == "__main__":
    main()


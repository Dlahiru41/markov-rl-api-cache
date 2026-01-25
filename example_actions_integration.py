"""
Integration example showing how to use the action space with an RL agent.

This demonstrates a complete decision cycle: state → action → execution → reward.
"""

import numpy as np
from src.rl.actions import CacheAction, ActionSpace, ActionConfig, ActionHistory
from src.rl.state import StateBuilder, StateConfig


class MockRLAgent:
    """Simple mock RL agent for demonstration."""

    def __init__(self, action_space):
        self.action_space = action_space
        # Simplified Q-table (in reality, this would be a neural network)
        self.q_values = np.random.rand(action_space.n)

    def select_action(self, state, valid_actions, epsilon=0.1):
        """Select action using epsilon-greedy policy."""
        if np.random.random() < epsilon:
            # Explore: sample from valid actions
            return np.random.choice(valid_actions)
        else:
            # Exploit: choose best valid action
            valid_q = [(a, self.q_values[a]) for a in valid_actions]
            return max(valid_q, key=lambda x: x[1])[0]

    def update(self, state, action, reward, next_state):
        """Update Q-values (simplified)."""
        learning_rate = 0.1
        discount = 0.99

        # Simple Q-learning update
        current_q = self.q_values[action]
        max_next_q = np.max(self.q_values)
        new_q = current_q + learning_rate * (reward + discount * max_next_q - current_q)
        self.q_values[action] = new_q


class MockCacheSystem:
    """Simulates a cache system for demonstration."""

    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.entries = {}
        self.lru_order = []
        self.total_requests = 0
        self.cache_hits = 0

    @property
    def size(self):
        return len(self.entries)

    @property
    def utilization(self):
        return self.size / self.capacity

    @property
    def hit_rate(self):
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    def request(self, api):
        """Simulate an API request."""
        self.total_requests += 1

        if api in self.entries:
            self.cache_hits += 1
            # Update LRU order
            self.lru_order.remove(api)
            self.lru_order.append(api)
            return True  # Cache hit
        else:
            return False  # Cache miss

    def cache_api(self, api):
        """Cache an API response."""
        if api not in self.entries:
            self.entries[api] = {'data': f'response_for_{api}'}
            self.lru_order.append(api)

            # Evict if over capacity
            if self.size > self.capacity:
                self.evict_lru(1)

    def prefetch(self, api):
        """Prefetch an API response."""
        self.cache_api(api)

    def evict_lru(self, count):
        """Evict least-recently-used entries."""
        for _ in range(min(count, self.size)):
            if self.lru_order:
                api_to_evict = self.lru_order.pop(0)
                del self.entries[api_to_evict]

    def evict_low_probability(self, count, predictions):
        """Evict entries with lowest predicted probability."""
        pred_dict = {api: prob for api, prob in predictions}

        # Score cached entries by prediction probability
        scored_entries = []
        for api in self.entries.keys():
            prob = pred_dict.get(api, 0.0)  # 0 if not in predictions
            scored_entries.append((api, prob))

        # Sort by probability (ascending) and evict lowest
        scored_entries.sort(key=lambda x: x[1])
        for i in range(min(count, len(scored_entries))):
            api_to_evict = scored_entries[i][0]
            if api_to_evict in self.lru_order:
                self.lru_order.remove(api_to_evict)
            del self.entries[api_to_evict]

    def get_metrics(self):
        """Return cache metrics."""
        return {
            'utilization': self.utilization,
            'hit_rate': self.hit_rate,
            'entries': self.size,
            'eviction_rate': 0.0  # Would track this in real system
        }


def execute_action(decoded, cache_system, current_api=None):
    """Execute a decoded action on the cache system."""

    if decoded['action_type'] == 'none':
        pass  # Do nothing

    elif decoded['action_type'] == 'cache' and current_api:
        cache_system.cache_api(current_api)

    elif decoded['action_type'] == 'prefetch':
        for api in decoded['apis_to_prefetch']:
            cache_system.prefetch(api)

    elif decoded['action_type'] == 'evict':
        if decoded['eviction_strategy'] == 'lru':
            cache_system.evict_lru(decoded['eviction_count'])
        elif decoded['eviction_strategy'] == 'low_prob':
            # Would use predictions from context
            cache_system.evict_low_probability(decoded['eviction_count'], [])


def compute_reward(old_metrics, new_metrics, action_cost=0.01):
    """
    Compute reward based on cache performance improvement.

    Reward = hit_rate_improvement - action_cost
    """
    hit_rate_delta = new_metrics['hit_rate'] - old_metrics['hit_rate']

    # Positive reward for improving hit rate
    # Small penalty for taking action (to encourage efficiency)
    reward = hit_rate_delta - action_cost

    return reward


def main():
    """Run integration demonstration."""
    print("="*70)
    print("Action Space Integration Example")
    print("="*70)
    print()

    # Setup components
    print("1. Initializing components...")

    # API vocabulary
    api_vocab = [
        'login', 'logout', 'profile', 'settings',
        'browse', 'search', 'product', 'category',
        'cart', 'wishlist', 'checkout', 'payment'
    ]

    # State builder
    state_config = StateConfig(markov_top_k=5)
    state_builder = StateBuilder(state_config)
    state_builder.fit(api_vocab)

    # Action space
    action_config = ActionConfig()
    action_space = ActionSpace(config=action_config)

    # Cache system
    cache = MockCacheSystem(capacity=100)

    # RL agent
    agent = MockRLAgent(action_space)

    # Action history
    history = ActionHistory()

    print(f"   - API vocabulary: {len(api_vocab)} APIs")
    print(f"   - State dimension: {state_config.state_dim}")
    print(f"   - Action space: {action_space.n} actions")
    print(f"   - Cache capacity: {cache.capacity}")
    print()

    # Simulate episodes
    print("2. Running decision cycles...")
    print()

    num_episodes = 5
    requests_per_episode = 10

    for episode in range(num_episodes):
        print(f"Episode {episode + 1}:")
        print("-" * 70)

        episode_reward = 0

        for step in range(requests_per_episode):
            # Simulate incoming request
            current_api = np.random.choice(api_vocab[:8])  # Bias toward first 8 APIs

            # Mock predictions (in reality, from Markov predictor)
            predictions = [
                (api_vocab[i], max(0.1, 1.0 - i*0.15))
                for i in range(min(6, len(api_vocab)))
            ]

            # Build state
            state = state_builder.build_state(
                markov_predictions=predictions[:5],
                cache_metrics=cache.get_metrics(),
                system_metrics={'cpu': 0.3, 'memory': 0.5},
                context={'user_type': 'premium', 'hour': 14, 'day': 2}
            )

            # Get valid actions
            valid_actions = action_space.get_valid_actions(
                cache_utilization=cache.utilization,
                has_predictions=len(predictions) > 0,
                cache_size=cache.size
            )

            # Agent selects action
            action = agent.select_action(state, valid_actions, epsilon=0.2)
            action_name = CacheAction.get_name(action)

            # Get old metrics
            old_metrics = cache.get_metrics()

            # Decode and execute action
            decoded = action_space.decode_action(action, predictions)
            execute_action(decoded, cache, current_api)

            # Process actual request
            was_hit = cache.request(current_api)

            # Get new metrics and compute reward
            new_metrics = cache.get_metrics()
            reward = compute_reward(old_metrics, new_metrics)
            episode_reward += reward

            # Build next state
            next_state = state_builder.build_state(
                markov_predictions=predictions[:5],
                cache_metrics=cache.get_metrics(),
                system_metrics={'cpu': 0.3, 'memory': 0.5},
                context={'user_type': 'premium', 'hour': 14, 'day': 2}
            )

            # Agent learns
            agent.update(state, action, reward, next_state)

            # Record in history
            history.record(action, state, reward, context={'episode': episode, 'step': step})

            # Print step info
            if step % 3 == 0:
                print(f"  Step {step:2d}: API={current_api:10s} Action={action_name:25s} "
                      f"Hit={'[OK]' if was_hit else '[FAIL]'} Reward={reward:+.3f}")

        print(f"  Episode reward: {episode_reward:.3f}")
        print(f"  Cache: {cache.size}/{cache.capacity} entries, "
              f"Hit rate: {cache.hit_rate:.1%}")
        print()

    # Analyze results
    print("3. Analyzing agent behavior...")
    print("-" * 70)

    # Action distribution
    dist = history.get_action_distribution()
    print("\nAction Distribution:")
    for action_name, freq in sorted(dist.items(), key=lambda x: -x[1]):
        if freq > 0:
            print(f"  {action_name:25s}: {freq*100:5.1f}%")

    # Reward by action
    rewards = history.get_reward_by_action()
    print("\nAverage Reward by Action:")
    for action_name, avg_reward in sorted(rewards.items(), key=lambda x: -x[1]):
        if action_name in [a for a, f in dist.items() if f > 0]:
            print(f"  {action_name:25s}: {avg_reward:+.4f}")

    # Overall statistics
    stats = history.get_statistics()
    print(f"\nOverall Statistics:")
    print(f"  Total actions: {stats['total_actions']}")
    print(f"  Total reward: {stats['total_reward']:.3f}")
    print(f"  Average reward: {stats['average_reward']:.3f}")

    # Cache performance
    print(f"\nFinal Cache Performance:")
    print(f"  Total requests: {cache.total_requests}")
    print(f"  Cache hits: {cache.cache_hits}")
    print(f"  Hit rate: {cache.hit_rate:.1%}")
    print(f"  Cache size: {cache.size}/{cache.capacity}")

    # Agent Q-values
    print(f"\nLearned Q-values:")
    for i in range(action_space.n):
        action_name = CacheAction.get_name(i)
        print(f"  {action_name:25s}: {agent.q_values[i]:.4f}")

    print()
    print("="*70)
    print("Integration example completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()


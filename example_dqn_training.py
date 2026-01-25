"""
Complete Example: Training a DQN Agent for API Cache Management

This example demonstrates the full workflow of training a DQN agent
to learn optimal caching policies for an API gateway.
"""

import numpy as np
from src.rl.agents.dqn_agent import DQNAgent, DoubleDQNAgent, DQNConfig


class SimpleCacheEnvironment:
    """
    A simple mock environment for testing DQN agent.

    Simulates an API cache with:
    - State: cache metrics and predictions
    - Actions: cache, prefetch, evict decisions
    - Rewards: based on hit rate and resource usage
    """

    def __init__(self, state_dim=60, action_dim=7):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        """Reset environment to initial state."""
        self.step_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        return self._get_state()

    def _get_state(self):
        """Generate current state representation."""
        # Simulated state: random values representing cache metrics
        state = np.random.randn(self.state_dim).astype(np.float32)
        return state

    def step(self, action):
        """
        Execute action and return (next_state, reward, done, info).

        Args:
            action: Action index (0-6)

        Returns:
            next_state: New state after action
            reward: Reward for this step
            done: Whether episode ended
            info: Additional information dict
        """
        self.step_count += 1

        # Simulate action effects (simplified)
        if action == 1:  # CACHE_CURRENT
            # High reward for caching
            reward = np.random.uniform(5, 15)
            self.cache_hits += 1
        elif action in [2, 3, 4]:  # PREFETCH variants
            # Variable reward for prefetching
            reward = np.random.uniform(-2, 10)
        elif action in [5, 6]:  # EVICT variants
            # Small penalty for eviction
            reward = np.random.uniform(-5, 2)
        else:  # DO_NOTHING
            # Neutral or slightly negative
            reward = np.random.uniform(-1, 1)
            self.cache_misses += 1

        # Generate next state
        next_state = self._get_state()

        # Episode ends after 100 steps
        done = (self.step_count >= 100)

        # Info
        info = {
            'step': self.step_count,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        }

        return next_state, reward, done, info


def train_dqn_agent(
    num_episodes=100,
    agent_type='dqn',
    prioritized_replay=False,
    save_path='trained_cache_agent.pt'
):
    """
    Train a DQN agent on the cache environment.

    Args:
        num_episodes: Number of training episodes
        agent_type: 'dqn' or 'double_dqn'
        prioritized_replay: Whether to use prioritized experience replay
        save_path: Path to save trained agent
    """

    print("=" * 80)
    print(f"TRAINING {agent_type.upper()} AGENT")
    print("=" * 80)

    # Environment
    env = SimpleCacheEnvironment(state_dim=60, action_dim=7)

    # Configuration
    config = DQNConfig(
        state_dim=60,
        action_dim=7,
        hidden_dims=[128, 64],
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        prioritized_replay=prioritized_replay,
        target_update_freq=100,
        max_grad_norm=10.0,
        device='auto',
        seed=42
    )

    # Agent
    if agent_type == 'double_dqn':
        agent = DoubleDQNAgent(config, seed=42)
    else:
        agent = DQNAgent(config, seed=42)

    print(f"\nAgent: {type(agent).__name__}")
    print(f"Device: {agent.device}")
    print(f"Prioritized Replay: {prioritized_replay}")
    print(f"Episodes: {num_episodes}")
    print()

    # Training loop
    episode_rewards = []
    episode_losses = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = []
        done = False

        while not done:
            # Select action
            action = agent.select_action(state)

            # Execute action
            next_state, reward, done, info = env.step(action)

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Train
            metrics = agent.train_step()
            if metrics:
                episode_loss.append(metrics['loss'])

            episode_reward += reward
            state = next_state

        # Record metrics
        episode_rewards.append(episode_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0.0
        episode_losses.append(avg_loss)

        # Logging
        if (episode + 1) % 10 == 0:
            recent_rewards = episode_rewards[-10:]
            recent_losses = episode_losses[-10:]

            print(f"Episode {episode + 1:3d}/{num_episodes} | "
                  f"Reward: {np.mean(recent_rewards):7.2f} ± {np.std(recent_rewards):5.2f} | "
                  f"Loss: {np.mean(recent_losses):7.4f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Buffer: {len(agent.buffer):5d}")

    # Save trained agent
    agent.save(save_path)
    print(f"\n[OK] Agent saved to {save_path}")

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total Episodes: {num_episodes}")
    print(f"Final Epsilon: {agent.epsilon:.3f}")
    print(f"Final Buffer Size: {len(agent.buffer)}")
    print(f"Average Reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
    print(f"Average Loss (last 10): {np.mean(episode_losses[-10:]):.4f}")

    return agent, episode_rewards, episode_losses


def evaluate_agent(agent_path, num_episodes=10):
    """
    Evaluate a trained agent.

    Args:
        agent_path: Path to saved agent
        num_episodes: Number of evaluation episodes
    """

    print("\n" + "=" * 80)
    print("AGENT EVALUATION")
    print("=" * 80)

    # Load agent
    env = SimpleCacheEnvironment(state_dim=60, action_dim=7)
    config = DQNConfig(state_dim=60, action_dim=7, seed=42)
    agent = DQNAgent(config)
    agent.load(agent_path)

    print(f"[OK] Agent loaded from {agent_path}")
    print(f"Epsilon: {agent.epsilon:.3f}")
    print(f"Evaluating for {num_episodes} episodes...")
    print()

    # Evaluation
    episode_rewards = []
    hit_rates = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Pure greedy policy
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, info = env.step(action)

            episode_reward += reward
            state = next_state

            if done:
                hit_rates.append(info['hit_rate'])

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1:2d}: Reward={episode_reward:7.2f}, Hit Rate={info['hit_rate']:.1%}")

    # Summary
    print("\n" + "-" * 80)
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Hit Rate: {np.mean(hit_rates):.1%}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")


def compare_agents():
    """Compare different agent configurations."""

    print("\n" + "=" * 80)
    print("COMPARING AGENT CONFIGURATIONS")
    print("=" * 80)

    configs = [
        ('DQN (Standard)', 'dqn', False),
        ('DQN (Prioritized)', 'dqn', True),
        ('Double DQN', 'double_dqn', False),
        ('Double DQN (Prioritized)', 'double_dqn', True),
    ]

    results = {}

    for name, agent_type, prioritized in configs:
        print(f"\n--- Training {name} ---")
        agent, rewards, losses = train_dqn_agent(
            num_episodes=50,
            agent_type=agent_type,
            prioritized_replay=prioritized,
            save_path=f"agent_{agent_type}_{'per' if prioritized else 'standard'}.pt"
        )

        results[name] = {
            'final_reward': np.mean(rewards[-10:]),
            'final_loss': np.mean(losses[-10:]),
            'epsilon': agent.epsilon
        }

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Configuration':<30} {'Reward':<12} {'Loss':<12} {'Epsilon':<10}")
    print("-" * 80)

    for name, metrics in results.items():
        print(f"{name:<30} {metrics['final_reward']:<12.2f} "
              f"{metrics['final_loss']:<12.4f} {metrics['epsilon']:<10.3f}")


def main():
    """Run complete example."""

    print("\n" + "=" * 80)
    print("DQN AGENT - COMPLETE TRAINING EXAMPLE")
    print("=" * 80)

    # 1. Train standard DQN
    print("\n1. Training Standard DQN Agent")
    agent, rewards, losses = train_dqn_agent(
        num_episodes=100,
        agent_type='dqn',
        prioritized_replay=False,
        save_path='cache_agent_standard.pt'
    )

    # 2. Evaluate
    print("\n2. Evaluating Trained Agent")
    evaluate_agent('cache_agent_standard.pt', num_episodes=10)

    # 3. Train Double DQN with prioritized replay
    print("\n3. Training Double DQN with Prioritized Replay")
    agent2, rewards2, losses2 = train_dqn_agent(
        num_episodes=100,
        agent_type='double_dqn',
        prioritized_replay=True,
        save_path='cache_agent_double_per.pt'
    )

    # 4. Evaluate
    print("\n4. Evaluating Double DQN Agent")
    evaluate_agent('cache_agent_double_per.pt', num_episodes=10)

    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE!")
    print("=" * 80)
    print("\nThe DQN agent has been successfully trained and evaluated.")
    print("You can now integrate it with your actual caching environment.")


if __name__ == "__main__":
    # Run the complete example
    main()

    # Optional: Compare different configurations
    # compare_agents()


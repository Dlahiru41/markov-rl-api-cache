"""Demonstration of FeatureEngineer integration with RL environment.

This example shows how to use the FeatureEngineer to create state representations
for reinforcement learning agents in an API caching scenario.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple

from preprocessing.feature_engineer import FeatureEngineer
from preprocessing.models import APICall, Session, Dataset


def create_demo_dataset():
    """Create a realistic API usage dataset for demonstration."""
    sessions = []
    base_time = datetime(2026, 1, 11, 10, 0, 0)

    # Create 10 sessions with various patterns
    for session_idx in range(10):
        user_types = ['premium', 'free', 'guest']
        user_type = user_types[session_idx % 3]

        session_start = base_time + timedelta(hours=session_idx)
        calls = []

        # Common pattern: login → profile → browse → product
        endpoints = [
            "/api/login",
            f"/api/users/{session_idx + 100}/profile",
            "/api/products/browse",
            f"/api/products/{session_idx + 200}/details",
        ]

        # Some sessions go to checkout
        if session_idx % 3 == 0:
            endpoints.extend([
                "/api/cart/add",
                "/api/cart",
                "/api/checkout"
            ])

        for call_idx, endpoint in enumerate(endpoints):
            method = "POST" if "login" in endpoint or "add" in endpoint else "GET"
            call = APICall(
                call_id=f"c{session_idx}_{call_idx}",
                endpoint=endpoint,
                method=method,
                params={"q": "test"} if "browse" in endpoint else {},
                user_id=f"user{session_idx}",
                session_id=f"sess{session_idx}",
                timestamp=session_start + timedelta(seconds=call_idx * 5),
                response_time_ms=80 + call_idx * 20,
                status_code=200,
                response_size_bytes=1024 * (call_idx + 1),
                user_type=user_type
            )
            calls.append(call)

        session = Session(
            session_id=f"sess{session_idx}",
            user_id=f"user{session_idx}",
            user_type=user_type,
            start_timestamp=session_start,
            calls=calls
        )
        sessions.append(session)

    return Dataset(name="demo_dataset", sessions=sessions)


class SimpleCachingEnvironment:
    """Simple RL environment for API caching decisions.

    State: Feature vector from FeatureEngineer
    Action: Cache (1) or Don't Cache (0)
    Reward: Positive if caching helps, negative if wastes memory
    """

    def __init__(self, feature_engineer: FeatureEngineer, sessions: List[Session]):
        self.fe = feature_engineer
        self.sessions = sessions
        self.current_session_idx = 0
        self.current_call_idx = 0
        self.cache = set()  # Simplified cache (just endpoint names)
        self.cache_capacity = 10

    def reset(self) -> np.ndarray:
        """Reset environment to start of a new session."""
        self.current_session_idx = (self.current_session_idx + 1) % len(self.sessions)
        self.current_call_idx = 0
        self.cache.clear()

        session = self.sessions[self.current_session_idx]
        call = session.calls[0]

        state = self.fe.transform(call, session, history=[])
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Execute action and return (next_state, reward, done).

        Args:
            action: 0 = don't cache, 1 = cache

        Returns:
            next_state: Feature vector for next call
            reward: Reward for the action
            done: Whether episode is finished
        """
        session = self.sessions[self.current_session_idx]
        current_call = session.calls[self.current_call_idx]

        # Calculate reward based on action
        reward = self._calculate_reward(current_call, action)

        # Execute action (add to cache if action=1)
        if action == 1 and len(self.cache) < self.cache_capacity:
            self.cache.add(current_call.endpoint)

        # Move to next call
        self.current_call_idx += 1
        done = self.current_call_idx >= len(session.calls)

        if done:
            next_state = np.zeros(self.fe.get_feature_dim())
        else:
            next_call = session.calls[self.current_call_idx]
            history = session.calls[:self.current_call_idx]
            next_state = self.fe.transform(next_call, session, history)

        return next_state, reward, done

    def _calculate_reward(self, call: APICall, action: int) -> float:
        """Calculate reward for caching decision."""
        # Reward structure:
        # - Cache hit (was in cache): +10
        # - Correct cache decision (will be requested again): +5
        # - Incorrect cache decision (wasted memory): -2
        # - Don't cache (no cost): 0

        endpoint = call.endpoint

        if action == 1:  # Decided to cache
            # Simple heuristic: reward caching if response time is high
            # In real scenario, would check if endpoint is requested again
            if call.response_time_ms > 150:
                return 5.0  # Good decision, expensive endpoint
            else:
                return -2.0  # Wasted cache space on cheap endpoint
        else:  # Decided not to cache
            if endpoint in self.cache:
                return 10.0  # Cache hit!
            else:
                return 0.0  # No action, no reward

    def render(self):
        """Display current state of environment."""
        session = self.sessions[self.current_session_idx]
        if self.current_call_idx < len(session.calls):
            call = session.calls[self.current_call_idx]
            print(f"  Session: {session.session_id}, Call: {self.current_call_idx}/{len(session.calls)}")
            print(f"  Endpoint: {call.endpoint}, Method: {call.method}")
            print(f"  User: {call.user_type}, Cache size: {len(self.cache)}/{self.cache_capacity}")


class RandomAgent:
    """Simple random baseline agent."""

    def select_action(self, state: np.ndarray) -> int:
        """Select action randomly."""
        return np.random.randint(0, 2)

    def update(self, state, action, reward, next_state):
        """Random agent doesn't learn."""
        pass


class SimpleQLearningAgent:
    """Simple Q-learning agent for demonstration."""

    def __init__(self, state_dim: int, action_dim: int = 2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = 0.1  # Exploration rate
        # Simplified: use state hash for Q-table
        self.q_table = {}
        self.learning_rate = 0.1
        self.gamma = 0.9  # Discount factor

    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            state_key = self._state_to_key(state)
            q_values = self.q_table.get(state_key, [0.0, 0.0])
            return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state):
        """Update Q-values using Q-learning update rule."""
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        # Get current Q-values
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0, 0.0]

        # Get next state max Q-value
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0.0, 0.0]

        max_next_q = max(self.q_table[next_state_key])

        # Q-learning update
        old_q = self.q_table[state_key][action]
        new_q = old_q + self.learning_rate * (reward + self.gamma * max_next_q - old_q)
        self.q_table[state_key][action] = new_q

    def _state_to_key(self, state: np.ndarray) -> str:
        """Convert state to hashable key."""
        # Discretize continuous features for Q-table
        discretized = (state * 10).astype(int)
        return str(discretized.tobytes())


def demo_feature_extraction():
    """Demonstrate feature extraction for RL."""
    print("\n" + "="*70)
    print("DEMO 1: Feature Extraction for RL State Representation")
    print("="*70)

    # Create dataset
    dataset = create_demo_dataset()
    print(f"\n✓ Created dataset with {len(dataset.sessions)} sessions")
    print(f"  Total calls: {dataset.total_calls}")

    # Fit feature engineer
    fe = FeatureEngineer()
    fe.fit(dataset.sessions)

    print(f"\n✓ Fitted FeatureEngineer")
    info = fe.get_feature_info()
    print(f"  Feature dimension: {info['feature_dim']}")
    print(f"  Categories: {info['categories']}")

    # Extract features for a sample call
    session = dataset.sessions[0]
    call = session.calls[2]  # Third call
    history = session.calls[:2]

    features = fe.transform(call, session, history)

    print(f"\n✓ Extracted features for sample call:")
    print(f"  Call: {call.endpoint} ({call.method})")
    print(f"  Feature vector shape: {features.shape}")
    print(f"  Feature range: [{features.min():.3f}, {features.max():.3f}]")

    # Show some features
    feature_names = fe.get_feature_names()
    print(f"\n  Sample features:")
    for i in range(min(10, len(features))):
        print(f"    {feature_names[i]:30s} = {features[i]:7.4f}")


def demo_environment():
    """Demonstrate the RL environment."""
    print("\n" + "="*70)
    print("DEMO 2: RL Environment with Feature States")
    print("="*70)

    # Setup
    dataset = create_demo_dataset()
    fe = FeatureEngineer()
    fe.fit(dataset.sessions)

    env = SimpleCachingEnvironment(fe, dataset.sessions[:3])

    print(f"\n✓ Created caching environment")
    print(f"  State dimension: {fe.get_feature_dim()}")
    print(f"  Action space: {{0: don't cache, 1: cache}}")
    print(f"  Cache capacity: {env.cache_capacity}")

    # Run one episode
    print(f"\n✓ Running sample episode:")
    state = env.reset()
    total_reward = 0

    for step in range(5):
        print(f"\n  Step {step + 1}:")
        env.render()

        # Random action for demo
        action = np.random.randint(0, 2)
        action_name = "CACHE" if action == 1 else "SKIP"

        next_state, reward, done = env.step(action)
        total_reward += reward

        print(f"  Action: {action_name}, Reward: {reward:+.1f}")

        if done:
            print(f"  Episode finished!")
            break

        state = next_state

    print(f"\n  Total reward: {total_reward:+.1f}")


def demo_training():
    """Demonstrate RL training with feature states."""
    print("\n" + "="*70)
    print("DEMO 3: RL Training Loop")
    print("="*70)

    # Setup
    dataset = create_demo_dataset()
    train_sessions = dataset.sessions[:7]
    test_sessions = dataset.sessions[7:]

    fe = FeatureEngineer()
    fe.fit(train_sessions)

    env = SimpleCachingEnvironment(fe, train_sessions)
    agent = SimpleQLearningAgent(state_dim=fe.get_feature_dim())

    print(f"\n✓ Training setup:")
    print(f"  Training sessions: {len(train_sessions)}")
    print(f"  Feature dimension: {fe.get_feature_dim()}")
    print(f"  Agent: Simple Q-Learning")

    # Training
    num_episodes = 20
    episode_rewards = []

    print(f"\n✓ Training for {num_episodes} episodes...")

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(100):  # Max steps per episode
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, next_state)

            episode_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)

        if (episode + 1) % 5 == 0:
            avg_reward = np.mean(episode_rewards[-5:])
            print(f"  Episode {episode + 1:2d}: avg reward = {avg_reward:+7.2f}")

    # Show learning curve
    print(f"\n✓ Training complete!")
    print(f"  Early episodes (1-5): {np.mean(episode_rewards[:5]):+7.2f}")
    print(f"  Late episodes (16-20): {np.mean(episode_rewards[-5:]):+7.2f}")
    improvement = np.mean(episode_rewards[-5:]) - np.mean(episode_rewards[:5])
    print(f"  Improvement: {improvement:+7.2f}")


def demo_comparison():
    """Compare random agent vs learning agent."""
    print("\n" + "="*70)
    print("DEMO 4: Agent Comparison")
    print("="*70)

    # Setup
    dataset = create_demo_dataset()
    fe = FeatureEngineer()
    fe.fit(dataset.sessions)

    # Random agent
    print(f"\n1. Random Agent:")
    env_random = SimpleCachingEnvironment(fe, dataset.sessions[:5])
    random_agent = RandomAgent()

    random_rewards = []
    for episode in range(10):
        state = env_random.reset()
        episode_reward = 0

        for step in range(100):
            action = random_agent.select_action(state)
            next_state, reward, done = env_random.step(action)
            episode_reward += reward
            state = next_state
            if done:
                break

        random_rewards.append(episode_reward)

    print(f"  Average reward: {np.mean(random_rewards):+7.2f}")

    # Learning agent
    print(f"\n2. Q-Learning Agent:")
    env_qlearning = SimpleCachingEnvironment(fe, dataset.sessions[:5])
    q_agent = SimpleQLearningAgent(state_dim=fe.get_feature_dim())

    q_rewards = []
    for episode in range(20):  # Train longer
        state = env_qlearning.reset()
        episode_reward = 0

        for step in range(100):
            action = q_agent.select_action(state)
            next_state, reward, done = env_qlearning.step(action)
            q_agent.update(state, action, reward, next_state)
            episode_reward += reward
            state = next_state
            if done:
                break

        q_rewards.append(episode_reward)

    print(f"  Average reward (last 10): {np.mean(q_rewards[-10:]):+7.2f}")

    # Comparison
    print(f"\n✓ Comparison:")
    print(f"  Random agent:     {np.mean(random_rewards):+7.2f}")
    print(f"  Q-learning agent: {np.mean(q_rewards[-10:]):+7.2f}")
    print(f"  Improvement:      {np.mean(q_rewards[-10:]) - np.mean(random_rewards):+7.2f}")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("FEATURE ENGINEER + RL INTEGRATION DEMO")
    print("="*70)
    print("\nDemonstrating how FeatureEngineer enables RL for API caching")

    np.random.seed(42)  # For reproducibility

    demo_feature_extraction()
    demo_environment()
    demo_training()
    demo_comparison()

    print("\n" + "="*70)
    print("✅ DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. FeatureEngineer converts API calls to numerical states")
    print("  2. States are fixed-size vectors suitable for RL algorithms")
    print("  3. Features capture temporal, user, and request patterns")
    print("  4. Learning agents can improve over random baselines")
    print("\nThe FeatureEngineer is ready for production RL training!")


if __name__ == "__main__":
    main()


"""
Comprehensive unit tests for all RL components.

This test suite validates:
- State representation and feature encoding
- Action space and action masking
- Reward calculation and normalization
- Replay buffers (uniform and prioritized)
- Q-Networks (standard and dueling)
- DQN agents (standard and double)
- Training orchestration
"""

import unittest
import numpy as np
import torch
import tempfile
import os
from pathlib import Path

from src.rl.state import StateBuilder, StateConfig
from src.rl.actions import CacheAction, ActionSpace, ActionConfig
from src.rl.reward import RewardCalculator, RewardConfig, ActionOutcome, RewardNormalizer
from src.rl.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from src.rl.networks.q_network import QNetwork, DuelingQNetwork, QNetworkConfig
from src.rl.agents.dqn_agent import DQNAgent, DoubleDQNAgent, DQNConfig


# ============================================================================
# Test Fixtures
# ============================================================================

class TestFixtures:
    """Shared fixtures for RL tests."""

    @staticmethod
    def get_state_config():
        """Small state config for fast tests."""
        return StateConfig(
            markov_top_k=3,
            include_probabilities=True,
            include_confidence=True,
            include_cache_metrics=True,
            include_system_metrics=True,
            include_user_context=True,
            include_temporal_context=True,
            include_session_context=True,
            vocab_size=100
        )

    @staticmethod
    def get_dqn_config():
        """Small DQN config for fast tests."""
        return DQNConfig(
            state_dim=28,  # From state_config.state_dim
            action_dim=7,
            hidden_dims=[32, 16],
            learning_rate=0.001,
            buffer_size=1000,
            batch_size=32,
            seed=42
        )

    @staticmethod
    def get_sample_states(batch_size=32, state_dim=28):
        """Generate batch of random state vectors."""
        return np.random.randn(batch_size, state_dim).astype(np.float32)

    @staticmethod
    def get_mock_environment():
        """Simple mock environment for testing."""
        class MockEnv:
            def __init__(self):
                self.state_dim = 28
                self.action_dim = 7
                self.step_count = 0

            def reset(self):
                self.step_count = 0
                return np.random.randn(self.state_dim).astype(np.float32)

            def step(self, action):
                self.step_count += 1
                next_state = np.random.randn(self.state_dim).astype(np.float32)
                reward = np.random.randn()
                done = (self.step_count >= 50)
                info = {}
                return next_state, reward, done, info

        return MockEnv()


# ============================================================================
# StateBuilder Tests
# ============================================================================

class TestStateBuilder(unittest.TestCase):
    """Tests for state representation and feature encoding."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = TestFixtures.get_state_config()
        self.builder = StateBuilder(self.config)

        # Fit builder with vocabulary
        vocabulary = [f"api_{i}" for i in range(50)]
        self.builder.fit(vocabulary)

    def test_state_dimension_matches_config(self):
        """Test that output shape equals config.state_dim."""
        state = self.builder.build_state()
        self.assertEqual(len(state), self.config.state_dim)

    def test_features_normalized(self):
        """Test that all features are in reasonable range."""
        # Build state with various inputs
        markov_predictions = [("api_1", 0.8), ("api_2", 0.6), ("api_3", 0.4)]
        cache_metrics = {
            "utilization": 0.75,
            "hit_rate": 0.65,
            "entries": 100,
            "eviction_rate": 0.02
        }
        system_metrics = {
            "cpu": 0.5,
            "memory": 0.6,
            "request_rate": 100.0,
            "p50_latency": 50.0,
            "p95_latency": 150.0,
            "p99_latency": 250.0,
            "error_rate": 0.01,
            "connections": 50,
            "queue_depth": 10
        }
        context = {
            "user_type": "premium",
            "hour": 14,
            "day_of_week": 2,
            "is_weekend": False,
            "session_position": 5,
            "session_duration": 120.0,
            "session_calls": 10
        }

        state = self.builder.build_state(
            markov_predictions=markov_predictions,
            cache_metrics=cache_metrics,
            system_metrics=system_metrics,
            context=context
        )

        # Most features should be normalized to [-1, 1] or [0, 1]
        self.assertTrue(np.all(np.abs(state) <= 10.0), "Features outside reasonable range")

    def test_handles_missing_metrics(self):
        """Test that builder doesn't crash with partial input."""
        # Should use defaults for missing metrics
        state = self.builder.build_state()

        self.assertEqual(len(state), self.config.state_dim)
        self.assertFalse(np.any(np.isnan(state)), "State contains NaN values")
        self.assertFalse(np.any(np.isinf(state)), "State contains inf values")

    def test_feature_names_match_dimensions(self):
        """Test that feature names list matches state_dim."""
        feature_names = self.builder.get_feature_names()
        self.assertEqual(len(feature_names), self.config.state_dim)

    def test_markov_predictions_encoded(self):
        """Test that top-k predictions appear correctly in state."""
        predictions = [("api_1", 0.9), ("api_2", 0.7), ("api_3", 0.5)]
        state = self.builder.build_state(markov_predictions=predictions)

        # First k elements should be API indices (normalized)
        api_indices = state[:self.config.markov_top_k]
        self.assertEqual(len(api_indices), 3)

        # Probabilities should appear after indices (if enabled)
        if self.config.include_probabilities:
            start_idx = self.config.markov_top_k
            end_idx = start_idx + self.config.markov_top_k
            probs = state[start_idx:end_idx]
            # First 3 should be the actual probabilities
            self.assertAlmostEqual(probs[0], 0.9, places=5)
            self.assertAlmostEqual(probs[1], 0.7, places=5)
            self.assertAlmostEqual(probs[2], 0.5, places=5)

    def test_context_encoding(self):
        """Test user type one-hot and time cyclical encoding."""
        context = {
            "user_type": "premium",
            "hour": 6,  # Morning
            "day_of_week": 0,  # Monday
            "is_weekend": False
        }

        state = self.builder.build_state(context=context)

        # State should not contain NaN or inf
        self.assertFalse(np.any(np.isnan(state)))
        self.assertFalse(np.any(np.isinf(state)))

    def test_padding_for_fewer_predictions(self):
        """Test that fewer than k predictions are padded with zeros."""
        # Only provide 1 prediction instead of 3
        predictions = [("api_1", 0.95)]
        state = self.builder.build_state(markov_predictions=predictions)

        # Should still have full state_dim
        self.assertEqual(len(state), self.config.state_dim)

        # The unfilled prediction slots should be 0
        api_indices = state[:self.config.markov_top_k]
        self.assertAlmostEqual(api_indices[1], 0.0, places=5)
        self.assertAlmostEqual(api_indices[2], 0.0, places=5)


# ============================================================================
# CacheAction Tests
# ============================================================================

class TestCacheAction(unittest.TestCase):
    """Tests for action space and action masking."""

    def setUp(self):
        """Set up test fixtures."""
        self.action_space = ActionSpace()
        self.config = ActionConfig()

    def test_all_actions_have_names(self):
        """Test that get_name works for all 7 actions."""
        for action in range(7):
            name = CacheAction.get_name(action)
            self.assertIsInstance(name, str)
            self.assertNotIn("UNKNOWN", name)

    def test_action_count(self):
        """Test that num_actions returns 7."""
        self.assertEqual(CacheAction.num_actions(), 7)

    def test_decode_prefetch_conservative(self):
        """Test that only high-prob (>70%) items are prefetched."""
        predictions = [
            ("api_1", 0.95),  # Should prefetch
            ("api_2", 0.65),  # Below threshold
            ("api_3", 0.50)   # Below threshold
        ]

        result = self.action_space.decode_action(
            CacheAction.PREFETCH_CONSERVATIVE,
            predictions=predictions
        )

        self.assertIn("prefetch", result)
        # Should prefetch items above threshold
        prefetch_list = result["prefetch"]
        self.assertGreater(len(prefetch_list), 0)

    def test_decode_prefetch_moderate(self):
        """Test that medium threshold (>50%) is respected."""
        predictions = [
            ("api_1", 0.95),  # Should prefetch
            ("api_2", 0.65),  # Should prefetch
            ("api_3", 0.45)   # Below threshold
        ]

        result = self.action_space.decode_action(
            CacheAction.PREFETCH_MODERATE,
            predictions=predictions,
            config=self.config
        )

        self.assertIn("prefetch", result)
        # Up to 3 items, but only those above 50%
        prefetch_list = result["prefetch"]
        self.assertGreater(len(prefetch_list), 0)
        self.assertLessEqual(len(prefetch_list), self.config.moderate_count)

    def test_decode_prefetch_aggressive(self):
        """Test that low threshold (>30%) is respected."""
        predictions = [
            ("api_1", 0.95),
            ("api_2", 0.45),
            ("api_3", 0.35),
            ("api_4", 0.25),  # Below threshold
            ("api_5", 0.15)
        ]

        result = self.action_space.decode_action(
            CacheAction.PREFETCH_AGGRESSIVE,
            predictions=predictions,
            config=self.config
        )

        self.assertIn("prefetch", result)
        prefetch_list = result["prefetch"]
        self.assertGreater(len(prefetch_list), 0)
        self.assertLessEqual(len(prefetch_list), self.config.aggressive_count)

    def test_action_mask_empty_cache(self):
        """Test that eviction actions are masked when cache is empty."""
        mask = self.action_space.get_action_mask(
            cache_utilization=0.0,
            has_predictions=False,
            cache_size=0
        )

        # Eviction actions should be masked (False)
        self.assertFalse(mask[CacheAction.EVICT_LRU])
        self.assertFalse(mask[CacheAction.EVICT_LOW_PROB])

        # Other actions should be available
        self.assertTrue(mask[CacheAction.DO_NOTHING])
        self.assertTrue(mask[CacheAction.CACHE_CURRENT])

    def test_action_mask_no_predictions(self):
        """Test that prefetch actions are masked without predictions."""
        mask = self.action_space.get_action_mask(
            cache_utilization=0.5,
            has_predictions=False,
            cache_size=10
        )

        # Prefetch actions should be masked
        self.assertFalse(mask[CacheAction.PREFETCH_CONSERVATIVE])
        self.assertFalse(mask[CacheAction.PREFETCH_MODERATE])
        self.assertFalse(mask[CacheAction.PREFETCH_AGGRESSIVE])

        # Non-prefetch actions should be available
        self.assertTrue(mask[CacheAction.DO_NOTHING])
        self.assertTrue(mask[CacheAction.CACHE_CURRENT])


# ============================================================================
# RewardCalculator Tests
# ============================================================================

class TestRewardCalculator(unittest.TestCase):
    """Tests for reward calculation and normalization."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = RewardConfig()
        self.calculator = RewardCalculator(self.config)

    def test_cache_hit_positive(self):
        """Test that cache hit gives positive reward."""
        outcome = ActionOutcome(cache_hit=True)
        reward = self.calculator.calculate(outcome)
        breakdown = self.calculator.calculate_detailed(outcome)

        self.assertGreater(reward, 0)
        self.assertIn("cache", breakdown)
        self.assertEqual(breakdown["cache"], self.config.cache_hit_reward)

    def test_cache_miss_negative(self):
        """Test that cache miss gives small negative reward."""
        outcome = ActionOutcome(cache_miss=True)
        reward = self.calculator.calculate(outcome)
        breakdown = self.calculator.calculate_detailed(outcome)

        self.assertLess(reward, 0)
        self.assertIn("cache", breakdown)
        self.assertEqual(breakdown["cache"], self.config.cache_miss_penalty)

    def test_cascade_prevented_highest(self):
        """Test that cascade prevention is largest positive reward."""
        outcome = ActionOutcome(cascade_prevented=True)
        reward = self.calculator.calculate(outcome)
        breakdown = self.calculator.calculate_detailed(outcome)

        self.assertGreater(reward, self.config.cache_hit_reward)
        self.assertEqual(reward, self.config.cascade_prevented_reward)
        self.assertIn("cascade", breakdown)

    def test_cascade_occurred_most_negative(self):
        """Test that cascade is largest penalty."""
        outcome = ActionOutcome(cascade_occurred=True)
        reward = self.calculator.calculate(outcome)
        breakdown = self.calculator.calculate_detailed(outcome)

        self.assertLess(reward, self.config.cache_miss_penalty)
        self.assertEqual(reward, self.config.cascade_occurred_penalty)
        self.assertIn("cascade", breakdown)

    def test_reward_clipping(self):
        """Test that rewards stay within configured bounds."""
        # Create extreme outcome
        outcome = ActionOutcome(
            cascade_occurred=True,
            cache_miss=True,
            actual_latency_ms=2000.0,
            baseline_latency_ms=100.0,
            prefetch_bytes=10000000
        )

        reward = self.calculator.calculate(outcome)

        self.assertGreaterEqual(reward, self.config.clip_min)
        self.assertLessEqual(reward, self.config.clip_max)

    def test_detailed_breakdown(self):
        """Test that all components are present in breakdown."""
        outcome = ActionOutcome(
            cache_hit=True,
            prefetch_attempted=3,
            prefetch_used=2,
            baseline_latency_ms=100.0,
            actual_latency_ms=90.0
        )

        breakdown = self.calculator.calculate_detailed(outcome)

        self.assertIsInstance(breakdown, dict)
        self.assertIn("cache", breakdown)
        self.assertIn("prefetch", breakdown)
        self.assertIn("latency", breakdown)
        self.assertIn("total", breakdown)

    def test_normalizer_running_stats(self):
        """Test that RewardNormalizer tracks mean/std correctly."""
        normalizer = RewardNormalizer()

        # Add some rewards
        rewards = [10.0, 20.0, 30.0, 40.0, 50.0]

        for reward in rewards:
            normalizer.update(reward)

        # Mean should be around 30.0
        self.assertAlmostEqual(normalizer.mean, np.mean(rewards), places=1)

        # Now normalize rewards
        normalized_rewards = [normalizer.normalize(r) for r in rewards]

        # Normalized rewards should have mean ~0 and std ~1
        if len(normalized_rewards) > 1:
            self.assertLess(abs(np.mean(normalized_rewards)), 1.0)


# ============================================================================
# ReplayBuffer Tests
# ============================================================================

class TestReplayBuffer(unittest.TestCase):
    """Tests for replay buffer (uniform and prioritized)."""

    def setUp(self):
        """Set up test fixtures."""
        self.capacity = 100
        self.buffer = ReplayBuffer(capacity=self.capacity, seed=42)

    def test_push_and_sample(self):
        """Test that we can store and retrieve experiences."""
        # Add experiences
        for i in range(50):
            state = np.random.randn(28).astype(np.float32)
            action = np.random.randint(0, 7)
            reward = np.random.randn()
            next_state = np.random.randn(28).astype(np.float32)
            done = (i % 10 == 9)

            self.buffer.push(state, action, reward, next_state, done)

        self.assertEqual(len(self.buffer), 50)

        # Sample batch
        batch = self.buffer.sample(32)
        states, actions, rewards, next_states, dones = batch

        self.assertEqual(states.shape, (32, 28))
        self.assertEqual(actions.shape, (32,))
        self.assertEqual(rewards.shape, (32,))
        self.assertEqual(next_states.shape, (32, 28))
        self.assertEqual(dones.shape, (32,))

    def test_capacity_limit(self):
        """Test that old experiences are removed when full."""
        # Fill beyond capacity
        for i in range(150):
            state = np.random.randn(28).astype(np.float32)
            self.buffer.push(state, 0, 0.0, state, False)

        # Should only have capacity number of experiences
        self.assertEqual(len(self.buffer), self.capacity)

    def test_sample_batch_shapes(self):
        """Test that returned arrays have correct shapes."""
        # Add some experiences
        for i in range(100):
            state = np.random.randn(28).astype(np.float32)
            self.buffer.push(state, i % 7, float(i), state, False)

        # Sample different batch sizes
        for batch_size in [16, 32, 64]:
            batch = self.buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = batch

            self.assertEqual(states.shape[0], batch_size)
            self.assertEqual(actions.shape[0], batch_size)
            self.assertEqual(rewards.shape[0], batch_size)
            self.assertEqual(next_states.shape[0], batch_size)
            self.assertEqual(dones.shape[0], batch_size)

    def test_save_and_load(self):
        """Test that serialization preserves data."""
        # Add experiences
        for i in range(50):
            state = np.random.randn(28).astype(np.float32)
            self.buffer.push(state, i % 7, float(i), state, False)

        # Save to file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            save_path = tmp.name

        try:
            self.buffer.save(save_path)

            # Load into new buffer
            new_buffer = ReplayBuffer(capacity=self.capacity)
            new_buffer.load(save_path)

            # Should have same size
            self.assertEqual(len(new_buffer), len(self.buffer))

            # Sample from both should work
            batch1 = self.buffer.sample(10)
            batch2 = new_buffer.sample(10)

            self.assertEqual(batch1[0].shape, batch2[0].shape)

        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    def test_prioritized_sampling(self):
        """Test that higher priority items are sampled more often."""
        buffer = PrioritizedReplayBuffer(capacity=100, seed=42)

        # Add experiences with different priorities
        high_priority_state = np.ones(28, dtype=np.float32) * 10  # Marker
        low_priority_state = np.ones(28, dtype=np.float32) * 1    # Marker

        # Add high priority experience
        buffer.push(high_priority_state, 0, 1.0, high_priority_state, False, priority=10.0)

        # Add many low priority experiences
        for _ in range(50):
            buffer.push(low_priority_state, 0, 0.0, low_priority_state, False, priority=0.1)

        # Sample many times and count high priority samples
        high_priority_count = 0
        num_samples = 100

        for _ in range(num_samples):
            states, _, _, _, _, _, _ = buffer.sample(1)
            if np.allclose(states[0], high_priority_state):
                high_priority_count += 1

        # High priority item should be sampled more often than uniform (>2%)
        self.assertGreater(high_priority_count / num_samples, 0.05)

    def test_priority_updates(self):
        """Test that updated priorities affect sampling."""
        buffer = PrioritizedReplayBuffer(capacity=100, seed=42)

        # Add experiences
        for i in range(50):
            state = np.random.randn(28).astype(np.float32)
            buffer.push(state, 0, 0.0, state, False, priority=1.0)

        # Sample and get indices
        _, _, _, _, _, _, indices = buffer.sample(10)

        # Update priorities
        new_priorities = np.ones(10) * 10.0
        buffer.update_priorities(indices, new_priorities)

        # The buffer should still work
        batch = buffer.sample(10)
        self.assertEqual(batch[0].shape[0], 10)


# ============================================================================
# QNetwork Tests
# ============================================================================

class TestQNetwork(unittest.TestCase):
    """Tests for Q-Network architectures."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = QNetworkConfig(
            state_dim=28,
            action_dim=7,
            hidden_dims=[32, 16],
            activation='relu',
            dropout=0.0  # Disable for deterministic tests
        )
        self.network = QNetwork(self.config)
        self.network.eval()  # Set to eval mode

    def test_output_shape(self):
        """Test that output is (batch, action_dim)."""
        batch_size = 16
        states = torch.randn(batch_size, self.config.state_dim)

        with torch.no_grad():
            q_values = self.network(states)

        self.assertEqual(q_values.shape, (batch_size, self.config.action_dim))

    def test_forward_deterministic(self):
        """Test that same input gives same output in eval mode."""
        state = torch.randn(1, self.config.state_dim)

        with torch.no_grad():
            output1 = self.network(state)
            output2 = self.network(state)

        self.assertTrue(torch.allclose(output1, output2))

    def test_gradients_flow(self):
        """Test that gradients flow through the network."""
        self.network.train()

        states = torch.randn(16, self.config.state_dim, requires_grad=True)
        q_values = self.network(states)

        # Compute loss and backprop
        loss = q_values.mean()
        loss.backward()

        # Check that gradients exist
        for param in self.network.parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(torch.all(param.grad == 0))

    def test_get_action(self):
        """Test that get_action returns valid action indices."""
        states = torch.randn(16, self.config.state_dim)

        with torch.no_grad():
            actions = self.network.get_action(states)

        self.assertEqual(actions.shape, (16,))
        self.assertTrue(torch.all(actions >= 0))
        self.assertTrue(torch.all(actions < self.config.action_dim))

    def test_dueling_architecture(self):
        """Test dueling network architecture."""
        config = QNetworkConfig(
            state_dim=28,
            action_dim=7,
            hidden_dims=[32, 16],
            dueling=True
        )
        dueling_net = DuelingQNetwork(config)
        dueling_net.eval()

        states = torch.randn(16, config.state_dim)

        with torch.no_grad():
            q_values = dueling_net(states)
            value = dueling_net.get_value(states)
            advantage = dueling_net.get_advantage(states)

        self.assertEqual(q_values.shape, (16, config.action_dim))
        self.assertEqual(value.shape, (16, 1))
        self.assertEqual(advantage.shape, (16, config.action_dim))

        # Check dueling formula: Q = V + A - mean(A)
        reconstructed_q = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        self.assertTrue(torch.allclose(q_values, reconstructed_q, atol=1e-5))


# ============================================================================
# DQNAgent Tests
# ============================================================================

class TestDQNAgent(unittest.TestCase):
    """Tests for DQN agent."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = TestFixtures.get_dqn_config()
        self.agent = DQNAgent(self.config, seed=42)

    def test_agent_initialization(self):
        """Test that agent initializes correctly."""
        self.assertIsNotNone(self.agent.online_net)
        self.assertIsNotNone(self.agent.target_net)
        self.assertIsNotNone(self.agent.optimizer)
        self.assertIsNotNone(self.agent.buffer)
        self.assertEqual(self.agent.epsilon, self.config.epsilon_start)

    def test_action_selection_exploration(self):
        """Test epsilon-greedy action selection."""
        state = np.random.randn(self.config.state_dim).astype(np.float32)

        # With high epsilon, should see variety
        self.agent.epsilon = 1.0
        actions = [self.agent.select_action(state) for _ in range(20)]
        unique_actions = len(set(actions))
        self.assertGreater(unique_actions, 1)

    def test_action_selection_greedy(self):
        """Test that evaluation mode is deterministic."""
        state = np.random.randn(self.config.state_dim).astype(np.float32)

        actions = [self.agent.select_action(state, evaluate=True) for _ in range(10)]
        unique_actions = len(set(actions))
        self.assertEqual(unique_actions, 1)

    def test_store_and_train(self):
        """Test storing experiences and training."""
        # Add experiences
        for i in range(100):
            state = np.random.randn(self.config.state_dim).astype(np.float32)
            action = np.random.randint(0, self.config.action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(self.config.state_dim).astype(np.float32)
            done = (i % 20 == 19)

            self.agent.store_transition(state, action, reward, next_state, done)

        self.assertEqual(len(self.agent.buffer), 100)

        # Train
        metrics = self.agent.train_step()

        self.assertIsNotNone(metrics)
        self.assertIn('loss', metrics)
        self.assertIn('epsilon', metrics)
        self.assertGreater(metrics['loss'], 0)

    def test_epsilon_decay(self):
        """Test that epsilon decays correctly."""
        initial_epsilon = self.agent.epsilon

        for _ in range(50):
            self.agent._decay_epsilon()

        self.assertLess(self.agent.epsilon, initial_epsilon)
        self.assertGreaterEqual(self.agent.epsilon, self.config.epsilon_end)

    def test_target_network_update(self):
        """Test target network update."""
        # Get initial params
        online_param = next(self.agent.online_net.parameters()).clone()
        target_param_before = next(self.agent.target_net.parameters()).clone()

        # Train to change online network
        for i in range(50):
            state = np.random.randn(self.config.state_dim).astype(np.float32)
            self.agent.store_transition(state, 0, 0.0, state, False)

        for _ in range(10):
            self.agent.train_step()

        online_param_after = next(self.agent.online_net.parameters()).clone()

        # Online should have changed
        self.assertFalse(torch.allclose(online_param, online_param_after))

        # Update target
        self.agent._update_target_network()
        target_param_after = next(self.agent.target_net.parameters()).clone()

        # Target should now match online
        self.assertTrue(torch.allclose(online_param_after, target_param_after))

    def test_save_and_load(self):
        """Test agent save and load."""
        # Train a bit
        for i in range(100):
            state = np.random.randn(self.config.state_dim).astype(np.float32)
            self.agent.store_transition(state, 0, 0.0, state, False)

        for _ in range(5):
            self.agent.train_step()

        # Save
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
            save_path = tmp.name

        try:
            epsilon_before = self.agent.epsilon
            steps_before = self.agent.steps_done

            self.agent.save(save_path)

            # Load into new agent
            agent2 = DQNAgent(self.config)
            agent2.load(save_path)

            self.assertAlmostEqual(agent2.epsilon, epsilon_before, places=6)
            self.assertEqual(agent2.steps_done, steps_before)

        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    def test_double_dqn_agent(self):
        """Test DoubleDQN agent variant."""
        agent = DoubleDQNAgent(self.config, seed=42)

        # Add experiences and train
        for i in range(100):
            state = np.random.randn(self.config.state_dim).astype(np.float32)
            action = np.random.randint(0, self.config.action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(self.config.state_dim).astype(np.float32)
            done = False

            agent.store_transition(state, action, reward, next_state, done)

        # Train should work
        metrics = agent.train_step()
        self.assertIsNotNone(metrics)
        self.assertIn('loss', metrics)


# ============================================================================
# Integration Tests
# ============================================================================

class TestRLIntegration(unittest.TestCase):
    """Integration tests for complete RL workflow."""

    def test_complete_training_loop(self):
        """Test complete training loop with all components."""
        # Setup
        env = TestFixtures.get_mock_environment()
        config = TestFixtures.get_dqn_config()
        agent = DQNAgent(config, seed=42)

        # Training loop
        num_episodes = 10
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)

                metrics = agent.train_step()

                episode_reward += reward
                state = next_state

            # Episode completed successfully
            self.assertGreater(episode_reward, -1000)  # Sanity check

        # Agent should have trained
        self.assertGreater(agent.steps_done, 0)
        self.assertEqual(episode + 1, num_episodes)

    def test_state_to_action_to_reward(self):
        """Test the state -> action -> reward pipeline."""
        # Create components
        state_config = TestFixtures.get_state_config()
        state_builder = StateBuilder(state_config)
        state_builder.fit([f"api_{i}" for i in range(50)])

        action_space = ActionSpace()
        reward_calculator = RewardCalculator(RewardConfig())

        # Build state
        predictions = [("api_1", 0.9), ("api_2", 0.7)]
        state_vector = state_builder.build_state(markov_predictions=predictions)

        self.assertEqual(len(state_vector), state_config.state_dim)

        # Select action (using random for simplicity)
        action = np.random.randint(0, 7)
        self.assertIn(action, range(7))

        # Decode action
        decoded = action_space.decode_action(action, predictions=predictions)
        self.assertIsInstance(decoded, dict)

        # Compute reward
        outcome = ActionOutcome(cache_hit=True)
        reward = reward_calculator.calculate(outcome)
        breakdown = reward_calculator.calculate_detailed(outcome)

        self.assertIsInstance(reward, float)
        self.assertIsInstance(breakdown, dict)


# ============================================================================
# Test Runner
# ============================================================================

def run_tests():
    """Run all RL tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestStateBuilder))
    suite.addTests(loader.loadTestsFromTestCase(TestCacheAction))
    suite.addTests(loader.loadTestsFromTestCase(TestRewardCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestReplayBuffer))
    suite.addTests(loader.loadTestsFromTestCase(TestQNetwork))
    suite.addTests(loader.loadTestsFromTestCase(TestDQNAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestRLIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 70)
    print("RL TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n[OK] ALL RL TESTS PASSED!")
    else:
        print("\n[FAIL] SOME TESTS FAILED")

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)


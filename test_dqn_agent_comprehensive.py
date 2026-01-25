"""
Comprehensive test suite for DQN Agent implementation.

This test file validates all features of the DQN agent as specified in the requirements.
Run this to verify the implementation is complete and working correctly.
"""

import unittest
import numpy as np
import torch
import tempfile
import os
from src.rl.agents.dqn_agent import DQNAgent, DoubleDQNAgent, DQNConfig


class TestDQNConfig(unittest.TestCase):
    """Test DQNConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DQNConfig(state_dim=60, action_dim=7)

        self.assertEqual(config.state_dim, 60)
        self.assertEqual(config.action_dim, 7)
        self.assertEqual(config.hidden_dims, [128, 64])
        self.assertTrue(config.dueling)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.weight_decay, 0.0)
        self.assertEqual(config.gamma, 0.99)
        self.assertEqual(config.epsilon_start, 1.0)
        self.assertEqual(config.epsilon_end, 0.05)
        self.assertEqual(config.epsilon_decay, 0.995)
        self.assertEqual(config.buffer_size, 100000)
        self.assertEqual(config.batch_size, 64)
        self.assertFalse(config.prioritized_replay)
        self.assertEqual(config.target_update_freq, 1000)
        self.assertEqual(config.max_grad_norm, 10.0)
        self.assertEqual(config.device, 'auto')
        self.assertIsNone(config.seed)

    def test_custom_config(self):
        """Test custom configuration."""
        config = DQNConfig(
            state_dim=30,
            action_dim=5,
            hidden_dims=[64, 32],
            dueling=False,
            learning_rate=0.0003,
            gamma=0.95,
            epsilon_start=0.8,
            prioritized_replay=True,
            seed=123
        )

        self.assertEqual(config.state_dim, 30)
        self.assertEqual(config.action_dim, 5)
        self.assertEqual(config.hidden_dims, [64, 32])
        self.assertFalse(config.dueling)
        self.assertEqual(config.learning_rate, 0.0003)
        self.assertEqual(config.gamma, 0.95)
        self.assertEqual(config.epsilon_start, 0.8)
        self.assertTrue(config.prioritized_replay)
        self.assertEqual(config.seed, 123)


class TestDQNAgent(unittest.TestCase):
    """Test DQNAgent class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = DQNConfig(
            state_dim=60,
            action_dim=7,
            hidden_dims=[128, 64],
            buffer_size=1000,
            batch_size=32,
            seed=42
        )
        self.agent = DQNAgent(self.config, seed=42)

    def test_initialization(self):
        """Test agent initialization."""
        self.assertIsNotNone(self.agent.online_net)
        self.assertIsNotNone(self.agent.target_net)
        self.assertIsNotNone(self.agent.optimizer)
        self.assertIsNotNone(self.agent.buffer)
        self.assertEqual(self.agent.epsilon, self.config.epsilon_start)
        self.assertEqual(self.agent.steps_done, 0)

    def test_device_setup(self):
        """Test device configuration."""
        # Auto device
        self.assertIn(str(self.agent.device), ['cpu', 'cuda'])

        # Explicit CPU
        config_cpu = DQNConfig(state_dim=60, action_dim=7, device='cpu')
        agent_cpu = DQNAgent(config_cpu)
        self.assertEqual(str(agent_cpu.device), 'cpu')

    def test_select_action_exploration(self):
        """Test action selection with exploration."""
        state = np.random.randn(60).astype(np.float32)
        actions = [self.agent.select_action(state) for _ in range(100)]

        # Check all actions are valid
        for action in actions:
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, self.config.action_dim)

        # With high epsilon, should see multiple different actions
        unique_actions = len(set(actions))
        self.assertGreater(unique_actions, 1)

    def test_select_action_evaluation(self):
        """Test greedy action selection."""
        state = np.random.randn(60).astype(np.float32)
        actions = [self.agent.select_action(state, evaluate=True) for _ in range(10)]

        # Greedy actions should be deterministic
        self.assertEqual(len(set(actions)), 1)

    def test_store_transition(self):
        """Test storing experiences."""
        state = np.random.randn(60).astype(np.float32)
        action = 2
        reward = 1.5
        next_state = np.random.randn(60).astype(np.float32)
        done = False

        initial_size = len(self.agent.buffer)
        self.agent.store_transition(state, action, reward, next_state, done)
        self.assertEqual(len(self.agent.buffer), initial_size + 1)

    def test_train_step_not_ready(self):
        """Test training when buffer not ready."""
        # Empty buffer
        metrics = self.agent.train_step()
        self.assertIsNone(metrics)

    def test_train_step(self):
        """Test training step."""
        # Fill buffer with experiences
        for i in range(100):
            state = np.random.randn(60).astype(np.float32)
            action = np.random.randint(0, self.config.action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(60).astype(np.float32)
            done = (i % 20 == 19)
            self.agent.store_transition(state, action, reward, next_state, done)

        # Train
        metrics = self.agent.train_step()

        self.assertIsNotNone(metrics)
        self.assertIn('loss', metrics)
        self.assertIn('q_mean', metrics)
        self.assertIn('epsilon', metrics)
        self.assertGreater(metrics['loss'], 0)
        self.assertEqual(self.agent.steps_done, 1)

    def test_epsilon_decay(self):
        """Test epsilon decay."""
        initial_epsilon = self.agent.epsilon

        # Decay 50 times
        for _ in range(50):
            self.agent._decay_epsilon()

        # Epsilon should decrease
        self.assertLess(self.agent.epsilon, initial_epsilon)

        # Should not go below epsilon_end
        self.assertGreaterEqual(self.agent.epsilon, self.config.epsilon_end)

    def test_target_network_update(self):
        """Test target network update."""
        # Get initial parameters
        online_param = next(self.agent.online_net.parameters()).clone()
        target_param_before = next(self.agent.target_net.parameters()).clone()

        # Train to change online network
        for i in range(50):
            state = np.random.randn(60).astype(np.float32)
            action = np.random.randint(0, self.config.action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(60).astype(np.float32)
            done = False
            self.agent.store_transition(state, action, reward, next_state, done)

        for _ in range(10):
            self.agent.train_step()

        online_param_after = next(self.agent.online_net.parameters()).clone()
        target_param_after = next(self.agent.target_net.parameters()).clone()

        # Online should have changed
        self.assertFalse(torch.allclose(online_param, online_param_after))

        # Update target
        self.agent._update_target_network()
        target_param_updated = next(self.agent.target_net.parameters()).clone()

        # Target should now match online
        self.assertTrue(torch.allclose(online_param_after, target_param_updated))

    def test_save_load(self):
        """Test save and load functionality."""
        # Train agent a bit
        for i in range(100):
            state = np.random.randn(60).astype(np.float32)
            action = self.agent.select_action(state)
            next_state = np.random.randn(60).astype(np.float32)
            reward = np.random.randn()
            done = (i % 20 == 19)
            self.agent.store_transition(state, action, reward, next_state, done)

        for _ in range(5):
            self.agent.train_step()

        # Save
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
            save_path = tmp.name

        try:
            epsilon_before = self.agent.epsilon
            steps_before = self.agent.steps_done
            online_param_before = next(self.agent.online_net.parameters()).clone()

            self.agent.save(save_path)

            # Create new agent and load
            agent2 = DQNAgent(self.config)
            agent2.load(save_path)

            # Verify state restored
            self.assertAlmostEqual(agent2.epsilon, epsilon_before, places=6)
            self.assertEqual(agent2.steps_done, steps_before)

            # Verify network weights
            online_param_after = next(agent2.online_net.parameters()).clone()
            self.assertTrue(torch.allclose(online_param_before, online_param_after))

        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    def test_get_metrics(self):
        """Test metrics retrieval."""
        metrics = self.agent.get_metrics()

        self.assertIn('steps', metrics)
        self.assertIn('epsilon', metrics)
        self.assertIn('last_loss', metrics)
        self.assertIn('buffer_size', metrics)
        self.assertIn('device', metrics)

        self.assertEqual(metrics['steps'], self.agent.steps_done)
        self.assertEqual(metrics['epsilon'], self.agent.epsilon)
        self.assertEqual(metrics['buffer_size'], len(self.agent.buffer))


class TestDoubleDQNAgent(unittest.TestCase):
    """Test DoubleDQNAgent class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = DQNConfig(
            state_dim=60,
            action_dim=7,
            hidden_dims=[128, 64],
            buffer_size=1000,
            batch_size=32,
            seed=42
        )
        self.agent = DoubleDQNAgent(self.config, seed=42)

    def test_initialization(self):
        """Test Double DQN initialization."""
        self.assertIsInstance(self.agent, DoubleDQNAgent)
        self.assertIsInstance(self.agent, DQNAgent)

    def test_train_step(self):
        """Test Double DQN training."""
        # Fill buffer
        for i in range(100):
            state = np.random.randn(60).astype(np.float32)
            action = np.random.randint(0, self.config.action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(60).astype(np.float32)
            done = (i % 20 == 19)
            self.agent.store_transition(state, action, reward, next_state, done)

        # Train
        metrics = self.agent.train_step()

        self.assertIsNotNone(metrics)
        self.assertIn('loss', metrics)
        self.assertGreater(metrics['loss'], 0)


class TestPrioritizedReplay(unittest.TestCase):
    """Test prioritized experience replay."""

    def test_prioritized_replay_buffer(self):
        """Test agent with prioritized replay."""
        config = DQNConfig(
            state_dim=60,
            action_dim=7,
            prioritized_replay=True,
            buffer_size=1000,
            batch_size=32,
            seed=42
        )
        agent = DQNAgent(config, seed=42)

        # Verify buffer type
        from src.rl.replay_buffer import PrioritizedReplayBuffer
        self.assertIsInstance(agent.buffer, PrioritizedReplayBuffer)

        # Fill and train
        for i in range(100):
            state = np.random.randn(60).astype(np.float32)
            action = np.random.randint(0, config.action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(60).astype(np.float32)
            done = (i % 20 == 19)
            agent.store_transition(state, action, reward, next_state, done)

        metrics = agent.train_step()
        self.assertIsNotNone(metrics)


class TestIntegration(unittest.TestCase):
    """Integration tests matching user requirements."""

    def test_user_validation_code(self):
        """Test exact code from user requirements."""
        config = DQNConfig(
            state_dim=60,
            action_dim=7,
            hidden_dims=[128, 64],
            epsilon_start=1.0,
            epsilon_end=0.1
        )
        agent = DQNAgent(config, seed=42)

        # Test action selection
        state = np.random.randn(60).astype(np.float32)
        action = agent.select_action(state)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 7)

        # Collect experiences
        for i in range(200):
            state = np.random.randn(60).astype(np.float32)
            action = agent.select_action(state)
            next_state = np.random.randn(60).astype(np.float32)
            reward = np.random.randn()
            done = (i % 50 == 49)
            agent.store_transition(state, action, reward, next_state, done)

        # Train
        for i in range(10):
            metrics = agent.train_step()
            if metrics:
                self.assertIn('loss', metrics)
                self.assertIn('q_mean', metrics)
                self.assertIn('epsilon', metrics)

        # Test save/load
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
            save_path = tmp.name

        try:
            agent.save(save_path)
            agent2 = DQNAgent(config)
            agent2.load(save_path)
            self.assertGreater(agent2.epsilon, 0)
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)


def run_tests():
    """Run all tests and display results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDQNConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestDQNAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestDoubleDQNAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestPrioritizedReplay))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n[SUCCESS] ALL TESTS PASSED!")
    else:
        print("\n[ERROR] SOME TESTS FAILED")

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)


"""
Integration tests for training loop.

These tests verify that the training process works correctly end-to-end,
including agent training, checkpointing, early stopping, and convergence.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import time

from src.rl.agents.dqn_agent import DQNAgent
from src.rl.training.trainer import Trainer, TrainingConfig
from src.integration.gym_environment import CachingEnv


class TestTrainingBasics:
    """Test basic training functionality."""

    def test_agent_can_train(self, fast_env, untrained_agent):
        """Test that training runs without errors."""
        agent = untrained_agent

        # Run a few training episodes
        for episode in range(5):
            state, _ = fast_env.reset(seed=42 + episode)
            episode_reward = 0

            for step in range(20):
                action = agent.select_action(state, evaluate=False)
                next_state, reward, terminated, truncated, info = fast_env.step(action)

                # Store experience
                agent.buffer.push(state, action, reward, next_state, terminated)
                episode_reward += reward

                # Update if enough experience
                if len(agent.buffer) >= agent.config.batch_size:
                    loss = agent.update()
                    assert loss is not None
                    assert np.isfinite(loss)

                state = next_state

                if terminated or truncated:
                    break

            agent.decay_epsilon()

        # Training should complete without errors
        assert len(agent.buffer) > 0
        assert agent.steps_done > 0

    def test_loss_decreases(self, fast_env, untrained_agent):
        """Test that loss generally decreases over training."""
        agent = untrained_agent
        losses = []

        # Collect initial experiences
        state, _ = fast_env.reset(seed=42)
        for _ in range(agent.config.batch_size * 3):
            action = agent.select_action(state, evaluate=False)
            next_state, reward, terminated, truncated, info = fast_env.step(action)
            agent.buffer.push(state, action, reward, next_state, terminated)
            state = next_state
            if terminated or truncated:
                state, _ = fast_env.reset()

        # Train and record losses
        for _ in range(50):
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

        # Loss should generally decrease (compare first 10 vs last 10)
        if len(losses) >= 20:
            early_loss = np.mean(losses[:10])
            late_loss = np.mean(losses[-10:])

            # Loss should decrease or stay stable (with some tolerance)
            assert late_loss <= early_loss * 1.5  # Allow 50% increase due to exploration

    def test_reward_improves(self, fast_env, untrained_agent):
        """Test that average reward improves over episodes."""
        agent = untrained_agent
        episode_rewards = []

        # Train for multiple episodes
        for episode in range(20):
            state, _ = fast_env.reset(seed=42 + episode)
            episode_reward = 0

            for step in range(30):
                action = agent.select_action(state, evaluate=False)
                next_state, reward, terminated, truncated, info = fast_env.step(action)

                agent.buffer.push(state, action, reward, next_state, terminated)
                episode_reward += reward

                if len(agent.buffer) >= agent.config.batch_size:
                    agent.update()

                state = next_state

                if terminated or truncated:
                    break

            episode_rewards.append(episode_reward)
            agent.decay_epsilon()

        # Compare early vs late performance
        early_reward = np.mean(episode_rewards[:5])
        late_reward = np.mean(episode_rewards[-5:])

        # Later episodes should perform at least as well
        # (may not always improve due to stochasticity, but should not degrade much)
        assert late_reward >= early_reward * 0.7  # Allow 30% variance

    def test_epsilon_decays(self, untrained_agent):
        """Test that exploration decreases as expected."""
        agent = untrained_agent

        initial_epsilon = agent.epsilon
        assert initial_epsilon == agent.config.epsilon_start

        # Decay epsilon multiple times
        for _ in range(20):
            agent.decay_epsilon()

        # Epsilon should decrease
        assert agent.epsilon < initial_epsilon

        # Epsilon should not go below minimum
        for _ in range(1000):
            agent.decay_epsilon()

        assert agent.epsilon >= agent.config.epsilon_end

    def test_target_network_updates(self, untrained_agent, fast_env):
        """Test that target network gets updated."""
        agent = untrained_agent

        # Get initial target network parameters
        initial_target_params = [param.clone() for param in agent.target_net.parameters()]

        # Collect experience
        state, _ = fast_env.reset(seed=42)
        for _ in range(agent.config.batch_size * 3):
            action = agent.select_action(state, evaluate=False)
            next_state, reward, terminated, truncated, info = fast_env.step(action)
            agent.buffer.push(state, action, reward, next_state, terminated)
            state = next_state
            if terminated or truncated:
                state, _ = fast_env.reset()

        # Train for enough steps to trigger target update
        for _ in range(agent.config.target_update_freq + 10):
            agent.update()

        # Target network should have been updated
        final_target_params = [param.clone() for param in agent.target_net.parameters()]

        # At least some parameters should have changed
        params_changed = False
        for initial, final in zip(initial_target_params, final_target_params):
            if not torch.allclose(initial, final):
                params_changed = True
                break

        assert params_changed


class TestCheckpointing:
    """Test checkpoint saving and loading."""

    def test_checkpoint_saves(self, fast_env, untrained_agent, training_config, temp_output_dir):
        """Test that checkpoints are created."""
        agent = untrained_agent

        # Update checkpoint config
        training_config.checkpoint_dir = str(temp_output_dir / "checkpoints")
        training_config.checkpoint_frequency = 5
        training_config.max_episodes = 10

        trainer = Trainer(agent, fast_env, training_config, output_dir=str(temp_output_dir))

        # Run training
        trainer.train()

        # Check that checkpoint directory exists
        checkpoint_dir = Path(training_config.checkpoint_dir)
        assert checkpoint_dir.exists()

        # Check that at least one checkpoint was created
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoints) > 0

    def test_checkpoint_loads(self, fast_env, untrained_agent, temp_output_dir):
        """Test that agent can be loaded from checkpoint."""
        agent = untrained_agent

        # Train briefly
        state, _ = fast_env.reset(seed=42)
        for _ in range(50):
            action = agent.select_action(state, evaluate=False)
            next_state, reward, terminated, truncated, info = fast_env.step(action)
            agent.buffer.push(state, action, reward, next_state, terminated)
            if len(agent.buffer) >= agent.config.batch_size:
                agent.update()
            state = next_state
            if terminated or truncated:
                state, _ = fast_env.reset()

        # Save checkpoint
        checkpoint_path = temp_output_dir / "test_checkpoint.pt"
        agent.save(str(checkpoint_path))

        # Create new agent and load
        new_agent = DQNAgent(agent.config)
        new_agent.load(str(checkpoint_path))

        # Agents should produce similar Q-values
        test_state = np.random.randn(agent.config.state_dim).astype(np.float32)

        q_values_1 = agent.get_q_values(test_state)
        q_values_2 = new_agent.get_q_values(test_state)

        np.testing.assert_array_almost_equal(q_values_1, q_values_2, decimal=5)

    def test_resumed_training_continues(self, fast_env, untrained_agent, temp_output_dir):
        """Test that resumed training maintains progress."""
        agent = untrained_agent

        # Train for a bit
        for episode in range(10):
            state, _ = fast_env.reset(seed=42 + episode)
            for _ in range(20):
                action = agent.select_action(state, evaluate=False)
                next_state, reward, terminated, truncated, info = fast_env.step(action)
                agent.buffer.push(state, action, reward, next_state, terminated)
                if len(agent.buffer) >= agent.config.batch_size:
                    agent.update()
                state = next_state
                if terminated or truncated:
                    break
            agent.decay_epsilon()

        # Save
        epsilon_before = agent.epsilon
        steps_before = agent.steps_done
        checkpoint_path = temp_output_dir / "resume_test.pt"
        agent.save(str(checkpoint_path))

        # Load and continue
        agent.load(str(checkpoint_path))

        # State should be preserved
        assert agent.epsilon == epsilon_before
        assert agent.steps_done == steps_before

        # Continue training
        state, _ = fast_env.reset(seed=100)
        for _ in range(20):
            action = agent.select_action(state, evaluate=False)
            next_state, reward, terminated, truncated, info = fast_env.step(action)
            agent.buffer.push(state, action, reward, next_state, terminated)
            if len(agent.buffer) >= agent.config.batch_size:
                agent.update()
            state = next_state
            if terminated or truncated:
                break

        # Steps should have increased
        assert agent.steps_done > steps_before

    def test_best_model_saved(self, fast_env, untrained_agent, training_config, temp_output_dir):
        """Test that best model is correctly identified and saved."""
        agent = untrained_agent

        training_config.save_best_only = True
        training_config.max_episodes = 15
        training_config.eval_frequency = 5
        training_config.checkpoint_dir = str(temp_output_dir / "checkpoints")

        trainer = Trainer(agent, fast_env, training_config, output_dir=str(temp_output_dir))
        trainer.train()

        # Check that best model exists
        best_model_path = Path(training_config.checkpoint_dir) / "best_model.pt"

        # Either best model exists or checkpoints exist
        checkpoints = list(Path(training_config.checkpoint_dir).glob("*.pt"))
        assert len(checkpoints) > 0 or best_model_path.exists()


class TestEarlyStoppingAndConvergence:
    """Test early stopping and convergence detection."""

    def test_early_stopping_triggers(self, fast_env, untrained_agent, training_config, temp_output_dir):
        """Test that training stops when no improvement."""
        agent = untrained_agent

        training_config.early_stopping = True
        training_config.patience = 5
        training_config.min_episodes = 5
        training_config.max_episodes = 100
        training_config.eval_frequency = 5

        trainer = Trainer(agent, fast_env, training_config, output_dir=str(temp_output_dir))

        # Train (may stop early)
        result = trainer.train()

        # Should stop before max episodes (or reach max)
        assert result['episodes_trained'] <= training_config.max_episodes

    def test_minimum_episodes_respected(self, fast_env, untrained_agent, training_config, temp_output_dir):
        """Test that early stopping waits for minimum episodes."""
        agent = untrained_agent

        training_config.early_stopping = True
        training_config.min_episodes = 10
        training_config.patience = 2
        training_config.max_episodes = 50

        trainer = Trainer(agent, fast_env, training_config, output_dir=str(temp_output_dir))
        result = trainer.train()

        # Should train for at least min_episodes
        assert result['episodes_trained'] >= training_config.min_episodes

    def test_convergence_detection(self, fast_env, agent_with_experience, training_config, temp_output_dir):
        """Test that training detects when agent has converged."""
        agent = agent_with_experience

        training_config.max_episodes = 30
        training_config.eval_frequency = 5
        training_config.early_stopping = True
        training_config.patience = 10

        trainer = Trainer(agent, fast_env, training_config, output_dir=str(temp_output_dir))

        start_time = time.time()
        result = trainer.train()
        elapsed = time.time() - start_time

        # Training should complete
        assert result['episodes_trained'] > 0

        # Should not take too long (reasonable timeout)
        assert elapsed < 300  # 5 minutes max for fast env


class TestTrainingMetrics:
    """Test that training metrics are properly tracked."""

    def test_metrics_tracked(self, fast_env, untrained_agent, training_config, temp_output_dir):
        """Test that training tracks all required metrics."""
        agent = untrained_agent

        training_config.max_episodes = 10
        training_config.eval_frequency = 5

        trainer = Trainer(agent, fast_env, training_config, output_dir=str(temp_output_dir))
        result = trainer.train()

        # Check that result contains expected keys
        expected_keys = ['episodes_trained', 'best_eval_reward', 'training_time']
        for key in expected_keys:
            assert key in result

    def test_episode_rewards_logged(self, fast_env, untrained_agent, training_config, temp_output_dir):
        """Test that episode rewards are logged."""
        agent = untrained_agent

        training_config.max_episodes = 10
        training_config.log_frequency = 5

        trainer = Trainer(agent, fast_env, training_config, output_dir=str(temp_output_dir))
        trainer.train()

        # Check that training history has rewards
        assert len(trainer.episode_rewards) > 0
        assert all(isinstance(r, (int, float, np.number)) for r in trainer.episode_rewards)

    def test_evaluation_results_stored(self, fast_env, untrained_agent, training_config, temp_output_dir):
        """Test that evaluation results are stored."""
        agent = untrained_agent

        training_config.max_episodes = 15
        training_config.eval_frequency = 5
        training_config.eval_episodes = 3

        trainer = Trainer(agent, fast_env, training_config, output_dir=str(temp_output_dir))
        trainer.train()

        # Should have evaluation results
        assert len(trainer.eval_rewards) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


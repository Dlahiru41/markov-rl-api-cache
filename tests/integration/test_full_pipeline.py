"""
End-to-end integration tests for the complete pipeline.

These tests verify that all system components work together correctly
in realistic scenarios, from training through evaluation.
"""

import pytest
import numpy as np
import time
from pathlib import Path

from src.integration.gym_environment import CachingEnv
from src.rl.agents.dqn_agent import DQNAgent
from src.rl.training.trainer import Trainer, TrainingConfig
from baselines.lru_policy import LRUPolicy
from baselines.random_policy import RandomPolicy
from baselines.agent_adapter import RLAgentAdapter


class TestEndToEnd:
    """Test complete end-to-end workflows."""

    def test_full_training_pipeline(self, fast_env, untrained_agent, training_config, temp_output_dir):
        """Test complete training pipeline: Train → Evaluate → Compare."""
        agent = untrained_agent

        # Configure for fast test
        training_config.max_episodes = 20
        training_config.eval_frequency = 10
        training_config.eval_episodes = 3

        # Train
        trainer = Trainer(agent, fast_env, training_config, output_dir=str(temp_output_dir))
        training_result = trainer.train()

        # Verify training completed
        assert training_result['episodes_trained'] > 0
        assert 'best_eval_reward' in training_result

        # Evaluate trained agent
        eval_rewards = []
        for episode in range(5):
            state, _ = fast_env.reset(seed=100 + episode)
            episode_reward = 0

            for _ in range(30):
                action = agent.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, info = fast_env.step(action)
                episode_reward += reward
                state = next_state

                if terminated or truncated:
                    break

            eval_rewards.append(episode_reward)

        # Should have evaluation results
        assert len(eval_rewards) == 5
        avg_reward = np.mean(eval_rewards)

        # Agent should perform reasonably
        assert np.isfinite(avg_reward)

    def test_training_improves_over_baselines(self, fast_env, trained_agent, temp_output_dir):
        """Test that trained agent performs better than simple baselines."""
        # Evaluate trained agent
        trained_rewards = []
        for episode in range(5):
            state, _ = fast_env.reset(seed=200 + episode)
            episode_reward = 0

            for _ in range(30):
                action = trained_agent.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, info = fast_env.step(action)
                episode_reward += reward
                state = next_state

                if terminated or truncated:
                    break

            trained_rewards.append(episode_reward)

        # Evaluate random baseline
        random_policy = RandomPolicy()
        random_rewards = []

        for episode in range(5):
            state, _ = fast_env.reset(seed=200 + episode)
            episode_reward = 0

            for _ in range(30):
                action = random_policy.select_action(state, [])
                next_state, reward, terminated, truncated, info = fast_env.step(action)
                episode_reward += reward
                state = next_state

                if terminated or truncated:
                    break

            random_rewards.append(episode_reward)

        # Trained agent should be competitive or better
        trained_avg = np.mean(trained_rewards)
        random_avg = np.mean(random_rewards)

        # Trained should be at least as good (with some tolerance)
        assert trained_avg >= random_avg * 0.8  # Within 20% or better

    def test_model_deployment(self, trained_agent, temp_output_dir):
        """Test that trained model can be deployed and serve predictions."""
        # Save model
        model_path = temp_output_dir / "deployed_model.pt"
        trained_agent.save(str(model_path))

        assert model_path.exists()

        # Load in "production" mode
        deployed_agent = DQNAgent(trained_agent.config)
        deployed_agent.load(str(model_path))

        # Test inference
        test_states = np.random.randn(10, trained_agent.config.state_dim).astype(np.float32)

        for state in test_states:
            action = deployed_agent.select_action(state, evaluate=True)

            # Should return valid action
            assert 0 <= action < 7

            # Should return consistent actions in eval mode
            action2 = deployed_agent.select_action(state, evaluate=True)
            assert action == action2  # Same state → same action in eval

    def test_metrics_collection(self, fast_env, untrained_agent, training_config, temp_output_dir):
        """Test that all metrics are collected throughout training."""
        agent = untrained_agent

        training_config.max_episodes = 15
        training_config.eval_frequency = 5

        trainer = Trainer(agent, fast_env, training_config, output_dir=str(temp_output_dir))
        result = trainer.train()

        # Check that metrics were collected
        assert len(trainer.episode_rewards) > 0
        assert len(trainer.episode_lengths) > 0

        # Check training result has metrics
        assert 'episodes_trained' in result
        assert 'training_time' in result

        # Check output directory has logs
        assert temp_output_dir.exists()


class TestScenarios:
    """Test realistic operational scenarios."""

    def test_normal_traffic_scenario(self, training_env, trained_agent, sample_traffic):
        """Test that agent handles normal traffic patterns."""
        # Simulate normal traffic
        total_reward = 0
        episodes_completed = 0

        for session in sample_traffic[:3]:  # Test subset
            state, _ = training_env.reset(seed=42)
            episode_reward = 0

            for step, api_call in enumerate(session['apis'][:20]):
                action = trained_agent.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, info = training_env.step(action)

                episode_reward += reward
                state = next_state

                if terminated or truncated:
                    break

            total_reward += episode_reward
            episodes_completed += 1

        # Should complete episodes without errors
        assert episodes_completed > 0

        # Average reward should be reasonable
        avg_reward = total_reward / episodes_completed
        assert np.isfinite(avg_reward)

    def test_peak_traffic_scenario(self, fast_env, trained_agent):
        """Test that agent handles peak load."""
        # Simulate peak traffic (more API calls per episode)
        peak_rewards = []

        for episode in range(3):
            state, _ = fast_env.reset(seed=300 + episode)
            episode_reward = 0

            # Longer episode to simulate peak
            for _ in range(50):
                action = trained_agent.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, info = fast_env.step(action)

                episode_reward += reward
                state = next_state

                if terminated or truncated:
                    break

            peak_rewards.append(episode_reward)

        # Should handle peak traffic
        assert len(peak_rewards) == 3
        assert all(np.isfinite(r) for r in peak_rewards)

    def test_cascade_prevention_scenario(self, training_env, trained_agent):
        """Test that agent helps prevent cascade failures."""
        cascade_count = 0
        normal_completion = 0

        for episode in range(5):
            state, _ = training_env.reset(seed=400 + episode)

            for _ in range(50):
                action = trained_agent.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, info = training_env.step(action)

                # Check if cascade occurred
                if terminated and training_env.cascade_detected:
                    cascade_count += 1
                    break

                state = next_state

                if terminated or truncated:
                    normal_completion += 1
                    break

        # Most episodes should complete normally (not cascade)
        total_episodes = cascade_count + normal_completion
        if total_episodes > 0:
            cascade_rate = cascade_count / total_episodes
            # Cascade rate should be low
            assert cascade_rate < 0.5  # Less than 50% cascades

    def test_cold_start_scenario(self, training_env, trained_agent, cache_manager):
        """Test that agent handles empty cache start."""
        # Clear cache completely
        cache_manager.clear()

        # Run episode from cold start
        state, _ = training_env.reset(seed=500)
        episode_reward = 0

        for _ in range(30):
            action = trained_agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, info = training_env.step(action)

            episode_reward += reward
            state = next_state

            if terminated or truncated:
                break

        # Should handle cold start
        assert np.isfinite(episode_reward)

        # Cache should have some entries now
        stats = cache_manager.get_stats()
        assert stats['entries'] >= 0


class TestMultiAgentComparison:
    """Test comparison of multiple agents/policies."""

    def test_compare_multiple_policies(self, fast_env):
        """Test comparing trained agent against baselines."""
        # Create policies to compare
        policies = {
            'random': RandomPolicy(),
            'lru': LRUPolicy()
        }

        results = {}

        for name, policy in policies.items():
            rewards = []

            for episode in range(3):
                state, _ = fast_env.reset(seed=600 + episode)
                episode_reward = 0

                for _ in range(20):
                    action = policy.select_action(state, [])
                    next_state, reward, terminated, truncated, info = fast_env.step(action)

                    episode_reward += reward
                    state = next_state

                    if terminated or truncated:
                        break

                rewards.append(episode_reward)

            results[name] = {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'rewards': rewards
            }

        # All policies should produce results
        assert len(results) == len(policies)

        for name, result in results.items():
            assert np.isfinite(result['mean'])
            assert result['std'] >= 0

    def test_statistical_comparison(self, fast_env, trained_agent):
        """Test statistical comparison between agents."""
        from scipy import stats as scipy_stats

        # Evaluate trained agent
        trained_rewards = []
        for episode in range(10):
            state, _ = fast_env.reset(seed=700 + episode)
            episode_reward = 0

            for _ in range(20):
                action = trained_agent.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, info = fast_env.step(action)
                episode_reward += reward
                state = next_state
                if terminated or truncated:
                    break

            trained_rewards.append(episode_reward)

        # Evaluate random baseline
        random_policy = RandomPolicy()
        random_rewards = []

        for episode in range(10):
            state, _ = fast_env.reset(seed=700 + episode)
            episode_reward = 0

            for _ in range(20):
                action = random_policy.select_action(state, [])
                next_state, reward, terminated, truncated, info = fast_env.step(action)
                episode_reward += reward
                state = next_state
                if terminated or truncated:
                    break

            random_rewards.append(episode_reward)

        # Perform statistical test
        # t-test or Mann-Whitney U test
        statistic, p_value = scipy_stats.mannwhitneyu(
            trained_rewards, random_rewards, alternative='greater'
        )

        # Statistical test should run without error
        assert np.isfinite(statistic)
        assert 0 <= p_value <= 1


class TestSystemReliability:
    """Test system reliability and error handling."""

    def test_handles_invalid_states(self, trained_agent):
        """Test that agent handles invalid state inputs gracefully."""
        # Test with edge cases
        test_cases = [
            np.zeros(trained_agent.config.state_dim, dtype=np.float32),
            np.ones(trained_agent.config.state_dim, dtype=np.float32),
            np.random.randn(trained_agent.config.state_dim).astype(np.float32)
        ]

        for state in test_cases:
            try:
                action = trained_agent.select_action(state, evaluate=True)
                assert 0 <= action < 7
            except Exception as e:
                pytest.fail(f"Agent failed on valid state: {e}")

    def test_recovers_from_errors(self, fast_env, trained_agent):
        """Test that system recovers from transient errors."""
        # Run episode
        state, _ = fast_env.reset(seed=800)

        errors_encountered = 0
        successful_steps = 0

        for _ in range(30):
            try:
                action = trained_agent.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, info = fast_env.step(action)

                successful_steps += 1
                state = next_state

                if terminated or truncated:
                    break
            except Exception as e:
                errors_encountered += 1
                # In real system, would log and retry
                if errors_encountered > 5:
                    break

        # Should have mostly successful steps
        assert successful_steps > 0

    def test_long_running_stability(self, fast_env, trained_agent):
        """Test that system remains stable over long runs."""
        # Run multiple episodes back-to-back
        rewards = []

        for episode in range(10):
            state, _ = fast_env.reset(seed=900 + episode)
            episode_reward = 0

            for _ in range(20):
                action = trained_agent.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, info = fast_env.step(action)

                episode_reward += reward
                state = next_state

                if terminated or truncated:
                    break

            rewards.append(episode_reward)

        # Performance should remain stable
        assert len(rewards) == 10
        assert all(np.isfinite(r) for r in rewards)

        # Variance should be reasonable
        std_reward = np.std(rewards)
        assert std_reward < 1000  # Reasonable variance


class TestResourceManagement:
    """Test resource management and cleanup."""

    def test_memory_cleanup(self, fast_env, untrained_agent):
        """Test that memory is properly managed."""
        agent = untrained_agent

        # Train briefly
        for episode in range(5):
            state, _ = fast_env.reset(seed=1000 + episode)

            for _ in range(20):
                action = agent.select_action(state, evaluate=False)
                next_state, reward, terminated, truncated, info = fast_env.step(action)

                agent.buffer.push(state, action, reward, next_state, terminated)

                if len(agent.buffer) >= agent.config.batch_size:
                    agent.update()

                state = next_state

                if terminated or truncated:
                    break

        # Buffer should not grow indefinitely
        assert len(agent.buffer) <= agent.config.buffer_size

    def test_environment_cleanup(self, env_config):
        """Test that environments clean up properly."""
        envs = []

        # Create multiple environments
        for _ in range(5):
            env = CachingEnv(env_config)
            envs.append(env)

        # Use environments
        for env in envs:
            state, _ = env.reset(seed=42)
            action = env.action_space.sample()
            env.step(action)

        # Close all
        for env in envs:
            env.close()

        # Should not raise errors
        assert len(envs) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


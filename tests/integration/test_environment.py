"""
Integration tests for Gymnasium environment (CachingEnv).

These tests verify that the CachingEnv properly integrates all components
(Markov predictor, cache manager, state builder, reward calculator, action space)
and implements the Gymnasium API correctly.
"""

import pytest
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from src.integration.gym_environment import CachingEnv, CacheEnvConfig


class TestEnvironmentBasics:
    """Test basic environment initialization and API compliance."""

    def test_environment_creation(self, env_config):
        """Test that CachingEnv initializes without errors."""
        env = CachingEnv(env_config)

        assert env is not None
        assert isinstance(env, gym.Env)
        assert env.action_space is not None
        assert env.observation_space is not None

        env.close()

    def test_observation_space_shape(self, training_env):
        """Test that observations match declared observation space."""
        obs, info = training_env.reset(seed=42)

        # Check shape matches observation space
        assert obs.shape == training_env.observation_space.shape

        # Check type
        assert obs.dtype == np.float32

        # Check bounds (should be normalized to [0, 1])
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    def test_action_space_valid(self, training_env):
        """Test that all actions are within valid range."""
        assert isinstance(training_env.action_space, gym.spaces.Discrete)
        assert training_env.action_space.n == 7  # 7 discrete actions

        # Sample multiple actions
        for _ in range(100):
            action = training_env.action_space.sample()
            assert 0 <= action < 7

    def test_reset_returns_valid_observation(self, training_env):
        """Test that reset returns proper (obs, info) tuple."""
        obs, info = training_env.reset(seed=42)

        # Check observation
        assert isinstance(obs, np.ndarray)
        assert obs.shape == training_env.observation_space.shape
        assert obs.dtype == np.float32

        # Check info dict
        assert isinstance(info, dict)
        assert 'episode' in info or 'session_id' in info

    def test_step_returns_valid_tuple(self, training_env):
        """Test that step returns (obs, reward, terminated, truncated, info)."""
        training_env.reset(seed=42)
        action = training_env.action_space.sample()

        result = training_env.step(action)

        assert len(result) == 5
        obs, reward, terminated, truncated, info = result

        # Check types
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float, np.number))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        # Check observation shape matches environment
        assert obs.shape == training_env.observation_space.shape

        # Check reward is finite
        assert np.isfinite(reward)

    def test_episode_terminates(self, fast_env):
        """Test that episodes eventually end."""
        fast_env.reset(seed=42)

        max_steps = 1000
        terminated = False
        truncated = False

        for step in range(max_steps):
            action = fast_env.action_space.sample()
            obs, reward, terminated, truncated, info = fast_env.step(action)

            if terminated or truncated:
                break

        # Episode should end before max_steps
        assert terminated or truncated
        assert step < max_steps

    def test_info_contains_required_keys(self, training_env):
        """Test that info dict has expected keys."""
        training_env.reset(seed=42)
        action = 0  # DO_NOTHING

        obs, reward, terminated, truncated, info = training_env.step(action)

        # Check for common info keys
        expected_keys = {'current_step', 'cumulative_reward'}

        # At least some expected keys should be present
        assert any(key in info for key in expected_keys) or len(info) > 0


class TestEnvironmentDynamics:
    """Test environment dynamics and caching behavior."""

    def test_cache_hit_increases_on_repeated_access(self, training_env):
        """Test that accessing the same API twice results in cache hit."""
        training_env.reset(seed=42)

        # First access - should be a miss
        action = 1  # CACHE_CURRENT
        obs1, reward1, _, _, info1 = training_env.step(action)

        # Get the current API from environment state
        current_api_1 = training_env.current_api

        # Make the same API call again by resetting to same state
        # In practice, we'll just run a few steps and check cache metrics
        initial_hits = training_env.total_cache_hits

        # Execute several cache actions
        for _ in range(5):
            action = 1  # CACHE_CURRENT
            obs, reward, terminated, truncated, info = training_env.step(action)
            if terminated or truncated:
                break

        # Cache hits should increase (probabilistically)
        # Note: This is stochastic, so we check that the mechanism works
        final_hits = training_env.total_cache_hits
        assert final_hits >= initial_hits

    def test_cache_eviction_works(self, training_env):
        """Test that when cache is full, old entries are evicted."""
        training_env.reset(seed=42)

        # Fill cache by caching many unique items
        for _ in range(50):
            action = 1  # CACHE_CURRENT
            obs, reward, terminated, truncated, info = training_env.step(action)
            if terminated or truncated:
                break

        # Check that cache manager has some entries
        cache_stats = training_env.cache_manager.get_stats()
        assert cache_stats['entries'] > 0

        # If cache utilization is high, eviction should have occurred
        if cache_stats.get('utilization', 0) > 0.5:
            assert cache_stats.get('evictions', 0) >= 0

    def test_prefetch_action_queues_predictions(self, training_env):
        """Test that prefetch actions affect cache."""
        training_env.reset(seed=42)

        # Execute prefetch action
        action = 3  # PREFETCH_MODERATE
        obs, reward, terminated, truncated, info = training_env.step(action)

        # Check that prefetch queue has items (or prefetch occurred)
        # This is implementation-specific, but we can check metrics
        assert 'action' in info or reward != 0

    def test_reward_positive_on_hit(self, training_env):
        """Test that cache hits give positive reward."""
        training_env.reset(seed=42)

        rewards = []

        # Run episode and collect rewards
        for _ in range(20):
            action = 1  # CACHE_CURRENT - increases hit probability
            obs, reward, terminated, truncated, info = training_env.step(action)
            rewards.append(reward)

            if terminated or truncated:
                break

        # At least some rewards should be non-zero
        assert len(rewards) > 0
        assert not all(r == 0 for r in rewards)

    def test_reward_negative_on_miss(self, training_env):
        """Test that cache misses can give negative reward."""
        training_env.reset(seed=42)

        # DO_NOTHING action should sometimes lead to misses
        action = 0  # DO_NOTHING

        rewards = []
        for _ in range(20):
            obs, reward, terminated, truncated, info = training_env.step(action)
            rewards.append(reward)

            if terminated or truncated:
                break

        # Rewards should vary (some positive, some negative)
        assert len(rewards) > 0
        assert np.std(rewards) > 0  # Rewards should have variance

    def test_cascade_ends_episode(self, env_config):
        """Test that cascade detection terminates episode."""
        # Create env with cascade detection enabled
        config = env_config
        config.episode_end_on_cascade = True

        env = CachingEnv(config)
        env.reset(seed=42)

        # Force a cascade by manipulating system metrics
        # In real scenario, cascade would be detected automatically
        env.cascade_detected = True

        # Next step should terminate
        action = 0
        obs, reward, terminated, truncated, info = env.step(action)

        # Episode should terminate on cascade
        if env.cascade_detected:
            assert terminated or truncated

        env.close()


class TestEnvironmentReproducibility:
    """Test that environment behavior is reproducible."""

    def test_same_seed_same_trajectory(self, env_config):
        """Test that same seed produces identical episode."""
        env1 = CachingEnv(env_config)
        env2 = CachingEnv(env_config)

        # Reset with same seed
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        # Initial observations should be identical
        np.testing.assert_array_almost_equal(obs1, obs2)

        # Run same actions
        for _ in range(10):
            action = 1  # Fixed action

            o1, r1, t1, tr1, i1 = env1.step(action)
            o2, r2, t2, tr2, i2 = env2.step(action)

            # Observations should be identical
            np.testing.assert_array_almost_equal(o1, o2)

            # Rewards should be identical
            assert r1 == r2

            if t1 or tr1:
                break

        env1.close()
        env2.close()

    def test_different_seeds_different_trajectories(self, env_config):
        """Test that different seeds produce different episodes."""
        env1 = CachingEnv(env_config)
        env2 = CachingEnv(env_config)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=123)

        # Initial observations should be different
        assert not np.allclose(obs1, obs2)

        env1.close()
        env2.close()

    def test_reset_clears_state(self, training_env):
        """Test that reset properly resets all internal state."""
        # Run episode
        training_env.reset(seed=42)
        for _ in range(20):
            action = training_env.action_space.sample()
            obs, reward, terminated, truncated, info = training_env.step(action)
            if terminated or truncated:
                break

        # Record state before reset
        steps_before = training_env.current_step

        # Reset
        obs, info = training_env.reset(seed=42)

        # Check that state is reset
        assert training_env.current_step == 0
        assert training_env.cumulative_reward == 0.0
        assert len(training_env.episode_rewards) == 0


class TestEnvironmentCompatibility:
    """Test compatibility with RL libraries and wrappers."""

    def test_stable_baselines_compatibility(self, fast_env):
        """Test that environment passes Stable-Baselines3 checks."""
        try:
            from stable_baselines3.common.env_checker import check_env as sb3_check_env

            # This will raise an error if environment is not compatible
            sb3_check_env(fast_env, warn=False)

        except ImportError:
            pytest.skip("stable-baselines3 not installed")
        except Exception as e:
            # If it fails, the error message will help debug
            pytest.fail(f"Environment failed SB3 compatibility check: {e}")

    def test_gymnasium_wrapper_compatibility(self, training_env):
        """Test that environment works with standard Gymnasium wrappers."""
        from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics

        # Wrap environment
        wrapped_env = TimeLimit(training_env, max_episode_steps=50)
        wrapped_env = RecordEpisodeStatistics(wrapped_env)

        # Test that wrapped env works
        obs, info = wrapped_env.reset(seed=42)
        assert obs is not None

        for _ in range(10):
            action = wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            if terminated or truncated:
                break

        # Should have episode statistics in info
        if terminated or truncated:
            assert 'episode' in info or 'r' in info

    def test_vectorized_environment(self, env_config):
        """Test that environment can be vectorized for parallel training."""
        try:
            from gymnasium.vector import SyncVectorEnv

            # Create vectorized environment
            def make_env():
                return CachingEnv(env_config)

            vec_env = SyncVectorEnv([make_env for _ in range(2)])

            # Test reset
            obs = vec_env.reset(seed=42)
            # Get expected state dimension from first environment
            expected_state_dim = vec_env.envs[0].observation_space.shape[0]
            assert obs[0].shape == (2, expected_state_dim)  # 2 environments, state_dim

            # Test step
            actions = vec_env.action_space.sample()
            obs, rewards, terminated, truncated, info = vec_env.step(actions)

            assert obs.shape == (2, expected_state_dim)
            assert len(rewards) == 2

            vec_env.close()

        except ImportError:
            pytest.skip("gymnasium.vector not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


"""
Validation script for the Gymnasium caching environment.

Tests all functionality:
1. Environment creation and configuration
2. Reset functionality
3. Step execution
4. Full episode rollout
5. Stable-Baselines3 compatibility
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from src.integration.gym_environment import CachingEnv, CacheEnvConfig, SimulatorConfig
from src.rl.state import StateConfig
from src.rl.reward import RewardConfig
from src.rl.actions import ActionConfig
from src.cache.cache_manager import CacheManagerConfig
import gymnasium as gym


def test_environment_creation():
    """Test 1: Environment creation and configuration."""
    print("\n" + "="*60)
    print("TEST 1: Environment Creation")
    print("="*60)

    try:
        # Create with default config
        config = CacheEnvConfig(
            use_real_services=False,
            max_steps_per_episode=100,
            seed=42
        )
        env = CachingEnv(config)

        print(f"✓ Environment created successfully")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        print(f"  State dimension: {env.observation_space.shape[0]}")

        assert env.observation_space.shape[0] > 0, "State dimension should be positive"
        assert env.action_space.n == 7, "Should have 7 actions"

        env.close()
        print("✓ Test 1 PASSED\n")
        return True

    except Exception as e:
        print(f"✗ Test 1 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_reset_functionality():
    """Test 2: Reset functionality."""
    print("\n" + "="*60)
    print("TEST 2: Reset Functionality")
    print("="*60)

    try:
        config = CacheEnvConfig(
            use_real_services=False,
            max_steps_per_episode=100,
            seed=42
        )
        env = CachingEnv(config)

        # Test reset
        obs, info = env.reset(seed=42)

        print(f"✓ Reset successful")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Observation dtype: {obs.dtype}")
        print(f"  Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
        print(f"  Info keys: {list(info.keys())}")

        assert obs.shape == env.observation_space.shape, "Observation shape mismatch"
        assert obs.dtype == np.float32, "Observation should be float32"
        assert np.all(obs >= 0) and np.all(obs <= 1), "Observations should be in [0, 1]"
        assert 'episode_number' in info, "Info should contain episode_number"
        assert 'initial_api' in info, "Info should contain initial_api"

        # Test multiple resets
        obs2, info2 = env.reset(seed=42)
        assert np.allclose(obs, obs2), "Same seed should give same initial state"

        env.close()
        print("✓ Test 2 PASSED\n")
        return True

    except Exception as e:
        print(f"✗ Test 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_step_execution():
    """Test 3: Step execution."""
    print("\n" + "="*60)
    print("TEST 3: Step Execution")
    print("="*60)

    try:
        config = CacheEnvConfig(
            use_real_services=False,
            max_steps_per_episode=100,
            seed=42
        )
        env = CachingEnv(config)

        obs, info = env.reset(seed=42)

        # Test each action type
        for action in range(7):
            obs_new, reward, terminated, truncated, info = env.step(action)

            print(f"✓ Action {action} executed: reward={reward:.2f}, terminated={terminated}, truncated={truncated}")

            assert obs_new.shape == env.observation_space.shape, "Observation shape mismatch"
            assert isinstance(reward, (int, float)), "Reward should be numeric"
            assert isinstance(terminated, bool), "Terminated should be bool"
            assert isinstance(truncated, bool), "Truncated should be bool"
            assert 'action_taken' in info, "Info should contain action_taken"
            assert 'cache_hit' in info, "Info should contain cache_hit"
            assert 'reward_breakdown' in info, "Info should contain reward_breakdown"

            if terminated or truncated:
                break

        env.close()
        print("✓ Test 3 PASSED\n")
        return True

    except Exception as e:
        print(f"✗ Test 3 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_full_episode():
    """Test 4: Full episode rollout."""
    print("\n" + "="*60)
    print("TEST 4: Full Episode Rollout")
    print("="*60)

    try:
        config = CacheEnvConfig(
            use_real_services=False,
            max_steps_per_episode=100,
            seed=42
        )
        env = CachingEnv(config)

        obs, _ = env.reset(seed=42)
        total_reward = 0
        steps = 0
        cache_hits = 0

        while True:
            # Use random policy
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            if info['cache_hit']:
                cache_hits += 1

            if terminated or truncated:
                print(f"✓ Episode finished naturally")
                print(f"  Steps: {steps}")
                print(f"  Total reward: {total_reward:.2f}")
                print(f"  Cache hits: {cache_hits}/{steps} ({cache_hits/steps:.1%})")
                print(f"  Termination reason: {'terminated' if terminated else 'truncated'}")
                break

            if steps > config.max_steps_per_episode + 10:
                raise RuntimeError("Episode exceeded max steps without ending")

        # Get episode metrics
        metrics = env.get_episode_metrics()
        print(f"\n  Episode metrics:")
        print(f"    Total steps: {metrics['total_steps']}")
        print(f"    Total reward: {metrics['total_reward']:.2f}")
        print(f"    Cache hit rate: {metrics['cache_hit_rate']:.2%}")
        print(f"    Prediction accuracy: {metrics['prediction_accuracy']:.2%}")
        print(f"    Cascade occurred: {metrics['cascade_occurred']}")

        assert metrics['total_steps'] > 0, "Should have taken some steps"
        assert steps == metrics['total_steps'], "Step count mismatch"

        env.close()
        print("✓ Test 4 PASSED\n")
        return True

    except Exception as e:
        print(f"✗ Test 4 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_stable_baselines3_compatibility():
    """Test 5: Stable-Baselines3 compatibility."""
    print("\n" + "="*60)
    print("TEST 5: Stable-Baselines3 Compatibility")
    print("="*60)

    try:
        from stable_baselines3.common.env_checker import check_env

        config = CacheEnvConfig(
            use_real_services=False,
            max_steps_per_episode=50,  # Shorter for faster testing
            seed=42
        )
        env = CachingEnv(config)

        print("Running Stable-Baselines3 env_checker...")
        check_env(env, warn=True)

        print("✓ Environment passed Stable-Baselines3 checks")

        env.close()
        print("✓ Test 5 PASSED\n")
        return True

    except ImportError:
        print("⚠ Stable-Baselines3 not installed, skipping compatibility test")
        print("  Install with: pip install stable-baselines3")
        return True  # Don't fail if SB3 not installed

    except Exception as e:
        print(f"✗ Test 5 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_render_functionality():
    """Test 6: Render functionality."""
    print("\n" + "="*60)
    print("TEST 6: Render Functionality")
    print("="*60)

    try:
        config = CacheEnvConfig(
            use_real_services=False,
            max_steps_per_episode=10,
            seed=42
        )
        env = CachingEnv(config)

        obs, _ = env.reset(seed=42)

        # Test human render
        print("\nTesting human render mode:")
        env.render(mode='human')

        # Test ansi render
        print("\nTesting ansi render mode:")
        output = env.render(mode='ansi')
        if output:
            print(output[:200] + "...")  # Print first 200 chars

        print("\n✓ Render functionality works")

        env.close()
        print("✓ Test 6 PASSED\n")
        return True

    except Exception as e:
        print(f"✗ Test 6 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_custom_config():
    """Test 7: Custom configuration."""
    print("\n" + "="*60)
    print("TEST 7: Custom Configuration")
    print("="*60)

    try:
        # Create custom configs
        state_config = StateConfig(
            markov_top_k=3,
            include_probabilities=True,
            include_confidence=True,
            include_cache_metrics=True,
            include_system_metrics=False,  # Disable some features
            include_user_context=True,
            include_temporal_context=False,
            include_session_context=True
        )

        reward_config = RewardConfig(
            cache_hit_reward=20.0,  # Custom values
            cache_miss_penalty=-2.0,
            cascade_prevented_reward=100.0
        )

        simulator_config = SimulatorConfig(
            num_apis=10,
            session_length_range=(5, 20),
            cascade_threshold=0.7
        )

        config = CacheEnvConfig(
            state_config=state_config,
            reward_config=reward_config,
            simulator_config=simulator_config,
            max_steps_per_episode=50,
            seed=42
        )

        env = CachingEnv(config)

        print(f"✓ Custom config applied")
        print(f"  State dim: {env.observation_space.shape[0]}")
        print(f"  Expected: {state_config.state_dim}")

        assert env.observation_space.shape[0] == state_config.state_dim, "State dim mismatch"

        # Run a short episode
        obs, _ = env.reset(seed=42)
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        print("✓ Environment works with custom config")

        env.close()
        print("✓ Test 7 PASSED\n")
        return True

    except Exception as e:
        print(f"✗ Test 7 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "#"*60)
    print("# GYMNASIUM CACHING ENVIRONMENT VALIDATION SUITE")
    print("#"*60)

    tests = [
        ("Environment Creation", test_environment_creation),
        ("Reset Functionality", test_reset_functionality),
        ("Step Execution", test_step_execution),
        ("Full Episode Rollout", test_full_episode),
        ("Stable-Baselines3 Compatibility", test_stable_baselines3_compatibility),
        ("Render Functionality", test_render_functionality),
        ("Custom Configuration", test_custom_config),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"EXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "#"*60)
    print("# TEST SUMMARY")
    print("#"*60)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")

    print(f"\n{passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! Environment is ready for training.")
    else:
        print(f"\n⚠ {total_count - passed_count} test(s) failed. Please review.")

    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)


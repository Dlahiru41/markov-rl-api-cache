"""
Quick validation script to verify all integration test fixes work correctly.
"""

import sys
import numpy as np

def test_dqn_agent_get_q_values():
    """Test that DQNAgent has get_q_values method."""
    print("Testing DQNAgent.get_q_values()...")
    from src.rl.agents.dqn_agent import DQNAgent, DQNConfig

    config = DQNConfig(state_dim=10, action_dim=5)
    agent = DQNAgent(config)
    state = np.random.randn(10).astype(np.float32)

    q_values = agent.get_q_values(state)
    assert q_values.shape == (5,), f"Expected shape (5,), got {q_values.shape}"
    assert isinstance(q_values, np.ndarray), f"Expected numpy array, got {type(q_values)}"
    print("✓ DQNAgent.get_q_values() works correctly")
    return True

def test_trainer_properties():
    """Test that Trainer has backward compatibility properties."""
    print("\nTesting Trainer backward compatibility properties...")
    from src.rl.training.trainer import Trainer, TrainingConfig
    from src.rl.agents.dqn_agent import DQNAgent, DQNConfig
    from src.integration.gym_environment import CachingEnv

    # Create minimal environment and agent
    env = CachingEnv(capacity=10, api_count=5)
    agent_config = DQNConfig(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    agent = DQNAgent(agent_config)

    # Create trainer
    training_config = TrainingConfig(max_episodes=1, verbose=False)
    trainer = Trainer(agent, env, training_config, output_dir="test_output")

    # Check properties exist
    assert hasattr(trainer, 'episode_rewards'), "Missing episode_rewards property"
    assert hasattr(trainer, 'episode_lengths'), "Missing episode_lengths property"

    # Check they work (should be empty initially)
    assert trainer.episode_rewards == [], f"Expected [], got {trainer.episode_rewards}"
    assert trainer.episode_lengths == [], f"Expected [], got {trainer.episode_lengths}"

    # Add some data
    trainer.train_rewards.append(100.0)
    trainer.train_lengths.append(10)

    # Check properties reflect the data
    assert trainer.episode_rewards == [100.0], "episode_rewards property not working"
    assert trainer.episode_lengths == [10], "episode_lengths property not working"

    env.close()
    print("✓ Trainer backward compatibility properties work correctly")
    return True

def test_trainer_return_values():
    """Test that Trainer.train() returns expected keys."""
    print("\nTesting Trainer.train() return values...")
    from src.rl.training.trainer import Trainer, TrainingConfig
    from src.rl.agents.dqn_agent import DQNAgent, DQNConfig
    from src.integration.gym_environment import CachingEnv

    # Create minimal environment and agent
    env = CachingEnv(capacity=10, api_count=5)
    agent_config = DQNConfig(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    agent = DQNAgent(agent_config)

    # Create trainer with minimal episodes
    training_config = TrainingConfig(
        max_episodes=2,
        eval_frequency=1,
        eval_episodes=1,
        verbose=False,
        plot_frequency=1000
    )
    trainer = Trainer(agent, env, training_config, output_dir="test_output")

    # Run training
    result = trainer.train()

    # Check required keys
    required_keys = ['episodes_trained', 'training_time', 'total_episodes', 'total_time', 'best_eval_reward']
    for key in required_keys:
        assert key in result, f"Missing key '{key}' in training result"

    # Check values are reasonable
    assert result['episodes_trained'] == 2, f"Expected 2 episodes, got {result['episodes_trained']}"
    assert result['total_episodes'] == 2, f"Expected 2 total episodes, got {result['total_episodes']}"
    assert result['training_time'] > 0, f"Training time should be positive, got {result['training_time']}"

    env.close()
    print("✓ Trainer.train() returns all expected keys with correct values")
    return True

def main():
    """Run all validation tests."""
    print("=" * 80)
    print("INTEGRATION TEST FIXES VALIDATION")
    print("=" * 80)

    try:
        all_passed = True
        all_passed &= test_dqn_agent_get_q_values()
        all_passed &= test_trainer_properties()
        all_passed &= test_trainer_return_values()

        print("\n" + "=" * 80)
        if all_passed:
            print("✓ ALL VALIDATION TESTS PASSED")
            print("=" * 80)
            return 0
        else:
            print("✗ SOME VALIDATION TESTS FAILED")
            print("=" * 80)
            return 1

    except Exception as e:
        print(f"\n✗ VALIDATION FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())


"""
Demo and validation script for the training orchestration system.

This script demonstrates all features of the training system:
1. Fresh training run
2. Checkpoint saving
3. Training resumption
4. Evaluation-only mode
5. Plotting and metrics
"""

import sys
from pathlib import Path
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rl.agents.dqn_agent import DQNAgent, DQNConfig
from src.rl.training.trainer import Trainer, TrainingConfig
import numpy as np


class MockCacheEnvironment:
    """Mock environment for testing."""

    def __init__(self, state_dim=60, action_dim=7, seed=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        if seed is not None:
            np.random.seed(seed)
        self.step_count = 0
        self.max_steps = 100

    def reset(self):
        """Reset environment."""
        self.step_count = 0
        return np.random.randn(self.state_dim).astype(np.float32)

    def step(self, action):
        """Execute action."""
        self.step_count += 1

        # Simulate rewards
        if action == 1:
            reward = np.random.uniform(5, 15)
        elif action in [2, 3, 4]:
            reward = np.random.uniform(-2, 10)
        else:
            reward = np.random.uniform(-1, 5)

        # Add learning signal
        progress = self.step_count / self.max_steps
        reward += progress * 5

        next_state = np.random.randn(self.state_dim).astype(np.float32)
        done = (self.step_count >= self.max_steps)
        info = {'step': self.step_count}

        return next_state, reward, done, info


def test_fresh_training():
    """Test 1: Fresh training run."""
    print("\n" + "=" * 80)
    print("TEST 1: FRESH TRAINING RUN")
    print("=" * 80)

    # Setup
    output_dir = "results/test_fresh"
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)

    # Create components
    env = MockCacheEnvironment(seed=42)

    agent_config = DQNConfig(
        state_dim=60,
        action_dim=7,
        hidden_dims=[64, 32],
        seed=42
    )
    agent = DQNAgent(agent_config, seed=42)

    training_config = TrainingConfig(
        max_episodes=100,
        max_steps_per_episode=100,
        eval_frequency=20,
        eval_episodes=5,
        checkpoint_frequency=50,
        checkpoint_dir="checkpoints",
        log_frequency=10,
        early_stopping=False,
        seed=42
    )

    trainer = Trainer(agent, env, training_config, output_dir=output_dir)

    # Train
    print(f"\nTraining for {training_config.max_episodes} episodes...")
    final_stats = trainer.train()

    # Verify outputs
    print("\n[OK] Training completed")
    print(f"  - Total episodes: {final_stats['total_episodes']}")
    print(f"  - Best eval reward: {final_stats['best_eval_reward']:.2f}")
    print(f"  - Total time: {final_stats['total_time']:.2f}s")

    # Check outputs exist
    checkpoint_dir = Path(output_dir) / "checkpoints"
    log_dir = Path(output_dir) / "logs"

    assert checkpoint_dir.exists(), "Checkpoint directory not created"
    assert log_dir.exists(), "Log directory not created"
    assert (checkpoint_dir / "best.pt").exists(), "Best checkpoint not saved"
    assert (checkpoint_dir / "latest.pt").exists(), "Latest checkpoint not saved"
    assert (Path(output_dir) / "training_curves.png").exists(), "Training curves not saved"

    print("\n[OK] All outputs verified:")
    print(f"  - Checkpoints: {checkpoint_dir}")
    print(f"  - Logs: {log_dir}")
    print(f"  - Training curves: {output_dir}/training_curves.png")

    return output_dir, trainer


def test_resume_training(checkpoint_path: str):
    """Test 2: Resume training from checkpoint."""
    print("\n" + "=" * 80)
    print("TEST 2: RESUME TRAINING")
    print("=" * 80)

    print(f"\nResuming from: {checkpoint_path}")

    # Create components
    env = MockCacheEnvironment(seed=42)

    agent_config = DQNConfig(
        state_dim=60,
        action_dim=7,
        hidden_dims=[64, 32],
        seed=42
    )
    agent = DQNAgent(agent_config, seed=42)

    training_config = TrainingConfig(
        max_episodes=150,  # Train for 50 more episodes
        max_steps_per_episode=100,
        eval_frequency=20,
        eval_episodes=5,
        checkpoint_frequency=25,
        log_frequency=10,
        early_stopping=False,
        seed=42
    )

    output_dir = "results/test_resume"
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)

    trainer = Trainer(agent, env, training_config, output_dir=output_dir)

    # Load checkpoint
    episode = trainer.load_checkpoint(checkpoint_path)
    print(f"[OK] Checkpoint loaded, resuming from episode {episode}")

    # Resume training
    print(f"\nTraining for {training_config.max_episodes - episode} more episodes...")
    final_stats = trainer.train()

    print("\n[OK] Resumed training completed")
    print(f"  - Total episodes: {final_stats['total_episodes']}")
    print(f"  - Best eval reward: {final_stats['best_eval_reward']:.2f}")


def test_evaluation_only(checkpoint_path: str):
    """Test 3: Evaluation-only mode."""
    print("\n" + "=" * 80)
    print("TEST 3: EVALUATION-ONLY MODE")
    print("=" * 80)

    print(f"\nEvaluating checkpoint: {checkpoint_path}")

    # Create components
    env = MockCacheEnvironment(seed=42)

    agent_config = DQNConfig(
        state_dim=60,
        action_dim=7,
        hidden_dims=[64, 32],
        seed=42
    )
    agent = DQNAgent(agent_config, seed=42)

    training_config = TrainingConfig(
        eval_episodes=20,  # More episodes for evaluation
        seed=42
    )

    output_dir = "results/test_eval"
    trainer = Trainer(agent, env, training_config, output_dir=output_dir)

    # Evaluate
    eval_metrics = trainer.evaluate_only(checkpoint_path, num_episodes=20)

    print("\n[OK] Evaluation completed")
    print(f"  - Mean reward: {eval_metrics['mean_reward']:.2f} +/- {eval_metrics['std_reward']:.2f}")
    print(f"  - Min/Max: {eval_metrics['min_reward']:.2f} / {eval_metrics['max_reward']:.2f}")
    print(f"  - Mean length: {eval_metrics['mean_length']:.0f}")


def test_early_stopping():
    """Test 4: Early stopping."""
    print("\n" + "=" * 80)
    print("TEST 4: EARLY STOPPING")
    print("=" * 80)

    # Setup
    output_dir = "results/test_early_stop"
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)

    # Create components
    env = MockCacheEnvironment(seed=42)

    agent_config = DQNConfig(
        state_dim=60,
        action_dim=7,
        hidden_dims=[64, 32],
        seed=42
    )
    agent = DQNAgent(agent_config, seed=42)

    training_config = TrainingConfig(
        max_episodes=10000,  # Large number
        max_steps_per_episode=100,
        eval_frequency=10,
        eval_episodes=5,
        checkpoint_frequency=50,
        log_frequency=5,
        early_stopping=True,
        patience=30,  # Stop after 30 episodes without improvement
        min_episodes=50,  # Must train at least 50 episodes
        seed=42
    )

    trainer = Trainer(agent, env, training_config, output_dir=output_dir)

    # Train
    print(f"\nTraining with early stopping (patience={training_config.patience})...")
    final_stats = trainer.train()

    print("\n[OK] Training completed (early stopping)")
    print(f"  - Total episodes: {final_stats['total_episodes']}")
    print(f"  - Stopped early: {final_stats['total_episodes'] < training_config.max_episodes}")


def test_metrics_tracking():
    """Test 5: Metrics tracking and plotting."""
    print("\n" + "=" * 80)
    print("TEST 5: METRICS TRACKING AND PLOTTING")
    print("=" * 80)

    # Setup
    output_dir = "results/test_metrics"
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)

    # Create components
    env = MockCacheEnvironment(seed=42)

    agent_config = DQNConfig(
        state_dim=60,
        action_dim=7,
        hidden_dims=[64, 32],
        seed=42
    )
    agent = DQNAgent(agent_config, seed=42)

    training_config = TrainingConfig(
        max_episodes=100,
        max_steps_per_episode=100,
        eval_frequency=20,
        eval_episodes=5,
        checkpoint_frequency=50,
        log_frequency=10,
        plot_frequency=50,
        early_stopping=False,
        seed=42
    )

    trainer = Trainer(agent, env, training_config, output_dir=output_dir)

    # Train
    print(f"\nTraining and tracking metrics...")
    final_stats = trainer.train()

    # Check metrics
    print("\n[OK] Metrics tracked:")
    print(f"  - Train rewards: {len(trainer.train_rewards)} episodes")
    print(f"  - Eval rewards: {len(trainer.eval_rewards)} evaluations")
    print(f"  - Losses: {len(trainer.losses)} episodes")
    print(f"  - Epsilons: {len(trainer.epsilons)} episodes")

    # Check files
    metrics_file = Path(output_dir) / "metrics.json"
    curves_file = Path(output_dir) / "training_curves.png"

    assert metrics_file.exists(), "Metrics JSON not saved"
    assert curves_file.exists(), "Training curves not saved"

    print("\n[OK] Files created:")
    print(f"  - Metrics: {metrics_file}")
    print(f"  - Curves: {curves_file}")


def main():
    """Run all tests."""
    print("=" * 80)
    print("TRAINING ORCHESTRATION SYSTEM - VALIDATION")
    print("=" * 80)

    try:
        # Test 1: Fresh training
        output_dir, trainer = test_fresh_training()
        checkpoint_path = str(Path(output_dir) / "checkpoints" / "latest.pt")

        # Test 2: Resume training
        test_resume_training(checkpoint_path)

        # Test 3: Evaluation only
        best_checkpoint = str(Path(output_dir) / "checkpoints" / "best.pt")
        test_evaluation_only(best_checkpoint)

        # Test 4: Early stopping
        test_early_stopping()

        # Test 5: Metrics tracking
        test_metrics_tracking()

        # Summary
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! [OK]")
        print("=" * 80)
        print("\nThe training orchestration system is fully functional!")
        print("\nGenerated outputs:")
        print("  - results/test_fresh/")
        print("  - results/test_resume/")
        print("  - results/test_eval/")
        print("  - results/test_early_stop/")
        print("  - results/test_metrics/")
        print("\nNext steps:")
        print("  1. Try: python scripts/train.py --episodes 100 --output results/my_run")
        print("  2. Resume: python scripts/train.py --resume results/my_run/checkpoints/latest.pt")
        print("  3. Evaluate: python scripts/train.py --eval-only --checkpoint results/my_run/checkpoints/best.pt")

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()



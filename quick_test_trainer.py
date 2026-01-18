"""Quick test of training orchestration - validates core functionality."""

import sys
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent))

from src.rl.agents.dqn_agent import DQNAgent, DQNConfig
from src.rl.training.trainer import Trainer, TrainingConfig
import numpy as np


class MockEnv:
    """Minimal mock environment."""
    def __init__(self, seed=None):
        if seed:
            np.random.seed(seed)
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return np.random.randn(60).astype(np.float32)

    def step(self, action):
        self.step_count += 1
        reward = np.random.uniform(5, 15) if action == 1 else np.random.uniform(-1, 5)
        reward += self.step_count * 0.1  # Learning signal
        next_state = np.random.randn(60).astype(np.float32)
        done = self.step_count >= 50
        return next_state, reward, done, {}


def main():
    print("=" * 60)
    print("QUICK TRAINER TEST")
    print("=" * 60)

    # Setup
    output_dir = "results/quick_test"
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)

    print("\n1. Creating agent and environment...")
    env = MockEnv(seed=42)
    agent_config = DQNConfig(state_dim=60, action_dim=7, hidden_dims=[32, 16], seed=42)
    agent = DQNAgent(agent_config, seed=42)
    print(f"   Agent: {type(agent).__name__}, Device: {agent.device}")

    print("\n2. Creating trainer...")
    training_config = TrainingConfig(
        max_episodes=50,
        max_steps_per_episode=50,
        eval_frequency=20,
        eval_episodes=3,
        checkpoint_frequency=25,
        log_frequency=10,
        early_stopping=False,
        seed=42
    )
    trainer = Trainer(agent, env, training_config, output_dir=output_dir)
    print(f"   Output: {output_dir}")

    print("\n3. Training for 50 episodes...")
    final_stats = trainer.train()

    print("\n4. Verifying outputs...")
    checkpoint_dir = Path(output_dir) / "checkpoints"
    assert checkpoint_dir.exists(), "Checkpoints dir missing"
    assert (checkpoint_dir / "best.pt").exists(), "Best checkpoint missing"
    assert (checkpoint_dir / "latest.pt").exists(), "Latest checkpoint missing"
    assert (Path(output_dir) / "training_curves.png").exists(), "Curves missing"
    print("   [OK] All files created")

    print("\n5. Testing checkpoint resume...")
    output_dir2 = "results/quick_test_resume"
    if Path(output_dir2).exists():
        shutil.rmtree(output_dir2)

    env2 = MockEnv(seed=42)
    agent2 = DQNAgent(agent_config, seed=42)
    training_config2 = TrainingConfig(
        max_episodes=75,  # 25 more
        max_steps_per_episode=50,
        eval_frequency=20,
        log_frequency=10,
        early_stopping=False,
        seed=42
    )
    trainer2 = Trainer(agent2, env2, training_config2, output_dir=output_dir2)

    checkpoint_path = str(checkpoint_dir / "latest.pt")
    episode = trainer2.load_checkpoint(checkpoint_path)
    print(f"   [OK] Loaded from episode {episode}")

    print(f"\n6. Resuming training...")
    final_stats2 = trainer2.train()
    print(f"   [OK] Resumed training completed")

    print("\n7. Testing evaluation...")
    best_checkpoint = str(checkpoint_dir / "best.pt")
    eval_metrics = trainer2.evaluate_only(best_checkpoint, num_episodes=5)
    print(f"   [OK] Evaluation: {eval_metrics['mean_reward']:.2f}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print(f"\nFinal Results:")
    print(f"  Episodes: {final_stats['total_episodes']}")
    print(f"  Best Reward: {final_stats['best_eval_reward']:.2f}")
    print(f"  Time: {final_stats['total_time']:.1f}s")
    print(f"\nOutputs:")
    print(f"  {output_dir}/")
    print(f"  {output_dir2}/")
    print("\nTry the CLI:")
    print("  python scripts/train.py --episodes 100 --output results/my_run")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


"""
Example training script using the Gymnasium caching environment with Stable-Baselines3.

This demonstrates:
1. Creating a properly configured environment
2. Training multiple RL algorithms (PPO, DQN, A2C)
3. Evaluating trained agents
4. Comparing performance
5. Visualizing results
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from src.integration.gym_environment import CachingEnv, CacheEnvConfig, SimulatorConfig
from src.rl.state import StateConfig
from src.rl.reward import RewardConfig

try:
    from stable_baselines3 import PPO, DQN, A2C
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    print("⚠ Stable-Baselines3 not installed. Install with: pip install stable-baselines3")
    STABLE_BASELINES_AVAILABLE = False
    sys.exit(1)


def create_training_env(seed: int = 42):
    """Create a properly configured training environment."""
    config = CacheEnvConfig(
        # Use moderate complexity for training
        simulator_config=SimulatorConfig(
            num_apis=15,
            session_length_range=(10, 50),
            cascade_threshold=0.75
        ),
        # Moderate episode length
        max_steps_per_episode=200,
        # End on cascade to learn cascade prevention
        episode_end_on_cascade=True,
        # Enable logging for monitoring
        log_episode_metrics=True,
        seed=seed
    )

    env = CachingEnv(config)
    env = Monitor(env)  # Wrap with Monitor for SB3 logging
    return env


def create_eval_env(seed: int = 999):
    """Create evaluation environment with different seed."""
    config = CacheEnvConfig(
        simulator_config=SimulatorConfig(
            num_apis=15,
            session_length_range=(10, 50),
            cascade_threshold=0.75
        ),
        max_steps_per_episode=200,
        episode_end_on_cascade=True,
        log_episode_metrics=False,  # Disable logging during eval
        seed=seed
    )
    return Monitor(CachingEnv(config))


def train_ppo(total_timesteps: int = 50_000, save_dir: str = "models"):
    """Train a PPO agent."""
    print("\n" + "="*60)
    print("Training PPO Agent")
    print("="*60)

    # Create environments
    train_env = create_training_env(seed=42)
    eval_env = create_eval_env(seed=999)

    # Create callbacks
    Path(save_dir).mkdir(exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_dir}/ppo_best",
        log_path=f"{save_dir}/ppo_logs",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{save_dir}/ppo_checkpoints",
        name_prefix="ppo_model"
    )

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        verbose=1,
        tensorboard_log=f"{save_dir}/ppo_tensorboard"
    )

    # Train
    print(f"Training PPO for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )

    # Save final model
    model.save(f"{save_dir}/ppo_final")
    print(f"✓ PPO training complete. Model saved to {save_dir}/ppo_final")

    train_env.close()
    eval_env.close()

    return model


def train_dqn(total_timesteps: int = 50_000, save_dir: str = "models"):
    """Train a DQN agent."""
    print("\n" + "="*60)
    print("Training DQN Agent")
    print("="*60)

    # Create environments
    train_env = create_training_env(seed=43)
    eval_env = create_eval_env(seed=998)

    # Create callbacks
    Path(save_dir).mkdir(exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_dir}/dqn_best",
        log_path=f"{save_dir}/dqn_logs",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{save_dir}/dqn_checkpoints",
        name_prefix="dqn_model"
    )

    # Create DQN model
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=f"{save_dir}/dqn_tensorboard"
    )

    # Train
    print(f"Training DQN for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )

    # Save final model
    model.save(f"{save_dir}/dqn_final")
    print(f"✓ DQN training complete. Model saved to {save_dir}/dqn_final")

    train_env.close()
    eval_env.close()

    return model


def train_a2c(total_timesteps: int = 50_000, save_dir: str = "models"):
    """Train an A2C agent."""
    print("\n" + "="*60)
    print("Training A2C Agent")
    print("="*60)

    # Create environments
    train_env = create_training_env(seed=44)
    eval_env = create_eval_env(seed=997)

    # Create callbacks
    Path(save_dir).mkdir(exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_dir}/a2c_best",
        log_path=f"{save_dir}/a2c_logs",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{save_dir}/a2c_checkpoints",
        name_prefix="a2c_model"
    )

    # Create A2C model
    model = A2C(
        "MlpPolicy",
        train_env,
        learning_rate=7e-4,
        n_steps=5,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=f"{save_dir}/a2c_tensorboard"
    )

    # Train
    print(f"Training A2C for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )

    # Save final model
    model.save(f"{save_dir}/a2c_final")
    print(f"✓ A2C training complete. Model saved to {save_dir}/a2c_final")

    train_env.close()
    eval_env.close()

    return model


def evaluate_agent(model, n_episodes: int = 20, render: bool = False):
    """Evaluate a trained agent."""
    eval_env = create_eval_env(seed=12345)

    all_metrics = []
    episode_rewards = []

    for episode in range(n_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated

            if render and episode == 0:  # Render first episode only
                eval_env.envs[0].render(mode='human')

        episode_rewards.append(episode_reward)

        # Get episode metrics
        if 'episode' in info:
            metrics = eval_env.envs[0].get_episode_metrics()
            all_metrics.append(metrics)

    eval_env.close()

    # Compute statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    # Aggregate metrics
    avg_cache_hit_rate = np.mean([m['cache_hit_rate'] for m in all_metrics])
    avg_pred_accuracy = np.mean([m['prediction_accuracy'] for m in all_metrics])
    cascade_count = sum(m['cascade_occurred'] for m in all_metrics)

    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'episode_rewards': episode_rewards,
        'avg_cache_hit_rate': avg_cache_hit_rate,
        'avg_pred_accuracy': avg_pred_accuracy,
        'cascade_count': cascade_count,
        'n_episodes': n_episodes
    }


def compare_agents(models_dict: dict, n_episodes: int = 20):
    """Compare performance of multiple agents."""
    print("\n" + "="*60)
    print("Agent Performance Comparison")
    print("="*60)

    results = {}
    for name, model in models_dict.items():
        print(f"\nEvaluating {name}...")
        results[name] = evaluate_agent(model, n_episodes=n_episodes)

    # Print comparison table
    print("\n" + "-"*60)
    print(f"{'Agent':<15} {'Mean Reward':<15} {'Hit Rate':<15} {'Cascades':<10}")
    print("-"*60)

    for name, result in results.items():
        print(
            f"{name:<15} "
            f"{result['mean_reward']:>7.2f} ± {result['std_reward']:<5.2f} "
            f"{result['avg_cache_hit_rate']:>6.1%}        "
            f"{result['cascade_count']:>3}/{result['n_episodes']}"
        )

    print("-"*60)

    # Determine best agent
    best_agent = max(results.items(), key=lambda x: x[1]['mean_reward'])
    print(f"\n🏆 Best Agent: {best_agent[0]} with mean reward {best_agent[1]['mean_reward']:.2f}")

    return results


def plot_training_comparison(results: dict, save_path: str = "comparison.png"):
    """Plot comparison of training results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("RL Agent Performance Comparison", fontsize=16)

    agents = list(results.keys())

    # Plot 1: Mean Rewards
    ax = axes[0, 0]
    means = [results[agent]['mean_reward'] for agent in agents]
    stds = [results[agent]['std_reward'] for agent in agents]
    ax.bar(agents, means, yerr=stds, capsize=5, alpha=0.7)
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Average Reward")
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Cache Hit Rate
    ax = axes[0, 1]
    hit_rates = [results[agent]['avg_cache_hit_rate'] * 100 for agent in agents]
    ax.bar(agents, hit_rates, alpha=0.7, color='green')
    ax.set_ylabel("Cache Hit Rate (%)")
    ax.set_title("Average Cache Hit Rate")
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Cascade Events
    ax = axes[1, 0]
    cascades = [results[agent]['cascade_count'] for agent in agents]
    ax.bar(agents, cascades, alpha=0.7, color='red')
    ax.set_ylabel("Number of Cascades")
    ax.set_title("Cascade Failure Events")
    ax.grid(axis='y', alpha=0.3)

    # Plot 4: Reward Distribution
    ax = axes[1, 1]
    for agent in agents:
        ax.hist(results[agent]['episode_rewards'], alpha=0.5, label=agent, bins=15)
    ax.set_xlabel("Episode Reward")
    ax.set_ylabel("Frequency")
    ax.set_title("Reward Distribution")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to {save_path}")
    plt.close()


def main():
    """Main training and evaluation pipeline."""
    print("\n" + "#"*60)
    print("# RL Training Pipeline for Intelligent Caching")
    print("#"*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Configuration
    TRAIN_TIMESTEPS = 50_000  # Adjust based on available time
    N_EVAL_EPISODES = 20
    SAVE_DIR = "trained_models"

    # Train agents
    models = {}

    print("\n🚀 Starting training...")

    # Train PPO
    models['PPO'] = train_ppo(total_timesteps=TRAIN_TIMESTEPS, save_dir=SAVE_DIR)

    # Train DQN
    models['DQN'] = train_dqn(total_timesteps=TRAIN_TIMESTEPS, save_dir=SAVE_DIR)

    # Train A2C
    models['A2C'] = train_a2c(total_timesteps=TRAIN_TIMESTEPS, save_dir=SAVE_DIR)

    # Compare agents
    results = compare_agents(models, n_episodes=N_EVAL_EPISODES)

    # Plot comparison
    plot_training_comparison(results, save_path=f"{SAVE_DIR}/comparison.png")

    print("\n" + "#"*60)
    print("# Training Complete!")
    print("#"*60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nModels saved to: {SAVE_DIR}/")
    print("To use trained models:")
    print(f"  from stable_baselines3 import PPO")
    print(f"  model = PPO.load('{SAVE_DIR}/ppo_final')")
    print(f"  # Then use model.predict(obs) for inference")


if __name__ == "__main__":
    main()


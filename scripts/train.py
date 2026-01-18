"""
Training script for DQN agent.

Command-line interface for training, evaluation, and resuming training.

Examples:
    # Start fresh training
    python scripts/train.py --episodes 1000 --output results/run1

    # Use config file
    python scripts/train.py --config configs/default.yaml --episodes 5000

    # Resume from checkpoint
    python scripts/train.py --resume results/run1/checkpoints/latest.pt --episodes 10000

    # Evaluate trained model
    python scripts/train.py --eval-only --checkpoint results/run1/checkpoints/best.pt

    # With Weights & Biases logging
    python scripts/train.py --wandb --episodes 10000
"""

import click
import yaml
import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl.agents.dqn_agent import DQNAgent, DoubleDQNAgent, DQNConfig
from src.rl.training.trainer import Trainer, TrainingConfig


class MockCacheEnvironment:
    """
    Mock caching environment for training validation.

    This will be replaced with the actual caching environment in Phase 6.
    For now, provides a simple environment for testing the training pipeline.
    """

    def __init__(self, state_dim=60, action_dim=7, seed=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0
        self.max_steps = 100

    def reset(self):
        """Reset environment to initial state."""
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        """Generate current state representation."""
        # Simulated state: random values representing cache metrics
        state = np.random.randn(self.state_dim).astype(np.float32)
        return state

    def step(self, action):
        """
        Execute action and return (next_state, reward, done, info).

        Args:
            action: Action index (0-6)

        Returns:
            next_state: New state after action
            reward: Reward for this step
            done: Whether episode ended
            info: Additional information dict
        """
        self.step_count += 1

        # Simulate action effects with varying rewards
        if action == 1:  # CACHE_CURRENT
            reward = np.random.uniform(5, 15)
        elif action in [2, 3, 4]:  # PREFETCH variants
            reward = np.random.uniform(-2, 10)
        elif action in [5, 6]:  # EVICT variants
            reward = np.random.uniform(-5, 2)
        else:  # DO_NOTHING
            reward = np.random.uniform(-1, 1)

        # Add some learning signal: reward increases over time (simulating learning)
        progress = self.step_count / self.max_steps
        reward += progress * 5  # Gradual improvement

        # Generate next state
        next_state = self._get_state()

        # Episode ends after max_steps
        done = (self.step_count >= self.max_steps)

        # Info
        info = {
            'step': self.step_count,
        }

        return next_state, reward, done, info


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_agent_from_config(config: dict, seed: int = None) -> DQNAgent:
    """Create DQN agent from configuration."""
    # Extract agent config
    agent_type = config.get('rl', {}).get('algorithm', 'dqn')

    # Network config
    network_config = config.get('rl', {}).get('network', {})
    hidden_dims = network_config.get('hidden_sizes', [256, 256])

    # Training config
    learning_rate = config.get('rl', {}).get('learning_rate', 0.001)
    discount = config.get('rl', {}).get('discount', 0.99)

    # Exploration config
    exploration = config.get('rl', {}).get('exploration', {})
    epsilon_start = exploration.get('initial_epsilon', 1.0)
    epsilon_end = exploration.get('final_epsilon', 0.01)
    epsilon_decay_steps = exploration.get('decay_steps', 100000)

    # Compute epsilon decay rate from steps
    # epsilon_new = epsilon_old * decay_rate
    # After decay_steps: epsilon_end = epsilon_start * decay_rate^decay_steps
    # decay_rate = (epsilon_end / epsilon_start)^(1/decay_steps)
    if epsilon_decay_steps > 0:
        epsilon_decay = (epsilon_end / epsilon_start) ** (1.0 / epsilon_decay_steps)
    else:
        epsilon_decay = 0.995  # Default

    # Buffer config
    buffer_size = config.get('rl', {}).get('replay_buffer_size', 100000)
    batch_size = config.get('rl', {}).get('batch_size', 64)

    # Target network update
    target_update_freq = config.get('rl', {}).get('target_update_freq', 1000)

    # Create agent config
    agent_config = DQNConfig(
        state_dim=60,  # Match mock environment
        action_dim=7,  # 7 caching actions
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
        gamma=discount,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        buffer_size=buffer_size,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        device='auto',
        seed=seed,
    )

    # Create agent
    if agent_type == 'double_dqn':
        agent = DoubleDQNAgent(agent_config, seed=seed)
    else:
        agent = DQNAgent(agent_config, seed=seed)

    return agent


def create_training_config_from_dict(config: dict, overrides: dict = None) -> TrainingConfig:
    """Create training config from dictionary with optional overrides."""
    training_config = config.get('training', {})

    # Default values
    params = {
        'max_episodes': training_config.get('max_episodes', 5000),
        'max_steps_per_episode': training_config.get('max_steps_per_episode', 1000),
        'eval_frequency': training_config.get('eval_frequency', 100),
        'checkpoint_frequency': training_config.get('checkpoint_frequency', 500),
        'eval_episodes': training_config.get('eval_episodes', 10),
        'early_stopping': training_config.get('early_stopping', True),
        'patience': training_config.get('patience', 50),
        'min_episodes': training_config.get('min_episodes', 1000),
        'log_frequency': training_config.get('log_frequency', 10),
        'checkpoint_dir': training_config.get('checkpoint_dir', 'checkpoints'),
    }

    # Apply overrides
    if overrides:
        params.update(overrides)

    return TrainingConfig(**params)


@click.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to config file')
@click.option('--episodes', '-e', type=int, help='Override max_episodes')
@click.option('--resume', '-r', type=click.Path(exists=True), help='Resume from checkpoint path')
@click.option('--eval-only', is_flag=True, help='Just evaluate a trained model')
@click.option('--checkpoint', type=click.Path(exists=True), help='Path to model for --eval-only')
@click.option('--output', '-o', type=str, default='results/training_run', help='Output directory')
@click.option('--seed', type=int, help='Random seed')
@click.option('--wandb', is_flag=True, help='Enable Weights & Biases logging')
@click.option('--agent-type', type=click.Choice(['dqn', 'double_dqn']), default='dqn', help='Agent type')
@click.option('--eval-episodes', type=int, help='Number of evaluation episodes')
def main(config, episodes, resume, eval_only, checkpoint, output, seed, wandb, agent_type, eval_episodes):
    """
    Train or evaluate a DQN agent for cache management.

    Examples:

        # Start fresh training
        python scripts/train.py --episodes 1000 --output results/run1

        # Use config file
        python scripts/train.py --config configs/default.yaml

        # Resume from checkpoint
        python scripts/train.py --resume results/run1/checkpoints/latest.pt

        # Evaluate trained model
        python scripts/train.py --eval-only --checkpoint results/run1/checkpoints/best.pt
    """

    click.echo("=" * 80)
    click.echo("DQN TRAINING SCRIPT")
    click.echo("=" * 80)

    # Load config if provided
    if config:
        click.echo(f"Loading config from {config}")
        config_dict = load_config(config)
    else:
        click.echo("Using default configuration")
        config_dict = {}

    # Set seed
    if seed is None:
        seed = config_dict.get('seed', 42)

    click.echo(f"Random seed: {seed}")

    # Evaluation-only mode
    if eval_only:
        if not checkpoint:
            click.echo("Error: --checkpoint required for --eval-only mode", err=True)
            sys.exit(1)

        click.echo(f"\nEVALUATION MODE")
        click.echo(f"Checkpoint: {checkpoint}")

        # Create environment
        env = MockCacheEnvironment(state_dim=60, action_dim=7, seed=seed)

        # Create agent (architecture must match checkpoint)
        agent = create_agent_from_config(config_dict, seed=seed)

        # Create training config (for evaluation parameters)
        training_overrides = {}
        if eval_episodes:
            training_overrides['eval_episodes'] = eval_episodes

        training_config = create_training_config_from_dict(config_dict, training_overrides)

        # Create trainer
        trainer = Trainer(agent, env, training_config, output_dir=output)

        # Evaluate
        eval_metrics = trainer.evaluate_only(checkpoint, num_episodes=eval_episodes)

        click.echo("\n" + "=" * 80)
        click.echo("EVALUATION COMPLETE")
        click.echo("=" * 80)
        click.echo(f"Mean Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
        click.echo(f"Min/Max: {eval_metrics['min_reward']:.2f} / {eval_metrics['max_reward']:.2f}")

        return

    # Training mode
    click.echo(f"\nTRAINING MODE")
    click.echo(f"Output directory: {output}")

    # Create environment
    env = MockCacheEnvironment(state_dim=60, action_dim=7, seed=seed)
    click.echo(f"Environment: MockCacheEnvironment (state_dim=60, action_dim=7)")

    # Create agent
    agent = create_agent_from_config(config_dict, seed=seed)
    click.echo(f"Agent: {type(agent).__name__}")
    click.echo(f"Device: {agent.device}")

    # Create training config with overrides
    training_overrides = {}
    if episodes:
        training_overrides['max_episodes'] = episodes
    if seed:
        training_overrides['seed'] = seed
    if wandb:
        training_overrides['use_wandb'] = True

    training_config = create_training_config_from_dict(config_dict, training_overrides)
    click.echo(f"Max episodes: {training_config.max_episodes}")
    click.echo(f"Evaluation frequency: {training_config.eval_frequency}")

    # Create trainer
    trainer = Trainer(agent, env, training_config, output_dir=output)

    # Resume from checkpoint if specified
    if resume:
        click.echo(f"\nResuming from checkpoint: {resume}")
        trainer.load_checkpoint(resume)
        click.echo(f"Resuming from episode {trainer.current_episode}")

    # Start training
    click.echo("\n" + "=" * 80)
    click.echo("STARTING TRAINING")
    click.echo("=" * 80)

    try:
        final_stats = trainer.train()

        # Print final results
        click.echo("\n" + "=" * 80)
        click.echo("TRAINING COMPLETE")
        click.echo("=" * 80)
        click.echo(f"Total episodes: {final_stats['total_episodes']}")
        click.echo(f"Best eval reward: {final_stats['best_eval_reward']:.2f}")
        click.echo(f"Total time: {final_stats['total_time']:.2f}s")
        click.echo(f"Avg episode time: {final_stats['avg_episode_time']:.3f}s")
        click.echo(f"\nCheckpoints saved to: {trainer.checkpoint_dir}")
        click.echo(f"Logs saved to: {trainer.log_dir}")
        click.echo(f"Training curves: {trainer.output_dir / 'training_curves.png'}")

    except KeyboardInterrupt:
        click.echo("\n\nTraining interrupted by user")
        click.echo(f"Emergency checkpoint saved to: {trainer.checkpoint_dir}")

    except Exception as e:
        click.echo(f"\n\nTraining failed with error: {e}", err=True)
        click.echo(f"Emergency checkpoint saved to: {trainer.checkpoint_dir}")
        raise


if __name__ == "__main__":
    main()


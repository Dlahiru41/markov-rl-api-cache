"""
Training orchestration for DQN agents.

This module manages the complete training process including:
- Episode execution
- Experience collection
- Periodic evaluation
- Checkpointing
- Early stopping
- Logging and metrics tracking
"""

import os
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import logging

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Configuration for DQN training orchestration.

    Attributes:
        max_episodes: Maximum training episodes
        max_steps_per_episode: Step limit per episode
        eval_frequency: Evaluate every N episodes
        eval_episodes: Number of episodes for evaluation
        checkpoint_frequency: Save checkpoint every N episodes
        checkpoint_dir: Directory to save checkpoints
        early_stopping: Whether to stop if no improvement
        patience: Episodes without improvement before stopping
        min_episodes: Minimum episodes before early stopping
        seed: Random seed for reproducibility
        log_frequency: How often to log progress (episodes)
        use_wandb: Whether to log to Weights & Biases
    """
    max_episodes: int = 50000
    max_steps_per_episode: int = 1000
    eval_frequency: int = 100
    eval_episodes: int = 10
    checkpoint_frequency: int = 500
    checkpoint_dir: str = "checkpoints"
    early_stopping: bool = True
    patience: int = 50
    min_episodes: int = 1000
    seed: Optional[int] = None
    log_frequency: int = 10
    use_wandb: bool = False

    # Additional useful options
    save_best_only: bool = True
    verbose: bool = True
    plot_frequency: int = 1000  # Plot every N episodes


class Trainer:
    """
    Training orchestrator for DQN agents.

    Manages the complete training loop including episode execution,
    evaluation, checkpointing, and early stopping.
    """

    def __init__(
        self,
        agent,
        environment,
        config: TrainingConfig,
        output_dir: Optional[str] = None
    ):
        """
        Initialize trainer.

        Args:
            agent: DQN agent to train
            environment: Training environment
            config: Training configuration
            output_dir: Base output directory for logs and checkpoints
        """
        self.agent = agent
        self.env = environment
        self.config = config

        # Setup output directory
        if output_dir is None:
            output_dir = f"training_run_{int(time.time())}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup checkpoint directory
        self.checkpoint_dir = self.output_dir / config.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup logs directory
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Set random seeds
        if config.seed is not None:
            self._set_seed(config.seed)

        # Training state
        self.current_episode = 0
        self.best_eval_reward = -np.inf
        self.episodes_without_improvement = 0

        # Metrics tracking
        self.train_rewards = []
        self.train_lengths = []
        self.eval_rewards = []
        self.eval_episodes_list = []
        self.losses = []
        self.epsilons = []
        self.q_values = []

        # Timing
        self.start_time = None
        self.episode_times = deque(maxlen=100)

        # Weights & Biases
        self.use_wandb = config.use_wandb
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
                self.wandb_initialized = False
            except ImportError:
                logger.warning("wandb not installed, disabling wandb logging")
                self.use_wandb = False

        logger.info(f"Trainer initialized with output dir: {self.output_dir}")
        logger.info(f"Config: {config}")

    def _setup_logging(self):
        """Setup file logging."""
        log_file = self.log_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        logger.info(f"Random seed set to {seed}")

    def train(self) -> Dict[str, Any]:
        """
        Run the main training loop.

        Returns:
            Dictionary with training statistics and metrics
        """
        logger.info("=" * 80)
        logger.info("STARTING TRAINING")
        logger.info("=" * 80)
        logger.info(f"Max episodes: {self.config.max_episodes}")
        logger.info(f"Max steps per episode: {self.config.max_steps_per_episode}")
        logger.info(f"Evaluation frequency: {self.config.eval_frequency}")
        logger.info(f"Checkpoint frequency: {self.config.checkpoint_frequency}")

        # Initialize wandb if enabled
        if self.use_wandb and not self.wandb_initialized:
            self._init_wandb()

        self.start_time = time.time()

        try:
            # Main training loop
            while self.current_episode < self.config.max_episodes:
                episode_start_time = time.time()

                # Run training episode
                episode_metrics = self._run_episode(evaluate=False)

                # Record metrics
                self.train_rewards.append(episode_metrics['total_reward'])
                self.train_lengths.append(episode_metrics['episode_length'])
                if 'loss' in episode_metrics:
                    self.losses.append(episode_metrics['loss'])
                if 'epsilon' in episode_metrics:
                    self.epsilons.append(episode_metrics['epsilon'])
                if 'q_mean' in episode_metrics:
                    self.q_values.append(episode_metrics['q_mean'])

                # Track episode time
                episode_time = time.time() - episode_start_time
                self.episode_times.append(episode_time)

                self.current_episode += 1

                # Logging
                if self.current_episode % self.config.log_frequency == 0:
                    self._log_progress(episode_metrics)

                # Evaluation
                if self.current_episode % self.config.eval_frequency == 0:
                    eval_metrics = self._evaluate()
                    self._log_evaluation(eval_metrics)

                    # Check for improvement and save best model
                    mean_reward = eval_metrics['mean_reward']
                    if mean_reward > self.best_eval_reward:
                        logger.info(f"New best model! Reward: {mean_reward:.2f} (previous: {self.best_eval_reward:.2f})")
                        self.best_eval_reward = mean_reward
                        self.episodes_without_improvement = 0
                        self._save_checkpoint(self.current_episode, is_best=True)
                    else:
                        self.episodes_without_improvement += self.config.eval_frequency

                    # Early stopping check
                    if self._should_stop_early():
                        logger.info(f"Early stopping triggered after {self.episodes_without_improvement} episodes without improvement")
                        break

                # Regular checkpointing
                if self.current_episode % self.config.checkpoint_frequency == 0:
                    self._save_checkpoint(self.current_episode, is_best=False)

                # Plot training curves periodically
                if self.current_episode % self.config.plot_frequency == 0:
                    self.plot_training_curves()

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self._save_checkpoint(self.current_episode, is_best=False, emergency=True)

        except Exception as e:
            logger.error(f"Training failed with error: {e}", exc_info=True)
            self._save_checkpoint(self.current_episode, is_best=False, emergency=True)
            raise

        finally:
            # Final cleanup
            self._finalize_training()

        # Compute final statistics
        final_stats = self._compute_final_statistics()

        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total episodes: {self.current_episode}")
        logger.info(f"Best eval reward: {self.best_eval_reward:.2f}")
        logger.info(f"Total time: {final_stats['total_time']:.2f}s")

        return final_stats

    def _run_episode(self, evaluate: bool = False) -> Dict[str, Any]:
        """
        Run a single episode.

        Args:
            evaluate: If True, run in evaluation mode (no exploration, no learning)

        Returns:
            Dictionary with episode metrics
        """
        state = self.env.reset()
        total_reward = 0.0
        episode_length = 0
        done = False

        episode_losses = []
        episode_q_values = []

        while not done and episode_length < self.config.max_steps_per_episode:
            # Select action
            action = self.agent.select_action(state, evaluate=evaluate)

            # Execute action
            next_state, reward, done, info = self.env.step(action)

            total_reward += reward
            episode_length += 1

            # Store transition and train (only during training)
            if not evaluate:
                self.agent.store_transition(state, action, reward, next_state, done)

                # Train agent
                train_metrics = self.agent.train_step()
                if train_metrics:
                    episode_losses.append(train_metrics['loss'])
                    if 'q_mean' in train_metrics:
                        episode_q_values.append(train_metrics['q_mean'])

            state = next_state

        # Compile episode metrics
        metrics = {
            'total_reward': total_reward,
            'episode_length': episode_length,
            'done': done,
        }

        if episode_losses:
            metrics['loss'] = np.mean(episode_losses)

        if episode_q_values:
            metrics['q_mean'] = np.mean(episode_q_values)

        # Add agent metrics
        agent_metrics = self.agent.get_metrics()
        if 'epsilon' in agent_metrics:
            metrics['epsilon'] = agent_metrics['epsilon']

        return metrics

    def _evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the current policy.

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating agent (episode {self.current_episode})...")

        eval_rewards = []
        eval_lengths = []

        for _ in range(self.config.eval_episodes):
            episode_metrics = self._run_episode(evaluate=True)
            eval_rewards.append(episode_metrics['total_reward'])
            eval_lengths.append(episode_metrics['episode_length'])

        eval_metrics = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'episode': self.current_episode,
        }

        # Track for plotting
        self.eval_rewards.append(eval_metrics['mean_reward'])
        self.eval_episodes_list.append(self.current_episode)

        return eval_metrics

    def _should_stop_early(self) -> bool:
        """Check if early stopping criteria are met."""
        if not self.config.early_stopping:
            return False

        if self.current_episode < self.config.min_episodes:
            return False

        if self.episodes_without_improvement >= self.config.patience:
            return True

        return False

    def _save_checkpoint(
        self,
        episode: int,
        is_best: bool = False,
        emergency: bool = False
    ):
        """
        Save a training checkpoint.

        Args:
            episode: Current episode number
            is_best: Whether this is the best model so far
            emergency: Whether this is an emergency save (e.g., interrupted training)
        """
        # Prepare checkpoint data
        checkpoint = {
            'episode': episode,
            'agent_state': self.agent.online_net.state_dict(),
            'target_state': self.agent.target_net.state_dict(),
            'optimizer_state': self.agent.optimizer.state_dict(),
            'epsilon': self.agent.epsilon,
            'steps_done': self.agent.steps_done,
            'best_eval_reward': self.best_eval_reward,
            'episodes_without_improvement': self.episodes_without_improvement,
            'train_rewards': self.train_rewards,
            'train_lengths': self.train_lengths,
            'eval_rewards': self.eval_rewards,
            'eval_episodes_list': self.eval_episodes_list,
            'losses': self.losses,
            'epsilons': self.epsilons,
            'config': asdict(self.config),
        }

        # Save paths
        if emergency:
            checkpoint_path = self.checkpoint_dir / f"emergency_ep{episode}.pt"
            logger.warning(f"Saving emergency checkpoint: {checkpoint_path}")
        elif is_best:
            checkpoint_path = self.checkpoint_dir / "best.pt"
            logger.info(f"Saving best checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_ep{episode}.pt"
            logger.info(f"Saving checkpoint: {checkpoint_path}")

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)

        # Always update latest checkpoint
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        # Save metadata as JSON for easy inspection
        metadata = {
            'episode': episode,
            'best_eval_reward': self.best_eval_reward,
            'total_episodes': self.current_episode,
            'epsilon': float(self.agent.epsilon),
            'timestamp': time.time(),
        }
        metadata_path = checkpoint_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load a training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Episode number to resume from
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # PyTorch 2.6+ defaults to weights_only=True, but we need to load full checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.agent.device, weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions
            checkpoint = torch.load(checkpoint_path, map_location=self.agent.device)

        # Restore agent state
        self.agent.online_net.load_state_dict(checkpoint['agent_state'])
        self.agent.target_net.load_state_dict(checkpoint['target_state'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.agent.epsilon = checkpoint['epsilon']
        self.agent.steps_done = checkpoint['steps_done']

        # Restore training state
        self.current_episode = checkpoint['episode']
        self.best_eval_reward = checkpoint['best_eval_reward']
        self.episodes_without_improvement = checkpoint['episodes_without_improvement']

        # Restore metrics
        self.train_rewards = checkpoint['train_rewards']
        self.train_lengths = checkpoint['train_lengths']
        self.eval_rewards = checkpoint['eval_rewards']
        self.eval_episodes_list = checkpoint['eval_episodes_list']
        self.losses = checkpoint['losses']
        self.epsilons = checkpoint['epsilons']

        logger.info(f"Checkpoint loaded. Resuming from episode {self.current_episode}")
        logger.info(f"Best eval reward: {self.best_eval_reward:.2f}")

        return self.current_episode

    def _log_progress(self, episode_metrics: Dict[str, Any]):
        """Log training progress."""
        # Compute moving averages
        recent_rewards = self.train_rewards[-self.config.log_frequency:]
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)

        recent_lengths = self.train_lengths[-self.config.log_frequency:]
        mean_length = np.mean(recent_lengths)

        # Compute time estimates
        elapsed_time = time.time() - self.start_time
        avg_episode_time = np.mean(self.episode_times)
        episodes_remaining = self.config.max_episodes - self.current_episode
        eta = avg_episode_time * episodes_remaining

        # Log to console
        if self.config.verbose:
            log_msg = (
                f"Episode {self.current_episode}/{self.config.max_episodes} | "
                f"Reward: {mean_reward:.2f}+/-{std_reward:.2f} | "
                f"Length: {mean_length:.0f} | "
            )

            if 'epsilon' in episode_metrics:
                log_msg += f"eps: {episode_metrics['epsilon']:.3f} | "

            if 'loss' in episode_metrics:
                log_msg += f"Loss: {episode_metrics['loss']:.4f} | "

            log_msg += f"Time: {elapsed_time:.0f}s | ETA: {eta:.0f}s"

            logger.info(log_msg)

        # Log to wandb
        if self.use_wandb:
            wandb_metrics = {
                'episode': self.current_episode,
                'train/reward': mean_reward,
                'train/reward_std': std_reward,
                'train/episode_length': mean_length,
                'train/episodes_per_second': 1.0 / avg_episode_time,
            }

            if 'epsilon' in episode_metrics:
                wandb_metrics['train/epsilon'] = episode_metrics['epsilon']

            if 'loss' in episode_metrics:
                wandb_metrics['train/loss'] = episode_metrics['loss']

            if 'q_mean' in episode_metrics:
                wandb_metrics['train/q_mean'] = episode_metrics['q_mean']

            self.wandb.log(wandb_metrics)

    def _log_evaluation(self, eval_metrics: Dict[str, Any]):
        """Log evaluation results."""
        logger.info("-" * 80)
        logger.info(f"EVALUATION (Episode {self.current_episode})")
        logger.info(f"  Mean Reward: {eval_metrics['mean_reward']:.2f} +/- {eval_metrics['std_reward']:.2f}")
        logger.info(f"  Min/Max: {eval_metrics['min_reward']:.2f} / {eval_metrics['max_reward']:.2f}")
        logger.info(f"  Mean Length: {eval_metrics['mean_length']:.0f}")
        logger.info(f"  Best So Far: {self.best_eval_reward:.2f}")
        logger.info("-" * 80)

        # Log to wandb
        if self.use_wandb:
            wandb_metrics = {
                'episode': self.current_episode,
                'eval/mean_reward': eval_metrics['mean_reward'],
                'eval/std_reward': eval_metrics['std_reward'],
                'eval/min_reward': eval_metrics['min_reward'],
                'eval/max_reward': eval_metrics['max_reward'],
                'eval/mean_length': eval_metrics['mean_length'],
                'eval/best_reward': self.best_eval_reward,
            }
            self.wandb.log(wandb_metrics)

    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            self.wandb.init(
                project="markov-rl-api-cache",
                config=asdict(self.config),
                dir=str(self.output_dir),
            )
            self.wandb_initialized = True
            logger.info("Weights & Biases initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            self.use_wandb = False

    def _finalize_training(self):
        """Finalize training - save final plots and logs."""
        # Plot final training curves
        self.plot_training_curves(output_path=self.output_dir / "training_curves.png")

        # Save final metrics
        self._save_metrics()

        # Close wandb
        if self.use_wandb and self.wandb_initialized:
            self.wandb.finish()

    def _compute_final_statistics(self) -> Dict[str, Any]:
        """Compute final training statistics."""
        total_time = time.time() - self.start_time

        stats = {
            'total_episodes': self.current_episode,
            'best_eval_reward': self.best_eval_reward,
            'total_time': total_time,
            'avg_episode_time': np.mean(self.episode_times) if self.episode_times else 0,
            'final_epsilon': self.agent.epsilon,
        }

        if self.train_rewards:
            stats['final_train_reward_mean'] = np.mean(self.train_rewards[-100:])
            stats['final_train_reward_std'] = np.std(self.train_rewards[-100:])

        if self.eval_rewards:
            stats['final_eval_reward'] = self.eval_rewards[-1]

        return stats

    def _save_metrics(self):
        """Save all metrics to file."""
        metrics = {
            'train_rewards': self.train_rewards,
            'train_lengths': self.train_lengths,
            'eval_rewards': self.eval_rewards,
            'eval_episodes': self.eval_episodes_list,
            'losses': self.losses,
            'epsilons': self.epsilons,
            'q_values': self.q_values,
        }

        metrics_path = self.output_dir / "metrics.json"

        # Convert numpy types to native Python types for JSON serialization
        metrics_serializable = {}
        for key, value in metrics.items():
            if isinstance(value, list):
                metrics_serializable[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in value]
            else:
                metrics_serializable[key] = value

        with open(metrics_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)

        logger.info(f"Metrics saved to {metrics_path}")

    def plot_training_curves(self, output_path: Optional[str] = None):
        """
        Plot training curves.

        Args:
            output_path: Path to save the plot. If None, shows interactively.
        """
        if not self.train_rewards:
            logger.warning("No training data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)

        # Helper function for moving average
        def moving_average(data, window=100):
            if len(data) < window:
                window = len(data)
            return np.convolve(data, np.ones(window)/window, mode='valid')

        # Plot 1: Training Reward
        ax = axes[0, 0]
        episodes = np.arange(len(self.train_rewards))
        ax.plot(episodes, self.train_rewards, alpha=0.3, label='Raw')
        if len(self.train_rewards) > 10:
            smoothed = moving_average(self.train_rewards, window=min(100, len(self.train_rewards)//10))
            ax.plot(episodes[len(episodes)-len(smoothed):], smoothed, linewidth=2, label='Smoothed (MA-100)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Training Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Evaluation Reward
        ax = axes[0, 1]
        if self.eval_rewards:
            ax.plot(self.eval_episodes_list, self.eval_rewards, marker='o', linewidth=2)
            ax.axhline(y=self.best_eval_reward, color='r', linestyle='--', label=f'Best: {self.best_eval_reward:.2f}')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Mean Reward')
            ax.set_title('Evaluation Reward')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No evaluation data yet', ha='center', va='center', transform=ax.transAxes)

        # Plot 3: Loss
        ax = axes[1, 0]
        if self.losses:
            loss_episodes = np.arange(len(self.losses))
            ax.plot(loss_episodes, self.losses, alpha=0.3, label='Raw')
            if len(self.losses) > 10:
                smoothed = moving_average(self.losses, window=min(100, len(self.losses)//10))
                ax.plot(loss_episodes[len(loss_episodes)-len(smoothed):], smoothed, linewidth=2, label='Smoothed')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No loss data yet', ha='center', va='center', transform=ax.transAxes)

        # Plot 4: Epsilon Decay
        ax = axes[1, 1]
        if self.epsilons:
            epsilon_episodes = np.arange(len(self.epsilons))
            ax.plot(epsilon_episodes, self.epsilons, linewidth=2)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Epsilon')
            ax.set_title('Exploration Rate (Epsilon)')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No epsilon data yet', ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Training curves saved to {output_path}")
            plt.close(fig)
        else:
            plt.show()

    def evaluate_only(self, checkpoint_path: str, num_episodes: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate a trained model without training.

        Args:
            checkpoint_path: Path to model checkpoint
            num_episodes: Number of episodes to evaluate (uses config if None)

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("=" * 80)
        logger.info("EVALUATION MODE")
        logger.info("=" * 80)

        # Load checkpoint
        self.load_checkpoint(checkpoint_path)

        # Override eval episodes if specified
        if num_episodes is not None:
            original_eval_episodes = self.config.eval_episodes
            self.config.eval_episodes = num_episodes

        # Run evaluation
        eval_metrics = self._evaluate()

        # Detailed logging
        logger.info("=" * 80)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Model: {checkpoint_path}")
        logger.info(f"Episodes: {self.config.eval_episodes}")
        logger.info(f"Mean Reward: {eval_metrics['mean_reward']:.2f} +/- {eval_metrics['std_reward']:.2f}")
        logger.info(f"Min Reward: {eval_metrics['min_reward']:.2f}")
        logger.info(f"Max Reward: {eval_metrics['max_reward']:.2f}")
        logger.info(f"Mean Episode Length: {eval_metrics['mean_length']:.0f}")
        logger.info("=" * 80)

        # Restore original config
        if num_episodes is not None:
            self.config.eval_episodes = original_eval_episodes

        return eval_metrics


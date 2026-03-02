"""
Integration controller that orchestrates all system components.

This module provides a unified interface for training, evaluation, and deployment
of the intelligent caching system. It manages the lifecycle of all components
(Markov predictor, RL agent, cache manager, simulator) and coordinates their interactions.
"""

import logging
import time
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from enum import Enum

import numpy as np

from .gym_environment import CachingEnv, CacheEnvConfig
from ..markov.predictor import MarkovPredictor
from ..cache.cache_manager import CacheManager, CacheManagerConfig
from ..rl.agents.dqn_agent import DQNAgent, DQNConfig
from ..rl.training.trainer import Trainer, TrainingConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OperatingMode(str, Enum):
    """Operating modes for the controller."""
    TRAINING = "training"
    EVALUATION = "evaluation"
    DEPLOYMENT = "deployment"
    DEMO = "demo"


@dataclass
class ControllerConfig:
    """Configuration for the integration controller."""

    # Operating mode
    mode: str = "training"  # 'training', 'evaluation', 'deployment', 'demo'

    # Component configurations
    env_config: Optional[CacheEnvConfig] = None
    agent_config: Optional[DQNConfig] = None
    training_config: Optional[TrainingConfig] = None

    # Model paths
    markov_model_path: Optional[str] = None
    agent_model_path: Optional[str] = None

    # Output configuration
    output_dir: str = "results/default"

    # Optional features
    enable_monitoring: bool = False
    enable_api: bool = False

    # Logging
    log_level: str = "INFO"

    # API configuration
    api_port: int = 8080
    api_host: str = "0.0.0.0"

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate mode
        if self.mode not in [m.value for m in OperatingMode]:
            raise ValueError(f"Invalid mode: {self.mode}. Must be one of {[m.value for m in OperatingMode]}")

        # Create default configs if not provided
        if self.env_config is None:
            self.env_config = CacheEnvConfig()

        if self.agent_config is None:
            self.agent_config = DQNConfig()

        if self.training_config is None:
            self.training_config = TrainingConfig()

        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Validate model paths if in evaluation/deployment mode
        if self.mode in [OperatingMode.EVALUATION, OperatingMode.DEPLOYMENT]:
            if self.agent_model_path is None:
                logger.warning(f"No agent model path specified for {self.mode} mode")

    def save(self, path: str):
        """Save configuration to file."""
        config_dict = {
            'mode': self.mode,
            'output_dir': self.output_dir,
            'markov_model_path': self.markov_model_path,
            'agent_model_path': self.agent_model_path,
            'enable_monitoring': self.enable_monitoring,
            'enable_api': self.enable_api,
            'log_level': self.log_level,
            'api_port': self.api_port,
            'api_host': self.api_host
        }

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Configuration saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'ControllerConfig':
        """Load configuration from file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)

        return cls(**config_dict)


class IntegrationController:
    """
    Main controller that orchestrates all system components.

    Manages lifecycle of:
    - Markov predictor
    - Cache manager
    - Gymnasium environment
    - RL agent
    - Trainer
    - Monitoring (optional)
    - Control API (optional)

    Provides unified interface for:
    - Training
    - Evaluation
    - Deployment
    - Demonstration
    """

    def __init__(self, config: ControllerConfig):
        """
        Initialize the controller.

        Args:
            config: Controller configuration
        """
        self.config = config

        # Set up logging
        log_level = getattr(logging, config.log_level.upper())
        logging.getLogger().setLevel(log_level)
        logger.setLevel(log_level)

        # Component references (initialized in setup())
        self.markov_predictor: Optional[MarkovPredictor] = None
        self.cache_manager: Optional[CacheManager] = None
        self.env: Optional[CachingEnv] = None
        self.agent: Optional[DQNAgent] = None
        self.trainer: Optional[Trainer] = None

        # Control API and monitoring
        self.api_server = None
        self.metrics_registry = None
        self.metrics_collector = None   # MetricsCollector (set in _setup_monitoring)

        # State tracking
        self._is_setup = False
        self._is_running = False
        self._start_time = None
        self._training_interrupted = False

        # Performance tracking
        self.episode_count = 0
        self.total_steps = 0
        self.total_reward = 0.0
        self.best_eval_reward = -float('inf')

        logger.info(f"IntegrationController initialized in {config.mode} mode")
        logger.info(f"Output directory: {config.output_dir}")

    def setup(self) -> bool:
        """
        Initialize all components in the correct order.

        Returns:
            True if setup successful, False otherwise
        """
        if self._is_setup:
            logger.warning("Controller already set up")
            return True

        logger.info("Setting up IntegrationController...")

        try:
            # 1. Load or create Markov predictor
            self._setup_markov_predictor()

            # 2. Create cache manager (if needed)
            if self.config.mode in [OperatingMode.TRAINING, OperatingMode.EVALUATION, OperatingMode.DEMO]:
                self._setup_cache_manager()

            # 3. Create Gymnasium environment (if needed)
            if self.config.mode in [OperatingMode.TRAINING, OperatingMode.EVALUATION, OperatingMode.DEMO]:
                self._setup_environment()

            # 4. Load or create RL agent
            self._setup_agent()

            # 5. Set up trainer (if training mode)
            if self.config.mode == OperatingMode.TRAINING:
                self._setup_trainer()

            # 6. Set up monitoring (if enabled)
            if self.config.enable_monitoring:
                self._setup_monitoring()

            # 7. Start control API (if enabled)
            if self.config.enable_api:
                self._setup_api()

            self._is_setup = True
            logger.info("✓ Setup completed successfully")
            return True

        except Exception as e:
            logger.error(f"Setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _setup_markov_predictor(self):
        """Initialize or load Markov predictor."""
        logger.info("Setting up Markov predictor...")

        if self.config.markov_model_path and Path(self.config.markov_model_path).exists():
            # Load pre-trained model
            logger.info(f"Loading Markov model from {self.config.markov_model_path}")
            self.markov_predictor = MarkovPredictor.load(self.config.markov_model_path)
        else:
            # Create new predictor
            logger.info("Creating new Markov predictor")
            markov_config = self.config.env_config.markov_config or {}
            self.markov_predictor = MarkovPredictor(**markov_config)

            # If we have a model path but file doesn't exist, warn
            if self.config.markov_model_path:
                logger.warning(f"Markov model path specified but not found: {self.config.markov_model_path}")

        logger.info("✓ Markov predictor ready")

    def _setup_cache_manager(self):
        """Initialize cache manager."""
        logger.info("Setting up cache manager...")

        cache_config = self.config.env_config.cache_config or CacheManagerConfig()
        self.cache_manager = CacheManager(cache_config)

        # Start the cache manager
        if not self.cache_manager.start():
            raise RuntimeError("Failed to start cache manager")

        logger.info("✓ Cache manager ready")

    def _setup_environment(self):
        """Initialize Gymnasium environment."""
        logger.info("Setting up Gymnasium environment...")

        self.env = CachingEnv(self.config.env_config)

        logger.info(f"✓ Environment ready (obs_space={self.env.observation_space.shape}, action_space={self.env.action_space.n})")

    def _setup_agent(self):
        """Initialize or load RL agent."""
        logger.info("Setting up RL agent...")

        # Get state and action dimensions from environment
        if self.env:
            state_dim = self.env.observation_space.shape[0]
            action_dim = self.env.action_space.n
        else:
            # Use defaults if no environment (deployment mode)
            state_dim = self.config.agent_config.state_dim or 60
            action_dim = self.config.agent_config.action_dim or 7

        # Update agent config with dimensions
        self.config.agent_config.state_dim = state_dim
        self.config.agent_config.action_dim = action_dim

        # Create agent
        self.agent = DQNAgent(self.config.agent_config)

        # Load pre-trained model if specified
        if self.config.agent_model_path and Path(self.config.agent_model_path).exists():
            logger.info(f"Loading agent from {self.config.agent_model_path}")
            self.agent.load(self.config.agent_model_path)
        else:
            if self.config.agent_model_path:
                logger.warning(f"Agent model path specified but not found: {self.config.agent_model_path}")

        logger.info("✓ RL agent ready")

    def _setup_trainer(self):
        """Initialize trainer for training mode."""
        logger.info("Setting up trainer...")

        if not self.env or not self.agent:
            raise RuntimeError("Environment and agent must be set up before trainer")

        self.trainer = Trainer(
            env=self.env,
            agent=self.agent,
            config=self.config.training_config
        )

        logger.info("✓ Trainer ready")

    def _setup_monitoring(self):
        """
        Set up Prometheus metrics monitoring using the full MetricsCollector.

        Creates a MetricsCollector, starts the HTTP /metrics server on
        port 9200 (configurable via METRICS_PORT env var), and attaches the
        collector to this controller so downstream code can call
        ``self.metrics_collector.record_*()``.
        """
        logger.info("Setting up monitoring...")

        try:
            import os
            from src.monitoring.metrics import MetricsCollector, start_metrics_server
            from prometheus_client import generate_latest

            metrics_port = int(os.environ.get("METRICS_PORT", 9200))

            # Create collector (uses its own private registry)
            self.metrics_collector = MetricsCollector(service="api-cache")
            self.metrics_registry = self.metrics_collector.registry

            # Start HTTP server so Prometheus can scrape /metrics
            start_metrics_server(
                port=metrics_port,
                registry=self.metrics_collector.registry,
            )

            # Legacy dict kept for backward-compat with any existing callers
            # that access controller.metrics['cache_hit_rate'] etc.
            self.metrics = {
                'collector': self.metrics_collector,
            }

            logger.info(
                f"✓ Monitoring ready – Prometheus scrape endpoint: "
                f"http://0.0.0.0:{metrics_port}/metrics"
            )

        except ImportError as exc:
            logger.warning(
                f"prometheus_client not installed ({exc}), monitoring disabled. "
                "Install with: pip install prometheus-client"
            )
            self.config.enable_monitoring = False
        except Exception as exc:
            logger.warning(f"Monitoring setup failed ({exc}), continuing without metrics")

    def _setup_api(self):
        """Set up FastAPI control API."""
        logger.info("Setting up control API...")

        try:
            # Import is done here to avoid dependency if not needed
            from .api import create_app

            # Create API app with reference to this controller
            self.api_server = create_app(self)

            logger.info(f"✓ Control API ready (will serve on {self.config.api_host}:{self.config.api_port})")

        except ImportError:
            logger.warning("FastAPI not installed, API disabled")
            self.config.enable_api = False

    def start(self):
        """
        Begin active operation based on mode.

        Raises:
            RuntimeError: If not set up or mode not recognized
        """
        if not self._is_setup:
            raise RuntimeError("Controller not set up. Call setup() first.")

        if self._is_running:
            logger.warning("Controller already running")
            return

        self._is_running = True
        self._start_time = time.time()

        logger.info(f"Starting controller in {self.config.mode} mode...")

        try:
            if self.config.mode == OperatingMode.TRAINING:
                # Training is started explicitly via train() method
                logger.info("Training mode ready. Call train() to begin.")

            elif self.config.mode == OperatingMode.EVALUATION:
                # Evaluation is run explicitly via evaluate() method
                logger.info("Evaluation mode ready. Call evaluate() to begin.")

            elif self.config.mode == OperatingMode.DEPLOYMENT:
                logger.info("Deployment mode active. Ready to serve predictions.")
                # In deployment mode, we just wait for API calls

            elif self.config.mode == OperatingMode.DEMO:
                logger.info("Demo mode ready. Call run_demo() or step_demo().")

        except Exception as e:
            logger.error(f"Error starting controller: {e}")
            self._is_running = False
            raise

    def stop(self):
        """Gracefully shut down all components."""
        logger.info("Stopping IntegrationController...")

        self._is_running = False
        self._training_interrupted = True

        # Stop trainer if running
        if self.trainer and hasattr(self.trainer, 'stop'):
            self.trainer.stop()

        # Close environment
        if self.env:
            self.env.close()
            logger.info("✓ Environment closed")

        # Stop cache manager
        if self.cache_manager:
            self.cache_manager.stop()
            logger.info("✓ Cache manager stopped")

        # Save current state if appropriate
        if self.config.mode == OperatingMode.TRAINING and self.agent:
            checkpoint_path = Path(self.config.output_dir) / "checkpoint_final.pt"
            self.agent.save(str(checkpoint_path))
            logger.info(f"✓ Final checkpoint saved to {checkpoint_path}")

        logger.info("✓ Controller stopped")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current system status.

        Returns:
            Dictionary with status information
        """
        status = {
            'is_setup': self._is_setup,
            'is_running': self._is_running,
            'mode': self.config.mode,
            'uptime_seconds': time.time() - self._start_time if self._start_time else 0,
            'component_health': {
                'markov_predictor': self.markov_predictor is not None,
                'cache_manager': self.cache_manager is not None and self.cache_manager.is_running if self.cache_manager else False,
                'environment': self.env is not None,
                'agent': self.agent is not None,
                'trainer': self.trainer is not None,
            },
            'training_progress': {
                'episode_count': self.episode_count,
                'total_steps': self.total_steps,
                'total_reward': self.total_reward,
                'best_eval_reward': self.best_eval_reward if self.best_eval_reward > -float('inf') else None,
            } if self.config.mode == OperatingMode.TRAINING else None,
        }

        # Add performance metrics if available
        if self.env and hasattr(self.env, 'total_cache_hits'):
            total_requests = self.env.total_cache_hits + self.env.total_cache_misses
            status['performance_metrics'] = {
                'cache_hit_rate': self.env.total_cache_hits / total_requests if total_requests > 0 else 0.0,
                'total_cache_hits': self.env.total_cache_hits,
                'total_cache_misses': self.env.total_cache_misses,
            }

        return status

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics from all components.

        Returns:
            Dictionary with metrics from all components
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'controller': {
                'mode': self.config.mode,
                'uptime_seconds': time.time() - self._start_time if self._start_time else 0,
            }
        }

        # Markov predictor metrics
        if self.markov_predictor:
            metrics['markov'] = {
                'prediction_count': self.markov_predictor.prediction_count,
                'accuracy': self.markov_predictor.correct_predictions.get(1, 0) / max(1, self.markov_predictor.prediction_count),
                'vocab_size': self.markov_predictor.vocab_size,
            }

        # Cache metrics
        if self.cache_manager:
            cache_metrics = self.cache_manager.get_metrics()
            metrics['cache'] = cache_metrics

        # RL agent metrics
        if self.agent:
            metrics['agent'] = {
                'epsilon': self.agent.epsilon,
                'replay_buffer_size': len(self.agent.replay_buffer) if hasattr(self.agent, 'replay_buffer') else 0,
                'training_steps': self.agent.training_steps if hasattr(self.agent, 'training_steps') else 0,
            }

        # Training metrics
        if self.config.mode == OperatingMode.TRAINING:
            metrics['training'] = {
                'episode_count': self.episode_count,
                'total_steps': self.total_steps,
                'total_reward': self.total_reward,
                'average_reward': self.total_reward / max(1, self.episode_count),
                'best_eval_reward': self.best_eval_reward if self.best_eval_reward > -float('inf') else None,
            }

        # Environment metrics
        if self.env:
            metrics['environment'] = {
                'total_cache_hits': getattr(self.env, 'total_cache_hits', 0),
                'total_cache_misses': getattr(self.env, 'total_cache_misses', 0),
                'cascade_detected': getattr(self.env, 'cascade_detected', False),
            }

        return metrics

    def train(self, num_episodes: Optional[int] = None) -> Dict[str, Any]:
        """
        Run training for specified episodes.

        Args:
            num_episodes: Number of episodes to train (None = use config)

        Returns:
            Training summary dictionary
        """
        if not self._is_setup:
            raise RuntimeError("Controller not set up. Call setup() first.")

        if self.config.mode != OperatingMode.TRAINING:
            raise RuntimeError(f"Cannot train in {self.config.mode} mode")

        if not self.trainer:
            raise RuntimeError("Trainer not initialized")

        logger.info(f"Starting training for {num_episodes or self.config.training_config.num_episodes} episodes...")

        # Use config episodes if not specified
        if num_episodes is None:
            num_episodes = self.config.training_config.num_episodes

        # Run training
        self._training_interrupted = False
        training_history = self.trainer.train(num_episodes=num_episodes)

        # Update tracking
        self.episode_count += num_episodes

        # Save final model
        model_path = Path(self.config.output_dir) / "final_model.pt"
        self.agent.save(str(model_path))
        logger.info(f"Final model saved to {model_path}")

        # Create summary
        summary = {
            'num_episodes': num_episodes,
            'total_steps': sum(training_history.get('episode_lengths', [])),
            'mean_reward': np.mean(training_history.get('episode_rewards', [])),
            'std_reward': np.std(training_history.get('episode_rewards', [])),
            'final_epsilon': self.agent.epsilon,
            'training_time_seconds': training_history.get('training_time', 0),
            'best_eval_reward': self.best_eval_reward if self.best_eval_reward > -float('inf') else None,
        }

        # ── Push training history to Prometheus metrics ───────────────────
        if self.metrics_collector is not None:
            rewards = training_history.get('episode_rewards', [])
            lengths = training_history.get('episode_lengths', [])
            hit_rates = training_history.get('cache_hit_rates', [])
            cascades = training_history.get('cascade_events', [])

            for i, r in enumerate(rewards):
                length = lengths[i] if i < len(lengths) else 0
                hit_rate = hit_rates[i] if i < len(hit_rates) else 0.0
                cascade = cascades[i] if i < len(cascades) else False
                self.metrics_collector.record_episode(
                    reward=r,
                    length=length,
                    hit_rate=hit_rate,
                    cascade_occurred=bool(cascade),
                )

            # Final epsilon + loss
            self.metrics_collector.update_epsilon(self.agent.epsilon)
            if training_history.get('losses'):
                last_loss = training_history['losses'][-1]
                if last_loss is not None:
                    self.metrics_collector.record_training_step(
                        loss=last_loss,
                        epsilon=self.agent.epsilon,
                    )

        logger.info(f"Training complete: mean_reward={summary['mean_reward']:.2f}, std={summary['std_reward']:.2f}")

        return summary

    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Run evaluation episodes with trained agent.

        Args:
            num_episodes: Number of episodes to evaluate

        Returns:
            Evaluation results dictionary
        """
        if not self._is_setup:
            raise RuntimeError("Controller not set up. Call setup() first.")

        if not self.env or not self.agent:
            raise RuntimeError("Environment and agent required for evaluation")

        logger.info(f"Starting evaluation for {num_episodes} episodes...")

        episode_rewards = []
        episode_lengths = []
        cache_hit_rates = []
        cascade_counts = 0

        # Save current epsilon and set to 0 (greedy)
        old_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            steps = 0
            done = False

            while not done:
                # Greedy action selection
                action = self.agent.select_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)

                episode_reward += reward
                steps += 1
                done = terminated or truncated

            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)

            # Get episode metrics
            metrics = self.env.get_episode_metrics()
            cache_hit_rates.append(metrics['cache_hit_rate'])
            if metrics['cascade_occurred']:
                cascade_counts += 1

            # Push to Prometheus
            if self.metrics_collector is not None:
                self.metrics_collector.record_episode(
                    reward=episode_reward,
                    length=steps,
                    hit_rate=metrics['cache_hit_rate'],
                    cascade_occurred=metrics['cascade_occurred'],
                )

            logger.info(f"Eval episode {episode + 1}/{num_episodes}: reward={episode_reward:.2f}, steps={steps}")

        # Restore epsilon
        self.agent.epsilon = old_epsilon

        # Compile results
        results = {
            'num_episodes': num_episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_cache_hit_rate': np.mean(cache_hit_rates),
            'cascade_count': cascade_counts,
            'cascade_rate': cascade_counts / num_episodes,
        }

        # Update best eval reward
        if results['mean_reward'] > self.best_eval_reward:
            self.best_eval_reward = results['mean_reward']

            # Save best model
            if self.config.mode == OperatingMode.TRAINING:
                best_model_path = Path(self.config.output_dir) / "best_model.pt"
                self.agent.save(str(best_model_path))
                logger.info(f"New best model saved to {best_model_path}")

        logger.info(f"Evaluation complete: mean_reward={results['mean_reward']:.2f}, hit_rate={results['mean_cache_hit_rate']:.2%}")

        return results

    def predict_action(self, state: np.ndarray) -> int:
        """
        Get recommended action for given state (for deployment).

        Args:
            state: Current state observation

        Returns:
            Action index
        """
        if not self.agent:
            raise RuntimeError("Agent not initialized")

        # Use greedy policy in deployment
        old_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0
        action = self.agent.select_action(state)
        self.agent.epsilon = old_epsilon

        return action

    def process_api_call(self, endpoint: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Complete processing of an API call (for deployment).

        Args:
            endpoint: API endpoint called
            context: Request context (user type, time, etc.)

        Returns:
            Processing results including action taken and predictions
        """
        if not self._is_setup:
            raise RuntimeError("Controller not set up")

        context = context or {}

        # 1. Update Markov predictor
        self.markov_predictor.observe(endpoint, context=context)

        # 2. Get predictions
        predictions = self.markov_predictor.predict(k=5, context=context)

        # 3. Build state (simplified - in production would use full StateBuilder)
        # For now, just use basic features
        state = self._build_deployment_state(predictions, context)

        # 4. Let RL agent choose action
        action = self.predict_action(state)

        # 5. Execute action (would interact with actual cache in production)
        action_name = self._get_action_name(action)

        # 6. Return results
        return {
            'endpoint': endpoint,
            'predictions': [(api, float(prob)) for api, prob in predictions],
            'action_taken': action_name,
            'action_index': int(action),
            'timestamp': datetime.now().isoformat(),
        }

    def _build_deployment_state(self, predictions: List[Tuple[str, float]], context: Dict[str, Any]) -> np.ndarray:
        """Build state vector for deployment (simplified version)."""
        # This is a simplified version - in production would use full StateBuilder
        state = np.zeros(60, dtype=np.float32)

        # Add predictions
        for i, (api, prob) in enumerate(predictions[:5]):
            state[i] = i / 100.0  # Normalized API index
            state[i + 5] = prob  # Probability

        return state

    def _get_action_name(self, action: int) -> str:
        """Get human-readable action name."""
        action_names = {
            0: "DO_NOTHING",
            1: "CACHE_CURRENT",
            2: "PREFETCH_CONSERVATIVE",
            3: "PREFETCH_MODERATE",
            4: "PREFETCH_AGGRESSIVE",
            5: "EVICT_LRU",
            6: "EVICT_LOW_PROB"
        }
        return action_names.get(action, f"UNKNOWN_{action}")

    def run_demo(self, scenario: str = 'normal') -> Dict[str, Any]:
        """
        Run an interactive demonstration.

        Args:
            scenario: Demo scenario ('normal', 'cascade', 'burst')

        Returns:
            Demo results
        """
        if not self._is_setup:
            raise RuntimeError("Controller not set up")

        if self.config.mode != OperatingMode.DEMO:
            logger.warning(f"Running demo in {self.config.mode} mode")

        logger.info(f"Starting demo with scenario: {scenario}")

        # Run a demo episode with visualization
        obs, _ = self.env.reset()
        demo_results = {
            'scenario': scenario,
            'steps': [],
            'total_reward': 0.0,
        }

        for step in range(100):  # 100 step demo
            # Get action
            action = self.agent.select_action(obs)

            # Execute
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Record step
            step_info = {
                'step': step,
                'action': self._get_action_name(action),
                'reward': float(reward),
                'cache_hit': info.get('cache_hit', False),
                'predictions': info.get('predictions', [])[:3],  # Top 3
                'cascade_risk': info.get('cascade_risk', 0.0),
            }
            demo_results['steps'].append(step_info)
            demo_results['total_reward'] += reward

            # Render if in terminal
            if step % 10 == 0:
                self.env.render(mode='human')

            if terminated or truncated:
                logger.info(f"Demo episode ended at step {step}")
                break

        logger.info(f"Demo complete: {len(demo_results['steps'])} steps, total_reward={demo_results['total_reward']:.2f}")

        return demo_results

    def step_demo(self) -> Dict[str, Any]:
        """
        Execute one demo step and return detailed state.

        Returns:
            Detailed state dictionary for visualization
        """
        if not self._is_setup:
            raise RuntimeError("Controller not set up")

        if not hasattr(self, '_demo_obs'):
            # Initialize demo
            self._demo_obs, _ = self.env.reset()
            self._demo_step = 0

        # Get action
        action = self.agent.select_action(self._demo_obs)

        # Execute
        self._demo_obs, reward, terminated, truncated, info = self.env.step(action)
        self._demo_step += 1

        # Build detailed state
        state = {
            'step': self._demo_step,
            'action': self._get_action_name(action),
            'reward': float(reward),
            'terminated': terminated,
            'truncated': truncated,
            'cache_hit': info.get('cache_hit', False),
            'predictions': info.get('predictions', []),
            'cache_metrics': info.get('cache_metrics', {}),
            'system_metrics': info.get('system_metrics', {}),
            'cascade_risk': info.get('cascade_risk', 0.0),
        }

        # Reset if episode ended
        if terminated or truncated:
            self._demo_obs, _ = self.env.reset()
            self._demo_step = 0
            state['episode_ended'] = True
        else:
            state['episode_ended'] = False

        return state

    def __enter__(self):
        """Context manager entry."""
        self.setup()
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


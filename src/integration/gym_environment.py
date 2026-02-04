"""
Custom Gymnasium (OpenAI Gym) environment for RL-based caching system training.

This module provides a standard Gym interface that wraps our entire caching system,
allowing RL agents to interact with the Markov predictor, cache manager, and
microservices through the standardized Gymnasium API.

The environment translates between RL concepts (states, actions, rewards) and
our caching system operations, making it compatible with popular RL libraries
like Stable-Baselines3, RLlib, etc.
"""

import gymnasium as gym
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import random
import time

from ..markov.predictor import MarkovPredictor
from ..cache.cache_manager import CacheManager, CacheManagerConfig
from ..rl.state import StateBuilder, StateConfig
from ..rl.reward import RewardCalculator, RewardConfig, ActionOutcome
from ..rl.actions import ActionSpace, ActionConfig, CacheAction

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class SimulatorConfig:
    """Configuration for the microservices simulator."""
    num_apis: int = 20  # Number of different API endpoints
    user_types: List[str] = field(default_factory=lambda: ['guest', 'free', 'premium'])
    session_length_range: Tuple[int, int] = (10, 100)  # Min/max calls per session
    cascade_threshold: float = 0.8  # System load threshold for cascade risk
    base_latency_ms: float = 50.0  # Base response time
    cache_hit_latency_ms: float = 5.0  # Cached response time
    error_rate_threshold: float = 0.1  # Error rate indicating problems
    mock_responses: bool = True  # Use synthetic data generation


@dataclass
class CacheEnvConfig:
    """Configuration for the caching environment."""

    # Component configurations
    markov_config: Optional[Dict[str, Any]] = None
    cache_config: Optional[CacheManagerConfig] = None
    simulator_config: Optional[SimulatorConfig] = None
    state_config: Optional[StateConfig] = None
    reward_config: Optional[RewardConfig] = None
    action_config: Optional[ActionConfig] = None

    # Episode parameters
    max_steps_per_episode: int = 1000
    use_real_services: bool = False  # Whether to call actual services or use mock
    episode_end_on_cascade: bool = True
    normalize_rewards: bool = False

    # Training parameters
    seed: Optional[int] = None
    log_episode_metrics: bool = True
    render_mode: Optional[str] = None  # 'human', 'ansi', None

    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.markov_config is None:
            self.markov_config = {
                'order': 1,
                'context_aware': True,
                'context_features': ['user_type', 'hour'],
                'smoothing': 0.001,
                'history_size': 10
            }

        if self.cache_config is None:
            self.cache_config = CacheManagerConfig(
                backend_type='memory',
                default_ttl=300,
                compression_enabled=True
            )

        if self.simulator_config is None:
            self.simulator_config = SimulatorConfig()

        if self.state_config is None:
            self.state_config = StateConfig()

        if self.reward_config is None:
            self.reward_config = RewardConfig()

        if self.action_config is None:
            self.action_config = ActionConfig()


class CachingEnv(gym.Env):
    """
    Gymnasium environment for RL-based intelligent caching.

    Observation Space:
        Box space of shape (state_dim,) with normalized values in [0, 1] or [-1, 1].
        State includes:
        - Markov predictions (API indices and probabilities)
        - Cache metrics (utilization, hit rate, etc.)
        - System metrics (CPU, memory, latency, etc.)
        - Context (user type, time, session info)

    Action Space:
        Discrete(7) representing:
        0: DO_NOTHING - Let normal LRU behavior happen
        1: CACHE_CURRENT - Explicitly cache current response
        2: PREFETCH_CONSERVATIVE - Prefetch top-1 if prob > 70%
        3: PREFETCH_MODERATE - Prefetch top-3 if prob > 50%
        4: PREFETCH_AGGRESSIVE - Prefetch top-5 if prob > 30%
        5: EVICT_LRU - Proactively evict least-recently-used
        6: EVICT_LOW_PROB - Evict entries with low predicted probability

    Rewards:
        Multi-objective reward combining:
        - Cache hits (+10.0) and misses (-1.0)
        - Cascade prevention (+50.0) and occurrence (-100.0)
        - Prefetch efficiency
        - Latency optimization
        - Resource management

    Episode Termination:
        - Natural: Session ends, cascade failure occurs
        - Truncated: Max steps reached
    """

    metadata = {
        'render_modes': ['human', 'ansi'],
        'render_fps': 1
    }

    def __init__(self, config: CacheEnvConfig):
        """
        Initialize the caching environment.

        Args:
            config: Environment configuration
        """
        super().__init__()

        self.config = config

        # Initialize random number generator
        self._np_random = np.random.RandomState()
        if config.seed is not None:
            self._np_random.seed(config.seed)

        # Define action and observation spaces (required by Gymnasium)
        self.action_space = gym.spaces.Discrete(7)

        # Observation space: normalized state vector
        state_dim = config.state_config.state_dim
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(state_dim,),
            dtype=np.float32
        )

        # Initialize components
        self._initialize_components()

        # Episode state
        self.current_step = 0
        self.episode_number = 0
        self.current_api = None
        self.session_context = {}
        self.session_apis = []  # APIs in current session
        self.session_position = 0
        self.session_start_time = 0.0

        # Episode tracking
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_outcomes = []
        self.cumulative_reward = 0.0

        # System state
        self.system_metrics = self._initialize_system_metrics()
        self.cascade_detected = False
        self.cascade_risk_score = 0.0

        # Performance tracking
        self.total_cache_hits = 0
        self.total_cache_misses = 0
        self.total_prefetch_hits = 0
        self.total_prefetch_wasted = 0
        self.correct_predictions = 0
        self.total_predictions = 0

        logger.info(f"CachingEnv initialized with state_dim={state_dim}, action_space=Discrete(7)")

    def _initialize_components(self):
        """Initialize all subsystem components."""
        # Markov predictor
        self.predictor = MarkovPredictor(**self.config.markov_config)

        # Cache manager
        self.cache_manager = CacheManager(self.config.cache_config)
        self.cache_manager.start()

        # State builder
        self.state_builder = StateBuilder(self.config.state_config)

        # Reward calculator
        self.reward_calculator = RewardCalculator(self.config.reward_config)

        # Action space
        self.action_space_handler = ActionSpace(self.config.action_config)

        # Build API vocabulary
        self._build_api_vocabulary()

        # Fit state builder on vocabulary
        if hasattr(self, 'api_vocabulary'):
            self.state_builder.fit(self.api_vocabulary)

        logger.info("All environment components initialized")

    def _build_api_vocabulary(self):
        """Build or load API vocabulary for the environment."""
        # Generate synthetic API vocabulary based on config
        num_apis = self.config.simulator_config.num_apis
        self.api_vocabulary = [f"api_{i}" for i in range(num_apis)]

        # Common e-commerce APIs for realism
        common_apis = [
            '/api/auth/login',
            '/api/auth/logout',
            '/api/user/profile',
            '/api/products/list',
            '/api/products/search',
            '/api/products/{id}',
            '/api/cart/view',
            '/api/cart/add',
            '/api/cart/update',
            '/api/cart/remove',
            '/api/orders/list',
            '/api/orders/create',
            '/api/orders/{id}',
            '/api/payment/process',
            '/api/reviews/list',
            '/api/reviews/create',
            '/api/wishlist/view',
            '/api/recommendations',
            '/api/categories',
            '/api/search/autocomplete'
        ]

        # Use common APIs if num_apis matches, otherwise use generated names
        if num_apis <= len(common_apis):
            self.api_vocabulary = common_apis[:num_apis]

        logger.info(f"API vocabulary built with {len(self.api_vocabulary)} endpoints")

    def _initialize_system_metrics(self) -> Dict[str, float]:
        """Initialize system metrics to baseline values."""
        return {
            'cpu': 0.3,  # 30% baseline
            'memory': 0.4,  # 40% baseline
            'request_rate': 100.0,
            'p50_latency': 50.0,
            'p95_latency': 100.0,
            'p99_latency': 200.0,
            'error_rate': 0.01,
            'connections': 50.0,
            'queue_depth': 5.0
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to start a new episode.

        Args:
            seed: Random seed for reproducibility
            options: Optional configuration for reset

        Returns:
            observation: Initial state vector
            info: Additional information dictionary
        """
        # Seed the environment
        if seed is not None:
            self._np_random.seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        # Reset episode counters
        self.current_step = 0
        self.episode_number += 1
        self.cumulative_reward = 0.0

        # Reset tracking
        self.episode_rewards.clear()
        self.episode_actions.clear()
        self.episode_outcomes.clear()

        # Reset cache statistics (optionally keep cache contents)
        reset_cache = options.get('reset_cache', False) if options else False
        if reset_cache:
            # Clear cache completely
            self.cache_manager.stop()
            self.cache_manager = CacheManager(self.config.cache_config)
            self.cache_manager.start()

        # Reset Markov predictor history (new user session)
        self.predictor.reset_history()

        # Start new session
        self._start_new_session()

        # Reset system metrics
        self.system_metrics = self._initialize_system_metrics()
        self.cascade_detected = False
        self.cascade_risk_score = 0.0

        # Generate first API call
        self.current_api = self._generate_api_call()
        self.predictor.observe(self.current_api, context=self._get_current_context())

        # Build initial observation
        observation = self._build_observation()

        # Info dict
        info = {
            'episode_number': self.episode_number,
            'session_id': self.session_context.get('session_id'),  # Add at top level for test compatibility
            'session_context': self.session_context.copy(),
            'initial_api': self.current_api,
            'cache_state': self._get_cache_state()
        }

        logger.info(f"Episode {self.episode_number} started - User: {self.session_context.get('user_type')}")

        return observation, info

    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep of the environment.

        Args:
            action: Action index (0-6)

        Returns:
            observation: New state vector
            reward: Reward for this step
            terminated: Whether episode ended naturally
            truncated: Whether episode ended due to step limit
            info: Additional information
        """
        self.current_step += 1

        # Get predictions from Markov predictor
        predictions = self.predictor.predict(
            k=5,
            context=self._get_current_context()
        )

        # Execute the action
        action_outcome = self._execute_action(action, predictions)

        # Get the actual next API call
        next_api = self._generate_api_call()

        # Check if it was a cache hit or miss
        cache_key = self._make_cache_key(next_api)
        cached_value = self.cache_manager.get(cache_key)

        if cached_value is not None:
            action_outcome.cache_hit = True
            action_outcome.actual_latency_ms = self.config.simulator_config.cache_hit_latency_ms
            self.total_cache_hits += 1

            # Check if this was from a prefetch
            if cache_key in getattr(self, '_prefetched_keys', set()):
                action_outcome.prefetch_used += 1
                self.total_prefetch_hits += 1
        else:
            action_outcome.cache_miss = True
            action_outcome.actual_latency_ms = self.config.simulator_config.base_latency_ms
            self.total_cache_misses += 1

            # Cache the response for next time
            self.cache_manager.set(cache_key, {'api': next_api, 'data': 'mock_response'})

        action_outcome.baseline_latency_ms = self.config.simulator_config.base_latency_ms

        # Check prediction accuracy
        if predictions and predictions[0][0] == next_api:
            action_outcome.prediction_was_correct = True
            action_outcome.prediction_confidence = predictions[0][1]
            self.correct_predictions += 1
        self.total_predictions += 1

        # Update system metrics based on cache performance
        self._update_system_metrics(action_outcome)

        # Check for cascade failure conditions
        cascade_occurred, cascade_risk = self._check_cascade_conditions()
        action_outcome.cascade_occurred = cascade_occurred
        action_outcome.cascade_risk_detected = cascade_risk > 0.5

        if cascade_risk > 0.7 and not cascade_occurred:
            action_outcome.cascade_prevented = True

        self.cascade_detected = cascade_occurred
        self.cascade_risk_score = cascade_risk

        # Get cache metrics
        cache_metrics = self.cache_manager.get_metrics()
        action_outcome.cache_utilization = cache_metrics.get('current_size_bytes', 0) / (100 * 1024 * 1024)  # Assuming 100MB max

        # Calculate reward
        reward = self.reward_calculator.calculate(action_outcome)
        reward_breakdown = self.reward_calculator.calculate_detailed(action_outcome)

        self.cumulative_reward += reward
        self.episode_rewards.append(reward)
        self.episode_actions.append(action)
        self.episode_outcomes.append(action_outcome)

        # Update predictor with observed API
        self.current_api = next_api
        self.predictor.observe(next_api, context=self._get_current_context())
        self.session_position += 1

        # Build new observation
        observation = self._build_observation()

        # Check if episode should end
        terminated, truncated, end_reason = self._should_end_episode()

        # Info dict
        info = {
            'step': self.current_step,
            'action_taken': CacheAction.get_name(action),
            'api': next_api,
            'cache_hit': action_outcome.cache_hit,
            'predictions': predictions[:3] if predictions else [],
            'reward': reward,
            'reward_breakdown': reward_breakdown,
            'cumulative_reward': self.cumulative_reward,
            'cascade_risk': cascade_risk,
            'cascade_occurred': cascade_occurred,
            'cache_metrics': cache_metrics,
            'system_metrics': self.system_metrics.copy(),
            'prediction_accuracy': self.correct_predictions / max(1, self.total_predictions)
        }

        if terminated or truncated:
            info['episode_summary'] = self.get_episode_metrics()
            if self.config.log_episode_metrics:
                self._log_episode_summary(info['episode_summary'])

        return observation, reward, terminated, truncated, info

    def _start_new_session(self):
        """Initialize a new user session."""
        # Select user type
        user_types = self.config.simulator_config.user_types
        user_type = self._np_random.choice(user_types)

        # Determine session length
        min_len, max_len = self.config.simulator_config.session_length_range
        session_length = self._np_random.randint(min_len, max_len)

        # Initialize session context
        self.session_context = {
            'user_type': user_type,
            'session_length': session_length,
            'hour': self._np_random.randint(0, 24),
            'day': self._np_random.randint(0, 7),
            'session_id': f"session_{self.episode_number}_{time.time()}"
        }

        self.session_position = 0
        self.session_start_time = time.time()

        # Generate session API sequence
        self._generate_session_sequence()

        logger.debug(f"New session started: {self.session_context}")

    def _generate_session_sequence(self):
        """Generate a realistic sequence of API calls for this session."""
        # For now, use a simple pattern based on user type
        user_type = self.session_context['user_type']
        session_length = self.session_context['session_length']

        # Common patterns
        if user_type == 'guest':
            # Browse-heavy pattern
            patterns = [
                '/api/products/list',
                '/api/products/search',
                '/api/products/{id}',
                '/api/categories',
                '/api/search/autocomplete'
            ]
        elif user_type == 'free':
            # Browse and some cart activity
            patterns = [
                '/api/auth/login',
                '/api/products/list',
                '/api/products/{id}',
                '/api/cart/view',
                '/api/cart/add',
                '/api/wishlist/view',
                '/api/reviews/list'
            ]
        else:  # premium
            # Full purchase flow
            patterns = [
                '/api/auth/login',
                '/api/user/profile',
                '/api/products/list',
                '/api/recommendations',
                '/api/products/{id}',
                '/api/cart/add',
                '/api/cart/view',
                '/api/orders/create',
                '/api/payment/process',
                '/api/orders/list'
            ]

        # Map patterns to actual vocabulary
        self.session_apis = []
        for _ in range(session_length):
            if patterns and all(p in self.api_vocabulary for p in patterns):
                # Use pattern
                api = self._np_random.choice(patterns)
            else:
                # Random from vocabulary
                api = self._np_random.choice(self.api_vocabulary)
            self.session_apis.append(api)

    def _generate_api_call(self) -> str:
        """
        Generate the next API call in the episode.

        Returns:
            API endpoint string
        """
        # Use session sequence if available
        if self.session_position < len(self.session_apis):
            return self.session_apis[self.session_position]

        # Otherwise random (session extended beyond expected)
        return self._np_random.choice(self.api_vocabulary)

    def _execute_action(
        self,
        action: int,
        predictions: List[Tuple[str, float]]
    ) -> ActionOutcome:
        """
        Execute the chosen action on the cache system.

        Args:
            action: Action index
            predictions: Current Markov predictions

        Returns:
            ActionOutcome describing what happened
        """
        outcome = ActionOutcome()

        # Decode action
        action_instructions = self.action_space_handler.decode_action(action, predictions)

        # Execute based on action type
        if action_instructions['action_type'] == 'cache':
            # Cache current response
            if self.current_api:
                cache_key = self._make_cache_key(self.current_api)
                self.cache_manager.set(cache_key, {'api': self.current_api, 'data': 'mock_response'})

        elif action_instructions['action_type'] == 'prefetch':
            # Prefetch predicted APIs
            apis_to_prefetch = action_instructions['apis_to_prefetch']
            outcome.prefetch_attempted = len(apis_to_prefetch)

            # Track prefetched keys
            if not hasattr(self, '_prefetched_keys'):
                self._prefetched_keys = set()

            for api in apis_to_prefetch:
                cache_key = self._make_cache_key(api)
                # Check if already cached
                if self.cache_manager.get(cache_key) is None:
                    # Prefetch (simulate fetching and caching)
                    self.cache_manager.set(cache_key, {'api': api, 'data': 'prefetched_response'})
                    self._prefetched_keys.add(cache_key)
                    outcome.prefetch_successful += 1
                    outcome.prefetch_bytes += 1024  # Assume 1KB per response

        elif action_instructions['action_type'] == 'evict':
            # Evict entries
            eviction_strategy = action_instructions['eviction_strategy']
            eviction_count = action_instructions['eviction_count']

            if eviction_strategy == 'lru':
                evicted = self.cache_manager.evict_lru(count=eviction_count)
                outcome.evictions_triggered = evicted
            elif eviction_strategy == 'low_prob':
                # Build probability map from predictions
                prob_map = {self._make_cache_key(api): prob for api, prob in predictions}
                evicted = self.cache_manager.evict_low_probability(prob_map, count=eviction_count)
                outcome.evictions_triggered = evicted

        return outcome

    def _update_system_metrics(self, outcome: ActionOutcome):
        """Update system metrics based on recent performance."""
        # Adjust metrics based on cache performance
        if outcome.cache_hit:
            # Cache hits reduce load
            self.system_metrics['cpu'] *= 0.95
            self.system_metrics['memory'] *= 0.98
            self.system_metrics['p95_latency'] *= 0.9
            self.system_metrics['error_rate'] *= 0.95
            self.system_metrics['queue_depth'] *= 0.9
        else:
            # Cache misses increase load
            self.system_metrics['cpu'] = min(1.0, self.system_metrics['cpu'] * 1.02)
            self.system_metrics['memory'] = min(1.0, self.system_metrics['memory'] * 1.01)
            self.system_metrics['p95_latency'] *= 1.05
            self.system_metrics['error_rate'] = min(0.5, self.system_metrics['error_rate'] * 1.02)
            self.system_metrics['queue_depth'] *= 1.05

        # Add some noise for realism
        for key in ['cpu', 'memory', 'error_rate']:
            noise = self._np_random.normal(0, 0.01)
            self.system_metrics[key] = np.clip(self.system_metrics[key] + noise, 0.0, 1.0)

    def _check_cascade_conditions(self) -> Tuple[bool, float]:
        """
        Check for cascade failure conditions.

        Returns:
            (cascade_occurred, risk_score)
        """
        # Calculate risk score based on multiple factors
        risk_factors = []

        # High CPU usage
        if self.system_metrics['cpu'] > 0.8:
            risk_factors.append((self.system_metrics['cpu'] - 0.8) / 0.2)

        # High error rate
        if self.system_metrics['error_rate'] > self.config.simulator_config.error_rate_threshold:
            risk_factors.append(
                (self.system_metrics['error_rate'] - self.config.simulator_config.error_rate_threshold) /
                (1.0 - self.config.simulator_config.error_rate_threshold)
            )

        # High latency
        if self.system_metrics['p95_latency'] > 500:
            risk_factors.append(min(1.0, (self.system_metrics['p95_latency'] - 500) / 1000))

        # Deep queue
        if self.system_metrics['queue_depth'] > 100:
            risk_factors.append(min(1.0, (self.system_metrics['queue_depth'] - 100) / 200))

        # Calculate overall risk
        risk_score = np.mean(risk_factors) if risk_factors else 0.0

        # Cascade occurs if risk is very high
        cascade_occurred = risk_score > self.config.simulator_config.cascade_threshold

        return cascade_occurred, risk_score

    def _build_observation(self) -> np.ndarray:
        """
        Build the observation vector for the current state.

        Returns:
            Numpy array of normalized state features
        """
        # Get predictions
        predictions = self.predictor.predict(k=5, context=self._get_current_context())

        # Get cache metrics
        cache_metrics = self.cache_manager.get_metrics()
        cache_metric_dict = {
            'utilization': cache_metrics.get('current_size_bytes', 0) / (100 * 1024 * 1024),
            'hit_rate': cache_metrics.get('hit_rate', 0.0),
            'entries': cache_metrics.get('current_entries', 0),
            'eviction_rate': 0.0  # Would need to track this over time
        }

        # Build context
        context = self._get_current_context()
        context['session_position'] = self.session_position
        context['session_duration'] = time.time() - self.session_start_time
        context['call_count'] = self.current_step

        # Build state vector
        state = self.state_builder.build_state(
            markov_predictions=predictions,
            cache_metrics=cache_metric_dict,
            system_metrics=self.system_metrics,
            context=context
        )

        return state.astype(np.float32)

    def _get_current_context(self) -> Dict[str, Any]:
        """Get current context for predictions."""
        return {
            'user_type': self.session_context.get('user_type', 'guest'),
            'hour': self.session_context.get('hour', 12),
            'day': self.session_context.get('day', 0)
        }

    def _should_end_episode(self) -> Tuple[bool, bool, str]:
        """
        Determine if episode should end.

        Returns:
            (terminated, truncated, reason)
        """
        # Check cascade failure
        if self.cascade_detected and self.config.episode_end_on_cascade:
            return True, False, "cascade_failure"

        # Check session ended naturally
        if self.session_position >= self.session_context['session_length']:
            return True, False, "session_complete"

        # Check step limit
        if self.current_step >= self.config.max_steps_per_episode:
            return False, True, "step_limit"

        return False, False, "ongoing"

    def _make_cache_key(self, api: str, params: Optional[Dict] = None) -> str:
        """Generate cache key for an API call."""
        if params:
            import json
            return f"{api}:{json.dumps(params, sort_keys=True)}"
        return api

    def _get_cache_state(self) -> Dict[str, Any]:
        """Get current cache state summary."""
        metrics = self.cache_manager.get_metrics()
        return {
            'entries': metrics.get('current_entries', 0),
            'size_bytes': metrics.get('current_size_bytes', 0),
            'hit_rate': metrics.get('hit_rate', 0.0),
            'hits': metrics.get('hits', 0),
            'misses': metrics.get('misses', 0)
        }

    def get_episode_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics for the completed episode.

        Returns:
            Dictionary of episode metrics
        """
        cache_metrics = self.cache_manager.get_metrics()

        # Action distribution
        action_counts = defaultdict(int)
        for action in self.episode_actions:
            action_counts[CacheAction.get_name(action)] += 1

        # Reward statistics
        total_reward = sum(self.episode_rewards)
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0

        # Performance metrics
        total_requests = self.total_cache_hits + self.total_cache_misses
        cache_hit_rate = self.total_cache_hits / total_requests if total_requests > 0 else 0.0

        metrics = {
            'episode_number': self.episode_number,
            'total_steps': self.current_step,
            'total_reward': total_reward,
            'average_reward': avg_reward,
            'cumulative_reward': self.cumulative_reward,

            # Cache performance
            'cache_hit_rate': cache_hit_rate,
            'total_cache_hits': self.total_cache_hits,
            'total_cache_misses': self.total_cache_misses,

            # Prefetch performance
            'total_prefetch_hits': self.total_prefetch_hits,
            'total_prefetch_wasted': self.total_prefetch_wasted,
            'prefetch_efficiency': (
                self.total_prefetch_hits / (self.total_prefetch_hits + self.total_prefetch_wasted)
                if (self.total_prefetch_hits + self.total_prefetch_wasted) > 0 else 0.0
            ),

            # Prediction accuracy
            'prediction_accuracy': self.correct_predictions / max(1, self.total_predictions),
            'correct_predictions': self.correct_predictions,
            'total_predictions': self.total_predictions,

            # Actions taken
            'action_distribution': dict(action_counts),

            # Cascade events
            'cascade_occurred': self.cascade_detected,
            'final_cascade_risk': self.cascade_risk_score,

            # Session info
            'session_context': self.session_context,

            # Final system state
            'final_system_metrics': self.system_metrics.copy(),
            'final_cache_metrics': cache_metrics
        }

        return metrics

    def _log_episode_summary(self, metrics: Dict[str, Any]):
        """Log episode summary to logger."""
        logger.info(
            f"Episode {metrics['episode_number']} complete: "
            f"Steps={metrics['total_steps']}, "
            f"Reward={metrics['total_reward']:.2f}, "
            f"Hit Rate={metrics['cache_hit_rate']:.2%}, "
            f"Pred Acc={metrics['prediction_accuracy']:.2%}, "
            f"Cascade={metrics['cascade_occurred']}"
        )

    def render(self, mode: str = 'human'):
        """
        Render the environment state.

        Args:
            mode: Render mode ('human' or 'ansi')
        """
        if mode not in self.metadata['render_modes']:
            return

        output = []
        output.append("=" * 60)
        output.append(f"Episode {self.episode_number} - Step {self.current_step}")
        output.append("=" * 60)

        # Session info
        output.append(f"User: {self.session_context.get('user_type')}")
        output.append(f"Session: {self.session_position}/{self.session_context.get('session_length')}")
        output.append(f"Current API: {self.current_api}")
        output.append("")

        # Cache state
        cache_state = self._get_cache_state()
        output.append("Cache State:")
        output.append(f"  Entries: {cache_state['entries']}")
        output.append(f"  Hit Rate: {cache_state['hit_rate']:.2%}")
        output.append("")

        # System metrics
        output.append("System Metrics:")
        output.append(f"  CPU: {self.system_metrics['cpu']:.1%}")
        output.append(f"  Memory: {self.system_metrics['memory']:.1%}")
        output.append(f"  P95 Latency: {self.system_metrics['p95_latency']:.1f}ms")
        output.append(f"  Error Rate: {self.system_metrics['error_rate']:.2%}")
        output.append(f"  Cascade Risk: {self.cascade_risk_score:.2%}")
        output.append("")

        # Episode performance
        output.append("Episode Performance:")
        output.append(f"  Cumulative Reward: {self.cumulative_reward:.2f}")
        output.append(f"  Avg Reward: {np.mean(self.episode_rewards) if self.episode_rewards else 0:.2f}")
        output.append(f"  Cache Hits: {self.total_cache_hits}")
        output.append(f"  Cache Misses: {self.total_cache_misses}")
        output.append("")

        # Recent actions
        if self.episode_actions:
            recent_actions = self.episode_actions[-5:]
            output.append("Recent Actions:")
            for i, action in enumerate(recent_actions):
                output.append(f"  {len(self.episode_actions)-len(recent_actions)+i}: {CacheAction.get_name(action)}")

        output.append("=" * 60)

        render_output = "\n".join(output)

        if mode == 'human':
            print(render_output)
        elif mode == 'ansi':
            return render_output

    def close(self):
        """Clean up resources."""
        if self.cache_manager and self.cache_manager.is_running:
            self.cache_manager.stop()

        logger.info("CachingEnv closed")

    def seed(self, seed: Optional[int] = None):
        """
        Set random seed (for backward compatibility with older Gym versions).

        Args:
            seed: Random seed
        """
        if seed is not None:
            self._np_random.seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        return [seed]


"""
Shared fixtures for integration tests.

This module provides common fixtures used across integration test modules,
including environment configurations, pre-trained agents, mock services, and test data.
"""

import pytest
import numpy as np
import torch
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

from src.integration.gym_environment import CachingEnv, CacheEnvConfig, SimulatorConfig
from src.cache.cache_manager import CacheManager, CacheManagerConfig
from src.rl.agents.dqn_agent import DQNAgent, DQNConfig
from src.rl.training.trainer import Trainer, TrainingConfig
from src.markov.predictor import MarkovPredictor
from src.rl.state import StateConfig
from src.rl.reward import RewardConfig
from src.rl.actions import ActionConfig
# Note: simulator imports are not needed for fixtures


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def env_config() -> CacheEnvConfig:
    """Standard environment configuration for testing."""
    return CacheEnvConfig(
        max_steps_per_episode=100,
        use_real_services=False,  # Use mock for fast tests
        episode_end_on_cascade=True,
        normalize_rewards=False,
        seed=42,
        log_episode_metrics=False,  # Disable logging in tests
        markov_config={
            'order': 1,
            'context_aware': True,
            'context_features': ['user_type', 'hour'],
            'smoothing': 0.001,
            'history_size': 10
        },
        cache_config=CacheManagerConfig(
            backend_type='memory',
            default_ttl=300,
            compression_enabled=False  # Disable for faster tests
        ),
        simulator_config=SimulatorConfig(
            num_apis=20,
            session_length_range=(10, 50),
            mock_responses=True
        )
    )


@pytest.fixture(scope="session")
def fast_env_config() -> CacheEnvConfig:
    """Fast environment configuration for quick tests."""
    return CacheEnvConfig(
        max_steps_per_episode=20,
        use_real_services=False,
        episode_end_on_cascade=True,
        seed=42,
        log_episode_metrics=False,
        markov_config={
            'order': 1,
            'context_aware': False,
            'smoothing': 0.01,
            'history_size': 5
        },
        cache_config=CacheManagerConfig(
            backend_type='memory',
            default_ttl=60,
            compression_enabled=False
        ),
        simulator_config=SimulatorConfig(
            num_apis=10,
            session_length_range=(5, 15),
            mock_responses=True
        )
    )


@pytest.fixture(scope="session")
def dqn_config() -> DQNConfig:
    """DQN configuration for testing."""
    return DQNConfig(
        state_dim=36,  # Matches actual environment state dimension
        action_dim=7,
        hidden_dims=[64, 32],
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        buffer_size=1000,
        batch_size=32,
        prioritized_replay=False,
        target_update_freq=100,
        device='cpu',  # Force CPU for reproducibility
        seed=42
    )


@pytest.fixture(scope="function")
def training_config(tmp_path) -> TrainingConfig:
    """Training configuration for testing."""
    return TrainingConfig(
        max_episodes=100,
        max_steps_per_episode=50,
        eval_frequency=20,
        eval_episodes=3,
        checkpoint_frequency=50,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        early_stopping=True,
        patience=20,
        min_episodes=10,
        seed=42,
        log_frequency=10,
        use_wandb=False,
        verbose=False
    )


# ============================================================================
# Environment Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def training_env(env_config):
    """Ready-to-use training environment."""
    env = CachingEnv(env_config)
    yield env
    env.close()


@pytest.fixture(scope="function")
def fast_env(fast_env_config):
    """Fast environment for quick tests."""
    env = CachingEnv(fast_env_config)
    yield env
    env.close()


@pytest.fixture(scope="function")
def multiple_envs(env_config):
    """Multiple environments for parallel testing."""
    envs = [CachingEnv(env_config) for _ in range(3)]
    yield envs
    for env in envs:
        env.close()


# ============================================================================
# Agent Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def untrained_agent(dqn_config):
    """Untrained DQN agent for testing."""
    return DQNAgent(dqn_config)


@pytest.fixture(scope="session")
def trained_agent(env_config, dqn_config):
    """
    Pre-trained agent for testing (train once, use in many tests).

    This fixture trains a small agent once per test session to save time.
    """
    # Create environment
    env = CachingEnv(env_config)

    # Create agent
    agent = DQNAgent(dqn_config, seed=42)

    # Quick training (just enough to learn something)
    print("\n[Fixture] Training agent for integration tests...")
    for episode in range(20):
        state, _ = env.reset(seed=42 + episode)
        episode_reward = 0

        for step in range(50):
            action = agent.select_action(state, evaluate=False)
            next_state, reward, terminated, truncated, info = env.step(action)

            agent.buffer.push(state, action, reward, next_state, terminated)
            episode_reward += reward

            if len(agent.buffer) >= agent.config.batch_size:
                loss = agent.update()

            state = next_state

            if terminated or truncated:
                break

        agent.decay_epsilon()

    print(f"[Fixture] Agent trained for 20 episodes. Final epsilon: {agent.epsilon:.3f}")

    env.close()
    return agent


@pytest.fixture(scope="function")
def agent_with_experience(untrained_agent, fast_env):
    """Agent with some experience in replay buffer."""
    agent = untrained_agent
    state, _ = fast_env.reset(seed=42)

    for _ in range(100):
        action = agent.select_action(state, evaluate=False)
        next_state, reward, terminated, truncated, _ = fast_env.step(action)
        agent.buffer.push(state, action, reward, next_state, terminated)

        state = next_state
        if terminated or truncated:
            state, _ = fast_env.reset()

    return agent


# ============================================================================
# Cache and Predictor Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def cache_manager():
    """Cache manager with memory backend."""
    config = CacheManagerConfig(
        backend_type='memory',
        default_ttl=300,
        compression_enabled=False
    )
    manager = CacheManager(config)
    manager.start()
    yield manager
    manager.stop()


@pytest.fixture(scope="function")
def markov_predictor():
    """Trained Markov predictor for testing."""
    predictor = MarkovPredictor(
        order=1,
        context_aware=False,
        smoothing=0.01,
        history_size=10
    )

    # Train on simple sequences
    sequences = [
        ['api_0', 'api_1', 'api_2'],
        ['api_0', 'api_1', 'api_3'],
        ['api_0', 'api_2', 'api_3'],
        ['api_1', 'api_2', 'api_3'],
    ] * 10  # Repeat for better statistics

    predictor.fit(sequences)
    return predictor


@pytest.fixture(scope="function")
def context_aware_predictor():
    """Context-aware Markov predictor for testing."""
    predictor = MarkovPredictor(
        order=1,
        context_aware=True,
        context_features=['user_type'],
        smoothing=0.01,
        history_size=10
    )

    # Train with contexts
    sequences = [
        ['login', 'profile', 'orders'],
        ['login', 'search', 'product'],
        ['browse', 'cart', 'checkout'],
    ] * 5

    contexts = [
        {'user_type': 'premium'},
        {'user_type': 'free'},
        {'user_type': 'guest'},
    ] * 5

    predictor.fit(sequences, contexts)
    return predictor


# ============================================================================
# Mock Service Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def mock_services():
    """Mocked microservices for fast testing."""
    services = {}

    api_endpoints = [
        '/api/auth/login',
        '/api/user/profile',
        '/api/products/list',
        '/api/products/123',
        '/api/cart/view',
        '/api/orders/list'
    ]

    for endpoint in api_endpoints:
        services[endpoint] = MockService(endpoint)

    return services


class MockService:
    """Simple mock service for testing."""

    def __init__(self, endpoint: str, latency_ms: float = 10.0):
        self.endpoint = endpoint
        self.latency_ms = latency_ms
        self.call_count = 0
        self.error_rate = 0.0

    def call(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Simulate service call."""
        self.call_count += 1

        # Simulate latency
        time.sleep(self.latency_ms / 1000.0)

        # Simulate errors
        if np.random.random() < self.error_rate:
            raise Exception(f"Service error at {self.endpoint}")

        return {
            'endpoint': self.endpoint,
            'data': f'Response from {self.endpoint}',
            'params': params,
            'timestamp': time.time()
        }

    def set_error_rate(self, rate: float):
        """Set error rate for failure injection."""
        self.error_rate = rate

    def set_latency(self, latency_ms: float):
        """Set response latency."""
        self.latency_ms = latency_ms

    def reset(self):
        """Reset service state."""
        self.call_count = 0
        self.error_rate = 0.0


# ============================================================================
# Traffic Generation Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def sample_traffic():
    """Pre-generated traffic data for testing."""
    return SampleTrafficGenerator.generate(
        num_sessions=10,
        apis_per_session=20,
        num_apis=10
    )


class SampleTrafficGenerator:
    """Helper to generate sample traffic patterns."""

    @staticmethod
    def generate(num_sessions: int, apis_per_session: int, num_apis: int) -> List[Dict[str, Any]]:
        """Generate sample traffic sessions."""
        sessions = []

        for session_id in range(num_sessions):
            session = {
                'session_id': f'session_{session_id}',
                'user_type': np.random.choice(['guest', 'free', 'premium']),
                'start_time': time.time(),
                'apis': []
            }

            for _ in range(apis_per_session):
                api_idx = np.random.randint(0, num_apis)
                session['apis'].append({
                    'endpoint': f'api_{api_idx}',
                    'timestamp': time.time(),
                    'params': {}
                })

            sessions.append(session)

        return sessions


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def temp_output_dir(tmp_path):
    """Temporary directory for test outputs."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture(scope="function")
def random_seed():
    """Set random seeds for reproducibility."""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed


@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset random state before each test."""
    np.random.seed(42)
    torch.manual_seed(42)
    yield
    # Cleanup after test
    pass


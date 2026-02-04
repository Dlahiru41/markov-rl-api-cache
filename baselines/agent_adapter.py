"""
Adapter to make RL agents compatible with the CachingPolicy interface.

This module provides wrappers to integrate trained RL agents (from stable-baselines3
or other libraries) with the baseline comparison framework.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from .base_policy import CachingPolicy


class RLAgentAdapter(CachingPolicy):
    """
    Adapter to make trained RL agents compatible with CachingPolicy interface.

    This allows fair comparison between baseline policies and trained RL agents.

    The agent must have:
    - predict(observation) method that returns (action, _)

    Compatible with:
    - Stable-Baselines3 agents (DQN, PPO, A2C, etc.)
    - Custom RL agents with similar interface

    Example:
        >>> from stable_baselines3 import DQN
        >>> trained_agent = DQN.load('path/to/model.zip')
        >>> policy = RLAgentAdapter(trained_agent, 'DQN')
        >>> action = policy.select_action(state, predictions)
    """

    def __init__(
        self,
        agent: Any,
        name: str = "RL Agent",
        deterministic: bool = True
    ):
        """
        Initialize RL agent adapter.

        Args:
            agent: Trained RL agent with predict() method
            name: Name for the agent (for logging/display)
            deterministic: Whether to use deterministic policy (True for evaluation)
        """
        self.agent = agent
        self._name = name
        self.deterministic = deterministic

        # Validate agent interface
        if not hasattr(agent, 'predict'):
            raise ValueError(
                f"Agent must have predict(observation) method. "
                f"Got type: {type(agent)}"
            )

        # Statistics
        self._step_count = 0

    def select_action(self, state: np.ndarray, predictions: List[Tuple[str, float]]) -> int:
        """
        Select action using the trained RL agent.

        Args:
            state: State vector (used by agent)
            predictions: Markov predictions (not used by agent, already in state)

        Returns:
            Action selected by the agent
        """
        self._step_count += 1

        # Call agent's predict method
        action, _ = self.agent.predict(state, deterministic=self.deterministic)

        # Convert to int if necessary
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)

        return action

    def get_name(self) -> str:
        """Return agent name."""
        return self._name

    def reset(self):
        """Reset for new episode."""
        self._step_count = 0
        # RL agents typically don't need explicit reset between episodes

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'step_count': self._step_count,
            'deterministic': self.deterministic,
            'agent_type': type(self.agent).__name__
        }


class TorchAgentAdapter(CachingPolicy):
    """
    Adapter for custom PyTorch agents.

    For agents that use PyTorch models directly without stable-baselines3.

    Example:
        >>> import torch
        >>> from src.rl.agents.dqn_agent import DQNAgent
        >>> agent = DQNAgent(state_dim=60, action_dim=7)
        >>> agent.load('path/to/checkpoint.pt')
        >>> policy = TorchAgentAdapter(agent, 'Custom DQN')
    """

    def __init__(
        self,
        agent: Any,
        name: str = "PyTorch Agent",
        device: str = 'cpu'
    ):
        """
        Initialize PyTorch agent adapter.

        Args:
            agent: Agent with select_action(state) method
            name: Agent name
            device: Device for inference ('cpu' or 'cuda')
        """
        import torch

        self.agent = agent
        self._name = name
        self.device = device
        self.torch = torch

        # Validate interface
        if not hasattr(agent, 'select_action'):
            raise ValueError(
                f"Agent must have select_action(state) method. "
                f"Got type: {type(agent)}"
            )

        # Set to evaluation mode if possible
        if hasattr(agent, 'eval'):
            agent.eval()

        self._step_count = 0

    def select_action(self, state: np.ndarray, predictions: List[Tuple[str, float]]) -> int:
        """Select action using PyTorch agent."""
        self._step_count += 1

        # Ensure numpy array input for agent
        state_np = np.asarray(state, dtype=np.float32)

        # Greedy action selection; most agents respect evaluate flag
        action = self.agent.select_action(state_np, evaluate=True)

        return int(action)

    def get_name(self) -> str:
        """Return agent name."""
        return self._name

    def reset(self):
        """Reset for new episode."""
        self._step_count = 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            'step_count': self._step_count,
            'device': self.device,
            'agent_type': type(self.agent).__name__
        }


class EnsembleAgentAdapter(CachingPolicy):
    """
    Adapter for ensemble of multiple agents.

    Combines predictions from multiple agents using voting or averaging.

    Parameters:
        agents: List of agents to ensemble
        method: Ensemble method ('voting' or 'average')

    Example:
        >>> agent1 = DQN.load('model1.zip')
        >>> agent2 = DQN.load('model2.zip')
        >>> agent3 = DQN.load('model3.zip')
        >>> policy = EnsembleAgentAdapter([agent1, agent2, agent3], method='voting')
    """

    def __init__(
        self,
        agents: List[Any],
        name: str = "Ensemble",
        method: str = 'voting'
    ):
        """Initialize ensemble adapter."""
        if not agents:
            raise ValueError("agents list cannot be empty")
        if method not in ['voting', 'average']:
            raise ValueError(f"method must be 'voting' or 'average', got {method}")

        self.agents = agents
        self._name = name
        self.method = method
        self._step_count = 0

    def select_action(self, state: np.ndarray, predictions: List[Tuple[str, float]]) -> int:
        """Select action using ensemble."""
        self._step_count += 1

        if self.method == 'voting':
            # Each agent votes for an action, majority wins
            votes = []
            for agent in self.agents:
                action, _ = agent.predict(state, deterministic=True)
                votes.append(int(action))

            # Return most common action
            from collections import Counter
            action = Counter(votes).most_common(1)[0][0]
            return action

        else:  # average
            # Average Q-values if available, otherwise fall back to voting
            # This requires agents to expose Q-values, which might not always be possible
            # For simplicity, use voting as fallback
            votes = []
            for agent in self.agents:
                action, _ = agent.predict(state, deterministic=True)
                votes.append(int(action))

            from collections import Counter
            action = Counter(votes).most_common(1)[0][0]
            return action

    def get_name(self) -> str:
        """Return ensemble name."""
        return f"{self._name} ({len(self.agents)} agents)"

    def reset(self):
        """Reset ensemble."""
        self._step_count = 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get ensemble statistics."""
        return {
            'step_count': self._step_count,
            'num_agents': len(self.agents),
            'method': self.method
        }


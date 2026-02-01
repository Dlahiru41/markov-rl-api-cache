"""
Abstract base class for caching policies and policy wrapper with statistics tracking.

This module defines the interface that all caching policies (baselines and RL agents)
must implement, enabling fair comparison in the evaluation framework.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class PolicyStatistics:
    """Statistics tracked for a caching policy."""

    # Action distribution
    action_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    total_decisions: int = 0

    # Reward tracking per action
    action_rewards: Dict[int, List[float]] = field(default_factory=lambda: defaultdict(list))
    total_reward: float = 0.0

    # Episode tracking
    episodes_completed: int = 0
    total_steps: int = 0

    def update_action(self, action: int, reward: float):
        """Record an action and its reward."""
        self.action_counts[action] += 1
        self.action_rewards[action].append(reward)
        self.total_decisions += 1
        self.total_reward += reward

    def get_action_distribution(self) -> Dict[int, float]:
        """Return normalized action distribution."""
        if self.total_decisions == 0:
            return {}
        return {
            action: count / self.total_decisions
            for action, count in self.action_counts.items()
        }

    def get_average_reward_per_action(self) -> Dict[int, float]:
        """Return average reward for each action."""
        return {
            action: np.mean(rewards) if rewards else 0.0
            for action, rewards in self.action_rewards.items()
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive statistics summary."""
        return {
            'total_decisions': self.total_decisions,
            'total_reward': self.total_reward,
            'average_reward': self.total_reward / self.total_decisions if self.total_decisions > 0 else 0.0,
            'episodes_completed': self.episodes_completed,
            'total_steps': self.total_steps,
            'action_distribution': self.get_action_distribution(),
            'average_reward_per_action': self.get_average_reward_per_action()
        }

    def reset(self):
        """Reset all statistics."""
        self.action_counts.clear()
        self.action_rewards.clear()
        self.total_decisions = 0
        self.total_reward = 0.0
        self.episodes_completed = 0
        self.total_steps = 0


class CachingPolicy(ABC):
    """
    Abstract base class for all caching policies.

    All baseline policies and RL agents must implement this interface
    to be compatible with the evaluation framework.
    """

    @abstractmethod
    def select_action(self, state: np.ndarray, predictions: List[Tuple[str, float]]) -> int:
        """
        Select a caching action given the current state and predictions.

        Args:
            state: State vector from the environment (shape: state_dim,)
            predictions: List of (api_endpoint, probability) tuples from Markov predictor,
                        sorted by probability in descending order

        Returns:
            Action index (0-6) corresponding to CacheAction enum
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Return the policy name for logging and display.

        Returns:
            Human-readable policy name
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset any internal state.

        Called at the start of each episode to ensure clean state.
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Return policy-specific statistics.

        Returns:
            Dictionary with statistics relevant to this policy
        """
        pass


class PolicyWrapper:
    """
    Wrapper that adds statistics tracking to any CachingPolicy.

    This wrapper transparently tracks action distributions, rewards, and other
    metrics without requiring the policy to implement tracking logic.

    Example:
        >>> policy = LRUPolicy()
        >>> wrapped = PolicyWrapper(policy)
        >>> action = wrapped.select_action(state, predictions)
        >>> wrapped.record_reward(action, reward)
        >>> stats = wrapped.get_statistics()
    """

    def __init__(self, policy: CachingPolicy):
        """
        Initialize the wrapper.

        Args:
            policy: The caching policy to wrap
        """
        self.policy = policy
        self.stats = PolicyStatistics()
        self._last_action: Optional[int] = None

    def select_action(self, state: np.ndarray, predictions: List[Tuple[str, float]]) -> int:
        """
        Select action using wrapped policy and track the decision.

        Args:
            state: State vector from environment
            predictions: Markov predictions

        Returns:
            Selected action
        """
        action = self.policy.select_action(state, predictions)
        self._last_action = action
        self.stats.total_steps += 1
        return action

    def record_reward(self, action: int, reward: float):
        """
        Record the reward received for an action.

        Args:
            action: Action that was taken
            reward: Reward received
        """
        self.stats.update_action(action, reward)

    def record_episode_end(self):
        """Mark the end of an episode."""
        self.stats.episodes_completed += 1

    def get_name(self) -> str:
        """Get the wrapped policy's name."""
        return self.policy.get_name()

    def reset(self):
        """Reset both policy and statistics."""
        self.policy.reset()
        self._last_action = None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get combined statistics from wrapper and policy.

        Returns:
            Dictionary with both wrapper statistics and policy-specific statistics
        """
        stats = self.stats.get_summary()
        stats['policy_specific'] = self.policy.get_statistics()
        return stats

    def reset_statistics(self):
        """Reset statistics without resetting policy state."""
        self.stats.reset()
        self._last_action = None


class StatefulPolicy(CachingPolicy):
    """
    Base class for policies that maintain internal state.

    Provides common functionality for policies that track history or metrics.
    """

    def __init__(self, name: str):
        """
        Initialize stateful policy.

        Args:
            name: Policy name
        """
        self._name = name
        self._step_count = 0
        self._episode_count = 0

    def get_name(self) -> str:
        """Return policy name."""
        return self._name

    def reset(self):
        """Reset state counters."""
        self._step_count = 0
        self._episode_count += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get basic statistics."""
        return {
            'step_count': self._step_count,
            'episode_count': self._episode_count
        }

    def _increment_step(self):
        """Increment step counter."""
        self._step_count += 1


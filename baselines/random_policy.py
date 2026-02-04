"""
Random caching policy baseline.

This is the lower bound - our RL agent should easily beat this.
Useful for sanity checking the environment and understanding the value of intelligence.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from .base_policy import CachingPolicy


class RandomPolicy(CachingPolicy):
    """
    Random action selection baseline.

    Selects actions uniformly at random (or with optional weights).
    This represents the absolute baseline - any intelligent policy should beat this.

    Parameters:
        action_weights: Optional weights for non-uniform random selection.
                       If None, uses uniform distribution.
                       Example: [0.2, 0.3, 0.1, 0.2, 0.1, 0.05, 0.05] for 7 actions
        seed: Random seed for reproducibility

    Example:
        >>> # Uniform random
        >>> policy = RandomPolicy()
        >>> action = policy.select_action(state, predictions)

        >>> # Weighted random (favor DO_NOTHING and CACHE_CURRENT)
        >>> weights = [0.3, 0.4, 0.1, 0.1, 0.05, 0.03, 0.02]
        >>> policy = RandomPolicy(action_weights=weights)
    """

    def __init__(
        self,
        action_weights: Optional[List[float]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize random policy.

        Args:
            action_weights: Optional action weights (must sum to 1.0)
            seed: Random seed for reproducibility
        """
        self.action_weights = action_weights
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        # Validate weights
        if action_weights is not None:
            if len(action_weights) != 7:
                raise ValueError(f"action_weights must have 7 elements, got {len(action_weights)}")
            if not np.isclose(sum(action_weights), 1.0):
                raise ValueError(f"action_weights must sum to 1.0, got {sum(action_weights)}")

        # Statistics
        self._step_count = 0
        self._action_counts: Dict[int, int] = {i: 0 for i in range(7)}

    def select_action(self, state: np.ndarray, predictions: List[Tuple[str, float]]) -> int:
        """
        Select a random action.

        Args:
            state: State vector (not used)
            predictions: Predictions (not used)

        Returns:
            Random action index
        """
        self._step_count += 1

        if self.action_weights is not None:
            # Weighted random selection
            action = self._rng.choice(7, p=self.action_weights)
        else:
            # Uniform random selection
            action = self._rng.randint(0, 7)

        self._action_counts[int(action)] += 1
        return int(action)

    def get_name(self) -> str:
        """Return policy name."""
        if self.action_weights is not None:
            return "Random (Weighted)"
        return "Random (Uniform)"

    def reset(self):
        """Reset for new episode."""
        self._step_count = 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get random policy statistics."""
        return {
            'step_count': self._step_count,
            'action_counts': self._action_counts.copy(),
            'action_weights': self.action_weights,
            'seed': self.seed
        }


class EpsilonRandomPolicy(CachingPolicy):
    """
    Epsilon-random policy: Random with probability epsilon, otherwise use base policy.

    This wraps another policy and adds randomness for exploration.

    Parameters:
        base_policy: The base policy to use (1-epsilon) of the time
        epsilon: Probability of selecting random action
        decay_rate: How much to decay epsilon after each episode
        min_epsilon: Minimum epsilon value

    Example:
        >>> from baselines.lru_policy import LRUPolicy
        >>> base = LRUPolicy()
        >>> policy = EpsilonRandomPolicy(base, epsilon=0.1)
        >>> # 10% random, 90% LRU
    """

    def __init__(
        self,
        base_policy: CachingPolicy,
        epsilon: float = 0.1,
        decay_rate: float = 0.99,
        min_epsilon: float = 0.01,
        seed: Optional[int] = None
    ):
        """Initialize epsilon-random policy."""
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        if not 0.0 <= decay_rate <= 1.0:
            raise ValueError(f"decay_rate must be in [0, 1], got {decay_rate}")
        if not 0.0 <= min_epsilon <= epsilon:
            raise ValueError(f"min_epsilon must be in [0, epsilon], got {min_epsilon}")

        self.base_policy = base_policy
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self._rng = np.random.RandomState(seed)

        self._step_count = 0
        self._random_count = 0
        self._base_count = 0
        self._episode_count = 0

    def select_action(self, state: np.ndarray, predictions: List[Tuple[str, float]]) -> int:
        """Select action with epsilon-random exploration."""
        self._step_count += 1

        if self._rng.random() < self.epsilon:
            # Random action
            self._random_count += 1
            return int(self._rng.randint(0, 7))
        else:
            # Base policy action
            self._base_count += 1
            return self.base_policy.select_action(state, predictions)

    def get_name(self) -> str:
        """Return policy name."""
        return f"ε-{self.base_policy.get_name()} (ε={self.epsilon:.3f})"

    def reset(self):
        """Reset for new episode and decay epsilon."""
        self._step_count = 0
        self.base_policy.reset()

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
        self._episode_count += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            'step_count': self._step_count,
            'random_count': self._random_count,
            'base_count': self._base_count,
            'random_rate': self._random_count / max(1, self._step_count),
            'current_epsilon': self.epsilon,
            'episode_count': self._episode_count,
            'base_policy_stats': self.base_policy.get_statistics()
        }


class BiasedRandomPolicy(CachingPolicy):
    """
    Biased random policy that avoids certain actions.

    Useful for testing - e.g., never prefetch aggressively, or never evict.

    Parameters:
        excluded_actions: List of action indices to never select
        seed: Random seed

    Example:
        >>> # Never prefetch aggressively (action 4) or evict (actions 5,6)
        >>> policy = BiasedRandomPolicy(excluded_actions=[4, 5, 6])
    """

    def __init__(
        self,
        excluded_actions: Optional[List[int]] = None,
        seed: Optional[int] = None
    ):
        """Initialize biased random policy."""
        self.excluded_actions = set(excluded_actions or [])
        self._rng = np.random.RandomState(seed)

        # Validate excluded actions
        if not all(0 <= a < 7 for a in self.excluded_actions):
            raise ValueError("excluded_actions must be in range [0, 6]")

        # Create valid action list
        self.valid_actions = [a for a in range(7) if a not in self.excluded_actions]
        if not self.valid_actions:
            raise ValueError("Cannot exclude all actions")

        self._step_count = 0

    def select_action(self, state: np.ndarray, predictions: List[Tuple[str, float]]) -> int:
        """Select random action from valid actions."""
        self._step_count += 1
        return int(self._rng.choice(self.valid_actions))

    def get_name(self) -> str:
        """Return policy name."""
        from src.rl.actions import CacheAction
        excluded_names = [CacheAction.get_name(a) for a in self.excluded_actions]
        return f"Biased Random (exclude: {', '.join(excluded_names)})"

    def reset(self):
        """Reset for new episode."""
        self._step_count = 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            'step_count': self._step_count,
            'excluded_actions': list(self.excluded_actions),
            'num_valid_actions': len(self.valid_actions)
        }


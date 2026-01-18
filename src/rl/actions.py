"""
Action space definition for the caching RL agent.

This module defines what decisions the RL agent can make regarding caching,
prefetching, and eviction policies.
"""

import numpy as np
from enum import IntEnum
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import random


class CacheAction(IntEnum):
    """
    Enumeration of possible caching actions for the RL agent.

    Uses IntEnum for easy integration with neural networks and numeric indexing.
    """
    DO_NOTHING = 0           # Let normal LRU behavior happen
    CACHE_CURRENT = 1        # Explicitly cache the current API response
    PREFETCH_CONSERVATIVE = 2  # Prefetch top-1 if prob > 70%
    PREFETCH_MODERATE = 3    # Prefetch top-3 if prob > 50%
    PREFETCH_AGGRESSIVE = 4  # Prefetch top-5 if prob > 30%
    EVICT_LRU = 5           # Proactively evict least-recently-used
    EVICT_LOW_PROB = 6      # Evict entries with lowest predicted probability

    @classmethod
    def num_actions(cls) -> int:
        """Return the total number of actions available."""
        return 7

    @classmethod
    def get_name(cls, action: int) -> str:
        """
        Return human-readable name for an action.

        Args:
            action: Action index (0-6)

        Returns:
            Human-readable action name
        """
        action_names = {
            0: "DO_NOTHING",
            1: "CACHE_CURRENT",
            2: "PREFETCH_CONSERVATIVE",
            3: "PREFETCH_MODERATE",
            4: "PREFETCH_AGGRESSIVE",
            5: "EVICT_LRU",
            6: "EVICT_LOW_PROB"
        }
        return action_names.get(action, f"UNKNOWN_ACTION_{action}")

    @classmethod
    def get_description(cls, action: int) -> str:
        """
        Return detailed description of what an action does.

        Args:
            action: Action index (0-6)

        Returns:
            Description of the action's behavior
        """
        descriptions = {
            0: "Take no caching action, let normal LRU behavior happen",
            1: "Explicitly cache the current API response",
            2: "Prefetch only the top-1 prediction, and only if its probability exceeds 70%",
            3: "Prefetch top-3 predictions with probability > 50%",
            4: "Prefetch top-5 predictions with probability > 30%",
            5: "Proactively evict least-recently-used entries to make room",
            6: "Evict entries with lowest predicted future access probability"
        }
        return descriptions.get(action, f"Unknown action with index {action}")


@dataclass
class ActionConfig:
    """
    Configuration for action execution with tunable thresholds.

    These thresholds control how conservative or aggressive the prefetching
    and eviction strategies are.
    """
    # Probability thresholds for prefetching
    conservative_threshold: float = 0.7
    moderate_threshold: float = 0.5
    aggressive_threshold: float = 0.3

    # Number of predictions to prefetch for each strategy
    conservative_count: int = 1
    moderate_count: int = 3
    aggressive_count: int = 5

    # Eviction parameters
    eviction_batch_size: int = 10  # How many entries to evict at once


class ActionSpace:
    """
    Defines the action space for the RL agent and provides utilities
    for action validation, sampling, and decoding.
    """

    def __init__(self, config: Optional[ActionConfig] = None):
        """
        Initialize the action space.

        Args:
            config: ActionConfig with thresholds, or None for defaults
        """
        self.config = config or ActionConfig()
        self._rng = random.Random()

    @property
    def n(self) -> int:
        """Return the number of actions in the space."""
        return CacheAction.num_actions()

    def sample(self) -> int:
        """
        Sample a random action for exploration.

        Returns:
            Random action index (0-6)
        """
        return self._rng.randint(0, self.n - 1)

    def get_valid_actions(
        self,
        cache_utilization: float,
        has_predictions: bool,
        cache_size: int
    ) -> List[int]:
        """
        Return list of currently valid action indices based on system state.

        Args:
            cache_utilization: Current cache capacity usage (0.0-1.0)
            has_predictions: Whether Markov predictions are available
            cache_size: Current number of entries in cache

        Returns:
            List of valid action indices
        """
        valid_actions = []

        # DO_NOTHING and CACHE_CURRENT are always valid
        valid_actions.append(CacheAction.DO_NOTHING)
        valid_actions.append(CacheAction.CACHE_CURRENT)

        # Prefetch actions only valid if we have predictions
        if has_predictions:
            valid_actions.append(CacheAction.PREFETCH_CONSERVATIVE)
            valid_actions.append(CacheAction.PREFETCH_MODERATE)
            valid_actions.append(CacheAction.PREFETCH_AGGRESSIVE)

        # Eviction actions only valid if cache has entries
        if cache_size > 0:
            valid_actions.append(CacheAction.EVICT_LRU)

            # EVICT_LOW_PROB needs predictions to determine which entries have low probability
            if has_predictions:
                valid_actions.append(CacheAction.EVICT_LOW_PROB)

        return valid_actions

    def get_action_mask(
        self,
        cache_utilization: float,
        has_predictions: bool,
        cache_size: int
    ) -> np.ndarray:
        """
        Return boolean mask array indicating valid actions.

        This is useful for masking invalid actions in the policy network output.

        Args:
            cache_utilization: Current cache capacity usage (0.0-1.0)
            has_predictions: Whether Markov predictions are available
            cache_size: Current number of entries in cache

        Returns:
            Boolean numpy array of shape (n,) with True for valid actions
        """
        mask = np.zeros(self.n, dtype=bool)
        valid_actions = self.get_valid_actions(cache_utilization, has_predictions, cache_size)
        mask[valid_actions] = True
        return mask

    def decode_action(
        self,
        action: int,
        predictions: Optional[List[Tuple[str, float]]] = None
    ) -> Dict[str, Any]:
        """
        Convert an action index into concrete execution instructions.

        Args:
            action: Action index (0-6)
            predictions: List of (api_name, probability) tuples from Markov predictor

        Returns:
            Dictionary with execution instructions containing:
                - action_type: 'none', 'cache', 'prefetch', or 'evict'
                - cache_current: bool
                - apis_to_prefetch: list of API endpoints to prefetch
                - eviction_strategy: 'lru', 'low_prob', or None
                - eviction_count: number of entries to evict
        """
        predictions = predictions or []

        result = {
            'action_type': 'none',
            'cache_current': False,
            'apis_to_prefetch': [],
            'eviction_strategy': None,
            'eviction_count': 0
        }

        if action == CacheAction.DO_NOTHING:
            result['action_type'] = 'none'

        elif action == CacheAction.CACHE_CURRENT:
            result['action_type'] = 'cache'
            result['cache_current'] = True

        elif action == CacheAction.PREFETCH_CONSERVATIVE:
            result['action_type'] = 'prefetch'
            result['apis_to_prefetch'] = self._filter_predictions(
                predictions,
                threshold=self.config.conservative_threshold,
                top_k=self.config.conservative_count
            )

        elif action == CacheAction.PREFETCH_MODERATE:
            result['action_type'] = 'prefetch'
            result['apis_to_prefetch'] = self._filter_predictions(
                predictions,
                threshold=self.config.moderate_threshold,
                top_k=self.config.moderate_count
            )

        elif action == CacheAction.PREFETCH_AGGRESSIVE:
            result['action_type'] = 'prefetch'
            result['apis_to_prefetch'] = self._filter_predictions(
                predictions,
                threshold=self.config.aggressive_threshold,
                top_k=self.config.aggressive_count
            )

        elif action == CacheAction.EVICT_LRU:
            result['action_type'] = 'evict'
            result['eviction_strategy'] = 'lru'
            result['eviction_count'] = self.config.eviction_batch_size

        elif action == CacheAction.EVICT_LOW_PROB:
            result['action_type'] = 'evict'
            result['eviction_strategy'] = 'low_prob'
            result['eviction_count'] = self.config.eviction_batch_size

        return result

    def _filter_predictions(
        self,
        predictions: List[Tuple[str, float]],
        threshold: float,
        top_k: int
    ) -> List[str]:
        """
        Filter predictions by probability threshold and return top-k API names.

        Args:
            predictions: List of (api_name, probability) tuples
            threshold: Minimum probability to include
            top_k: Maximum number of predictions to return

        Returns:
            List of API names that meet the threshold, up to top_k entries
        """
        # Filter by threshold
        filtered = [(api, prob) for api, prob in predictions if prob > threshold]

        # Take top-k
        filtered = filtered[:top_k]

        # Return just the API names
        return [api for api, prob in filtered]


class ActionHistory:
    """
    Records and analyzes action history for debugging and evaluation.

    Helps understand what actions the agent is taking and whether they
    lead to good rewards.
    """

    def __init__(self):
        """Initialize empty action history."""
        self.history = []
        self._action_counts = {i: 0 for i in range(CacheAction.num_actions())}
        self._action_rewards = {i: [] for i in range(CacheAction.num_actions())}

    def record(
        self,
        action: int,
        state: np.ndarray,
        reward: float,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Record an action taken by the agent.

        Args:
            action: Action index that was taken
            state: State vector at time of action
            reward: Reward received for this action
            context: Optional additional context (e.g., timestamp, user_id)
        """
        record = {
            'action': action,
            'action_name': CacheAction.get_name(action),
            'state': state.copy() if isinstance(state, np.ndarray) else state,
            'reward': reward,
            'context': context or {}
        }

        self.history.append(record)
        self._action_counts[action] += 1
        self._action_rewards[action].append(reward)

    def get_action_distribution(self) -> Dict[str, float]:
        """
        Return distribution of actions taken (as percentages).

        Returns:
            Dictionary mapping action_name -> frequency (0.0-1.0)
        """
        total = sum(self._action_counts.values())
        if total == 0:
            return {CacheAction.get_name(i): 0.0 for i in range(CacheAction.num_actions())}

        distribution = {}
        for action_idx in range(CacheAction.num_actions()):
            action_name = CacheAction.get_name(action_idx)
            frequency = self._action_counts[action_idx] / total
            distribution[action_name] = frequency

        return distribution

    def get_reward_by_action(self) -> Dict[str, float]:
        """
        Return average reward for each action type.

        Returns:
            Dictionary mapping action_name -> average_reward
        """
        avg_rewards = {}
        for action_idx in range(CacheAction.num_actions()):
            action_name = CacheAction.get_name(action_idx)
            rewards = self._action_rewards[action_idx]
            avg_rewards[action_name] = np.mean(rewards) if rewards else 0.0

        return avg_rewards

    def get_statistics(self) -> Dict[str, Any]:
        """
        Return comprehensive statistics about action history.

        Returns:
            Dictionary with various statistics
        """
        total_actions = len(self.history)
        total_reward = sum(record['reward'] for record in self.history)

        return {
            'total_actions': total_actions,
            'total_reward': total_reward,
            'average_reward': total_reward / total_actions if total_actions > 0 else 0.0,
            'action_distribution': self.get_action_distribution(),
            'reward_by_action': self.get_reward_by_action()
        }

    def clear(self):
        """Clear all recorded history."""
        self.history.clear()
        self._action_counts = {i: 0 for i in range(CacheAction.num_actions())}
        self._action_rewards = {i: [] for i in range(CacheAction.num_actions())}

    def get_recent_actions(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Return the n most recent actions.

        Args:
            n: Number of recent actions to return

        Returns:
            List of recent action records
        """
        return self.history[-n:] if self.history else []


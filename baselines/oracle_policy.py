"""
Oracle policy - perfect knowledge upper bound.

This policy "cheats" by having access to future API calls, representing the
theoretical best possible performance. Useful for understanding improvement potential.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from .base_policy import CachingPolicy


class OraclePolicy(CachingPolicy):
    """
    Oracle policy with perfect future knowledge.

    This policy "cheats" by knowing what API calls will come next. It represents
    the theoretical upper bound on performance - the best any policy could possibly do.

    Strategy:
    - Has access to future API calls (passed via set_future_calls)
    - Prefetches items that WILL be accessed
    - Never wastes cache space on items that WON'T be accessed
    - Perfect cache management

    Note: This requires special handling in the evaluation framework to pass
    future information. In normal operation, this information wouldn't be available.

    Parameters:
        lookahead_window: How many future steps to consider
        prefetch_if_within: Prefetch if item will be accessed within N steps

    Example:
        >>> policy = OraclePolicy(lookahead_window=10)
        >>> # In evaluation loop:
        >>> policy.set_future_calls(['api1', 'api2', 'api3'])
        >>> action = policy.select_action(state, predictions)
        >>> # Oracle will prefetch api1, api2, api3 since it knows they're coming
    """

    def __init__(
        self,
        lookahead_window: int = 10,
        prefetch_if_within: int = 5,
        eviction_threshold: float = 0.9
    ):
        """
        Initialize oracle policy.

        Args:
            lookahead_window: How many future steps to look ahead
            prefetch_if_within: Prefetch if access within N steps
            eviction_threshold: Cache utilization for eviction
        """
        self.lookahead_window = lookahead_window
        self.prefetch_if_within = prefetch_if_within
        self.eviction_threshold = eviction_threshold

        # Future information (must be set externally)
        self._future_calls: List[str] = []
        self._cached_items: set = set()

        # Statistics
        self._step_count = 0
        self._perfect_prefetches = 0
        self._wasted_prefetches = 0
        self._perfect_cache_hits = 0

    def set_future_calls(self, future_calls: List[str]):
        """
        Set the future API calls (used by evaluation framework).

        Args:
            future_calls: List of future API endpoint names
        """
        self._future_calls = future_calls[:self.lookahead_window]

    def select_action(self, state: np.ndarray, predictions: List[Tuple[str, float]]) -> int:
        """
        Select action with perfect knowledge.

        Args:
            state: State vector
            predictions: Markov predictions (oracle doesn't need these)

        Returns:
            Optimal action based on future knowledge
        """
        from src.rl.actions import CacheAction

        self._step_count += 1

        # Extract cache utilization
        cache_utilization = 0.5
        if len(state) >= 12:
            cache_utilization = float(state[11])

        # Priority 1: Evict if cache too full
        if cache_utilization > self.eviction_threshold:
            return int(CacheAction.EVICT_LRU)

        # Priority 2: Prefetch based on future knowledge
        if not self._future_calls:
            # No future information available - just cache current
            return int(CacheAction.CACHE_CURRENT)

        # Check how soon the top predictions will be accessed
        near_future = self._future_calls[:self.prefetch_if_within]

        # Count how many predicted items are in near future
        future_in_predictions = 0
        if predictions:
            for api, prob in predictions[:5]:  # Check top-5 predictions
                if api in near_future:
                    future_in_predictions += 1

        # Decide prefetch aggressiveness based on matches
        if future_in_predictions >= 3:
            # Many matches - prefetch moderately
            self._perfect_prefetches += 1
            return int(CacheAction.PREFETCH_MODERATE)
        elif future_in_predictions >= 1:
            # Some matches - prefetch conservatively
            self._perfect_prefetches += 1
            return int(CacheAction.PREFETCH_CONSERVATIVE)
        else:
            # No matches - just cache current
            return int(CacheAction.CACHE_CURRENT)

    def advance_step(self, actual_call: str):
        """
        Advance the oracle's knowledge after a step.

        Args:
            actual_call: The API that was actually called this step
        """
        # Remove the call that just happened from future calls
        if self._future_calls and self._future_calls[0] == actual_call:
            self._future_calls.pop(0)
            self._perfect_cache_hits += 1

    def get_name(self) -> str:
        """Return policy name."""
        return f"Oracle (lookahead={self.lookahead_window})"

    def reset(self):
        """Reset for new episode."""
        self._step_count = 0
        self._future_calls = []
        self._cached_items.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get oracle statistics."""
        return {
            'step_count': self._step_count,
            'perfect_prefetches': self._perfect_prefetches,
            'perfect_cache_hits': self._perfect_cache_hits,
            'lookahead_window': self.lookahead_window,
            'prefetch_if_within': self.prefetch_if_within,
            'perfect_prefetch_rate': self._perfect_prefetches / max(1, self._step_count)
        }


class PartialOraclePolicy(CachingPolicy):
    """
    Partial oracle policy with limited future knowledge.

    More realistic than full oracle - has some probability of knowing future calls.
    Represents what you could achieve with partial prediction accuracy.

    Parameters:
        knowledge_probability: Probability of having correct future knowledge
        lookahead_window: How many steps to look ahead
        fallback_policy: Policy to use when no future knowledge available
    """

    def __init__(
        self,
        knowledge_probability: float = 0.8,
        lookahead_window: int = 5,
        fallback_policy: Optional[CachingPolicy] = None,
        seed: Optional[int] = None
    ):
        """Initialize partial oracle policy."""
        if not 0.0 <= knowledge_probability <= 1.0:
            raise ValueError(f"knowledge_probability must be in [0, 1], got {knowledge_probability}")

        self.knowledge_probability = knowledge_probability
        self.lookahead_window = lookahead_window
        self.fallback_policy = fallback_policy
        self._rng = np.random.RandomState(seed)

        self._future_calls: List[str] = []
        self._step_count = 0
        self._oracle_decisions = 0
        self._fallback_decisions = 0

    def set_future_calls(self, future_calls: List[str]):
        """Set future calls."""
        self._future_calls = future_calls[:self.lookahead_window]

    def select_action(self, state: np.ndarray, predictions: List[Tuple[str, float]]) -> int:
        """Select action with partial knowledge."""
        from src.rl.actions import CacheAction

        self._step_count += 1

        # Probabilistically have future knowledge
        has_knowledge = self._rng.random() < self.knowledge_probability

        if has_knowledge and self._future_calls:
            # Use oracle knowledge
            self._oracle_decisions += 1

            # Simple oracle logic: prefetch if items in future
            if len(self._future_calls) >= 3:
                return int(CacheAction.PREFETCH_MODERATE)
            elif len(self._future_calls) >= 1:
                return int(CacheAction.PREFETCH_CONSERVATIVE)
            else:
                return int(CacheAction.CACHE_CURRENT)
        else:
            # Use fallback policy
            self._fallback_decisions += 1

            if self.fallback_policy:
                return self.fallback_policy.select_action(state, predictions)
            else:
                # Default fallback: cache current
                return int(CacheAction.CACHE_CURRENT)

    def advance_step(self, actual_call: str):
        """Advance oracle knowledge."""
        if self._future_calls and self._future_calls[0] == actual_call:
            self._future_calls.pop(0)

    def get_name(self) -> str:
        """Return policy name."""
        return f"Partial Oracle (p={self.knowledge_probability:.2f})"

    def reset(self):
        """Reset for new episode."""
        self._step_count = 0
        self._future_calls = []
        if self.fallback_policy:
            self.fallback_policy.reset()

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            'step_count': self._step_count,
            'oracle_decisions': self._oracle_decisions,
            'fallback_decisions': self._fallback_decisions,
            'oracle_rate': self._oracle_decisions / max(1, self._step_count),
            'knowledge_probability': self.knowledge_probability
        }


class NoisyOraclePolicy(CachingPolicy):
    """
    Noisy oracle policy - has future knowledge but with errors.

    Represents imperfect prediction. The oracle knows future calls but they
    might be in wrong order or have some missing.

    Parameters:
        noise_rate: Probability of error in future knowledge
        lookahead_window: How many steps to look ahead
    """

    def __init__(
        self,
        noise_rate: float = 0.2,
        lookahead_window: int = 10,
        seed: Optional[int] = None
    ):
        """Initialize noisy oracle policy."""
        if not 0.0 <= noise_rate <= 1.0:
            raise ValueError(f"noise_rate must be in [0, 1], got {noise_rate}")

        self.noise_rate = noise_rate
        self.lookahead_window = lookahead_window
        self._rng = np.random.RandomState(seed)

        self._future_calls: List[str] = []
        self._step_count = 0

    def set_future_calls(self, future_calls: List[str]):
        """Set future calls with noise."""
        clean_future = future_calls[:self.lookahead_window]

        # Add noise: randomly shuffle some elements
        noisy_future = clean_future.copy()
        for i in range(len(noisy_future)):
            if self._rng.random() < self.noise_rate:
                # Introduce error - swap with random position
                j = self._rng.randint(0, len(noisy_future))
                noisy_future[i], noisy_future[j] = noisy_future[j], noisy_future[i]

        self._future_calls = noisy_future

    def select_action(self, state: np.ndarray, predictions: List[Tuple[str, float]]) -> int:
        """Select action with noisy future knowledge."""
        from src.rl.actions import CacheAction

        self._step_count += 1

        if not self._future_calls:
            return int(CacheAction.CACHE_CURRENT)

        # Use noisy future knowledge
        near_future = self._future_calls[:5]

        if len(near_future) >= 3:
            return int(CacheAction.PREFETCH_MODERATE)
        elif len(near_future) >= 1:
            return int(CacheAction.PREFETCH_CONSERVATIVE)
        else:
            return int(CacheAction.CACHE_CURRENT)

    def advance_step(self, actual_call: str):
        """Advance knowledge."""
        if self._future_calls and self._future_calls[0] == actual_call:
            self._future_calls.pop(0)

    def get_name(self) -> str:
        """Return policy name."""
        return f"Noisy Oracle (noise={self.noise_rate:.2f})"

    def reset(self):
        """Reset for new episode."""
        self._step_count = 0
        self._future_calls = []

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            'step_count': self._step_count,
            'noise_rate': self.noise_rate
        }

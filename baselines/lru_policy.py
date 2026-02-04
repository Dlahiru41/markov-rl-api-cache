"""
Least Recently Used (LRU) caching policy.

Classic LRU baseline - the standard against which most caching algorithms are compared.
This policy always caches current responses and evicts least recently used items when full.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from collections import OrderedDict

from .base_policy import CachingPolicy


class LRUPolicy(CachingPolicy):
    """
    Classic Least Recently Used (LRU) caching policy.

    Strategy:
    - Always cache the current API response (CACHE_CURRENT)
    - When cache utilization exceeds threshold, evict LRU items (EVICT_LRU)
    - Never proactively prefetch (conservative approach)

    This is the standard baseline that most real-world caches use.

    Parameters:
        eviction_threshold: Cache utilization (0.0-1.0) that triggers eviction
        conservative: If True, only use CACHE_CURRENT. If False, also use EVICT_LRU

    Example:
        >>> policy = LRUPolicy(eviction_threshold=0.9)
        >>> action = policy.select_action(state, predictions)
        >>> # Returns CACHE_CURRENT normally, EVICT_LRU when cache is full
    """

    def __init__(self, eviction_threshold: float = 0.9, conservative: bool = False):
        """
        Initialize LRU policy.

        Args:
            eviction_threshold: Cache utilization to trigger eviction (default 0.9)
            conservative: If True, never use EVICT_LRU (default False)
        """
        if not 0.0 <= eviction_threshold <= 1.0:
            raise ValueError(f"eviction_threshold must be in [0, 1], got {eviction_threshold}")

        self.eviction_threshold = eviction_threshold
        self.conservative = conservative

        # Track access history for LRU ordering
        self._access_history: OrderedDict[str, int] = OrderedDict()
        self._step_count = 0

        # Statistics
        self._eviction_count = 0
        self._cache_decision_count = 0

    def select_action(self, state: np.ndarray, predictions: List[Tuple[str, float]]) -> int:
        """
        Select action based on LRU logic.

        Args:
            state: State vector (we extract cache utilization from it)
            predictions: Markov predictions (not used by LRU)

        Returns:
            1 (CACHE_CURRENT) normally, 5 (EVICT_LRU) when cache is full
        """
        from src.rl.actions import CacheAction

        self._step_count += 1

        # Extract cache utilization from state
        # Based on StateConfig: cache metrics start after predictions (indices 10+)
        # State structure: [api_indices(5), probabilities(5), confidence(1),
        #                   cache_utilization, hit_rate, entries, eviction_rate, ...]
        cache_utilization = 0.5  # Default
        if len(state) >= 12:
            cache_utilization = float(state[11])  # Cache utilization is at index 11

        # Decision logic
        if not self.conservative and cache_utilization > self.eviction_threshold:
            # Cache is full - trigger eviction
            self._eviction_count += 1
            return int(CacheAction.EVICT_LRU)
        else:
            # Normal operation - cache current response
            self._cache_decision_count += 1
            return int(CacheAction.CACHE_CURRENT)

    def get_name(self) -> str:
        """Return policy name."""
        suffix = " (Conservative)" if self.conservative else ""
        return f"LRU{suffix}"

    def reset(self):
        """Reset internal state for new episode."""
        self._access_history.clear()
        self._step_count = 0
        # Don't reset cumulative statistics (eviction_count, cache_decision_count)

    def get_statistics(self) -> Dict[str, Any]:
        """Get LRU-specific statistics."""
        return {
            'eviction_count': self._eviction_count,
            'cache_decision_count': self._cache_decision_count,
            'eviction_threshold': self.eviction_threshold,
            'conservative': self.conservative,
            'step_count': self._step_count,
            'eviction_rate': self._eviction_count / max(1, self._step_count)
        }


class AdaptiveLRUPolicy(CachingPolicy):
    """
    Adaptive LRU policy that adjusts eviction threshold based on performance.

    Strategy:
    - Start with initial eviction threshold
    - If cache hit rate is low, lower threshold (evict more aggressively)
    - If cache hit rate is high, raise threshold (keep more items)
    - Adapt within min/max bounds

    This represents a smarter LRU variant that learns from performance.

    Parameters:
        initial_threshold: Starting eviction threshold
        adaptation_rate: How quickly to adjust threshold
        min_threshold: Minimum eviction threshold
        max_threshold: Maximum eviction threshold
        window_size: Number of steps to consider for hit rate
    """

    def __init__(
        self,
        initial_threshold: float = 0.9,
        adaptation_rate: float = 0.01,
        min_threshold: float = 0.7,
        max_threshold: float = 0.95,
        window_size: int = 100
    ):
        """Initialize adaptive LRU policy."""
        self.threshold = initial_threshold
        self.initial_threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.window_size = window_size

        self._step_count = 0
        self._recent_hit_rates: List[float] = []
        self._adaptation_count = 0

    def select_action(self, state: np.ndarray, predictions: List[Tuple[str, float]]) -> int:
        """Select action using adaptive LRU logic."""
        from src.rl.actions import CacheAction

        self._step_count += 1

        # Extract cache metrics from state
        cache_utilization = 0.5
        cache_hit_rate = 0.5
        if len(state) >= 13:
            cache_utilization = float(state[11])
            cache_hit_rate = float(state[12])

        # Track hit rate for adaptation
        self._recent_hit_rates.append(cache_hit_rate)
        if len(self._recent_hit_rates) > self.window_size:
            self._recent_hit_rates.pop(0)

        # Adapt threshold every window_size steps
        if self._step_count % self.window_size == 0 and len(self._recent_hit_rates) >= self.window_size:
            avg_hit_rate = np.mean(self._recent_hit_rates)

            # If hit rate is low, lower threshold (evict more)
            # If hit rate is high, raise threshold (keep more)
            if avg_hit_rate < 0.5:
                self.threshold = max(self.min_threshold, self.threshold - self.adaptation_rate)
                self._adaptation_count += 1
            elif avg_hit_rate > 0.7:
                self.threshold = min(self.max_threshold, self.threshold + self.adaptation_rate)
                self._adaptation_count += 1

        # Decision logic
        if cache_utilization > self.threshold:
            return int(CacheAction.EVICT_LRU)
        else:
            return int(CacheAction.CACHE_CURRENT)

    def get_name(self) -> str:
        """Return policy name."""
        return "Adaptive LRU"

    def reset(self):
        """Reset for new episode."""
        self._step_count = 0
        self._recent_hit_rates.clear()
        self.threshold = self.initial_threshold

    def get_statistics(self) -> Dict[str, Any]:
        """Get adaptive LRU statistics."""
        return {
            'current_threshold': self.threshold,
            'adaptation_count': self._adaptation_count,
            'window_size': self.window_size,
            'step_count': self._step_count,
            'recent_avg_hit_rate': np.mean(self._recent_hit_rates) if self._recent_hit_rates else 0.0
        }

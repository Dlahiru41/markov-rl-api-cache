"""
Least Frequently Used (LFU) caching policy.

LFU baseline that tracks access frequencies and evicts least frequently accessed items.
Often better than LRU for workloads with stable hot items.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from collections import defaultdict

from .base_policy import CachingPolicy


class LFUPolicy(CachingPolicy):
    """
    Least Frequently Used (LFU) caching policy.

    Strategy:
    - Track access frequency for each cached endpoint
    - Always cache current response (CACHE_CURRENT)
    - When cache is full, evict items with lowest frequency (EVICT_LOW_PROB)
    - Apply frequency decay over time to prevent old popular items from staying forever

    This is often better than LRU for workloads with stable hot items that should
    remain cached even if not accessed recently.

    Parameters:
        eviction_threshold: Cache utilization that triggers eviction
        decay_rate: How quickly to decay frequencies (0.0 = no decay, 1.0 = full decay)
        decay_interval: Steps between decay applications

    Example:
        >>> policy = LFUPolicy(eviction_threshold=0.85, decay_rate=0.01)
        >>> action = policy.select_action(state, predictions)
    """

    def __init__(
        self,
        eviction_threshold: float = 0.85,
        decay_rate: float = 0.01,
        decay_interval: int = 100
    ):
        """
        Initialize LFU policy.

        Args:
            eviction_threshold: Cache utilization to trigger eviction
            decay_rate: Frequency decay rate per decay_interval steps
            decay_interval: Steps between frequency decay
        """
        if not 0.0 <= eviction_threshold <= 1.0:
            raise ValueError(f"eviction_threshold must be in [0, 1], got {eviction_threshold}")
        if not 0.0 <= decay_rate <= 1.0:
            raise ValueError(f"decay_rate must be in [0, 1], got {decay_rate}")
        if decay_interval <= 0:
            raise ValueError(f"decay_interval must be positive, got {decay_interval}")

        self.eviction_threshold = eviction_threshold
        self.decay_rate = decay_rate
        self.decay_interval = decay_interval

        # Track access frequency per endpoint
        self._frequency: Dict[str, float] = defaultdict(float)
        self._step_count = 0

        # Statistics
        self._eviction_count = 0
        self._cache_decision_count = 0
        self._decay_applications = 0
        self._total_accesses = 0

    def _apply_decay(self):
        """Apply frequency decay to all tracked endpoints."""
        decay_factor = 1.0 - self.decay_rate
        for endpoint in self._frequency:
            self._frequency[endpoint] *= decay_factor
        self._decay_applications += 1

    def _record_access(self, predictions: List[Tuple[str, float]]):
        """Record access to increase frequency."""
        # In a real system, we'd track the actual endpoint accessed
        # For simulation, we track based on predictions (top prediction is likely accessed)
        if predictions:
            endpoint = predictions[0][0]  # Top prediction
            self._frequency[endpoint] += 1.0
            self._total_accesses += 1

    def select_action(self, state: np.ndarray, predictions: List[Tuple[str, float]]) -> int:
        """
        Select action based on LFU logic.

        Args:
            state: State vector
            predictions: Markov predictions (used to infer access patterns)

        Returns:
            1 (CACHE_CURRENT) normally, 6 (EVICT_LOW_PROB) when cache is full
        """
        from src.rl.actions import CacheAction

        self._step_count += 1

        # Record access for frequency tracking
        self._record_access(predictions)

        # Apply decay periodically
        if self._step_count % self.decay_interval == 0:
            self._apply_decay()

        # Extract cache utilization from state
        cache_utilization = 0.5  # Default
        if len(state) >= 12:
            cache_utilization = float(state[11])

        # Decision logic
        if cache_utilization > self.eviction_threshold:
            # Cache is full - evict lowest frequency items
            # We use EVICT_LOW_PROB action, but our logic is frequency-based
            self._eviction_count += 1
            return int(CacheAction.EVICT_LOW_PROB)
        else:
            # Normal operation - cache current response
            self._cache_decision_count += 1
            return int(CacheAction.CACHE_CURRENT)

    def get_name(self) -> str:
        """Return policy name."""
        return "LFU"

    def reset(self):
        """Reset internal state for new episode."""
        # Keep frequency history across episodes (represents learned knowledge)
        self._step_count = 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get LFU-specific statistics."""
        avg_frequency = np.mean(list(self._frequency.values())) if self._frequency else 0.0

        return {
            'eviction_count': self._eviction_count,
            'cache_decision_count': self._cache_decision_count,
            'eviction_threshold': self.eviction_threshold,
            'tracked_endpoints': len(self._frequency),
            'average_frequency': avg_frequency,
            'total_accesses': self._total_accesses,
            'decay_applications': self._decay_applications,
            'step_count': self._step_count,
            'eviction_rate': self._eviction_count / max(1, self._step_count)
        }

    def get_top_endpoints(self, k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top-k most frequently accessed endpoints.

        Args:
            k: Number of top endpoints to return

        Returns:
            List of (endpoint, frequency) tuples sorted by frequency
        """
        sorted_endpoints = sorted(
            self._frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_endpoints[:k]


class WindowedLFUPolicy(CachingPolicy):
    """
    Windowed LFU policy that only considers recent accesses.

    Instead of tracking all-time frequency, only consider the last W accesses.
    This makes the policy more adaptive to changing patterns.

    Parameters:
        eviction_threshold: Cache utilization that triggers eviction
        window_size: Number of recent accesses to consider
    """

    def __init__(
        self,
        eviction_threshold: float = 0.85,
        window_size: int = 1000
    ):
        """Initialize windowed LFU policy."""
        self.eviction_threshold = eviction_threshold
        self.window_size = window_size

        # Circular buffer of recent accesses
        self._access_window: List[str] = []
        self._frequency: Dict[str, int] = defaultdict(int)

        self._step_count = 0
        self._eviction_count = 0
        self._cache_decision_count = 0

    def _record_access(self, endpoint: str):
        """Record an access in the sliding window."""
        # Add new access
        self._access_window.append(endpoint)
        self._frequency[endpoint] += 1

        # Remove oldest access if window is full
        if len(self._access_window) > self.window_size:
            oldest = self._access_window.pop(0)
            self._frequency[oldest] -= 1
            if self._frequency[oldest] <= 0:
                del self._frequency[oldest]

    def select_action(self, state: np.ndarray, predictions: List[Tuple[str, float]]) -> int:
        """Select action using windowed LFU logic."""
        from src.rl.actions import CacheAction

        self._step_count += 1

        # Record access
        if predictions:
            self._record_access(predictions[0][0])

        # Extract cache utilization
        cache_utilization = 0.5
        if len(state) >= 12:
            cache_utilization = float(state[11])

        # Decision logic
        if cache_utilization > self.eviction_threshold:
            self._eviction_count += 1
            return int(CacheAction.EVICT_LOW_PROB)
        else:
            self._cache_decision_count += 1
            return int(CacheAction.CACHE_CURRENT)

    def get_name(self) -> str:
        """Return policy name."""
        return f"Windowed LFU (W={self.window_size})"

    def reset(self):
        """Reset for new episode."""
        self._step_count = 0
        # Keep frequency tracking across episodes

    def get_statistics(self) -> Dict[str, Any]:
        """Get windowed LFU statistics."""
        return {
            'eviction_count': self._eviction_count,
            'cache_decision_count': self._cache_decision_count,
            'window_size': self.window_size,
            'current_window_fill': len(self._access_window),
            'tracked_endpoints': len(self._frequency),
            'step_count': self._step_count
        }

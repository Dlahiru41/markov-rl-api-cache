"""
Adaptive heuristic caching policy.

This policy adjusts its behavior based on observed performance metrics, representing
a sophisticated hand-crafted approach that's smarter than static rules but doesn't
use machine learning.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from collections import deque

from .base_policy import CachingPolicy


class AdaptivePolicy(CachingPolicy):
    """
    Adaptive heuristic policy that adjusts behavior based on observed metrics.

    This represents a more sophisticated baseline than static rules. It tracks
    recent performance and adapts its strategy accordingly.

    Adaptation Logic:
    - Track cache hit rate over recent window
    - If hit rate is dropping: become more aggressive with prefetching
    - If hit rate is stable and high: reduce prefetching to save bandwidth
    - If seeing cascade patterns (high system load): prioritize critical endpoints
    - Adjust thresholds based on performance

    Parameters:
        window_size: Number of steps to consider for adaptation
        aggression_step: How much to change thresholds when adapting
        min_hit_rate: Hit rate threshold below which we become aggressive
        max_hit_rate: Hit rate threshold above which we become conservative

    Example:
        >>> policy = AdaptivePolicy(window_size=100, aggression_step=0.05)
        >>> action = policy.select_action(state, predictions)
        >>> # Policy adapts based on recent performance
    """

    def __init__(
        self,
        window_size: int = 100,
        aggression_step: float = 0.05,
        min_hit_rate: float = 0.5,
        max_hit_rate: float = 0.8,
        initial_conservative_threshold: float = 0.7,
        initial_moderate_threshold: float = 0.5
    ):
        """
        Initialize adaptive policy.

        Args:
            window_size: Steps to consider for adaptation
            aggression_step: Threshold adjustment magnitude
            min_hit_rate: Hit rate below which to become aggressive
            max_hit_rate: Hit rate above which to become conservative
            initial_conservative_threshold: Starting threshold for conservative prefetch
            initial_moderate_threshold: Starting threshold for moderate prefetch
        """
        self.window_size = window_size
        self.aggression_step = aggression_step
        self.min_hit_rate = min_hit_rate
        self.max_hit_rate = max_hit_rate

        # Adaptive thresholds
        self.conservative_threshold = initial_conservative_threshold
        self.moderate_threshold = initial_moderate_threshold
        self.initial_conservative = initial_conservative_threshold
        self.initial_moderate = initial_moderate_threshold

        # Performance tracking
        self._recent_hit_rates: deque = deque(maxlen=window_size)
        self._recent_cpu_loads: deque = deque(maxlen=window_size)
        self._recent_latencies: deque = deque(maxlen=window_size)

        # Statistics
        self._step_count = 0
        self._adaptation_count = 0
        self._aggressive_phases = 0
        self._conservative_phases = 0
        self._action_counts: Dict[str, int] = {
            'prefetch_conservative': 0,
            'prefetch_moderate': 0,
            'prefetch_aggressive': 0,
            'cache_current': 0,
            'evict': 0
        }

    def _update_metrics(self, state: np.ndarray):
        """Extract and track recent metrics from state."""
        # Extract metrics from state vector
        # State structure: [predictions, confidence, cache_metrics, system_metrics, ...]

        if len(state) >= 13:
            cache_hit_rate = float(state[12])
            self._recent_hit_rates.append(cache_hit_rate)

        if len(state) >= 15:
            cpu_load = float(state[15])  # Approximate CPU index
            self._recent_cpu_loads.append(cpu_load)

        if len(state) >= 18:
            latency_p50 = float(state[18])  # Approximate latency index
            self._recent_latencies.append(latency_p50)

    def _adapt_thresholds(self):
        """Adapt thresholds based on recent performance."""
        if len(self._recent_hit_rates) < self.window_size // 2:
            return  # Not enough data yet

        avg_hit_rate = np.mean(list(self._recent_hit_rates))
        avg_cpu = np.mean(list(self._recent_cpu_loads)) if self._recent_cpu_loads else 0.5

        # Adaptation logic
        if avg_hit_rate < self.min_hit_rate:
            # Low hit rate - become more aggressive with prefetching
            # Lower thresholds = prefetch more often
            self.conservative_threshold = max(0.3, self.conservative_threshold - self.aggression_step)
            self.moderate_threshold = max(0.2, self.moderate_threshold - self.aggression_step)
            self._aggressive_phases += 1
            self._adaptation_count += 1

        elif avg_hit_rate > self.max_hit_rate and avg_cpu < 0.7:
            # High hit rate and not stressed - become more conservative
            # Raise thresholds = prefetch less often (save resources)
            self.conservative_threshold = min(0.9, self.conservative_threshold + self.aggression_step)
            self.moderate_threshold = min(0.8, self.moderate_threshold + self.aggression_step)
            self._conservative_phases += 1
            self._adaptation_count += 1

    def select_action(self, state: np.ndarray, predictions: List[Tuple[str, float]]) -> int:
        """
        Select action with adaptive logic.

        Args:
            state: State vector with metrics
            predictions: Markov predictions

        Returns:
            Action based on adaptive thresholds
        """
        from ..src.rl.actions import CacheAction

        self._step_count += 1

        # Update metrics tracking
        self._update_metrics(state)

        # Adapt every window_size steps
        if self._step_count % self.window_size == 0:
            self._adapt_thresholds()

        # Extract current state features
        cache_utilization = float(state[11]) if len(state) >= 12 else 0.5
        cache_hit_rate = float(state[12]) if len(state) >= 13 else 0.5
        cpu_load = float(state[15]) if len(state) >= 16 else 0.5

        # Get prediction confidence
        confidence = predictions[0][1] if predictions else 0.0

        # Priority 1: Handle critical situations
        if cache_utilization > 0.95:
            # Cache critically full - evict
            self._action_counts['evict'] += 1
            return int(CacheAction.EVICT_LRU)

        if cpu_load > 0.85:
            # System under stress - be conservative, just cache current
            self._action_counts['cache_current'] += 1
            return int(CacheAction.CACHE_CURRENT)

        # Priority 2: Cascade risk detection
        if cpu_load > 0.75 and cache_hit_rate < 0.6:
            # Risk of cascade - prefetch moderately to improve hit rate
            self._action_counts['prefetch_moderate'] += 1
            return int(CacheAction.PREFETCH_MODERATE)

        # Priority 3: Normal operation with adaptive thresholds
        if confidence >= self.conservative_threshold:
            # High confidence - prefetch conservatively
            self._action_counts['prefetch_conservative'] += 1
            return int(CacheAction.PREFETCH_CONSERVATIVE)

        elif confidence >= self.moderate_threshold:
            # Medium confidence - prefetch moderately
            self._action_counts['prefetch_moderate'] += 1
            return int(CacheAction.PREFETCH_MODERATE)

        elif confidence >= 0.3 and cache_hit_rate < 0.5:
            # Low confidence but poor hit rate - try aggressive to explore
            self._action_counts['prefetch_aggressive'] += 1
            return int(CacheAction.PREFETCH_AGGRESSIVE)

        else:
            # Default - just cache current
            self._action_counts['cache_current'] += 1
            return int(CacheAction.CACHE_CURRENT)

    def get_name(self) -> str:
        """Return policy name."""
        return "Adaptive Heuristic"

    def reset(self):
        """Reset for new episode."""
        self._step_count = 0
        self._recent_hit_rates.clear()
        self._recent_cpu_loads.clear()
        self._recent_latencies.clear()
        # Reset thresholds to initial values
        self.conservative_threshold = self.initial_conservative
        self.moderate_threshold = self.initial_moderate

    def get_statistics(self) -> Dict[str, Any]:
        """Get adaptive policy statistics."""
        avg_hit_rate = np.mean(list(self._recent_hit_rates)) if self._recent_hit_rates else 0.0
        avg_cpu = np.mean(list(self._recent_cpu_loads)) if self._recent_cpu_loads else 0.0

        return {
            'step_count': self._step_count,
            'adaptation_count': self._adaptation_count,
            'aggressive_phases': self._aggressive_phases,
            'conservative_phases': self._conservative_phases,
            'current_thresholds': {
                'conservative': self.conservative_threshold,
                'moderate': self.moderate_threshold
            },
            'recent_avg_hit_rate': avg_hit_rate,
            'recent_avg_cpu': avg_cpu,
            'action_counts': self._action_counts.copy()
        }


class MultiObjectiveAdaptivePolicy(CachingPolicy):
    """
    Multi-objective adaptive policy that balances multiple goals.

    Considers:
    - Cache hit rate (performance)
    - System CPU load (resource usage)
    - Latency (user experience)
    - Prediction confidence (accuracy)

    Uses weighted combination to make decisions.

    Parameters:
        hit_rate_weight: Weight for cache hit rate objective
        latency_weight: Weight for latency objective
        cpu_weight: Weight for CPU usage objective
        confidence_weight: Weight for prediction confidence
    """

    def __init__(
        self,
        hit_rate_weight: float = 0.4,
        latency_weight: float = 0.3,
        cpu_weight: float = 0.2,
        confidence_weight: float = 0.1,
        window_size: int = 50
    ):
        """Initialize multi-objective adaptive policy."""
        # Normalize weights
        total = hit_rate_weight + latency_weight + cpu_weight + confidence_weight
        self.hit_rate_weight = hit_rate_weight / total
        self.latency_weight = latency_weight / total
        self.cpu_weight = cpu_weight / total
        self.confidence_weight = confidence_weight / total

        self.window_size = window_size

        # Tracking
        self._recent_scores: deque = deque(maxlen=window_size)
        self._step_count = 0

    def _compute_score(self, state: np.ndarray, predictions: List[Tuple[str, float]]) -> float:
        """Compute multi-objective score."""
        # Extract features
        cache_hit_rate = float(state[12]) if len(state) >= 13 else 0.5
        latency = float(state[18]) if len(state) >= 19 else 0.5
        cpu_load = float(state[15]) if len(state) >= 16 else 0.5
        confidence = predictions[0][1] if predictions else 0.0

        # Compute weighted score (higher is better)
        score = (
            self.hit_rate_weight * cache_hit_rate +
            self.latency_weight * (1.0 - latency) +  # Lower latency is better
            self.cpu_weight * (1.0 - cpu_load) +      # Lower CPU is better
            self.confidence_weight * confidence
        )

        return score

    def select_action(self, state: np.ndarray, predictions: List[Tuple[str, float]]) -> int:
        """Select action based on multi-objective score."""
        from ..src.rl.actions import CacheAction

        self._step_count += 1

        # Compute current score
        score = self._compute_score(state, predictions)
        self._recent_scores.append(score)

        # Extract features
        cache_utilization = float(state[11]) if len(state) >= 12 else 0.5
        confidence = predictions[0][1] if predictions else 0.0

        # Evict if too full
        if cache_utilization > 0.95:
            return int(CacheAction.EVICT_LRU)

        # Compute average recent score
        avg_score = np.mean(list(self._recent_scores)) if self._recent_scores else 0.5

        # Decision based on score trend and confidence
        if avg_score > 0.7:
            # Doing well - be conservative
            if confidence > 0.7:
                return int(CacheAction.PREFETCH_CONSERVATIVE)
            else:
                return int(CacheAction.CACHE_CURRENT)
        elif avg_score > 0.5:
            # Moderate performance - moderate prefetch
            if confidence > 0.5:
                return int(CacheAction.PREFETCH_MODERATE)
            else:
                return int(CacheAction.CACHE_CURRENT)
        else:
            # Poor performance - try aggressive
            if confidence > 0.3:
                return int(CacheAction.PREFETCH_AGGRESSIVE)
            else:
                return int(CacheAction.PREFETCH_MODERATE)

    def get_name(self) -> str:
        """Return policy name."""
        return "Multi-Objective Adaptive"

    def reset(self):
        """Reset for new episode."""
        self._step_count = 0
        self._recent_scores.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            'step_count': self._step_count,
            'recent_avg_score': np.mean(list(self._recent_scores)) if self._recent_scores else 0.0,
            'weights': {
                'hit_rate': self.hit_rate_weight,
                'latency': self.latency_weight,
                'cpu': self.cpu_weight,
                'confidence': self.confidence_weight
            }
        }


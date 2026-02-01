"""
Static Markov policy - uses Markov predictions with fixed heuristic rules.

This baseline uses the Markov predictor but applies hand-crafted rules instead of
learned behavior. It isolates the value of RL vs just using Markov predictions.
"""

import numpy as np
from typing import List, Tuple, Dict, Any

from .base_policy import CachingPolicy


class StaticMarkovPolicy(CachingPolicy):
    """
    Static policy that uses Markov predictions with fixed thresholds.

    This represents what a reasonable hand-crafted policy using Markov predictions
    would look like. It isolates the value of RL by showing what you can achieve
    with just predictions and fixed rules.

    Strategy:
    - Use Markov prediction confidence to decide prefetch aggressiveness
    - High confidence (>0.7): Prefetch conservatively (top-1)
    - Medium confidence (>0.5): Prefetch moderately (top-3)
    - Low confidence (>0.3): Prefetch aggressively (top-5, try more options)
    - Very low confidence: Just cache current, don't prefetch
    - Also evict when cache is too full (>0.95)

    Parameters:
        conservative_threshold: Confidence threshold for conservative prefetch
        moderate_threshold: Confidence threshold for moderate prefetch
        aggressive_threshold: Confidence threshold for aggressive prefetch
        eviction_threshold: Cache utilization threshold for eviction

    Example:
        >>> policy = StaticMarkovPolicy()
        >>> action = policy.select_action(state, predictions)
        >>> # Returns prefetch action based on prediction confidence
    """

    def __init__(
        self,
        conservative_threshold: float = 0.7,
        moderate_threshold: float = 0.5,
        aggressive_threshold: float = 0.3,
        eviction_threshold: float = 0.95
    ):
        """
        Initialize static Markov policy.

        Args:
            conservative_threshold: Min confidence for conservative prefetch
            moderate_threshold: Min confidence for moderate prefetch
            aggressive_threshold: Min confidence for aggressive prefetch
            eviction_threshold: Cache utilization for eviction
        """
        self.conservative_threshold = conservative_threshold
        self.moderate_threshold = moderate_threshold
        self.aggressive_threshold = aggressive_threshold
        self.eviction_threshold = eviction_threshold

        # Statistics
        self._action_counts: Dict[str, int] = {
            'cache_current': 0,
            'prefetch_conservative': 0,
            'prefetch_moderate': 0,
            'prefetch_aggressive': 0,
            'evict_lru': 0
        }
        self._step_count = 0
        self._high_confidence_count = 0
        self._medium_confidence_count = 0
        self._low_confidence_count = 0

    def select_action(self, state: np.ndarray, predictions: List[Tuple[str, float]]) -> int:
        """
        Select action based on prediction confidence and cache state.

        Args:
            state: State vector
            predictions: List of (api, probability) sorted by probability

        Returns:
            Action index based on confidence thresholds
        """
        from ..src.rl.actions import CacheAction

        self._step_count += 1

        # Extract cache utilization from state
        cache_utilization = 0.5
        if len(state) >= 12:
            cache_utilization = float(state[11])

        # Priority 1: Evict if cache is critically full
        if cache_utilization > self.eviction_threshold:
            self._action_counts['evict_lru'] += 1
            return int(CacheAction.EVICT_LRU)

        # Get prediction confidence (max probability)
        confidence = 0.0
        if predictions:
            confidence = predictions[0][1]  # Top prediction probability

        # Priority 2: Select action based on confidence
        if confidence >= self.conservative_threshold:
            # High confidence - prefetch conservatively (just top-1)
            self._high_confidence_count += 1
            self._action_counts['prefetch_conservative'] += 1
            return int(CacheAction.PREFETCH_CONSERVATIVE)

        elif confidence >= self.moderate_threshold:
            # Medium confidence - prefetch moderately (top-3)
            self._medium_confidence_count += 1
            self._action_counts['prefetch_moderate'] += 1
            return int(CacheAction.PREFETCH_MODERATE)

        elif confidence >= self.aggressive_threshold:
            # Low confidence - prefetch aggressively (top-5, try more options)
            self._low_confidence_count += 1
            self._action_counts['prefetch_aggressive'] += 1
            return int(CacheAction.PREFETCH_AGGRESSIVE)

        else:
            # Very low confidence - just cache current, don't waste resources
            self._action_counts['cache_current'] += 1
            return int(CacheAction.CACHE_CURRENT)

    def get_name(self) -> str:
        """Return policy name."""
        return "Static Markov"

    def reset(self):
        """Reset for new episode."""
        self._step_count = 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get static Markov policy statistics."""
        return {
            'step_count': self._step_count,
            'action_counts': self._action_counts.copy(),
            'confidence_distribution': {
                'high': self._high_confidence_count,
                'medium': self._medium_confidence_count,
                'low': self._low_confidence_count
            },
            'thresholds': {
                'conservative': self.conservative_threshold,
                'moderate': self.moderate_threshold,
                'aggressive': self.aggressive_threshold,
                'eviction': self.eviction_threshold
            }
        }


class InverseStaticMarkovPolicy(CachingPolicy):
    """
    Inverse static Markov policy - prefetches MORE when confidence is LOW.

    This is a counter-intuitive baseline that tests whether the standard approach
    is actually correct. When predictions are uncertain, maybe we should try more
    options to increase hit probability.

    Strategy:
    - Low confidence: Prefetch aggressively (cast a wide net)
    - Medium confidence: Prefetch moderately
    - High confidence: Prefetch conservatively (we're pretty sure)
    """

    def __init__(
        self,
        conservative_threshold: float = 0.3,
        moderate_threshold: float = 0.5,
        aggressive_threshold: float = 0.7,
        eviction_threshold: float = 0.95
    ):
        """Initialize inverse static Markov policy."""
        self.conservative_threshold = conservative_threshold
        self.moderate_threshold = moderate_threshold
        self.aggressive_threshold = aggressive_threshold
        self.eviction_threshold = eviction_threshold

        self._step_count = 0
        self._action_counts: Dict[str, int] = {
            'cache_current': 0,
            'prefetch_conservative': 0,
            'prefetch_moderate': 0,
            'prefetch_aggressive': 0,
            'evict_lru': 0
        }

    def select_action(self, state: np.ndarray, predictions: List[Tuple[str, float]]) -> int:
        """Select action with inverse confidence logic."""
        from ..src.rl.actions import CacheAction

        self._step_count += 1

        # Extract cache utilization
        cache_utilization = 0.5
        if len(state) >= 12:
            cache_utilization = float(state[11])

        # Evict if too full
        if cache_utilization > self.eviction_threshold:
            self._action_counts['evict_lru'] += 1
            return int(CacheAction.EVICT_LRU)

        # Get confidence
        confidence = predictions[0][1] if predictions else 0.0

        # INVERSE LOGIC: Low confidence = aggressive prefetch
        if confidence < self.conservative_threshold:
            # Very uncertain - try many options
            self._action_counts['prefetch_aggressive'] += 1
            return int(CacheAction.PREFETCH_AGGRESSIVE)

        elif confidence < self.moderate_threshold:
            # Somewhat uncertain - try moderate options
            self._action_counts['prefetch_moderate'] += 1
            return int(CacheAction.PREFETCH_MODERATE)

        elif confidence < self.aggressive_threshold:
            # Pretty confident - just prefetch top-1
            self._action_counts['prefetch_conservative'] += 1
            return int(CacheAction.PREFETCH_CONSERVATIVE)

        else:
            # Very confident - just cache current
            self._action_counts['cache_current'] += 1
            return int(CacheAction.CACHE_CURRENT)

    def get_name(self) -> str:
        """Return policy name."""
        return "Inverse Static Markov"

    def reset(self):
        """Reset for new episode."""
        self._step_count = 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            'step_count': self._step_count,
            'action_counts': self._action_counts.copy()
        }


class BalancedStaticMarkovPolicy(CachingPolicy):
    """
    Balanced static Markov policy with multiple decision factors.

    Considers both prediction confidence AND cache state to make decisions.
    More sophisticated than pure confidence-based approach.

    Strategy:
    - High confidence + low cache utilization: Prefetch moderately
    - High confidence + high cache utilization: Prefetch conservatively
    - Low confidence + low cache utilization: Prefetch aggressively
    - Low confidence + high cache utilization: Just cache current
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        utilization_threshold: float = 0.8
    ):
        """Initialize balanced static Markov policy."""
        self.confidence_threshold = confidence_threshold
        self.utilization_threshold = utilization_threshold

        self._step_count = 0

    def select_action(self, state: np.ndarray, predictions: List[Tuple[str, float]]) -> int:
        """Select action based on confidence AND cache state."""
        from ..src.rl.actions import CacheAction

        self._step_count += 1

        # Extract state features
        cache_utilization = float(state[11]) if len(state) >= 12 else 0.5
        confidence = predictions[0][1] if predictions else 0.0

        # Evict if critically full
        if cache_utilization > 0.95:
            return int(CacheAction.EVICT_LRU)

        # Decision matrix
        high_confidence = confidence >= self.confidence_threshold
        high_utilization = cache_utilization >= self.utilization_threshold

        if high_confidence and not high_utilization:
            # Confident and have space - prefetch moderately
            return int(CacheAction.PREFETCH_MODERATE)
        elif high_confidence and high_utilization:
            # Confident but constrained - prefetch conservatively
            return int(CacheAction.PREFETCH_CONSERVATIVE)
        elif not high_confidence and not high_utilization:
            # Uncertain with space - try aggressive to explore
            return int(CacheAction.PREFETCH_AGGRESSIVE)
        else:
            # Uncertain and constrained - just cache current
            return int(CacheAction.CACHE_CURRENT)

    def get_name(self) -> str:
        """Return policy name."""
        return "Balanced Static Markov"

    def reset(self):
        """Reset for new episode."""
        self._step_count = 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            'step_count': self._step_count,
            'confidence_threshold': self.confidence_threshold,
            'utilization_threshold': self.utilization_threshold
        }


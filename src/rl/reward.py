"""
Multi-objective reward function for the caching RL agent.

This module implements a carefully designed reward function that balances multiple
competing objectives:
1. Cache performance (hits vs misses)
2. Cascade prevention (most critical)
3. Prefetch efficiency
4. Latency optimization
5. Resource management

The reward magnitudes are intentionally designed so that cascade prevention
dominates - a single cascade can undo the benefit of hundreds of cache hits.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque


@dataclass
class RewardConfig:
    """
    Configuration for reward function weights.

    All weights are tunable to adapt the reward function to different scenarios
    and priorities. The default values encode our priorities:

    1. CASCADE PREVENTION IS MOST IMPORTANT
       - cascade_prevented_reward (50.0) is 5x a cache hit
       - cascade_occurred_penalty (-100.0) is 100x a cache miss
       - Rationale: One cascade can affect thousands of requests

    2. Cache performance matters
       - cache_hit_reward (10.0) is baseline "good" signal
       - cache_miss_penalty (-1.0) is small penalty (misses are normal)
       - Rationale: 10:1 ratio encourages hitting cache without being too conservative

    3. Prefetch efficiency matters moderately
       - prefetch_used_reward (5.0) is half a cache hit
       - prefetch_wasted_penalty (-3.0) is moderate
       - Rationale: Want to prefetch useful items, but waste is not catastrophic

    4. Latency improvements are valuable
       - Weight per ms saved/added incentivizes faster responses
       - Asymmetric weights: degradation hurts more than improvement helps

    5. Resource costs should be tracked but not dominate
       - Small penalties for bandwidth and cache pressure
       - Rationale: Resources matter but shouldn't prevent effective caching
    """

    # Cache performance rewards
    cache_hit_reward: float = 10.0
    cache_miss_penalty: float = -1.0

    # Cascade prevention (MOST IMPORTANT)
    cascade_prevented_reward: float = 50.0
    cascade_occurred_penalty: float = -100.0

    # Prefetch efficiency
    prefetch_used_reward: float = 5.0
    prefetch_wasted_penalty: float = -3.0

    # Latency optimization
    latency_improvement_weight: float = 0.1  # Per ms saved
    latency_degradation_weight: float = -0.2  # Per ms added (asymmetric)

    # Resource management
    bandwidth_penalty_weight: float = -0.01  # Per KB used for prefetching
    cache_full_penalty: float = -5.0

    # Reward shaping (helps learning)
    enable_shaping: bool = True
    correct_prediction_bonus: float = 1.0
    exploration_bonus: float = 0.1

    # Bounds (prevent extreme rewards from destabilizing training)
    clip_min: float = -100.0
    clip_max: float = 100.0


@dataclass
class ActionOutcome:
    """
    Captures the complete outcome of taking an action in the caching system.

    This comprehensive state captures everything needed to compute a detailed
    reward signal, including both immediate effects (cache hit/miss) and
    downstream consequences (cascade prevention, prefetch usage).
    """

    # Cache performance
    cache_hit: bool = False
    cache_miss: bool = False

    # Prefetch metrics
    prefetch_attempted: int = 0
    prefetch_successful: int = 0
    prefetch_used: int = 0
    prefetch_wasted: int = 0
    prefetch_bytes: int = 0

    # Cascade detection/prevention
    cascade_risk_detected: bool = False
    cascade_prevented: bool = False
    cascade_occurred: bool = False

    # Latency metrics
    actual_latency_ms: float = 0.0
    baseline_latency_ms: float = 0.0

    # Cache state
    cache_utilization: float = 0.0
    evictions_triggered: int = 0

    # Prediction quality
    prediction_was_correct: bool = False
    prediction_confidence: float = 0.0


class RewardCalculator:
    """
    Calculates rewards based on action outcomes.

    The reward function is designed to be:
    1. Multi-objective: Balances competing goals
    2. Interpretable: Each component has clear meaning
    3. Stable: Clipped to prevent extreme values
    4. Informative: Provides detailed breakdowns for analysis
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        """
        Initialize calculator with configuration.

        Args:
            config: Reward configuration, or None for defaults
        """
        self.config = config or RewardConfig()

    def calculate(self, outcome: ActionOutcome) -> float:
        """
        Calculate total reward for an action outcome.

        This is the main method used during training. It sums all applicable
        reward components and clips the result to configured bounds.

        Args:
            outcome: ActionOutcome describing what happened

        Returns:
            Single float reward value
        """
        breakdown = self.calculate_detailed(outcome)
        return breakdown['total']

    def calculate_detailed(self, outcome: ActionOutcome) -> Dict[str, float]:
        """
        Calculate reward with detailed breakdown by component.

        This is essential for understanding what's driving the agent's
        learning signal. Use this for debugging and analysis.

        Args:
            outcome: ActionOutcome describing what happened

        Returns:
            Dictionary with components:
                - cache: Cache hit/miss rewards
                - cascade: Cascade prevention/occurrence
                - prefetch: Prefetch efficiency
                - latency: Latency improvements/degradations
                - bandwidth: Bandwidth usage penalties
                - shaping: Auxiliary reward shaping signals
                - total: Sum of all components (clipped)
        """
        breakdown = {
            'cache': 0.0,
            'cascade': 0.0,
            'prefetch': 0.0,
            'latency': 0.0,
            'bandwidth': 0.0,
            'shaping': 0.0,
            'total': 0.0
        }

        # 1. Cache performance
        if outcome.cache_hit:
            breakdown['cache'] += self.config.cache_hit_reward
        if outcome.cache_miss:
            breakdown['cache'] += self.config.cache_miss_penalty

        # 2. CASCADE PREVENTION (highest priority)
        if outcome.cascade_prevented:
            breakdown['cascade'] += self.config.cascade_prevented_reward
        if outcome.cascade_occurred:
            breakdown['cascade'] += self.config.cascade_occurred_penalty

        # 3. Prefetch efficiency
        breakdown['prefetch'] += (
            outcome.prefetch_used * self.config.prefetch_used_reward
        )
        breakdown['prefetch'] += (
            outcome.prefetch_wasted * self.config.prefetch_wasted_penalty
        )

        # 4. Latency optimization
        latency_delta = outcome.baseline_latency_ms - outcome.actual_latency_ms
        if latency_delta > 0:
            # We improved latency (saved time)
            breakdown['latency'] += (
                latency_delta * self.config.latency_improvement_weight
            )
        elif latency_delta < 0:
            # We degraded latency (added time)
            breakdown['latency'] += (
                abs(latency_delta) * self.config.latency_degradation_weight
            )

        # 5. Bandwidth usage
        if outcome.prefetch_bytes > 0:
            prefetch_kb = outcome.prefetch_bytes / 1024.0
            breakdown['bandwidth'] += (
                prefetch_kb * self.config.bandwidth_penalty_weight
            )

        # 6. Cache pressure penalty
        if outcome.cache_utilization > 0.95:
            breakdown['bandwidth'] += self.config.cache_full_penalty

        # 7. Reward shaping (auxiliary signals to help learning)
        if self.config.enable_shaping:
            if outcome.prediction_was_correct:
                breakdown['shaping'] += self.config.correct_prediction_bonus

            # Small exploration bonus (encourages trying things)
            breakdown['shaping'] += self.config.exploration_bonus

        # Sum all components
        total = sum(breakdown.values())

        # Clip to bounds for training stability
        total = np.clip(total, self.config.clip_min, self.config.clip_max)
        breakdown['total'] = total

        return breakdown


class RewardNormalizer:
    """
    Normalizes rewards for training stability.

    Neural networks train more stably when inputs have zero mean and unit
    variance. This class tracks running statistics of rewards and normalizes
    them accordingly.

    Uses Welford's online algorithm for numerical stability.
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize normalizer.

        Args:
            epsilon: Small constant to prevent division by zero
        """
        self.epsilon = epsilon
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squared differences from mean

    def update(self, reward: float):
        """
        Update running statistics with new reward.

        Uses Welford's online algorithm for numerical stability.

        Args:
            reward: New reward value
        """
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        """Calculate variance from running statistics."""
        if self.count < 2:
            return 1.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        """Calculate standard deviation."""
        return np.sqrt(self.variance)

    def normalize(self, reward: float) -> float:
        """
        Normalize a reward using running statistics.

        Args:
            reward: Raw reward value

        Returns:
            Normalized reward: (reward - mean) / std
        """
        if self.count == 0:
            return reward

        return (reward - self.mean) / (self.std + self.epsilon)

    def denormalize(self, normalized_reward: float) -> float:
        """
        Convert normalized reward back to original scale.

        Args:
            normalized_reward: Normalized reward value

        Returns:
            Original scale reward
        """
        if self.count == 0:
            return normalized_reward

        return normalized_reward * self.std + self.mean

    def reset(self):
        """Reset statistics."""
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0


class RewardTracker:
    """
    Tracks and analyzes reward history for monitoring and debugging.

    Maintains a sliding window of recent rewards and their breakdowns,
    providing statistics to help understand agent behavior and learning
    progress.
    """

    def __init__(self, window_size: int = 1000):
        """
        Initialize tracker.

        Args:
            window_size: Number of recent rewards to track
        """
        self.window_size = window_size
        self.rewards = deque(maxlen=window_size)
        self.breakdowns = deque(maxlen=window_size)

    def record(self, reward: float, breakdown: Dict[str, float]):
        """
        Record a reward and its breakdown.

        Args:
            reward: Total reward value
            breakdown: Dictionary with reward component breakdown
        """
        self.rewards.append(reward)
        self.breakdowns.append(breakdown.copy())

    def get_statistics(self) -> Dict[str, float]:
        """
        Get statistics over recent rewards.

        Returns:
            Dictionary with:
                - mean: Average reward
                - std: Standard deviation
                - min: Minimum reward
                - max: Maximum reward
                - count: Number of rewards tracked
        """
        if not self.rewards:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'count': 0
            }

        rewards_array = np.array(self.rewards)
        return {
            'mean': float(np.mean(rewards_array)),
            'std': float(np.std(rewards_array)),
            'min': float(np.min(rewards_array)),
            'max': float(np.max(rewards_array)),
            'count': len(self.rewards)
        }

    def get_component_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for each reward component.

        This helps identify which components are driving rewards and
        whether they're balanced appropriately.

        Returns:
            Dictionary mapping component name to statistics dict
        """
        if not self.breakdowns:
            return {}

        # Extract component names from first breakdown
        components = list(self.breakdowns[0].keys())

        stats = {}
        for component in components:
            if component == 'total':
                continue  # Skip total (it's in get_statistics)

            values = [bd[component] for bd in self.breakdowns]
            values_array = np.array(values)

            stats[component] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'sum': float(np.sum(values_array))
            }

        return stats

    def get_recent_rewards(self, n: int = 10) -> List[float]:
        """
        Get the n most recent rewards.

        Args:
            n: Number of recent rewards to return

        Returns:
            List of recent reward values
        """
        return list(self.rewards)[-n:]

    def get_recent_breakdowns(self, n: int = 10) -> List[Dict[str, float]]:
        """
        Get the n most recent reward breakdowns.

        Args:
            n: Number of recent breakdowns to return

        Returns:
            List of recent breakdown dictionaries
        """
        return list(self.breakdowns)[-n:]

    def clear(self):
        """Clear all tracked data."""
        self.rewards.clear()
        self.breakdowns.clear()

    def get_component_contributions(self) -> Dict[str, float]:
        """
        Calculate what percentage each component contributes to total reward.

        This helps identify if one component is dominating too much or
        if the reward function is well-balanced.

        Returns:
            Dictionary mapping component to percentage of total reward magnitude
        """
        if not self.breakdowns:
            return {}

        # Sum absolute values of each component across all breakdowns
        component_sums = {}
        total_magnitude = 0.0

        for breakdown in self.breakdowns:
            for component, value in breakdown.items():
                if component == 'total':
                    continue

                abs_value = abs(value)
                component_sums[component] = component_sums.get(component, 0.0) + abs_value
                total_magnitude += abs_value

        # Convert to percentages
        if total_magnitude == 0:
            return {comp: 0.0 for comp in component_sums}

        return {
            comp: (sum_val / total_magnitude) * 100.0
            for comp, sum_val in component_sums.items()
        }


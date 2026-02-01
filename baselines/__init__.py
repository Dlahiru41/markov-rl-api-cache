"""
Baseline caching policies for comparison against RL agents.

This module provides standard caching baselines (LRU, LFU, random, etc.) and a
comprehensive comparison framework for evaluating different caching strategies.
"""

from .base_policy import CachingPolicy, PolicyWrapper, PolicyStatistics, StatefulPolicy
from .lru_policy import LRUPolicy, AdaptiveLRUPolicy
from .lfu_policy import LFUPolicy, WindowedLFUPolicy
from .static_markov_policy import (
    StaticMarkovPolicy,
    InverseStaticMarkovPolicy,
    BalancedStaticMarkovPolicy
)
from .random_policy import RandomPolicy, EpsilonRandomPolicy, BiasedRandomPolicy
from .oracle_policy import OraclePolicy, PartialOraclePolicy, NoisyOraclePolicy
from .adaptive_policy import AdaptivePolicy, MultiObjectiveAdaptivePolicy
from .comparison import BaselineComparator, ComparisonConfig, PolicyResults
from .agent_adapter import RLAgentAdapter, TorchAgentAdapter, EnsembleAgentAdapter

__all__ = [
    # Base classes
    'CachingPolicy',
    'PolicyWrapper',
    'PolicyStatistics',
    'StatefulPolicy',

    # LRU variants
    'LRUPolicy',
    'AdaptiveLRUPolicy',

    # LFU variants
    'LFUPolicy',
    'WindowedLFUPolicy',

    # Static Markov variants
    'StaticMarkovPolicy',
    'InverseStaticMarkovPolicy',
    'BalancedStaticMarkovPolicy',

    # Random variants
    'RandomPolicy',
    'EpsilonRandomPolicy',
    'BiasedRandomPolicy',

    # Oracle variants
    'OraclePolicy',
    'PartialOraclePolicy',
    'NoisyOraclePolicy',

    # Adaptive variants
    'AdaptivePolicy',
    'MultiObjectiveAdaptivePolicy',

    # Comparison framework
    'BaselineComparator',
    'ComparisonConfig',
    'PolicyResults',

    # Agent adapters
    'RLAgentAdapter',
    'TorchAgentAdapter',
    'EnsembleAgentAdapter',
]



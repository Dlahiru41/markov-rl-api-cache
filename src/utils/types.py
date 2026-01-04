"""Type definitions for markov-rl-api-cache.

This module centralizes commonly used type aliases and TypedDict structures
used across the project to improve IDE autocompletion and static type checks.

Usage examples:
    from src.utils.types import APIEndpoint, Prediction, Experience

    def score_prediction(p: Prediction) -> float:
        endpoint, prob = p
        return prob

Note: these are lightweight aliases and TypedDict definitions intended
for readability and type-checking; they do not add runtime enforcement.
"""
from __future__ import annotations

from typing import Any, Dict, NewType, Optional, Sequence, Tuple, TypedDict, Union

# Primitive aliases
APIEndpoint = NewType("APIEndpoint", str)
SessionID = NewType("SessionID", str)
UserID = NewType("UserID", str)
Probability = NewType("Probability", float)

# State and action representations
# StateVector: sequence of floats representing environment state (features)
StateVector = Sequence[float]
# Action can be represented as an int (action index) or a small string label
Action = Union[int, str]
# Reward is a scalar float
Reward = float

# A prediction is a tuple of (APIEndpoint, probability)
Prediction = Tuple[APIEndpoint, Probability]

# Experience tuple used by RL replay buffers: (state, action, reward, next_state, done)
# - state, next_state: StateVector
# - action: Action
# - reward: Reward
# - done: bool
Experience = Tuple[StateVector, Action, Reward, StateVector, bool]

# TypedDicts for structured dictionaries
class APICallData(TypedDict, total=False):
    """Structured representation of a single API call/event.

    Fields
    ------
    endpoint: APIEndpoint
    method: str             # HTTP method (GET, POST, ...)
    params: Dict[str, Any]  # Query/body parameters (optional)
    user_id: Optional[UserID]
    session_id: Optional[SessionID]
    timestamp: float        # Epoch seconds
    latency_ms: float       # Observed latency in milliseconds
    status_code: int        # HTTP status code
    headers: Dict[str, str]
    payload_size: Optional[int]
    """

    endpoint: APIEndpoint
    method: str
    params: Dict[str, Any]
    user_id: Optional[UserID]
    session_id: Optional[SessionID]
    timestamp: float
    latency_ms: float
    status_code: int
    headers: Dict[str, str]
    payload_size: Optional[int]


class CacheMetrics(TypedDict, total=False):
    """Metrics emitted by the cache layer for monitoring and evaluation.

    Fields
    ------
    hits: int
    misses: int
    hit_rate: float
    memory_usage_bytes: int
    max_memory_bytes: Optional[int]
    eviction_count: Optional[int]
    """

    hits: int
    misses: int
    hit_rate: float
    memory_usage_bytes: int
    max_memory_bytes: Optional[int]
    eviction_count: Optional[int]


class RLTrainingMetrics(TypedDict, total=False):
    """Metrics recorded during RL training and evaluation.

    Fields
    ------
    episode: int
    step: int
    reward: float
    loss: Optional[float]
    epsilon: Optional[float]
    info: Dict[str, Any]
    """

    episode: int
    step: int
    reward: float
    loss: Optional[float]
    epsilon: Optional[float]
    info: Dict[str, Any]


__all__ = [
    "APIEndpoint",
    "SessionID",
    "UserID",
    "Probability",
    "StateVector",
    "Action",
    "Reward",
    "Prediction",
    "Experience",
    "APICallData",
    "CacheMetrics",
    "RLTrainingMetrics",
]

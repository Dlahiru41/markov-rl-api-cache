"""Custom exception hierarchy for markov-rl-api-cache.

Defines a base exception and domain-specific subclasses that attach an optional
context dictionary for easier debugging. Each exception overrides `__str__`
so the message and context are formatted nicely.

Usage examples:
    from src.utils.exceptions import ConfigError
    raise ConfigError("Missing required setting", context={"key": "rl.learning_rate"})
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class MarkovRLException(Exception):
    """Base class for all project-specific exceptions in markov-rl-api-cache.

    Parameters
    ----------
    message: Human-readable error message describing what went wrong.
    context: Optional mapping with additional debugging information (e.g.,
             offending value, filenames, indices).
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.context:
            return f"{self.__class__.__name__}: {self.message} | context={self.context}"
        return f"{self.__class__.__name__}: {self.message}"


# Configuration errors
class ConfigError(MarkovRLException):
    """Raised when configuration is missing or invalid.

    Use this when a required configuration key is missing, has the wrong type,
    or the configuration file cannot be parsed.
    """


# Data and dataset errors
class DataError(MarkovRLException):
    """Raised for issues related to datasets or data formatting.

    Use this when input data is malformed, columns are missing, or expected
    schema validation fails.
    """


# Markov chain specific errors
class MarkovChainError(MarkovRLException):
    """Base class for errors specific to Markov chain operations."""


class UnknownStateError(MarkovChainError):
    """Raised when the Markov model encounters a state that is not present in
    its transition table. Provide the unknown state in the context.
    """


class InsufficientDataError(MarkovChainError):
    """Raised when there is insufficient transition data to build or query
    a Markov chain model (e.g., not enough counts for reliable estimation).
    """


# RL agent errors
class RLAgentError(MarkovRLException):
    """Base class for reinforcement-learning agent issues."""


class TrainingFailureError(RLAgentError):
    """Raised when training fails to progress or a critical error occurs during
    training (e.g., NaNs in gradients, numerical instabilities).
    """


class InvalidActionError(RLAgentError):
    """Raised when an action selected or requested is invalid for the current
    environment (out of range, wrong type, etc.). Provide action and state in
    the context.
    """


# Cache errors
class CacheError(MarkovRLException):
    """Base class for cache-related errors (e.g., Redis access problems)."""


class CacheConnectionError(CacheError):
    """Raised when the cache backend cannot be reached or connection fails."""


class CacheCapacityError(CacheError):
    """Raised when cache capacity limits are reached or an eviction policy
    prevents storing critical data.
    """


# Simulator errors
class SimulatorError(MarkovRLException):
    """Base class for simulator/runtime errors."""


class ServiceUnavailableError(SimulatorError):
    """Raised when a simulated service is unavailable or repeatedly failing."""


class CascadeDetectedError(SimulatorError):
    """Raised when the simulator detects cascading failures across services
    (indicating a systemic issue rather than an isolated outage).
    """


__all__ = [
    "MarkovRLException",
    "ConfigError",
    "DataError",
    "MarkovChainError",
    "UnknownStateError",
    "InsufficientDataError",
    "RLAgentError",
    "TrainingFailureError",
    "InvalidActionError",
    "CacheError",
    "CacheConnectionError",
    "CacheCapacityError",
    "SimulatorError",
    "ServiceUnavailableError",
    "CascadeDetectedError",
]


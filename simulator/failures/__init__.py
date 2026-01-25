"""
Failure Injection Package

Tools for testing system resilience through controlled failure injection.
"""

from .injector import (
    FailureScenario,
    FailureInjector,
    CascadeSimulator,
)

__all__ = [
    "FailureScenario",
    "FailureInjector",
    "CascadeSimulator",
]


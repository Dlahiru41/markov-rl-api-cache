"""
Evaluation module for experiment management and result analysis.

Provides tools for running systematic experiments, comparing configurations,
and analyzing results for thesis evaluation.
"""

from .experiment_runner import (
    ExperimentRunner,
    ExperimentConfig,
    ExperimentResult
)

__all__ = [
    'ExperimentRunner',
    'ExperimentConfig',
    'ExperimentResult',
]


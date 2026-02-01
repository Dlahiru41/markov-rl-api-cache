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

from .analyzer import (
    ResultsAnalyzer,
    ResultsVisualizer
)

from .report_generator import (
    ReportGenerator
)

__all__ = [
    'ExperimentRunner',
    'ExperimentConfig',
    'ExperimentResult',
    'ResultsAnalyzer',
    'ResultsVisualizer',
    'ReportGenerator',
]


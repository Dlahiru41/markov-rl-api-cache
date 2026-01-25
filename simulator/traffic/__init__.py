"""
Traffic Generation Package

Simulates realistic user traffic patterns for e-commerce services.
"""

from .generator import (
    TrafficProfile,
    TrafficGenerator,
    SimulatedUser,
    BrowseWorkflow,
    PurchaseWorkflow,
    AccountWorkflow,
    QuickBuyWorkflow,
)

__all__ = [
    "TrafficProfile",
    "TrafficGenerator",
    "SimulatedUser",
    "BrowseWorkflow",
    "PurchaseWorkflow",
    "AccountWorkflow",
    "QuickBuyWorkflow",
]


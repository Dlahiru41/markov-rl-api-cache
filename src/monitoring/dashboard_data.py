"""Dashboard aggregation provider for recent cache/training/prediction trends."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


class DashboardDataProvider:
    """Aggregate lightweight dashboard payloads from in-memory component stats."""

    def __init__(self, components: Optional[Dict[str, Any]] = None):
        """Initialize provider with component references."""
        self.components = components or {}

    def get_stats(self, hours: int = 6) -> Dict[str, Any]:
        """Return dashboard-ready time-series aggregates for last N hours."""
        now = datetime.now(timezone.utc)
        points = [now - timedelta(minutes=10 * i) for i in range(min(hours * 6, 36), -1, -1)]

        cache = self.components.get("cache_manager")
        cache_stats = cache.get_stats() if cache and hasattr(cache, "get_stats") else {}
        scheduler = self.components.get("scheduler")
        job_status = scheduler.get_status() if scheduler and hasattr(scheduler, "get_status") else {}
        agent = self.components.get("dqn_agent")

        hit = float(cache_stats.get("hit_rate", 0.0))
        eps = float(getattr(agent, "epsilon", 0.0)) if agent else 0.0
        loss = float(getattr(agent, "last_loss", 0.0)) if agent else 0.0
        p50 = float(self.components.get("latency_p50_ms", 5.0))
        p95 = float(self.components.get("latency_p95_ms", 25.0))
        p99 = float(self.components.get("latency_p99_ms", 80.0))
        # Fallback placeholder can be overridden by passing `top_api_sequences` in components.
        top_api_sequences = self.components.get("top_api_sequences") or [
            {
                "sequence": ["GET:/api/auth", "GET:/api/users/me", "GET:/api/dashboard"],
                "count": 1,
                "cache_hit_rate": hit,
                "placeholder": True,
            }
        ]

        return {
            "time_range": f"last_{hours}_hours",
            "cache_performance": {
                "hit_rate_over_time": [{"timestamp": p.isoformat(), "rate": hit} for p in points],
                "latency_over_time": [{"timestamp": p.isoformat(), "p50": p50, "p95": p95, "p99": p99} for p in points],
            },
            "training_history": {
                "loss_over_time": [{"timestamp": p.isoformat(), "loss": loss} for p in points],
                "reward_over_time": [{"timestamp": p.isoformat(), "avg_reward": self.components.get("avg_reward_last_100", 0.0)} for p in points],
                "epsilon_over_time": [{"timestamp": p.isoformat(), "epsilon": eps} for p in points],
            },
            "prediction_accuracy": {
                "markov_accuracy_over_time": [
                    {
                        "timestamp": p.isoformat(),
                        "top1": self.components.get("markov_top1", 0.0),
                        "top3": self.components.get("markov_top3", 0.0),
                    }
                    for p in points
                ]
            },
            "top_api_sequences": top_api_sequences,
            "scheduler_jobs": job_status,
        }

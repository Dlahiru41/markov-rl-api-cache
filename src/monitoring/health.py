"""Health and Prometheus-style monitoring endpoints."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from fastapi.responses import PlainTextResponse

from src.monitoring.logger import get_severity_counters


class HealthMonitor:
    """Build liveness, detailed health, and metrics endpoint responses."""

    def __init__(self, components: Optional[Dict[str, Any]] = None, version: str = "1.0.0"):
        """Initialize monitor with component references."""
        self.started = time.time()
        self.components = components or {}
        self.version = version

    def health_check(self) -> Dict[str, Any]:
        """Return quick liveness payload."""
        return {
            "status": "healthy",
            "uptime_seconds": int(time.time() - self.started),
            "version": self.version,
        }

    def detailed_health(self) -> Dict[str, Any]:
        """Return detailed health payload for all managed components."""
        collector = self.components.get("collector")
        scheduler = self.components.get("scheduler")
        cache_manager = self.components.get("cache_manager")
        markov_chain = self.components.get("markov_chain")
        dqn_agent = self.components.get("dqn_agent")

        cache_stats = cache_manager.get_stats() if cache_manager is not None and hasattr(cache_manager, "get_stats") else {}
        collector_status = collector.get_status() if collector is not None and hasattr(collector, "get_status") else {}
        scheduler_status = scheduler.get_status() if scheduler is not None and hasattr(scheduler, "get_status") else {}
        markov_metrics = markov_chain.get_metrics() if markov_chain is not None and hasattr(markov_chain, "get_metrics") else {}
        agent_metrics = dqn_agent.get_metrics() if dqn_agent is not None and hasattr(dqn_agent, "get_metrics") else {}

        return {
            "status": "healthy",
            "uptime_seconds": int(time.time() - self.started),
            "components": {
                "gateway": self.components.get("gateway", {"status": "healthy"}),
                "cache": {
                    "status": "healthy",
                    "redis_connected": bool(cache_stats),
                    "redis_memory_mb": round(float(cache_stats.get("current_size_bytes", 0)) / (1024 * 1024), 2),
                    "hit_rate": cache_stats.get("hit_rate", 0.0),
                    "total_keys": cache_stats.get("entries", 0),
                    "prefetch_queue_size": cache_stats.get("prefetch_queue_size", 0),
                },
                "markov": {
                    "status": "healthy",
                    "total_transitions": markov_metrics.get("prediction_count", 0),
                    "unique_api_endpoints": getattr(markov_chain, "vocab_size", 0) if markov_chain is not None else 0,
                    "prediction_accuracy_last_1000": markov_metrics.get("top_1_accuracy", 0.0),
                    "last_update": self.components.get("last_markov_update"),
                },
                "rl_agent": {
                    "status": "healthy",
                    "epsilon": agent_metrics.get("epsilon", 0.0),
                    "replay_buffer_size": agent_metrics.get("buffer_size", 0),
                    "avg_reward_last_100": self.components.get("avg_reward_last_100", 0.0),
                    "last_training": self.components.get("last_training"),
                    "model_checkpoint": self.components.get("model_checkpoint"),
                },
                "scheduler": {
                    "status": "running" if scheduler is not None else "stopped",
                    "jobs": scheduler_status,
                },
                "collector": {
                    "status": "healthy",
                    **collector_status,
                    "active_sessions": self.components.get("session_tracker").active_session_count() if self.components.get("session_tracker") else 0,
                },
            },
            "system": self.components.get("resource_guard").get_system_snapshot() if self.components.get("resource_guard") else {},
        }

    def prometheus_metrics(self) -> PlainTextResponse:
        """Render Prometheus-like text metrics."""
        detailed = self.detailed_health()
        counters = get_severity_counters()
        cache = detailed["components"]["cache"]
        rl = detailed["components"]["rl_agent"]
        collector = detailed["components"]["collector"]
        system = detailed.get("system", {})
        scheduler_jobs = detailed["components"]["scheduler"]["jobs"]

        lines = [
            "# Cache metrics",
            f"markov_cache_hits_total {int(cache.get('hit_rate', 0.0) * max(1, cache.get('total_keys', 0)))}",
            f"markov_cache_misses_total {max(0, cache.get('total_keys', 0) - int(cache.get('hit_rate', 0.0) * max(1, cache.get('total_keys', 0))))}",
            "# RL metrics",
            f"rl_agent_epsilon {rl.get('epsilon', 0.0)}",
            f"rl_agent_replay_buffer_size {rl.get('replay_buffer_size', 0)}",
            "# Collector metrics",
            f"collector_buffer_usage_ratio {collector.get('buffer_usage_ratio', 0.0)}",
            f"collector_flush_total {collector.get('total_flushed', 0)}",
            f"collector_flush_errors_total {collector.get('flush_errors', 0)}",
            f"collector_active_sessions {collector.get('active_sessions', 0)}",
            "# Scheduler metrics",
        ]

        for job_name, st in scheduler_jobs.items():
            lines.append(f"scheduler_job_runs_total{{job=\"{job_name}\"}} {st.get('run_count', 0)}")
            lines.append(f"scheduler_job_errors_total{{job=\"{job_name}\"}} {st.get('error_count', 0)}")
            lines.append(f"scheduler_job_duration_seconds{{job=\"{job_name}\"}} {st.get('last_duration', 0.0)}")

        lines.extend(
            [
                "# System metrics",
                f"system_cpu_usage_percent {system.get('cpu_percent', 0)}",
                f"system_memory_usage_percent {system.get('memory_percent', 0)}",
                f"logs_warnings_total {counters['warnings']}",
                f"logs_errors_total {counters['errors']}",
                f"logs_critical_total {counters['critical']}",
            ]
        )

        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

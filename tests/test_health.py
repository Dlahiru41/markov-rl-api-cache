"""Tests for health monitor endpoints payloads."""

from src.monitoring.health import HealthMonitor


class DummyCollector:
    def get_status(self):
        return {
            "buffer_usage_ratio": 0.1,
            "buffer_usage_percent": 10,
            "total_collected": 1,
            "total_flushed": 1,
            "flush_errors": 0,
        }


class DummyScheduler:
    def get_status(self):
        return {"markov_update": {"run_count": 0, "error_count": 0, "last_duration": 0.0}}


class DummyGuard:
    def get_system_snapshot(self):
        return {"cpu_percent": 1, "memory_percent": 2, "disk_percent": 3}


class DummySessionTracker:
    def active_session_count(self):
        return 0


class DummyMarkov:
    def get_metrics(self):
        return {"prediction_count": 0, "top_1_accuracy": 0.0}


class DummyAgent:
    def get_metrics(self):
        return {"epsilon": 0.1, "buffer_size": 0}


def test_health_payloads():
    hm = HealthMonitor(
        components={
            "collector": DummyCollector(),
            "scheduler": DummyScheduler(),
            "resource_guard": DummyGuard(),
            "session_tracker": DummySessionTracker(),
            "markov_chain": DummyMarkov(),
            "dqn_agent": DummyAgent(),
            "gateway": {"status": "healthy"},
        }
    )
    basic = hm.health_check()
    detailed = hm.detailed_health()
    metrics = hm.prometheus_metrics()

    assert basic["status"] == "healthy"
    assert "components" in detailed
    assert metrics.status_code == 200
    assert "collector_flush_total" in metrics.body.decode("utf-8")

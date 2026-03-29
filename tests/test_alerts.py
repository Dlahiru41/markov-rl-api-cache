"""Tests for alert rule evaluation."""

from pathlib import Path

from src.monitoring.alerts import AlertManager


def test_alert_manager_evaluates_rules(tmp_path):
    rules = tmp_path / "alerts.yaml"
    rules.write_text(
        """
alerts:
  - name: high_cache_miss_rate
    condition: cache_hit_rate < 0.5
    severity: warning
    message: "Cache hit rate dropped below 50%: {value}"
""".strip()
    )

    manager = AlertManager(rules_path=str(rules))
    fired = manager.evaluate({"cache_hit_rate": 0.2})

    assert len(fired) == 1
    assert fired[0]["name"] == "high_cache_miss_rate"
    assert "0.2" in fired[0]["message"]

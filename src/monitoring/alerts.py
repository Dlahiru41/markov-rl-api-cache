"""Rule-based alert evaluation and active-alert storage."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.monitoring.logger import get_logger

logger = get_logger("monitor")


class AlertManager:
    """Evaluate configured alert rules and expose active alerts."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, redis_client: Any = None, rules_path: str = "config/alerts.yaml"):
        """Initialize manager and load alert rules."""
        self.config = config or {}
        self.redis = redis_client
        self.rules_path = Path(rules_path)
        self.rules = self._load_rules()
        self.active: List[Dict[str, Any]] = []
        self._fired_total: Dict[str, int] = {}

    def evaluate(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate all rules against current metric values."""
        fired: List[Dict[str, Any]] = []
        for rule in self.rules:
            value = self._extract_value(metrics, rule)
            if self._matches(rule.get("condition", ""), value):
                alert = {
                    "name": rule["name"],
                    "severity": rule.get("severity", "warning"),
                    "message": rule.get("message", "alert").format(value=value),
                    "value": value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                fired.append(alert)
                self._record(alert)
        return fired

    def get_active(self) -> Dict[str, Any]:
        """Return active alerts list."""
        if self.redis is not None:
            try:
                vals = self.redis.lrange("alerts:active", 0, -1)
                alerts = [json.loads(v.decode("utf-8") if isinstance(v, bytes) else v) for v in vals]
                return {"alerts": alerts}
            except Exception:
                pass
        return {"alerts": list(self.active)}

    def _load_rules(self) -> List[Dict[str, Any]]:
        """Load rule definitions from YAML file."""
        if not self.rules_path.exists():
            return []
        data = yaml.safe_load(self.rules_path.read_text(encoding="utf-8")) or {}
        return data.get("alerts", [])

    def _extract_value(self, metrics: Dict[str, Any], rule: Dict[str, Any]) -> Any:
        """Extract metric value referenced by simple rule condition."""
        cond = str(rule.get("condition", ""))
        field = cond.split()[0] if cond else ""
        return metrics.get(field)

    def _matches(self, condition: str, value: Any) -> bool:
        """Evaluate simple rule expressions like a < b, x == false."""
        tokens = condition.split()
        if len(tokens) < 3:
            return False
        op = tokens[1]
        rhs_text = tokens[2].lower()
        rhs: Any
        if rhs_text in ("true", "false"):
            rhs = rhs_text == "true"
        else:
            try:
                rhs = float(rhs_text)
            except ValueError:
                rhs = rhs_text

        if value is None:
            return False

        if op == "<":
            return float(value) < float(rhs)
        if op == ">":
            return float(value) > float(rhs)
        if op == "==":
            return value == rhs
        if op == "<=":
            return float(value) <= float(rhs)
        if op == ">=":
            return float(value) >= float(rhs)
        return False

    def _record(self, alert: Dict[str, Any]) -> None:
        """Store alert in memory/redis and increment counters and logs."""
        self.active.append(alert)
        key = f"{alert['name']}::{alert['severity']}"
        self._fired_total[key] = self._fired_total.get(key, 0) + 1

        sev = alert["severity"].lower()
        if sev == "critical":
            logger.critical("alert_fired", data=alert)
        elif sev == "error":
            logger.error("alert_fired", data=alert)
        elif sev == "warning":
            logger.warning("alert_fired", data=alert)
        else:
            logger.info("alert_fired", data=alert)

        if self.redis is not None:
            try:
                self.redis.rpush("alerts:active", json.dumps(alert))
                self.redis.expire("alerts:active", 24 * 3600)
            except Exception:
                logger.warning("alert_redis_store_failed", data={"name": alert["name"]})


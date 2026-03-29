"""System resource checks used as training pre-condition guards."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import psutil


class ResourceGuard:
    """Evaluate CPU/memory/disk/redis thresholds before training."""

    def __init__(
        self,
        redis_client: Any = None,
        max_cpu: float = 80.0,
        max_memory: float = 85.0,
        max_disk: float = 95.0,
    ):
        """Initialize guard thresholds and redis client."""
        self.redis_client = redis_client
        self.max_cpu = max_cpu
        self.max_memory = max_memory
        self.max_disk = max_disk

    def can_train(self) -> Tuple[bool, str]:
        """Check resource thresholds and redis connectivity."""
        snap = self.get_system_snapshot()

        if snap["cpu_percent"] > self.max_cpu:
            return False, f"CPU at {snap['cpu_percent']:.1f}%, threshold is {self.max_cpu:.1f}%"
        if snap["memory_percent"] > self.max_memory:
            return False, f"Memory at {snap['memory_percent']:.1f}%, threshold is {self.max_memory:.1f}%"
        if snap["disk_percent"] > self.max_disk:
            return False, f"Disk at {snap['disk_percent']:.1f}%, threshold is {self.max_disk:.1f}%"
        if not snap["redis_connected"]:
            return False, "Redis disconnected"
        return True, "ok"

    def get_system_snapshot(self) -> Dict[str, Any]:
        """Return point-in-time system and redis metrics."""
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        disk = psutil.disk_usage("/").percent

        redis_connected = False
        redis_memory_used = 0
        active_connections = 0

        if self.redis_client is not None:
            try:
                redis_connected = bool(self.redis_client.ping())
                info = self.redis_client.info() if hasattr(self.redis_client, "info") else {}
                redis_memory_used = int(info.get("used_memory", 0))
                active_connections = int(info.get("connected_clients", 0))
            except Exception:
                redis_connected = False

        return {
            "cpu_percent": cpu,
            "memory_percent": mem,
            "disk_percent": disk,
            "redis_connected": redis_connected,
            "redis_memory_used": redis_memory_used,
            "active_connections": active_connections,
        }


"""Tests for ResourceGuard threshold behavior."""

from src.scheduler.resource_guard import ResourceGuard


class RedisOK:
    def ping(self):
        return True

    def info(self):
        return {"used_memory": 123, "connected_clients": 2}


def test_resource_guard_thresholds(monkeypatch):
    guard = ResourceGuard(redis_client=RedisOK(), max_cpu=80, max_memory=85)

    monkeypatch.setattr("src.scheduler.resource_guard.psutil.cpu_percent", lambda interval=None: 92.0)
    monkeypatch.setattr("src.scheduler.resource_guard.psutil.virtual_memory", lambda: type("M", (), {"percent": 50.0})())
    monkeypatch.setattr("src.scheduler.resource_guard.psutil.disk_usage", lambda p: type("D", (), {"percent": 20.0})())

    ok, reason = guard.can_train()
    assert not ok
    assert "CPU" in reason


def test_resource_guard_snapshot(monkeypatch):
    guard = ResourceGuard(redis_client=RedisOK())

    monkeypatch.setattr("src.scheduler.resource_guard.psutil.cpu_percent", lambda interval=None: 10.0)
    monkeypatch.setattr("src.scheduler.resource_guard.psutil.virtual_memory", lambda: type("M", (), {"percent": 20.0})())
    monkeypatch.setattr("src.scheduler.resource_guard.psutil.disk_usage", lambda p: type("D", (), {"percent": 30.0})())

    snap = guard.get_system_snapshot()
    assert snap["redis_connected"] is True
    assert snap["active_connections"] == 2

"""Tests for APICallCollector ring-buffer behavior."""

from datetime import datetime, timezone

from src.pipeline.collector import APICallCollector, APICallRecord


def _record(i: int) -> APICallRecord:
    return APICallRecord(
        request_id=str(i),
        session_id="s1",
        timestamp=datetime.now(timezone.utc),
        method="GET",
        path=f"/api/{i}",
        query_params="",
        upstream_status=200,
        response_time_ms=1.0,
        cache_hit=False,
        cache_key=f"k{i}",
    )


def test_collector_buffer_overflow_tracks_drops(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = APICallCollector(config={"collector_buffer_size": 3, "collector_flush_interval_seconds": 3600}, redis_client=None)

    for i in range(6):
        c.record(_record(i))

    status = c.get_status()
    assert status["dropped"] >= 3
    assert status["total_collected"] == 6
    assert status["buffer_usage_percent"] == 100

"""
Unit tests for the API Gateway reverse proxy.

Validates:
- GET request → MISS → forwarded → cached → returns X-Cache: MISS
- Same GET again → HIT → not forwarded → returns X-Cache: HIT
- POST → forwarded → related GET cache keys invalidated
- Redis down → request still succeeds via upstream fallback
- Upstream down → returns 502
- Upstream timeout → returns 504
- Admin endpoints work
- Cache key building
"""

import json
import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from src.gateway.cache_keys import (
    build_cache_key,
    build_cache_key_raw,
    get_invalidation_pattern,
    get_ttl_for_path,
    is_cacheable,
)
from src.gateway.proxy import (
    _GatewayStats,
    _GATEWAY_CACHE_PREFIX,
    create_gateway_app,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RULES = [
    {"path": "/api/products*", "ttl": 600, "vary_by_user": False},
    {"path": "/api/users/*", "ttl": 120, "vary_by_user": True},
    {"path": "/api/orders/*", "ttl": 60, "vary_by_user": True},
    {"path": "/api/auth/*", "ttl": 0, "cache": False},
]


def _fake_upstream_response(status_code=200, body="upstream", headers=None):
    """Build an ``httpx.Response`` suitable for mocking."""
    headers = headers or {"content-type": "text/plain"}
    return httpx.Response(
        status_code=status_code,
        text=body,
        headers=headers,
        request=httpx.Request("GET", "http://upstream"),
    )


# ---------------------------------------------------------------------------
# Cache-key tests
# ---------------------------------------------------------------------------


class TestCacheKeys:
    """Test the cache-key builder module."""

    def test_basic_key(self):
        key = build_cache_key("GET", "/api/users/123")
        assert isinstance(key, str)
        assert len(key) == 32  # md5 hex

    def test_sorted_query_params(self):
        k1 = build_cache_key("GET", "/api/x", {"b": "2", "a": "1"})
        k2 = build_cache_key("GET", "/api/x", {"a": "1", "b": "2"})
        assert k1 == k2

    def test_different_paths_different_keys(self):
        k1 = build_cache_key("GET", "/api/a")
        k2 = build_cache_key("GET", "/api/b")
        assert k1 != k2

    def test_vary_by_user(self):
        headers = {"x-user-id": "42"}
        k1 = build_cache_key("GET", "/api/users/1", headers=headers, cache_rules=SAMPLE_RULES)
        k2 = build_cache_key("GET", "/api/users/1", cache_rules=SAMPLE_RULES)
        assert k1 != k2

    def test_ttl_for_path(self):
        assert get_ttl_for_path("/api/products/1", SAMPLE_RULES) == 600
        assert get_ttl_for_path("/api/users/1", SAMPLE_RULES) == 120
        assert get_ttl_for_path("/api/auth/login", SAMPLE_RULES) == 0
        assert get_ttl_for_path("/unknown", SAMPLE_RULES, default_ttl=300) == 300

    def test_is_cacheable(self):
        assert is_cacheable("/api/products/1", SAMPLE_RULES)
        assert not is_cacheable("/api/auth/login", SAMPLE_RULES)

    def test_invalidation_pattern(self):
        p = get_invalidation_pattern("/api/users")
        assert p == "GET:/api/users*"


# ---------------------------------------------------------------------------
# Proxy integration tests (FastAPI TestClient, mocked upstream + Redis)
# ---------------------------------------------------------------------------


class TestGatewayProxy:
    """Tests for the gateway reverse proxy, using mocked Redis and upstream."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Patch Redis so the gateway operates with an in-memory store."""
        self.store: Dict[str, bytes] = {}

        mock_redis = MagicMock()
        mock_redis.ping.return_value = True

        def fake_get(key):
            return self.store.get(key)

        def fake_setex(key, ttl, value):
            self.store[key] = value
            return True

        def fake_set(key, value):
            self.store[key] = value
            return True

        def fake_delete(*keys):
            count = 0
            for k in keys:
                key_str = k.decode() if isinstance(k, bytes) else k
                if key_str in self.store:
                    del self.store[key_str]
                    count += 1
            return count

        def fake_scan(cursor=0, match="*", count=100):
            import fnmatch
            matched = [k.encode() if isinstance(k, str) else k for k in self.store if fnmatch.fnmatch(k, match)]
            return (0, matched)

        mock_redis.get = fake_get
        mock_redis.setex = fake_setex
        mock_redis.set = fake_set
        mock_redis.delete = fake_delete
        mock_redis.scan = fake_scan

        with patch("src.gateway.proxy._get_redis_client", return_value=mock_redis):
            self.app = create_gateway_app(
                upstream_url="http://fake-upstream:9000",
                cache_default_ttl=300,
                cache_enabled=True,
                upstream_timeout_ms=5000,
            )
        self.app.state.redis = mock_redis
        self.client = TestClient(self.app)
        self.mock_redis = mock_redis

    # ---- GET caching --------------------------------------------------

    def test_get_miss_then_hit(self):
        """First GET → MISS (forwarded), second GET → HIT (from cache)."""
        fake_resp = _fake_upstream_response(200, '{"id":1}')

        with patch("httpx.AsyncClient.request", return_value=fake_resp):
            r1 = self.client.get("/api/products/1")

        assert r1.status_code == 200
        assert r1.headers.get("x-cache") == "MISS"
        assert r1.text == '{"id":1}'

        # Second request should come from cache – no upstream call
        r2 = self.client.get("/api/products/1")
        assert r2.status_code == 200
        assert r2.headers.get("x-cache") == "HIT"
        assert r2.text == '{"id":1}'

    def test_post_forwarded_and_cache_invalidated(self):
        """POST forwards to upstream and invalidates related GET keys."""
        get_resp = _fake_upstream_response(200, '{"id":1}')
        post_resp = _fake_upstream_response(201, '{"created":true}')

        # Populate cache with a GET
        with patch("httpx.AsyncClient.request", return_value=get_resp):
            self.client.get("/api/products/1")

        # Keys should exist in the store
        assert len(self.store) > 0

        # POST should invalidate the cached GET
        with patch("httpx.AsyncClient.request", return_value=post_resp):
            r = self.client.post("/api/products", content=b'{"name":"new"}')

        assert r.status_code == 201

        # Verify cache was invalidated — the next GET should be a MISS
        with patch("httpx.AsyncClient.request", return_value=get_resp):
            r = self.client.get("/api/products/1")
        assert r.headers.get("x-cache") == "MISS"

    def test_non_cacheable_path(self):
        """Paths marked cache=false should never be cached."""
        fake_resp = _fake_upstream_response(200, "token")

        with patch("httpx.AsyncClient.request", return_value=fake_resp):
            r1 = self.client.get("/api/auth/login")

        assert r1.status_code == 200
        # Not cached, so second call should also forward to upstream
        with patch("httpx.AsyncClient.request", return_value=fake_resp):
            r2 = self.client.get("/api/auth/login")
        assert r2.status_code == 200

    # ---- Resilience ---------------------------------------------------

    def test_redis_down_fallback(self):
        """When Redis is unavailable, requests still go through upstream."""
        self.app.state.redis = None

        fake_resp = _fake_upstream_response(200, "ok")
        with patch("src.gateway.proxy._get_redis_client", return_value=None):
            with patch("httpx.AsyncClient.request", return_value=fake_resp):
                r = self.client.get("/api/products/1")

        assert r.status_code == 200
        assert r.text == "ok"

    def test_upstream_down_returns_502(self):
        """When upstream is unreachable the gateway returns 502."""
        with patch(
            "httpx.AsyncClient.request",
            side_effect=httpx.ConnectError("connection refused"),
        ):
            r = self.client.get("/api/products/1")

        assert r.status_code == 502

    def test_upstream_timeout_returns_504(self):
        """When upstream times out the gateway returns 504."""
        with patch(
            "httpx.AsyncClient.request",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            r = self.client.get("/api/products/1")

        assert r.status_code == 504

    # ---- Admin endpoints ----------------------------------------------

    def test_admin_health(self):
        r = self.client.get("/admin/health")
        assert r.status_code == 200
        body = r.json()
        assert "redis" in body
        assert "upstream" in body

    def test_admin_stats(self):
        r = self.client.get("/admin/stats")
        assert r.status_code == 200
        body = r.json()
        assert "cache_hits" in body
        assert "cache_misses" in body

    def test_admin_config(self):
        r = self.client.get("/admin/config")
        assert r.status_code == 200
        body = r.json()
        assert body["upstream_url"] == "http://fake-upstream:9000"
        assert body["cache_default_ttl"] == 300

    def test_admin_cache_flush(self):
        # Put something in the store first
        self.store["gw_cache:test"] = b"data"
        r = self.client.post("/admin/cache/flush")
        assert r.status_code == 200

    def test_admin_cache_invalidate(self):
        r = self.client.post(
            "/admin/cache/invalidate",
            json={"pattern": "/api/users/*"},
        )
        assert r.status_code == 200
        body = r.json()
        assert "keys_deleted" in body


# ---------------------------------------------------------------------------
# Stats tests
# ---------------------------------------------------------------------------


class TestGatewayStats:
    def test_as_dict(self):
        s = _GatewayStats()
        s.hits = 8
        s.misses = 2
        d = s.as_dict()
        assert d["hit_rate"] == 0.8
        assert d["miss_rate"] == 0.2

    def test_zero_division(self):
        s = _GatewayStats()
        d = s.as_dict()
        assert d["hit_rate"] == 0.0

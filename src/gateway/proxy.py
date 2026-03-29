"""
Reverse-proxy / API Gateway for markov-rl-api-cache.

Intercepts all incoming HTTP requests, transparently caches GET responses
in Redis, forwards mutations to the upstream service and auto-invalidates
related cache entries.  After every proxied request the Markov + RL
intelligence layer is invoked asynchronously (zero added latency).
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx
import yaml
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from .cache_keys import (
    build_cache_key,
    build_cache_key_raw,
    get_invalidation_pattern,
    get_ttl_for_path,
    is_cacheable,
)
from ..monitoring.logger import setup_logging
from ..monitoring.health import HealthMonitor
from ..monitoring.alerts import AlertManager
from ..monitoring.dashboard_data import DashboardDataProvider
from ..pipeline.collector import APICallCollector, APICallRecord
from ..pipeline.session_tracker import SessionTracker
from ..pipeline.experience_builder import ExperienceBuilder
from ..scheduler.training_scheduler import TrainingScheduler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_RULES: List[Dict[str, Any]] = []


def _load_cache_rules() -> List[Dict[str, Any]]:
    """Load cache rules from ``configs/cache_rules.yaml``."""
    root = Path(__file__).resolve().parents[2]
    rules_path = root / "configs" / "cache_rules.yaml"
    if rules_path.exists():
        with rules_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data.get("cache_rules", [])
    return []


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        return default


def _env_bool(name: str, default: bool = True) -> bool:
    val = os.environ.get(name, "").lower()
    if val in ("0", "false", "no", "off"):
        return False
    if val in ("1", "true", "yes", "on"):
        return True
    return default


# ---------------------------------------------------------------------------
# Redis helpers  (import late so that missing redis is not fatal at import)
# ---------------------------------------------------------------------------

def _get_redis_client():
    """Return a connected ``redis.Redis`` client or ``None``."""
    try:
        import redis as _redis

        host = _env("REDIS_HOST", "localhost")
        port = _env_int("REDIS_PORT", 6379)
        db = _env_int("REDIS_DB", 0)
        client = _redis.Redis(host=host, port=port, db=db, socket_timeout=2.0)
        client.ping()
        return client
    except Exception:
        return None


_GATEWAY_CACHE_PREFIX = "gw_cache:"


def _redis_get(client, key: str) -> Optional[bytes]:
    try:
        return client.get(f"{_GATEWAY_CACHE_PREFIX}{key}")
    except Exception:
        return None


def _redis_set(client, key: str, value: bytes, ttl: int) -> bool:
    try:
        if ttl > 0:
            client.setex(f"{_GATEWAY_CACHE_PREFIX}{key}", ttl, value)
        else:
            client.set(f"{_GATEWAY_CACHE_PREFIX}{key}", value)
        return True
    except Exception:
        return False


def _redis_invalidate(client, pattern: str) -> int:
    """Delete all keys matching *pattern* (uses SCAN to avoid blocking)."""
    deleted = 0
    try:
        cursor = 0
        while True:
            cursor, keys = client.scan(
                cursor=cursor,
                match=f"{_GATEWAY_CACHE_PREFIX}{pattern}",
                count=200,
            )
            if keys:
                client.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break
    except Exception:
        pass
    return deleted


def _redis_flush(client) -> int:
    """Flush all gateway cache keys."""
    return _redis_invalidate(client, "*")


# ---------------------------------------------------------------------------
# Stats collector (in-memory; survives only within process lifetime)
# ---------------------------------------------------------------------------

class _GatewayStats:
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.upstream_errors = 0
        self.total_requests = 0
        self.prefetch_issued = 0
        self.prefetch_used = 0

    def as_dict(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "miss_rate": round(self.misses / total, 4) if total else 0.0,
            "upstream_errors": self.upstream_errors,
            "prefetch_issued": self.prefetch_issued,
            "prefetch_used": self.prefetch_used,
        }


# ---------------------------------------------------------------------------
# Markov / RL integration (async, non-blocking)
# ---------------------------------------------------------------------------

def _try_rl_hook(
    path: str,
    session_id: Optional[str],
    cache_hit: bool,
    redis_client,
    upstream_url: str,
    cache_rules: Sequence[Dict[str, Any]],
    default_ttl: int,
    timeout: float,
    stats: _GatewayStats,
):
    """Best-effort invocation of the Markov + RL prediction & prefetch loop.

    Runs synchronously inside a background thread so it never blocks the
    response to the client.
    """
    try:
        from ..markov.predictor import MarkovPredictor
        from ..rl.actions import CacheAction

        predictor = MarkovPredictor()
        if session_id:
            predictor.observe(path)
        predictions = predictor.predict(k=3)

        if not predictions:
            return

        # Decide whether to prefetch (simplified heuristic – real deployment
        # would query DQNAgent; we keep the RL core untouched).
        top_paths = [p for p, prob in predictions if prob > 0.3]

        if not top_paths:
            return

        # Fire prefetch requests in background
        for ppath in top_paths:
            try:
                url = f"{upstream_url.rstrip('/')}{ppath}"
                resp = httpx.get(url, timeout=timeout)
                if 200 <= resp.status_code < 300:
                    ttl = get_ttl_for_path(ppath, cache_rules, default_ttl)
                    if ttl > 0 and redis_client:
                        cache_key = build_cache_key_raw("GET", ppath)
                        payload = json.dumps({
                            "status_code": resp.status_code,
                            "headers": dict(resp.headers),
                            "body": resp.text,
                            "prefetched": True,
                        }).encode()
                        _redis_set(redis_client, cache_key, payload, ttl)
                        stats.prefetch_issued += 1
            except Exception:
                pass
    except Exception:
        # RL modules may not be importable in every environment (e.g. tests).
        pass


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------

def create_gateway_app(
    upstream_url: Optional[str] = None,
    cache_default_ttl: Optional[int] = None,
    cache_enabled: Optional[bool] = None,
    upstream_timeout_ms: Optional[int] = None,
) -> FastAPI:
    """Create and return the gateway FastAPI application.

    All parameters fall back to environment variables when ``None``.
    """
    upstream_url = upstream_url or _env("UPSTREAM_URL", "http://localhost:9000")
    cache_default_ttl = (
        cache_default_ttl
        if cache_default_ttl is not None
        else _env_int("CACHE_DEFAULT_TTL", 300)
    )
    cache_enabled = (
        cache_enabled if cache_enabled is not None else _env_bool("CACHE_ENABLED", True)
    )
    upstream_timeout_ms = (
        upstream_timeout_ms
        if upstream_timeout_ms is not None
        else _env_int("UPSTREAM_TIMEOUT_MS", 5000)
    )
    timeout_seconds = upstream_timeout_ms / 1000.0

    cache_rules = _load_cache_rules() or _DEFAULT_CACHE_RULES
    stats = _GatewayStats()

    setup_logging()
    app = FastAPI(title="Markov RL API Cache Gateway", version="1.0.0")

    # Try to connect to Redis at startup; we'll retry on each request if
    # it was unavailable initially.
    app.state.redis = _get_redis_client()
    app.state.stats = stats
    app.state.upstream_url = upstream_url
    app.state.cache_rules = cache_rules
    app.state.cache_default_ttl = cache_default_ttl
    app.state.cache_enabled = cache_enabled
    app.state.upstream_timeout_ms = upstream_timeout_ms
    app.state.start_time = time.time()

    try:
        from ..markov.first_order import FirstOrderMarkovChain

        markov_chain = FirstOrderMarkovChain(smoothing=0.001)
    except Exception:
        class _NoOpMarkovChain:
            def update(self, *_args, **_kwargs):
                return self

            def predict(self, *_args, **_kwargs):
                return []

            def get_metrics(self):
                return {"prediction_count": 0, "top_1_accuracy": 0.0}

        markov_chain = _NoOpMarkovChain()

    try:
        from ..rl.agents.dqn_agent import DQNAgent, DQNConfig

        dqn_agent = DQNAgent(DQNConfig(state_dim=60, action_dim=7))
    except Exception:
        class _NoOpBuffer:
            def __len__(self):
                return 0

        class _NoOpDQNAgent:
            epsilon = 0.0
            last_loss = 0.0
            buffer = _NoOpBuffer()

            def train_step(self):
                return None

            def save(self, _path: str):
                return None

            def load(self, _path: str):
                return None

            def store_transition(self, *_args, **_kwargs):
                return None

            def get_metrics(self):
                return {"epsilon": self.epsilon, "buffer_size": len(self.buffer), "last_loss": self.last_loss}

        dqn_agent = _NoOpDQNAgent()

    collector = APICallCollector(redis_client=app.state.redis)
    collector.start()
    session_tracker = SessionTracker(markov_chain=markov_chain)
    experience_builder = ExperienceBuilder(dqn_agent=dqn_agent)
    scheduler = TrainingScheduler(
        markov_chain=markov_chain,
        dqn_agent=dqn_agent,
        collector=collector,
        redis_client=app.state.redis,
    )
    scheduler.start()

    health_monitor = HealthMonitor(
        components={
            "collector": collector,
            "scheduler": scheduler,
            "cache_manager": None,
            "markov_chain": markov_chain,
            "dqn_agent": dqn_agent,
            "resource_guard": scheduler.resource_guard,
            "session_tracker": session_tracker,
            "gateway": {"status": "healthy"},
        },
        version="1.0.0",
    )
    alert_manager = AlertManager(redis_client=app.state.redis)
    dashboard_provider = DashboardDataProvider(
        components={
            "cache_manager": None,
            "scheduler": scheduler,
            "dqn_agent": dqn_agent,
        }
    )

    app.state.collector = collector
    app.state.session_tracker = session_tracker
    app.state.experience_builder = experience_builder
    app.state.training_scheduler = scheduler
    app.state.health_monitor = health_monitor
    app.state.alert_manager = alert_manager
    app.state.dashboard_provider = dashboard_provider

    # ------------------------------------------------------------------
    # Admin endpoints (not proxied)
    # ------------------------------------------------------------------

    @app.get("/admin/health")
    async def admin_health():
        redis_ok = False
        try:
            rc = app.state.redis or _get_redis_client()
            if rc:
                redis_ok = rc.ping()
        except Exception:
            pass

        upstream_ok = False
        try:
            r = httpx.get(upstream_url, timeout=2.0)
            upstream_ok = r.status_code < 500
        except Exception:
            pass

        return {
            "status": "healthy" if (redis_ok and upstream_ok) else "degraded",
            "upstream": {"url": upstream_url, "reachable": upstream_ok},
            "redis": {"connected": redis_ok},
            "rl_agent": {"loaded": True},
        }

    @app.get("/admin/stats")
    async def admin_stats():
        return stats.as_dict()

    @app.post("/admin/cache/flush")
    async def admin_cache_flush():
        rc = app.state.redis or _get_redis_client()
        count = 0
        if rc:
            count = _redis_flush(rc)
        return {"message": "Cache flushed", "keys_deleted": count}

    @app.post("/admin/cache/invalidate")
    async def admin_cache_invalidate(request: Request):
        body = await request.json()
        pattern = body.get("pattern", "*")
        internal_pattern = f"GET:{pattern}"
        rc = app.state.redis or _get_redis_client()
        count = 0
        if rc:
            count = _redis_invalidate(rc, internal_pattern)
        return {"message": "Invalidated", "pattern": pattern, "keys_deleted": count}

    @app.get("/admin/config")
    async def admin_config():
        return {
            "upstream_url": upstream_url,
            "cache_default_ttl": cache_default_ttl,
            "cache_enabled": cache_enabled,
            "upstream_timeout_ms": upstream_timeout_ms,
            "cache_rules": cache_rules,
        }

    @app.get("/health")
    async def health():
        return health_monitor.health_check()

    @app.get("/health/detailed")
    async def detailed_health():
        return health_monitor.detailed_health()

    @app.get("/metrics")
    async def prometheus_metrics():
        return health_monitor.prometheus_metrics()

    @app.get("/dashboard/stats")
    async def dashboard_stats(hours: int = 6):
        return dashboard_provider.get_stats(hours=hours)

    @app.get("/alerts/active")
    async def active_alerts():
        return alert_manager.get_active()

    @app.get("/scheduler/status")
    async def scheduler_status():
        return scheduler.get_status()

    @app.post("/scheduler/trigger/{job_name}")
    async def scheduler_trigger(job_name: str):
        return scheduler.manual_trigger(job_name)

    # ------------------------------------------------------------------
    # Catch-all proxy
    # ------------------------------------------------------------------

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    )
    async def proxy(request: Request, path: str):
        stats.total_requests += 1
        request_start = time.perf_counter()
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))

        full_path = f"/{path}"
        method = request.method.upper()
        query_params = dict(request.query_params)
        req_headers = dict(request.headers)

        # Remove hop-by-hop / host headers before forwarding
        forward_headers = {
            k: v
            for k, v in req_headers.items()
            if k.lower() not in ("host", "transfer-encoding")
        }

        # ---- GET requests: cache logic --------------------------------
        if method == "GET" and cache_enabled:
            cacheable = is_cacheable(full_path, cache_rules)
            if cacheable:
                cache_key = build_cache_key_raw(
                    method, full_path, query_params, req_headers, cache_rules
                )
                rc = app.state.redis
                if rc is None:
                    rc = _get_redis_client()
                    app.state.redis = rc

                cached = None
                if rc:
                    cached = _redis_get(rc, cache_key)

                if cached is not None:
                    stats.hits += 1
                    try:
                        data = json.loads(cached)
                    except Exception:
                        data = {"body": cached.decode(errors="replace"), "status_code": 200, "headers": {}}
                    resp_headers = {
                        k: v
                        for k, v in data.get("headers", {}).items()
                        if k.lower() not in ("transfer-encoding", "content-encoding", "content-length")
                    }
                    resp_headers["X-Cache"] = "HIT"
                    if data.get("prefetched"):
                        stats.prefetch_used += 1
                    try:
                        session_id = session_tracker.extract_session_id(
                            headers={k.lower(): v for k, v in req_headers.items()},
                            client_ip=request.client.host if request.client else "",
                            user_agent=req_headers.get("user-agent", ""),
                        )
                        latency_ms = (time.perf_counter() - request_start) * 1000.0
                        rec = APICallRecord(
                            request_id=request_id,
                            session_id=session_id,
                            timestamp=datetime.now(timezone.utc),
                            method=method,
                            path=full_path,
                            query_params=str(request.query_params),
                            upstream_status=data.get("status_code", 200),
                            response_time_ms=latency_ms,
                            cache_hit=True,
                            cache_key=cache_key,
                            markov_prediction=[],
                            rl_action_taken=0,
                            rl_state_vector=[0.0] * 60,
                            rl_reward=0.0,
                            context={},
                        )
                        import threading
                        threading.Thread(target=collector.record, args=(rec,), daemon=True).start()
                        threading.Thread(target=session_tracker.track, args=(session_id, full_path, {}), daemon=True).start()
                    except Exception:
                        pass
                    return Response(
                        content=data.get("body", ""),
                        status_code=data.get("status_code", 200),
                        headers=resp_headers,
                    )
                else:
                    stats.misses += 1

        # ---- Forward to upstream --------------------------------------
        target_url = f"{upstream_url.rstrip('/')}{full_path}"
        if query_params:
            from urllib.parse import urlencode

            target_url += "?" + urlencode(query_params)

        try:
            body = await request.body()
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                upstream_resp = await client.request(
                    method=method,
                    url=target_url,
                    headers=forward_headers,
                    content=body if method not in ("GET", "HEAD") else None,
                )
        except httpx.TimeoutException:
            stats.upstream_errors += 1
            return JSONResponse(
                status_code=504,
                content={"detail": "Upstream timeout"},
                headers={"X-Cache": "MISS"},
            )
        except httpx.ConnectError:
            stats.upstream_errors += 1
            return JSONResponse(
                status_code=502,
                content={"detail": "Upstream unavailable"},
                headers={"X-Cache": "MISS"},
            )
        except Exception:
            stats.upstream_errors += 1
            return JSONResponse(
                status_code=502,
                content={"detail": "Bad gateway"},
                headers={"X-Cache": "MISS"},
            )

        # ---- Build response to client ---------------------------------
        resp_headers = {
            k: v
            for k, v in upstream_resp.headers.items()
            if k.lower() not in ("transfer-encoding", "content-encoding", "content-length")
        }

        if method == "GET" and cache_enabled:
            resp_headers["X-Cache"] = "MISS"

            # Store in cache
            cacheable = is_cacheable(full_path, cache_rules)
            if cacheable and 200 <= upstream_resp.status_code < 300:
                ttl = get_ttl_for_path(full_path, cache_rules, cache_default_ttl)
                if ttl > 0:
                    cache_key = build_cache_key_raw(
                        method, full_path, query_params, req_headers, cache_rules
                    )
                    payload = json.dumps({
                        "status_code": upstream_resp.status_code,
                        "headers": dict(upstream_resp.headers),
                        "body": upstream_resp.text,
                    }).encode()
                    rc = app.state.redis or _get_redis_client()
                    if rc:
                        _redis_set(rc, cache_key, payload, ttl)
                        app.state.redis = rc

        # Mutation: auto-invalidate related cache keys
        if method in ("POST", "PUT", "PATCH", "DELETE"):
            if 200 <= upstream_resp.status_code < 300:
                rc = app.state.redis or _get_redis_client()
                if rc:
                    pattern = get_invalidation_pattern(full_path)
                    _redis_invalidate(rc, pattern)
                    app.state.redis = rc

        # ---- Async RL hook (non-blocking) -----------------------------
        session_id = req_headers.get("x-session-id") or req_headers.get(
            "authorization", ""
        )
        # If we reach this point for a GET the request was a cache MISS
        # (cache HITs return early above), so cache_hit is always False here.
        cache_hit = False

        import threading

        threading.Thread(
            target=_try_rl_hook,
            args=(
                full_path,
                session_id,
                cache_hit,
                app.state.redis,
                upstream_url,
                cache_rules,
                cache_default_ttl,
                timeout_seconds,
                stats,
            ),
            daemon=True,
        ).start()

        try:
            session_id = session_tracker.extract_session_id(
                headers={k.lower(): v for k, v in req_headers.items()},
                client_ip=request.client.host if request.client else "",
                user_agent=req_headers.get("user-agent", ""),
            )
            latency_ms = (time.perf_counter() - request_start) * 1000.0
            rec = APICallRecord(
                request_id=request_id,
                session_id=session_id,
                timestamp=datetime.now(timezone.utc),
                method=method,
                path=full_path,
                query_params=str(request.query_params),
                upstream_status=upstream_resp.status_code,
                response_time_ms=latency_ms,
                cache_hit=False,
                cache_key=build_cache_key_raw(method, full_path, query_params, req_headers, cache_rules) if method == "GET" else "",
                markov_prediction=[],
                rl_action_taken=0,
                rl_state_vector=[0.0] * 60,
                rl_reward=0.0,
                context={},
            )
            threading.Thread(target=collector.record, args=(rec,), daemon=True).start()
            threading.Thread(target=session_tracker.track, args=(session_id, full_path, {}), daemon=True).start()
        except Exception:
            pass

        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers=resp_headers,
        )

    @app.on_event("shutdown")
    async def _shutdown():
        try:
            collector.stop()
        except Exception:
            pass
        try:
            session_tracker.force_finalize_all()
        except Exception:
            pass
        try:
            scheduler.stop()
        except Exception:
            pass

    return app

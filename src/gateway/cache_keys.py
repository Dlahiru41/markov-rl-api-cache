"""
Smart cache-key builder for the API Gateway reverse proxy.

Builds deterministic keys from HTTP method, path, sorted query parameters,
and optionally the authenticated user identity.  Keys are hashed with MD5
for a fixed-length, safe Redis key.
"""

from __future__ import annotations

import hashlib
from fnmatch import fnmatch
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlencode


def build_cache_key(
    method: str,
    path: str,
    query_params: Optional[Mapping[str, str]] = None,
    headers: Optional[Mapping[str, str]] = None,
    cache_rules: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    """Return an MD5-hex cache key for the request.

    Parameters
    ----------
    method:
        HTTP method (e.g. ``"GET"``).
    path:
        Request path (e.g. ``"/api/users/123"``).
    query_params:
        Parsed query-string mapping.
    headers:
        Request headers (used to extract user identity when *vary_by_user*).
    cache_rules:
        List of rule dicts loaded from ``cache_rules.yaml``.

    Returns
    -------
    str
        Hex-encoded MD5 digest that serves as the Redis key.
    """
    base = f"{method.upper()}:{path}"

    if query_params:
        sorted_q = urlencode(sorted(query_params.items()))
        base += f"?{sorted_q}"

    # Optionally vary by user when the matching cache rule says so.
    if headers and cache_rules:
        vary = _should_vary_by_user(path, cache_rules)
        if vary:
            user_id = _extract_user_id(headers)
            if user_id:
                base += f":user:{user_id}"

    return hashlib.md5(base.encode()).hexdigest()


def build_cache_key_raw(
    method: str,
    path: str,
    query_params: Optional[Mapping[str, str]] = None,
    headers: Optional[Mapping[str, str]] = None,
    cache_rules: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    """Return an **unhashed** cache key for the request.

    The raw format allows Redis SCAN-based invalidation by path prefix.
    """
    base = f"{method.upper()}:{path}"

    if query_params:
        sorted_q = urlencode(sorted(query_params.items()))
        base += f"?{sorted_q}"

    if headers and cache_rules:
        vary = _should_vary_by_user(path, cache_rules)
        if vary:
            user_id = _extract_user_id(headers)
            if user_id:
                base += f":user:{user_id}"

    return base


def _should_vary_by_user(
    path: str, cache_rules: Sequence[Dict[str, Any]]
) -> bool:
    for rule in cache_rules:
        if fnmatch(path, rule.get("path", "")):
            return bool(rule.get("vary_by_user", False))
    return False


def _extract_user_id(headers: Mapping[str, str]) -> Optional[str]:
    """Best-effort extraction of user identity from request headers."""
    # Explicit header first
    user_id = headers.get("x-user-id") or headers.get("X-User-Id")
    if user_id:
        return user_id

    # Attempt to pull a 'sub' claim from a JWT Bearer token (no validation).
    auth = headers.get("authorization") or headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1]
        user_id = _extract_sub_from_jwt(token)
        if user_id:
            return user_id

    return None


def _extract_sub_from_jwt(token: str) -> Optional[str]:
    """Decode the *payload* section of a JWT (no signature verification)."""
    import base64
    import json

    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        payload_b64 = parts[1]
        # Add padding
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        payload_bytes = base64.urlsafe_b64decode(payload_b64)
        payload = json.loads(payload_bytes)
        return str(payload.get("sub")) if "sub" in payload else None
    except Exception:
        return None


def get_ttl_for_path(
    path: str,
    cache_rules: Sequence[Dict[str, Any]],
    default_ttl: int = 300,
) -> int:
    """Return the TTL (seconds) that applies to *path*.

    Returns ``0`` when caching is disabled for the path.
    """
    for rule in cache_rules:
        if fnmatch(path, rule.get("path", "")):
            if not rule.get("cache", True):
                return 0
            return int(rule.get("ttl", default_ttl))
    return default_ttl


def is_cacheable(
    path: str,
    cache_rules: Sequence[Dict[str, Any]],
) -> bool:
    """Return whether the path is eligible for caching."""
    for rule in cache_rules:
        if fnmatch(path, rule.get("path", "")):
            return rule.get("cache", True) and rule.get("ttl", 1) > 0
    return True


def get_invalidation_pattern(path: str) -> str:
    """Derive a glob pattern for keys that should be invalidated.

    For example ``POST /api/users`` → ``"GET:/api/users*"``.
    """
    # Strip trailing slash and trailing resource id for pattern
    return f"GET:{path.rstrip('/')}*"

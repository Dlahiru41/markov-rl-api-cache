"""Session lifecycle tracking for API sequences and Markov updates."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.monitoring.logger import get_logger

logger = get_logger("session_tracker")


@dataclass
class SessionState:
    """In-memory session state."""

    session_id: str
    created_at: datetime
    last_seen: datetime
    calls: List[str] = field(default_factory=list)
    contexts: List[Dict[str, Any]] = field(default_factory=list)


class SessionTracker:
    """Track sessions, expire/finalize them, and update Markov transitions."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        markov_chain: Any = None,
        on_session_finalized: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """Initialize tracker with TTL/max length limits."""
        self.config = config or {}
        self.markov_chain = markov_chain
        self.ttl_minutes = int(self.config.get("session_ttl_minutes", 30))
        self.max_calls = int(self.config.get("session_max_calls", 200))
        self.callback = on_session_finalized

        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()

        Path("data/sessions").mkdir(parents=True, exist_ok=True)

    def extract_session_id(self, headers: Dict[str, str], client_ip: str = "", user_agent: str = "") -> str:
        """Extract session identity from header/token/fallback hash."""
        if headers.get("x-session-id"):
            return headers["x-session-id"]
        if headers.get("authorization"):
            token = headers["authorization"].strip()
            return hashlib.sha1(token.encode("utf-8")).hexdigest()[:24]
        fallback = f"{client_ip}|{user_agent}".encode("utf-8")
        return hashlib.sha1(fallback).hexdigest()[:24]

    def track(self, session_id: str, path: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Record one API call in session and finalize when limits are reached."""
        now = datetime.now(timezone.utc)
        with self._lock:
            self._expire_locked(now)
            state = self._sessions.get(session_id)
            if state is None:
                state = SessionState(session_id=session_id, created_at=now, last_seen=now)
                self._sessions[session_id] = state
                logger.info("session_created", data={"session_id": session_id})

            state.last_seen = now
            state.calls.append(path)
            state.contexts.append(context or {})

            if len(state.calls) >= self.max_calls:
                self._finalize_locked(session_id, reason="max_length")

    def force_finalize_all(self) -> None:
        """Finalize all active sessions."""
        with self._lock:
            for sid in list(self._sessions.keys()):
                self._finalize_locked(sid, reason="shutdown")

    def active_session_count(self) -> int:
        """Return number of currently active sessions."""
        with self._lock:
            return len(self._sessions)

    def _expire_locked(self, now: datetime) -> None:
        """Expire sessions whose TTL window elapsed."""
        ttl = timedelta(minutes=self.ttl_minutes)
        expired = [sid for sid, st in self._sessions.items() if now - st.last_seen > ttl]
        for sid in expired:
            self._finalize_locked(sid, reason="ttl_expired")

    def _finalize_locked(self, session_id: str, reason: str) -> None:
        """Finalize one session, update Markov transitions, emit callback, persist log."""
        state = self._sessions.pop(session_id, None)
        if state is None:
            return

        transitions = 0
        if self.markov_chain is not None:
            for i in range(len(state.calls) - 1):
                try:
                    self.markov_chain.update(state.calls[i], state.calls[i + 1])
                    transitions += 1
                except Exception:
                    logger.error("session_markov_update_failed", data={"session_id": session_id}, exc_info=True)

        event = {
            "event": "session_finalized",
            "session_id": session_id,
            "created_at": state.created_at.isoformat(),
            "last_seen": state.last_seen.isoformat(),
            "call_count": len(state.calls),
            "calls": state.calls,
            "reason": reason,
            "transitions_added": transitions,
        }

        self._write_session_jsonl(event)
        logger.info("session_finalized", data=event)

        if self.callback is not None:
            try:
                self.callback(event)
            except Exception:
                logger.error("session_finalize_callback_failed", data={"session_id": session_id}, exc_info=True)

    def _write_session_jsonl(self, payload: Dict[str, Any]) -> None:
        """Persist finalized session payload to daily JSONL file."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_path = Path("data/sessions") / f"{date_str}.jsonl"
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")


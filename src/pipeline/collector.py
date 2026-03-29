"""Non-blocking API call collection with ring-buffer and periodic flush."""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from src.monitoring.logger import get_logger

logger = get_logger("collector")


@dataclass
class APICallRecord:
    """Captured details for one API call passing through the gateway."""

    request_id: str
    session_id: str
    timestamp: datetime
    method: str
    path: str
    query_params: str
    upstream_status: int
    response_time_ms: float
    cache_hit: bool
    cache_key: str
    markov_prediction: Optional[List[Tuple[str, float]]] = None
    rl_action_taken: int = 0
    rl_state_vector: List[float] = field(default_factory=list)
    rl_reward: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)


class APICallCollector:
    """Thread-safe collector that buffers records and flushes asynchronously."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, redis_client: Any = None):
        """Initialize collector with ring-buffer and background flusher."""
        self.config = config or {}
        self.redis_client = redis_client
        self.buffer_size = int(self.config.get("collector_buffer_size", 10000))
        self.flush_interval_seconds = int(self.config.get("collector_flush_interval_seconds", 60))

        self._buffer: Deque[APICallRecord] = deque(maxlen=self.buffer_size)
        self._lock = threading.Lock()
        self._running = False
        self._flush_thread: Optional[threading.Thread] = None

        self._total_collected = 0
        self._total_flushed = 0
        self._flush_errors = 0
        self._dropped = 0

        Path("data/api_calls").mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        """Start periodic background flushing."""
        if self._running:
            return
        self._running = True
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True, name="api-call-collector")
        self._flush_thread.start()

    def stop(self) -> None:
        """Stop flushing thread and flush pending records once."""
        self._running = False
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=2)
        self.flush_now()

    def record(self, record: APICallRecord) -> None:
        """Add a record without blocking the request path."""
        with self._lock:
            if len(self._buffer) >= self.buffer_size:
                self._dropped += 1
                logger.warning("collector_buffer_overflow", data={"buffer_size": self.buffer_size, "dropped": self._dropped})
            self._buffer.append(record)
            self._total_collected += 1

            usage = len(self._buffer) / max(1, self.buffer_size)
            if usage >= 0.8:
                logger.warning("collector_buffer_high_usage", data={"usage_ratio": round(usage, 3)})

    def new_record(self, **kwargs: Any) -> APICallRecord:
        """Build a record with defaults and generated request ID when missing."""
        kwargs.setdefault("request_id", str(uuid.uuid4()))
        kwargs.setdefault("timestamp", datetime.now(timezone.utc))
        kwargs.setdefault("session_id", "anonymous")
        kwargs.setdefault("method", "GET")
        kwargs.setdefault("path", "/")
        kwargs.setdefault("query_params", "")
        kwargs.setdefault("upstream_status", 200)
        kwargs.setdefault("response_time_ms", 0.0)
        kwargs.setdefault("cache_hit", False)
        kwargs.setdefault("cache_key", "")
        return APICallRecord(**kwargs)

    def flush_now(self) -> None:
        """Flush current buffer to Redis and local JSONL audit logs."""
        records = self._drain_buffer()
        if not records:
            return

        try:
            self._flush_to_jsonl(records)
            self._flush_to_redis(records)
            self._total_flushed += len(records)
            logger.info("collector_flush_success", data={"count": len(records)})
        except Exception as exc:
            self._flush_errors += 1
            with self._lock:
                for rec in reversed(records):
                    self._buffer.appendleft(rec)
            logger.error("collector_flush_failed", data={"error": str(exc), "retry": True}, exc_info=True)

    def get_status(self) -> Dict[str, Any]:
        """Return collector health and throughput counters."""
        with self._lock:
            buffer_len = len(self._buffer)
        return {
            "status": "healthy",
            "buffer_usage_percent": int((buffer_len / max(1, self.buffer_size)) * 100),
            "buffer_usage_ratio": buffer_len / max(1, self.buffer_size),
            "total_collected": self._total_collected,
            "total_flushed": self._total_flushed,
            "flush_errors": self._flush_errors,
            "dropped": self._dropped,
        }

    def _flush_loop(self) -> None:
        """Periodic flush loop that never raises to callers."""
        while self._running:
            time.sleep(self.flush_interval_seconds)
            try:
                self.flush_now()
            except Exception:
                self._flush_errors += 1
                logger.error("collector_flush_loop_error", exc_info=True)

    def _drain_buffer(self) -> List[APICallRecord]:
        """Atomically drain ring buffer into a list."""
        with self._lock:
            items = list(self._buffer)
            self._buffer.clear()
            return items

    def _flush_to_jsonl(self, records: List[APICallRecord]) -> None:
        """Append full records to daily JSONL audit file."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_path = Path("data/api_calls") / f"{date_str}.jsonl"
        with out_path.open("a", encoding="utf-8") as fh:
            for rec in records:
                payload = asdict(rec)
                payload["timestamp"] = rec.timestamp.isoformat()
                fh.write(json.dumps(payload, default=str) + "\n")

    def _flush_to_redis(self, records: List[APICallRecord]) -> None:
        """Flush compact session sequences and experiences into Redis lists."""
        if self.redis_client is None:
            return

        pipe = self.redis_client.pipeline(transaction=False)
        for rec in records:
            pipe.rpush(f"markov:api_sequences:{rec.session_id}", rec.path)

            exp = {
                "state": rec.rl_state_vector,
                "action": rec.rl_action_taken,
                "reward": rec.rl_reward,
                "next_state": rec.rl_state_vector,
                "done": False,
                "request_id": rec.request_id,
                "timestamp": rec.timestamp.isoformat(),
            }
            pipe.rpush("rl:experiences", json.dumps(exp))
        pipe.execute()


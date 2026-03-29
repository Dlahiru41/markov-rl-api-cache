"""Centralized structured JSON logging for the Markov-RL cache service."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional


_WARNING_COUNTER = 0
_ERROR_COUNTER = 0
_CRITICAL_COUNTER = 0


class JSONLogFormatter(logging.Formatter):
    """Format logs as one-line JSON records."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "component": getattr(record, "component", "app"),
            "event": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
            "data": getattr(record, "data", {}),
        }
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


class ComponentLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter adding component/event/data fields with safe defaults."""

    def process(self, msg: str, kwargs: Dict[str, Any]):
        extra = kwargs.setdefault("extra", {})
        extra.setdefault("component", self.extra.get("component", "app"))
        extra.setdefault("request_id", kwargs.pop("request_id", None))
        extra.setdefault("data", kwargs.pop("data", {}))
        _track_severity(kwargs.get("levelno", None), getattr(self.logger, "level", logging.INFO), extra)
        return msg, kwargs


class SafeLoggerAdapter(ComponentLoggerAdapter):
    """Adds structured convenience methods with event+data signature."""

    def debug(self, event: str, *, data: Optional[Dict[str, Any]] = None, request_id: Optional[str] = None):
        self.logger.debug(event, extra={"component": self.extra["component"], "data": data or {}, "request_id": request_id})

    def info(self, event: str, *, data: Optional[Dict[str, Any]] = None, request_id: Optional[str] = None):
        self.logger.info(event, extra={"component": self.extra["component"], "data": data or {}, "request_id": request_id})

    def warning(self, event: str, *, data: Optional[Dict[str, Any]] = None, request_id: Optional[str] = None):
        global _WARNING_COUNTER
        _WARNING_COUNTER += 1
        self.logger.warning(event, extra={"component": self.extra["component"], "data": data or {}, "request_id": request_id})

    def error(self, event: str, *, data: Optional[Dict[str, Any]] = None, request_id: Optional[str] = None, exc_info: bool = False):
        global _ERROR_COUNTER
        _ERROR_COUNTER += 1
        self.logger.error(
            event,
            extra={"component": self.extra["component"], "data": data or {}, "request_id": request_id},
            exc_info=exc_info,
        )

    def critical(self, event: str, *, data: Optional[Dict[str, Any]] = None, request_id: Optional[str] = None, exc_info: bool = False):
        global _CRITICAL_COUNTER
        _CRITICAL_COUNTER += 1
        self.logger.critical(
            event,
            extra={"component": self.extra["component"], "data": data or {}, "request_id": request_id},
            exc_info=exc_info,
        )


def _track_severity(*_args, **_kwargs) -> None:
    return


def _build_rotating_handler(path: Path, level: int) -> RotatingFileHandler:
    handler = RotatingFileHandler(path, maxBytes=50 * 1024 * 1024, backupCount=10)
    handler.setLevel(level)
    handler.setFormatter(JSONLogFormatter())
    return handler


def setup_logging(config: Optional[Any] = None) -> None:
    """Configure JSON logging for console and log files."""
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_level)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(JSONLogFormatter())
    root.addHandler(console)

    root.addHandler(_build_rotating_handler(logs_dir / "markov-rl-cache.log", log_level))

    for component_file in ("training.log", "scheduler.log", "gateway.log"):
        root.addHandler(_build_rotating_handler(logs_dir / component_file, log_level))


def get_logger(component: str) -> SafeLoggerAdapter:
    """Return a component-tagged structured logger."""
    return SafeLoggerAdapter(logging.getLogger(component), {"component": component})


def get_severity_counters() -> Dict[str, int]:
    """Return warning/error/critical counters for metrics exposure."""
    return {
        "warnings": _WARNING_COUNTER,
        "errors": _ERROR_COUNTER,
        "critical": _CRITICAL_COUNTER,
    }


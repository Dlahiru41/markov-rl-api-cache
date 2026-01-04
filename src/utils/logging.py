"""Logging utilities for markov-rl-api-cache using loguru.

Provides:
- setup_logger(...) to configure console + rotating file logging and to
  integrate with Python stdlib logging.
- MetricsLogger for experiment metric collection and CSV export.
- Decorators: @timed and @log_exceptions
- Helper functions for common logging patterns (cache events, training steps)

The module uses structured logging via loguru where possible.
"""
from __future__ import annotations

import csv
import functools
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional

from loguru import logger as _loguru_logger


# --------------------------- Logger setup ---------------------------------

class _InterceptHandler(logging.Handler):
    """Redirect stdlib logging to loguru."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - thin wrapper
        # Retrieve corresponding Loguru level if it exists
        try:
            level = _loguru_logger.level(record.levelname).name
        except Exception:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        # Walk up to find the caller frame outside logging
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        _loguru_logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logger(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: int | str | None = 5,
    colorize: bool = True,
) -> None:
    """Configure loguru and route Python stdlib logging to loguru.

    Parameters
    ----------
    level:
        Minimum log level for both console and file sinks (e.g., "DEBUG").
    log_file:
        Path to the log file. If None, a default `logs/app.log` under the
        project root will be used and the directory will be created.
    rotation:
        Log rotation setting passed to loguru (e.g., "10 MB" or a time)
    retention:
        How many rotated files to keep; passed directly to loguru (5 means keep
        5 files).
    colorize:
        Whether to use colorized output in the console sink.
    """
    # Remove existing handlers from loguru
    _loguru_logger.remove()

    # Console sink
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{module}</cyan>:{line} - <level>{message}</level>"
    )
    _loguru_logger.add(
        sys.stderr,
        level=level,
        format=console_format,
        colorize=colorize,
        enqueue=True,
    )

    # File sink
    if log_file is None:
        # Resolve logs dir relative to project root
        project_root = Path(__file__).resolve().parents[3]
        logs_dir = project_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(logs_dir / "app.log")
    _loguru_logger.add(
        log_file,
        level=level,
        rotation=rotation,
        retention=retention,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {module}:{line} - {message}",
        enqueue=True,
    )

    # Route the standard logging to loguru
    logging.root.handlers = [ _InterceptHandler() ]
    logging.root.setLevel(logging.NOTSET)

    # Optionally set verbosity for commonly noisy libraries
    for noisy in ("uvicorn.access", "asyncio", "urllib3", "botocore"):
        logging.getLogger(noisy).handlers = [ _InterceptHandler() ]


# --------------------------- Metrics logger -------------------------------

@dataclass
class MetricsLogger:
    """In-memory metrics logger for experiments with CSV export.

    Thread-safe and lightweight. Each metric logged is appended as a row with
    columns: timestamp, name, value, step, tags (JSON-ish string).
    """

    _rows: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def log_metric(self, name: str, value: float, step: int, tags: Optional[MutableMapping[str, Any]] = None) -> None:
        """Log a single scalar metric.

        Parameters
        ----------
        name: Metric name (e.g., "loss", "reward").
        value: Numeric value.
        step: Global step or episode step.
        tags: Optional dict of tags (e.g., {"env": "ecommerce"}).
        """
        row = {
            "timestamp": time.time(),
            "name": name,
            "value": float(value),
            "step": int(step),
            "tags": dict(tags) if tags is not None else {},
        }
        with self._lock:
            self._rows.append(row)
        _loguru_logger.debug("Metric logged: {name} @step={step} value={value} tags={tags}", **row)

    def log_episode_summary(self, episode: int, reward: float, loss: Optional[float] = None, epsilon: Optional[float] = None, **extras: Any) -> None:
        """Log a summary for a training episode.

        This creates multiple metric rows (reward, loss, epsilon) and any
        additional named extras provided.
        """
        tags = {k: v for k, v in extras.items()} if extras else {}
        self.log_metric("episode_reward", reward, step=episode, tags=tags)
        if loss is not None:
            self.log_metric("episode_loss", loss, step=episode, tags=tags)
        if epsilon is not None:
            self.log_metric("epsilon", epsilon, step=episode, tags=tags)
        _loguru_logger.info("Episode {episode} summary: reward={reward} loss={loss} epsilon={epsilon}", episode=episode, reward=reward, loss=loss, epsilon=epsilon)

    def export_csv(self, path: str | Path, fieldnames: Optional[Iterable[str]] = None) -> None:
        """Export all collected metrics to a CSV file.

        Parameters
        ----------
        path: Destination CSV file path.
        fieldnames: Optional sequence of field names; default will write timestamp,name,value,step,tags
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            rows_copy = list(self._rows)
        if not fieldnames:
            fieldnames = ("timestamp", "name", "value", "step", "tags")
        # Write CSV
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows_copy:
                # Ensure tags is stringified
                r_out = {k: (r.get(k) if k != "tags" else str(r.get("tags", {}))) for k in fieldnames}
                writer.writerow(r_out)
        _loguru_logger.info("Metrics exported to {path}", path=str(path))


# --------------------------- Decorators ----------------------------------

def timed(level: str = "INFO"):
    """Decorator that logs function execution time at given log level.

    Usage:
      @timed("DEBUG")
      def work(...):
          ...
    """

    def _decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.time() - start
                _loguru_logger.log(level, "{func} took {elapsed:.4f}s", func=func.__qualname__, elapsed=elapsed)

        return wrapper

    return _decorator


def log_exceptions(reraise: bool = True):
    """Decorator that logs exceptions with full traceback.

    If `reraise` is True (default) the original exception is re-raised after
    logging. Otherwise the exception is swallowed and the function returns None.
    """

    def _decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                _loguru_logger.exception("Exception in %s", func.__qualname__)
                if reraise:
                    raise
                return None

        return wrapper

    return _decorator


# --------------------------- Helper functions -----------------------------

def log_cache_event(hit: bool, api: str, latency_ms: float, **extra: Any) -> None:
    """Log a cache event (hit/miss).

    Parameters
    ----------
    hit: True for cache hit, False for miss
    api: API identifier/path
    latency_ms: Observed latency in milliseconds
    extra: Additional key/value information to include in the log
    """
    status = "HIT" if hit else "MISS"
    _loguru_logger.info(
        "CACHE {status} api={api} latency_ms={latency_ms:.2f} {extra}",
        status=status,
        api=api,
        latency_ms=latency_ms,
        extra=extra,
    )


def log_training_step(episode: int, step: int, action: Any, reward: float, **extra: Any) -> None:
    """Log a single training step.

    Parameters
    ----------
    episode: Episode number
    step: Step index within the episode
    action: Action taken (can be any serializable/printable)
    reward: Reward obtained
    extra: Additional info (e.g., state summary)
    """
    _loguru_logger.debug(
        "TRAIN step={step} episode={episode} action={action} reward={reward} {extra}",
        step=step,
        episode=episode,
        action=action,
        reward=reward,
        extra=extra,
    )


__all__ = [
    "setup_logger",
    "MetricsLogger",
    "timed",
    "log_exceptions",
    "log_cache_event",
    "log_training_step",
]


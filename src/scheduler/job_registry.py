"""Job registration and status tracking wrapper around APScheduler."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from apscheduler.schedulers.background import BackgroundScheduler


class JobRegistry:
    """Manage job lifecycle and status bookkeeping."""

    def __init__(self, scheduler: Optional[BackgroundScheduler] = None):
        """Initialize scheduler and internal job status map."""
        self.scheduler = scheduler or BackgroundScheduler()
        self.status: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, func: Callable[[], Any], trigger: str = "interval", **trigger_kwargs: Any) -> None:
        """Register a named job and wrap it for status accounting."""
        self.status.setdefault(
            name,
            {
                "next_run": None,
                "last_run": None,
                "last_status": None,
                "last_duration": 0.0,
                "run_count": 0,
                "error_count": 0,
            },
        )

        def wrapped() -> Any:
            start = time.perf_counter()
            self.status[name]["last_run"] = datetime.now(timezone.utc).isoformat()
            self.status[name]["run_count"] += 1
            try:
                result = func()
                self.status[name]["last_status"] = "success"
                return result
            except Exception:
                self.status[name]["last_status"] = "error"
                self.status[name]["error_count"] += 1
                raise
            finally:
                self.status[name]["last_duration"] = time.perf_counter() - start

        job = self.scheduler.add_job(wrapped, trigger, id=name, replace_existing=True, **trigger_kwargs)
        self.status[name]["next_run"] = job.next_run_time.isoformat() if job.next_run_time else None

    def start(self) -> None:
        """Start underlying APScheduler instance."""
        if not self.scheduler.running:
            self.scheduler.start()

    def stop(self) -> None:
        """Stop scheduler gracefully."""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)

    def pause_job(self, name: str) -> None:
        """Pause a registered job by name."""
        self.scheduler.pause_job(name)

    def resume_job(self, name: str) -> None:
        """Resume a paused job by name."""
        self.scheduler.resume_job(name)

    def trigger_job(self, name: str) -> None:
        """Trigger a one-off immediate run for a job."""
        self.scheduler.add_job(self.scheduler.get_job(name).func)

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Return status map with updated next_run values."""
        for name in list(self.status.keys()):
            job = self.scheduler.get_job(name)
            self.status[name]["next_run"] = job.next_run_time.isoformat() if job and job.next_run_time else None
        return self.status


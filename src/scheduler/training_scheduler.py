"""Background scheduler coordinating Markov updates, DQN training, eval and cleanup."""

from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.monitoring.logger import get_logger
from src.scheduler.job_registry import JobRegistry
from src.scheduler.resource_guard import ResourceGuard

logger = get_logger("scheduler")


class TrainingScheduler:
    """Register and run periodic jobs for model updates and maintenance."""

    def __init__(
        self,
        markov_chain: Any,
        dqn_agent: Any,
        collector: Any,
        config: Optional[Dict[str, Any]] = None,
        redis_client: Any = None,
    ):
        """Initialize scheduler dependencies and configuration."""
        self.markov_chain = markov_chain
        self.dqn_agent = dqn_agent
        self.collector = collector
        self.config = config or {}
        self.redis = redis_client

        self.registry = JobRegistry()
        self.resource_guard = ResourceGuard(
            redis_client=redis_client,
            max_cpu=float(self._cfg("MAX_CPU_FOR_TRAINING", 80)),
            max_memory=float(self._cfg("MAX_MEMORY_FOR_TRAINING", 85)),
        )

        self._register_jobs()

    def _cfg(self, key: str, default: Any) -> Any:
        return self.config.get(key.lower(), self.config.get(key, os.getenv(key, default)))

    def _register_jobs(self) -> None:
        """Register all periodic jobs with configured schedules."""
        self.registry.register(
            "markov_update",
            self.markov_update_job,
            "interval",
            seconds=int(self._cfg("MARKOV_UPDATE_INTERVAL_SECONDS", 300)),
        )
        self.registry.register(
            "dqn_training",
            self.dqn_training_job,
            "interval",
            seconds=int(self._cfg("DQN_TRAIN_INTERVAL_SECONDS", 900)),
        )
        self.registry.register(
            "model_evaluation",
            self.model_evaluation_job,
            "interval",
            seconds=int(self._cfg("EVAL_INTERVAL_SECONDS", 3600)),
        )
        self.registry.register(
            "data_cleanup",
            self.data_cleanup_job,
            "cron",
            hour=int(self._cfg("CLEANUP_CRON_HOUR", 2)),
            minute=int(self._cfg("CLEANUP_CRON_MINUTE", 0)),
        )

    def start(self) -> None:
        """Start periodic job execution."""
        self.registry.start()
        logger.info("scheduler_started")

    def stop(self) -> None:
        """Stop scheduler execution."""
        self.registry.stop()
        logger.info("scheduler_stopped")

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Expose job status map for monitoring APIs."""
        return self.registry.get_status()

    def manual_trigger(self, job_name: str) -> Dict[str, Any]:
        """Trigger named job manually."""
        self.registry.trigger_job(job_name)
        return {"triggered": job_name}

    def markov_update_job(self) -> Dict[str, Any]:
        """Update Markov transitions from finalized session sequences in Redis."""
        lock = "markov:training:lock"
        if not self._acquire_lock(lock, ttl=120):
            logger.warning("markov_update_skipped_lock_held")
            return {"skipped": "lock_held"}

        try:
            sessions = self._read_markov_sequences()
            transitions = 0
            top_counts: Dict[Tuple[str, str], int] = {}
            for seq in sessions:
                for i in range(len(seq) - 1):
                    cur, nxt = seq[i], seq[i + 1]
                    self.markov_chain.update(cur, nxt)
                    transitions += 1
                    top_counts[(cur, nxt)] = top_counts.get((cur, nxt), 0) + 1
            top5 = sorted(top_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(
                "markov_update_completed",
                data={
                    "sessions_processed": len(sessions),
                    "transitions_added": transitions,
                    "top_transitions": [((a, b), c) for (a, b), c in top5],
                },
            )
            self._clear_markov_sequences()
            return {"sessions_processed": len(sessions), "transitions_added": transitions}
        except Exception:
            logger.error("markov_update_failed", data={"traceback": traceback.format_exc()}, exc_info=True)
            self._inc_error_counter("scheduler.markov_update.errors")
            return {"error": True}
        finally:
            self._release_lock(lock)

    def dqn_training_job(self) -> Dict[str, Any]:
        """Run guarded DQN training cycle and checkpoint rotation."""
        min_exp = int(self._cfg("MIN_EXPERIENCES_FOR_TRAINING", 500))
        steps = int(self._cfg("TRAINING_STEPS_PER_CYCLE", 100))
        max_ckpts = int(self._cfg("MAX_CHECKPOINTS", 10))

        replay_size = len(getattr(self.dqn_agent, "buffer", [])) if self.dqn_agent is not None else 0
        if replay_size < min_exp:
            logger.warning("dqn_training_skipped", data={"reason": "insufficient_experiences", "buffer_size": replay_size})
            return {"skipped": "insufficient_experiences"}

        can_train, reason = self.resource_guard.can_train()
        if not can_train:
            logger.warning("dqn_training_skipped", data={"reason": reason})
            return {"skipped": reason}

        lock = "rl:training:lock"
        if not self._acquire_lock(lock, ttl=1200):
            logger.warning("dqn_training_skipped", data={"reason": "lock_held"})
            return {"skipped": "lock_held"}

        t0 = time.perf_counter()
        epsilon_before = float(getattr(self.dqn_agent, "epsilon", 0.0))
        metrics: List[Dict[str, Any]] = []
        try:
            logger.info(
                "dqn_training_started",
                data={"buffer_size": replay_size, "epsilon": epsilon_before, "system": self.resource_guard.get_system_snapshot()},
            )

            for step in range(1, steps + 1):
                step_t0 = time.perf_counter()
                out = self.dqn_agent.train_step()
                if out is None:
                    continue
                step_ms = (time.perf_counter() - step_t0) * 1000.0
                sample = {
                    "loss": float(out.get("loss", 0.0)),
                    "q_mean": float(out.get("q_mean", 0.0)),
                    "epsilon": float(out.get("epsilon", 0.0)),
                    "step_duration_ms": step_ms,
                }
                metrics.append(sample)
                if step % 10 == 0:
                    logger.info("dqn_training_progress", data={"step": step, "latest": sample})

            ckpt = self._save_checkpoint(prefix="dqn")
            self._trim_checkpoints(max_ckpts)

            epsilon_after = float(getattr(self.dqn_agent, "epsilon", epsilon_before))
            total = time.perf_counter() - t0
            avg_loss = sum(m["loss"] for m in metrics) / max(1, len(metrics))
            avg_q = sum(m["q_mean"] for m in metrics) / max(1, len(metrics))
            summary = {
                "avg_loss": avg_loss,
                "avg_q_mean": avg_q,
                "epsilon_before": epsilon_before,
                "epsilon_after": epsilon_after,
                "duration_seconds": total,
                "checkpoint": ckpt,
            }
            logger.info("dqn_training_completed", data=summary)
            return summary
        except Exception:
            emergency = self._save_checkpoint(prefix="emergency")
            logger.error("dqn_training_failed", data={"checkpoint": emergency, "traceback": traceback.format_exc()}, exc_info=True)
            self._inc_error_counter("scheduler.dqn_training.errors")
            return {"error": True, "checkpoint": emergency}
        finally:
            self._release_lock(lock)

    def model_evaluation_job(self) -> Dict[str, Any]:
        """Evaluate latest model checkpoint and persist evaluation artifact."""
        ckpt = self._latest_checkpoint()
        if ckpt is None:
            logger.warning("model_evaluation_skipped", data={"reason": "no_checkpoint"})
            return {"skipped": "no_checkpoint"}

        replay_size = len(getattr(self.dqn_agent, "buffer", [])) if self.dqn_agent is not None else 0
        if replay_size == 0:
            logger.warning("model_evaluation_skipped", data={"reason": "empty_replay"})
            return {"skipped": "empty_replay"}

        try:
            self.dqn_agent.load(str(ckpt))
            # Placeholder metrics until full offline evaluation over replay samples is integrated.
            avg_reward = 0.0
            hit_rate = 0.0
            prediction_accuracy = 0.0
            prefetch_efficiency = 0.0

            result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "checkpoint": ckpt.name,
                "average_reward": avg_reward,
                "cache_hit_rate": hit_rate,
                "prediction_accuracy": prediction_accuracy,
                "prefetch_efficiency": prefetch_efficiency,
            }
            Path("data/evaluations").mkdir(parents=True, exist_ok=True)
            name = datetime.now(timezone.utc).strftime("eval_%Y%m%d_%H%M%S.json")
            out = Path("data/evaluations") / name
            out.write_text(json.dumps(result, indent=2), encoding="utf-8")
            logger.info("model_evaluation_completed", data=result)
            return result
        except Exception:
            logger.error("model_evaluation_failed", data={"traceback": traceback.format_exc()}, exc_info=True)
            return {"error": True}

    def data_cleanup_job(self) -> Dict[str, Any]:
        """Delete old logs and compact data according to retention settings."""
        log_days = int(self._cfg("LOG_RETENTION_DAYS", 30))
        session_days = int(self._cfg("SESSION_RETENTION_DAYS", 14))

        now = time.time()
        deleted_files = 0
        reclaimed_bytes = 0

        for folder, retention in ((Path("data/api_calls"), log_days), (Path("data/sessions"), session_days)):
            if not folder.exists():
                continue
            cutoff = now - (retention * 86400)
            for file in folder.glob("*.jsonl"):
                if file.stat().st_mtime < cutoff:
                    reclaimed_bytes += file.stat().st_size
                    file.unlink(missing_ok=True)
                    deleted_files += 1

        summary = {
            "files_deleted": deleted_files,
            "disk_space_reclaimed_bytes": reclaimed_bytes,
            "redis_keys_removed": 0,
        }
        logger.info("data_cleanup_completed", data=summary)
        return summary

    def _save_checkpoint(self, prefix: str) -> str:
        """Save checkpoint under models/checkpoints timestamped filename."""
        Path("models/checkpoints").mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = Path("models/checkpoints") / f"{prefix}_{ts}.pt"
        self.dqn_agent.save(str(path))
        return str(path)

    def _trim_checkpoints(self, max_checkpoints: int) -> None:
        """Keep only newest configured number of checkpoints."""
        files = sorted(Path("models/checkpoints").glob("*.pt"), key=lambda p: p.stat().st_mtime)
        overflow = max(0, len(files) - max_checkpoints)
        for old in files[:overflow]:
            old.unlink(missing_ok=True)

    def _latest_checkpoint(self) -> Optional[Path]:
        """Return most recent checkpoint path."""
        files = sorted(Path("models/checkpoints").glob("*.pt"), key=lambda p: p.stat().st_mtime)
        return files[-1] if files else None

    def _acquire_lock(self, key: str, ttl: int) -> bool:
        """Acquire redis lock key with TTL, fallback true when redis unavailable."""
        if self.redis is None:
            return True
        try:
            return bool(self.redis.set(key, "1", nx=True, ex=ttl))
        except Exception:
            return False

    def _release_lock(self, key: str) -> None:
        """Release redis lock key best-effort."""
        if self.redis is None:
            return
        try:
            self.redis.delete(key)
        except Exception:
            return

    def _read_markov_sequences(self) -> List[List[str]]:
        """Read session sequences from redis keys markov:api_sequences:*."""
        if self.redis is None:
            return []
        sequences: List[List[str]] = []
        for key in self.redis.scan_iter("markov:api_sequences:*"):
            vals = self.redis.lrange(key, 0, -1)
            seq = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in vals]
            if seq:
                sequences.append(seq)
        return sequences

    def _clear_markov_sequences(self) -> None:
        """Delete all processed markov session sequence keys."""
        if self.redis is None:
            return
        for key in self.redis.scan_iter("markov:api_sequences:*"):
            self.redis.delete(key)

    def _inc_error_counter(self, key: str) -> None:
        """Increment redis-backed error counter if available."""
        if self.redis is None:
            return
        try:
            self.redis.incr(key)
        except Exception:
            return

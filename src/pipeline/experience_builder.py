"""Build asynchronous RL experiences from request decisions and outcomes."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.monitoring.logger import get_logger
from src.rl.reward import ActionOutcome, RewardCalculator

logger = get_logger("collector")


@dataclass
class PendingExperience:
    """Pending experience waiting for outcome and next state."""

    request_id: str
    state: List[float]
    action: int
    context: Dict[str, Any]


class ExperienceBuilder:
    """Construct (s, a, r, s', done) tuples across async request lifecycle."""

    def __init__(self, dqn_agent: Any = None):
        """Initialize pending queues and optional DQN sink."""
        self.dqn_agent = dqn_agent
        self.reward_calculator = RewardCalculator()
        self._pending: Dict[str, PendingExperience] = {}
        self._ready: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def start_decision(self, request_id: str, state: List[float], action: int, context: Optional[Dict[str, Any]] = None) -> None:
        """Record decision-time state/action for later completion."""
        with self._lock:
            self._pending[request_id] = PendingExperience(
                request_id=request_id,
                state=state,
                action=action,
                context=context or {},
            )

    def complete_outcome(self, request_id: str, outcome: Dict[str, Any], next_state: List[float], done: bool = False) -> Optional[Dict[str, Any]]:
        """Complete pending request with reward and next state and store transition."""
        with self._lock:
            pending = self._pending.pop(request_id, None)
        if pending is None:
            logger.warning("experience_missing_pending", data={"request_id": request_id})
            return None

        action_outcome = ActionOutcome(
            cache_hit=bool(outcome.get("cache_hit", False)),
            cache_miss=not bool(outcome.get("cache_hit", False)),
            prefetch_used=int(outcome.get("prefetch_used", 0)),
            prefetch_wasted=int(outcome.get("prefetch_wasted", 0)),
            prefetch_bytes=int(outcome.get("prefetch_bytes", 0)),
            cascade_prevented=bool(outcome.get("cascade_prevented", False)),
            cascade_occurred=bool(outcome.get("cascade_occurred", False)),
            actual_latency_ms=float(outcome.get("latency_ms", 0.0)),
            baseline_latency_ms=float(outcome.get("baseline_latency_ms", 0.0)),
            cache_utilization=float(outcome.get("cache_utilization", 0.0)),
            prediction_was_correct=bool(outcome.get("prediction_was_correct", False)),
        )
        reward = float(self.reward_calculator.calculate(action_outcome))

        transition = {
            "request_id": request_id,
            "state": pending.state,
            "action": pending.action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        }

        if self.dqn_agent is not None:
            try:
                self.dqn_agent.store_transition(pending.state, pending.action, reward, next_state, done)
            except Exception:
                logger.error("experience_dqn_store_failed", data={"request_id": request_id}, exc_info=True)

        with self._lock:
            self._ready.append(transition)

        return transition

    def pop_ready(self) -> List[Dict[str, Any]]:
        """Return and clear complete transitions built so far."""
        with self._lock:
            ready = list(self._ready)
            self._ready.clear()
            return ready

    def pending_count(self) -> int:
        """Return number of in-flight pending decisions."""
        with self._lock:
            return len(self._pending)


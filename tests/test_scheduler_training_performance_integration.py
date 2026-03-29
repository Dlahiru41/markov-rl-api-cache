"""
Integration suite for scheduler, training, and latency comparison.

Focus:
- Scheduler job wiring + executable training cycle
- End-to-end DQN training effect on policy behavior
- Latency comparison before vs after training data
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.cache.backend import InMemoryBackend
from src.rl.agents.dqn_agent import DQNAgent, DQNConfig
from src.scheduler.training_scheduler import TrainingScheduler


class _DummyMarkov:
    def update(self, *_args, **_kwargs):
        return self


class _DummyCollector:
    pass


def _make_dqn_agent(seed: int = 7) -> DQNAgent:
    config = DQNConfig(
        state_dim=8,
        action_dim=2,  # 0=NO_PREFETCH, 1=PREFETCH_NEXT
        hidden_dims=[32, 16],
        dueling=False,
        learning_rate=0.001,
        gamma=0.95,
        epsilon_start=0.0,  # deterministic eval for this suite
        epsilon_end=0.0,
        epsilon_decay=1.0,
        buffer_size=5000,
        batch_size=64,
        target_update_freq=50,
        device="cpu",
        seed=seed,
    )
    return DQNAgent(config, seed=seed)


def _state_for_key(index: int, total: int) -> np.ndarray:
    ratio = float(index) / float(max(1, total - 1))
    return np.array(
        [
            ratio,
            ratio**2,
            np.sin(ratio * np.pi),
            np.cos(ratio * np.pi),
            1.0 if index % 2 == 0 else 0.0,
            1.0 if index % 3 == 0 else 0.0,
            1.0 if index % 5 == 0 else 0.0,
            1.0,
        ],
        dtype=np.float32,
    )


def _build_training_data(samples: int = 300) -> List[np.ndarray]:
    return [_state_for_key(i, samples) for i in range(samples)]


def _train_prefetch_policy(agent: DQNAgent, states: List[np.ndarray], iterations: int = 250) -> None:
    for state in states:
        # Synthetic supervised-style shaping: keep dynamics stationary so reward signal
        # isolates the action preference (prefetch vs no-prefetch) for this suite.
        next_state = state
        agent.store_transition(state, 1, 2.0, next_state, False)
        agent.store_transition(state, 0, -1.0, next_state, False)

    for _ in range(iterations):
        agent.train_step()


def _simulate_request_latency(agent: DQNAgent, requests: List[int]) -> Dict[str, float]:
    cache = InMemoryBackend(max_size_bytes=8 * 1024 * 1024)
    latencies_ms: List[float] = []
    prefetch_actions = 0
    total = len(requests)

    for idx, req_key in enumerate(requests):
        state = _state_for_key(idx, total)
        action = int(agent.select_action(state, evaluate=True))

        if action == 1 and idx + 1 < total:
            prefetch_actions += 1
            nxt = requests[idx + 1]
            cache.set(f"req:{nxt}", b"prefetched", ttl=60)

        cache_key = f"req:{req_key}"
        cached = cache.get(cache_key)
        if cached is None:
            latencies_ms.append(50.0)
            cache.set(cache_key, b"origin", ttl=60)
        else:
            latencies_ms.append(5.0)

    return {
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "prefetch_ratio": float(prefetch_actions / max(1, total)),
    }


def test_scheduler_runs_training_cycle_and_writes_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    agent = _make_dqn_agent(seed=11)
    train_states = _build_training_data(samples=150)
    _train_prefetch_policy(agent, train_states, iterations=120)

    scheduler = TrainingScheduler(
        markov_chain=_DummyMarkov(),
        dqn_agent=agent,
        collector=_DummyCollector(),
        config={
            "MIN_EXPERIENCES_FOR_TRAINING": 64,
            "TRAINING_STEPS_PER_CYCLE": 20,
            "MAX_CHECKPOINTS": 2,
        },
        redis_client=None,
    )

    status = scheduler.get_status()
    assert set(status.keys()) == {"markov_update", "dqn_training", "model_evaluation", "data_cleanup"}

    # ResourceGuard requires redis connectivity by default; force trainable path for this isolated integration test.
    scheduler.resource_guard.can_train = lambda: (True, "ok")
    result = scheduler.dqn_training_job()
    assert "avg_loss" in result
    assert "duration_seconds" in result
    assert "checkpoint" in result

    checkpoint = Path(result["checkpoint"])
    assert checkpoint.exists()


def test_training_data_changes_policy_preference():
    agent = _make_dqn_agent(seed=21)
    probe = _state_for_key(8, 100)

    q_before = agent.get_q_values(probe)
    action_before = int(agent.select_action(probe, evaluate=True))

    _train_prefetch_policy(agent, _build_training_data(samples=240), iterations=260)

    q_after = agent.get_q_values(probe)
    action_after = int(agent.select_action(probe, evaluate=True))

    assert q_after[1] > q_after[0]
    assert q_after[1] > q_before[1]
    assert action_after == 1
    assert action_after >= action_before or q_after[1] - q_after[0] > 0.5


def test_latency_reduces_after_training_data():
    base_agent = _make_dqn_agent(seed=33)
    trained_agent = copy.deepcopy(base_agent)

    workload = list(range(180))

    before = _simulate_request_latency(base_agent, workload)

    _train_prefetch_policy(trained_agent, _build_training_data(samples=360), iterations=300)
    after = _simulate_request_latency(trained_agent, workload)

    assert after["prefetch_ratio"] >= before["prefetch_ratio"]
    assert after["mean_latency_ms"] <= before["mean_latency_ms"]
    assert after["p95_latency_ms"] <= before["p95_latency_ms"]

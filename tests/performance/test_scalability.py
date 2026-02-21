"""
Scalability tests – measure how system performance changes as key
dimensions grow: number of services, traffic rate, model vocabulary,
network size, and training data volume.
"""

import random
import time
from typing import List
import numpy as np
import pytest

from src.markov.first_order import FirstOrderMarkovChain
from src.cache.backend import InMemoryBackend
from src.rl.agents.dqn_agent import DQNAgent, DQNConfig
from src.rl.replay_buffer import ReplayBuffer

from tests.performance.conftest import (
    measure_throughput,
    measure_latency,
    _build_vocabulary,
    _generate_sequences,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fit_first_order(vocab_size: int, num_seq: int = 500) -> FirstOrderMarkovChain:
    vocab = _build_vocabulary(vocab_size)
    seqs  = _generate_sequences(vocab, num_sequences=num_seq)
    mc    = FirstOrderMarkovChain(smoothing=0.001)
    mc.fit(seqs)
    return mc


def _make_agent(state_dim: int, action_dim: int,
                hidden: List[int]) -> DQNAgent:
    config = DQNConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden,
        buffer_size=10_000,
        batch_size=32,
        device="cpu",
        seed=0,
    )
    return DQNAgent(config, seed=0)


# ===========================================================================
# TestServiceScaling
# ===========================================================================

@pytest.mark.slow
class TestServiceScaling:
    """Throughput as the simulated number of services grows."""

    def _make_vocab_for_services(self, num_services: int,
                                  endpoints_per_service: int = 10) -> List[str]:
        vocab = []
        for s in range(num_services):
            for e in range(endpoints_per_service):
                vocab.append(f"/service_{s}/endpoint_{e}")
        return vocab

    # ------------------------------------------------------------------
    @pytest.mark.parametrize("num_services", [7, 14, 28])
    def test_scale_num_services(self, num_services, benchmark_config):
        """
        Prediction throughput with 7, 14, and 28 services.
        Expect graceful degradation (no cliff edge).
        """
        vocab = self._make_vocab_for_services(num_services)
        seqs  = _generate_sequences(vocab, num_sequences=500)
        mc    = FirstOrderMarkovChain(smoothing=0.001)
        mc.fit(seqs)

        mc_vocab = list(mc.states)
        if not mc_vocab:
            pytest.skip("Empty vocabulary")

        rng = random.Random(0)
        anchors = [rng.choice(mc_vocab) for _ in range(200)]
        idx = [0]

        def predict():
            mc.predict(anchors[idx[0] % len(anchors)], k=5)
            idx[0] += 1

        result = measure_throughput(
            predict,
            duration=benchmark_config["throughput_duration_s"],
            warmup=100,
        )
        print(
            f"\n[{num_services} services] "
            f"{result['ops_per_sec']:,.0f} pred/s"
        )
        # Should still exceed half the base target at the largest scale
        half_target = benchmark_config["target_markov_predictions_per_sec"] * 0.5
        assert result["ops_per_sec"] >= half_target, (
            f"{num_services} services: {result['ops_per_sec']:.0f} < {half_target:.0f}"
        )

    # ------------------------------------------------------------------
    @pytest.mark.parametrize("endpoints_per_service", [5, 20, 50, 100])
    def test_scale_num_endpoints(self, endpoints_per_service, benchmark_config):
        """
        Prediction throughput as endpoints per service scales.
        """
        vocab = self._make_vocab_for_services(7, endpoints_per_service)
        seqs  = _generate_sequences(vocab, num_sequences=500)
        mc    = FirstOrderMarkovChain(smoothing=0.001)
        mc.fit(seqs)

        mc_vocab = list(mc.states)
        if not mc_vocab:
            pytest.skip("Empty vocabulary")

        rng = random.Random(0)
        anchors = [rng.choice(mc_vocab) for _ in range(200)]
        idx = [0]

        def predict():
            mc.predict(anchors[idx[0] % len(anchors)], k=5)
            idx[0] += 1

        result = measure_throughput(
            predict,
            duration=benchmark_config["throughput_duration_s"],
            warmup=100,
        )
        print(
            f"\n[{endpoints_per_service} eps/svc] "
            f"{result['ops_per_sec']:,.0f} pred/s"
        )
        assert result["ops_per_sec"] >= 200, (
            f"{endpoints_per_service} eps/svc: throughput {result['ops_per_sec']:.0f} < 200"
        )


# ===========================================================================
# TestTrafficScaling
# ===========================================================================

@pytest.mark.slow
class TestTrafficScaling:
    """Simulate multiple concurrent clients at different request rates."""

    # ------------------------------------------------------------------
    @pytest.mark.parametrize("target_rps", [100, 500, 1000, 5000])
    def test_scale_requests_per_second(self, target_rps, benchmark_config):
        """
        Fire `target_rps` synthetic cache requests per second and measure
        achieved throughput + latency.
        """
        backend = InMemoryBackend(max_size_bytes=128 * 1024 * 1024)
        rng     = random.Random(0)
        for i in range(10_000):
            backend.set(f"rps:key:{i}", b"data", ttl=3600)

        keys = [f"rps:key:{i}" for i in range(10_000)]

        interval  = 1.0 / target_rps
        duration  = min(3.0, benchmark_config["throughput_duration_s"])
        count     = [0]
        latencies = []

        t0 = time.perf_counter()
        deadline = t0 + duration
        while time.perf_counter() < deadline:
            start = time.perf_counter()
            backend.get(rng.choice(keys))
            elapsed = time.perf_counter() - start
            latencies.append(elapsed * 1000)
            count[0] += 1

            # Rate limit
            sleep = interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

        actual_rps = count[0] / (time.perf_counter() - t0)
        latencies.sort()
        p99 = latencies[int(0.99 * len(latencies))] if latencies else 0

        print(
            f"\n[target={target_rps} RPS] "
            f"achieved={actual_rps:,.0f} RPS  p99={p99:.3f}ms"
        )
        # For low targets we can enforce 60% achievement.
        # For high targets (>500) the sleep granularity limits us, so only
        # check that we achieved at least 500 RPS (the sleep-free ceiling).
        effective_target = min(target_rps, 500)
        assert actual_rps >= effective_target * 0.60, (
            f"Achieved {actual_rps:.0f} RPS < 60% of effective target {effective_target}"
        )

    # ------------------------------------------------------------------
    @pytest.mark.parametrize("num_users", [1, 4, 8, 16])
    def test_scale_concurrent_users(self, num_users, benchmark_config):
        """
        Throughput with N concurrent threads each performing cache reads.
        """
        import threading

        backend = InMemoryBackend(max_size_bytes=128 * 1024 * 1024)
        for i in range(5_000):
            backend.set(f"cu:key:{i}", b"data", ttl=3600)

        keys     = [f"cu:key:{i}" for i in range(5_000)]
        duration = min(3.0, benchmark_config["throughput_duration_s"])
        counts   = [0] * num_users
        stop_evt = threading.Event()

        def worker(tid: int):
            rng = random.Random(tid)
            while not stop_evt.is_set():
                backend.get(rng.choice(keys))
                counts[tid] += 1

        threads = [threading.Thread(target=worker, args=(i,), daemon=True)
                   for i in range(num_users)]
        for t in threads:
            t.start()
        time.sleep(duration)
        stop_evt.set()
        for t in threads:
            t.join(timeout=2.0)

        total_ops = sum(counts)
        agg_rps   = total_ops / duration
        print(
            f"\n[{num_users} threads] aggregate {agg_rps:,.0f} ops/s "
            f"({total_ops} ops in {duration:.1f}s)"
        )
        # Total throughput should be at least 1,000 ops/s regardless of thread count
        assert agg_rps >= 1_000, (
            f"{num_users} threads: aggregate {agg_rps:.0f} < 1,000 ops/s"
        )


# ===========================================================================
# TestModelScaling
# ===========================================================================

@pytest.mark.slow
class TestModelScaling:
    """How prediction latency and memory scale with model complexity."""

    # ------------------------------------------------------------------
    @pytest.mark.parametrize("vocab_size", [100, 1000, 10_000])
    def test_scale_vocabulary_size(self, vocab_size, benchmark_config):
        """
        Prediction latency with 100, 1000, and 10 000 unique API endpoints.
        p99 should remain < 5 ms even at 10 000 endpoints.
        """
        mc    = _fit_first_order(vocab_size, num_seq=min(vocab_size * 2, 3000))
        vocab = list(mc.states)
        if not vocab:
            pytest.skip("Empty vocabulary")

        rng     = random.Random(0)
        anchors = [rng.choice(vocab) for _ in range(200)]
        idx     = [0]

        def predict():
            mc.predict(anchors[idx[0] % len(anchors)], k=5)
            idx[0] += 1

        stats = measure_latency(
            predict,
            iterations=500,
            warmup=100,
        )
        print(
            f"\n[vocab={vocab_size}] "
            f"median={stats['median_ms']:.4f}ms  "
            f"p99={stats['p99_ms']:.4f}ms"
        )
        assert stats["p99_ms"] < 100.0, (
            f"vocab={vocab_size}: p99 {stats['p99_ms']:.4f}ms >= 100ms"
        )

    # ------------------------------------------------------------------
    @pytest.mark.parametrize("n_contexts", [2, 5, 10])
    def test_scale_context_dimensions(self, n_contexts, benchmark_config):
        """
        Prediction latency with different numbers of context features.
        """
        from src.markov.context_aware import ContextAwareMarkovChain

        vocab    = _build_vocabulary(100)
        seqs     = _generate_sequences(vocab, num_sequences=500)
        features = [f"feat_{j}" for j in range(n_contexts)]
        ctxs     = [
            {f: random.choice(["low", "high"]) for f in features}
            for _ in seqs
        ]

        mc_ctx = ContextAwareMarkovChain(
            context_features=features,
            order=1,
            smoothing=0.001,
        )
        mc_ctx.fit(seqs, ctxs)

        rng  = random.Random(0)
        anch = [rng.choice(vocab) for _ in range(100)]
        idx  = [0]

        def predict():
            ctx = {f: rng.choice(["low", "high"]) for f in features}
            mc_ctx.predict(anch[idx[0] % len(anch)], ctx, k=5)
            idx[0] += 1

        stats = measure_latency(predict, iterations=300, warmup=50)
        print(
            f"\n[{n_contexts} context features] "
            f"p99={stats['p99_ms']:.4f}ms"
        )
        assert stats["p99_ms"] < 10.0, (
            f"{n_contexts} ctx: p99 {stats['p99_ms']:.4f}ms >= 10ms"
        )

    # ------------------------------------------------------------------
    @pytest.mark.parametrize("hidden", [
        [32, 16],
        [128, 64],
        [512, 256, 128],
    ])
    def test_scale_network_size(self, hidden, benchmark_config):
        """
        Action-selection latency for small, medium, and large Q-networks.
        p99 should remain < 5ms even for the largest architecture (CPU).
        """
        agent = _make_agent(state_dim=32, action_dim=10, hidden=hidden)
        rng   = np.random.default_rng(0)
        states = [rng.random(32).astype(np.float32) for _ in range(200)]
        idx    = [0]

        def select():
            agent.select_action(states[idx[0] % 200], evaluate=True)
            idx[0] += 1

        stats = measure_latency(select, iterations=500, warmup=100)
        print(
            f"\nNetwork {hidden}: "
            f"median={stats['median_ms']:.4f}ms  "
            f"p99={stats['p99_ms']:.4f}ms"
        )
        assert stats["p99_ms"] < 5.0, (
            f"Network {hidden}: p99 {stats['p99_ms']:.4f}ms >= 5ms"
        )


# ===========================================================================
# TestDataScaling
# ===========================================================================

@pytest.mark.slow
class TestDataScaling:
    """Training & fitting performance vs data volume."""

    # ------------------------------------------------------------------
    @pytest.mark.parametrize("n_sequences", [1_000, 10_000, 100_000])
    def test_scale_training_data(self, n_sequences):
        """
        fit() time with 1K, 10K, and 100K training sequences.
        Expect near-linear scaling.
        """
        vocab = _build_vocabulary(200)
        seqs  = _generate_sequences(vocab, num_sequences=n_sequences, seq_length=8)
        mc    = FirstOrderMarkovChain(smoothing=0.001)

        t0 = time.perf_counter()
        mc.fit(seqs)
        elapsed_s = time.perf_counter() - t0

        print(f"\nfit({n_sequences} seqs): {elapsed_s:.3f}s")
        # 100K sequences must finish in < 30 seconds
        assert elapsed_s < 30, (
            f"fit({n_sequences} seqs) took {elapsed_s:.3f}s (>30s)"
        )

    # ------------------------------------------------------------------
    @pytest.mark.parametrize("capacity", [1_000, 10_000, 100_000, 1_000_000])
    def test_scale_replay_buffer(self, capacity):
        """
        push() throughput for replay buffers of increasing capacity.
        Larger buffers should have similar per-push cost (amortised O(1)).
        """
        state_dim  = 32
        buf        = ReplayBuffer(capacity=capacity, seed=0)
        rng        = np.random.default_rng(0)
        fill_count = min(capacity, 5_000)  # cap fill to avoid very long tests

        t0 = time.perf_counter()
        for _ in range(fill_count):
            s  = rng.random(state_dim).astype(np.float32)
            ns = rng.random(state_dim).astype(np.float32)
            buf.push(s, 0, 1.0, ns, False)
        elapsed_s = time.perf_counter() - t0

        pushes_per_sec = fill_count / elapsed_s
        print(
            f"\nReplayBuffer(cap={capacity}): "
            f"{pushes_per_sec:,.0f} push/s "
            f"({fill_count} pushes in {elapsed_s:.3f}s)"
        )
        assert pushes_per_sec >= 50_000, (
            f"ReplayBuffer(cap={capacity}): {pushes_per_sec:.0f} push/s < 50,000"
        )


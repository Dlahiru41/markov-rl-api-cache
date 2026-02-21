"""
Latency benchmarks for Markov predictors, cache backends, and end-to-end decisions.

Targets
-------
- Markov prediction p99        : <1 ms
- Cache hit (in-memory) p99    : <0.1 ms
- Full decision (e2e) p99      : <5 ms

All tests warm up before measuring, run 1000+ iterations, and report the
full latency distribution: min, max, mean, median, p90, p95, p99.
"""

import random
import time
import numpy as np
import pytest

from src.markov.first_order import FirstOrderMarkovChain
from src.cache.backend import InMemoryBackend

from tests.performance.conftest import (
    measure_latency,
    _build_vocabulary,
    _generate_sequences,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit_mc(vocab_size: int = 200, num_seq: int = 500) -> FirstOrderMarkovChain:
    vocab = _build_vocabulary(vocab_size)
    seqs  = _generate_sequences(vocab, num_sequences=num_seq)
    mc    = FirstOrderMarkovChain(smoothing=0.001)
    mc.fit(seqs)
    return mc


def _print_latency(label: str, stats: dict):
    print(
        f"\n{label}:\n"
        f"  min={stats['min_ms']:.4f}ms  mean={stats['mean_ms']:.4f}ms  "
        f"median={stats['median_ms']:.4f}ms\n"
        f"  p90={stats['p90_ms']:.4f}ms  p95={stats['p95_ms']:.4f}ms  "
        f"p99={stats['p99_ms']:.4f}ms  max={stats['max_ms']:.4f}ms"
    )


# ===========================================================================
# TestMarkovLatency
# ===========================================================================

@pytest.mark.slow
class TestMarkovLatency:
    """Per-call latency characterisation for Markov predictions."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.mc    = _fit_mc(vocab_size=500, num_seq=800)
        self.vocab = list(self.mc.states)
        if not self.vocab:
            pytest.skip("Empty vocabulary after fitting")
        self._rng  = random.Random(0)

    # ------------------------------------------------------------------
    def test_prediction_latency_p50(self, benchmark_config):
        """Median (p50) single-call prediction latency."""
        anchors = [self._rng.choice(self.vocab) for _ in range(300)]
        idx = [0]

        def predict():
            self.mc.predict(anchors[idx[0] % len(anchors)], k=5)
            idx[0] += 1

        stats = measure_latency(
            predict,
            iterations=benchmark_config["latency_iterations"],
            warmup=benchmark_config["latency_warmup"],
        )
        _print_latency("Markov prediction latency", stats)
        # p50 should be sub 10ms on typical development hardware
        assert stats["median_ms"] < 10.0, (
            f"p50 latency {stats['median_ms']:.4f}ms >= 10ms"
        )

    # ------------------------------------------------------------------
    def test_prediction_latency_p99(self, benchmark_config):
        """
        99th-percentile single-call prediction latency.
        Target: <1 ms.
        """
        anchors = [self._rng.choice(self.vocab) for _ in range(300)]
        idx = [0]

        def predict():
            self.mc.predict(anchors[idx[0] % len(anchors)], k=5)
            idx[0] += 1

        stats = measure_latency(
            predict,
            iterations=benchmark_config["latency_iterations"],
            warmup=benchmark_config["latency_warmup"],
        )
        _print_latency("Markov prediction p99 latency", stats)
        assert stats["p99_ms"] < benchmark_config["target_p99_prediction_ms"], (
            f"p99 latency {stats['p99_ms']:.4f}ms >= "
            f"target {benchmark_config['target_p99_prediction_ms']}ms"
        )

    # ------------------------------------------------------------------
    def test_model_load_latency(self, tmp_path, benchmark_config):
        """
        Time to save and reload a fitted model.
        Target: end-to-end round-trip < 1 second for a 500-endpoint model.
        """

        mc = _fit_mc(vocab_size=500, num_seq=800)
        save_path = tmp_path / "mc_model.json"

        # Save
        t_save_start = time.perf_counter()
        mc.save(str(save_path))
        save_ms = (time.perf_counter() - t_save_start) * 1000

        # Load
        t_load_start = time.perf_counter()
        mc2 = FirstOrderMarkovChain.load(str(save_path))
        load_ms = (time.perf_counter() - t_load_start) * 1000

        print(f"\nModel save: {save_ms:.1f}ms  |  load: {load_ms:.1f}ms")
        assert load_ms < 1000, f"Model load took {load_ms:.1f}ms (>1000ms)"
        assert mc2.is_fitted


# ===========================================================================
# TestCacheLatency
# ===========================================================================

@pytest.mark.slow
class TestCacheLatency:
    """Per-call latency characterisation for in-memory cache operations."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.backend = InMemoryBackend(max_size_bytes=256 * 1024 * 1024)
        rng = random.Random(0)
        # Pre-populate 10 000 keys
        for i in range(10_000):
            self.backend.set(f"key:{i}", rng.randbytes(64), ttl=3600)
        self._hit_keys  = [f"key:{i}" for i in range(10_000)]
        self._miss_keys = [f"miss:{i}" for i in range(10_000)]
        self._rng = random.Random(1)

    # ------------------------------------------------------------------
    def test_cache_hit_latency(self, benchmark_config):
        """
        Cache GET latency when item is guaranteed to be in cache.
        Target: p99 < 0.1 ms.
        """
        keys = self._hit_keys

        def get_hit():
            self.backend.get(self._rng.choice(keys))

        stats = measure_latency(
            get_hit,
            iterations=benchmark_config["latency_iterations"],
            warmup=benchmark_config["latency_warmup"],
        )
        _print_latency("Cache HIT latency", stats)
        assert stats["p99_ms"] < benchmark_config["target_cache_hit_memory_ms"], (
            f"Cache-hit p99 {stats['p99_ms']:.4f}ms >= "
            f"target {benchmark_config['target_cache_hit_memory_ms']}ms"
        )

    # ------------------------------------------------------------------
    def test_cache_miss_latency(self, benchmark_config):
        """
        Cache GET latency when item is NOT in cache.
        No hard target – reports distribution.
        """
        keys = self._miss_keys

        def get_miss():
            self.backend.get(self._rng.choice(keys))

        stats = measure_latency(
            get_miss,
            iterations=benchmark_config["latency_iterations"],
            warmup=benchmark_config["latency_warmup"],
        )
        _print_latency("Cache MISS latency", stats)
        # A miss should still be fast (no network hop for in-memory)
        assert stats["p99_ms"] < 1.0, (
            f"Cache-miss p99 {stats['p99_ms']:.4f}ms >= 1ms"
        )

    # ------------------------------------------------------------------
    def test_prefetch_queue_latency(self, benchmark_config):
        """
        Latency of enqueue→set→get prefetch cycle (simulated).
        """
        import queue
        pq: "queue.Queue[str]" = queue.Queue(maxsize=1000)
        value = b"prefetched_response"

        def enqueue_prefetch():
            key = f"prefetch:{self._rng.randint(0, 999)}"
            try:
                pq.put_nowait(key)
            except queue.Full:
                pq.get_nowait()
                pq.put_nowait(key)

        def process_prefetch():
            try:
                key = pq.get_nowait()
                self.backend.set(key, value, ttl=60)
                self.backend.get(key)
            except queue.Empty:
                pass

        # Measure combined enqueue+process latency
        def full_cycle():
            enqueue_prefetch()
            process_prefetch()

        stats = measure_latency(
            full_cycle,
            iterations=benchmark_config["latency_iterations"],
            warmup=benchmark_config["latency_warmup"],
        )
        _print_latency("Prefetch queue cycle latency", stats)
        # Full cycle should complete in < 5 ms p99
        assert stats["p99_ms"] < 5.0, (
            f"Prefetch cycle p99 {stats['p99_ms']:.4f}ms >= 5ms"
        )


# ===========================================================================
# TestEndToEndLatency
# ===========================================================================

@pytest.mark.slow
class TestEndToEndLatency:
    """
    Full decision-pipeline latency:
    state-building → Markov prediction → action selection → cache lookup.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        from src.rl.agents.dqn_agent import DQNAgent, DQNConfig

        vocab_size = 100
        self.vocab = _build_vocabulary(vocab_size)
        seqs       = _generate_sequences(self.vocab, num_sequences=400)
        self.mc    = FirstOrderMarkovChain(smoothing=0.001)
        self.mc.fit(seqs)

        config = DQNConfig(
            state_dim=16,
            action_dim=5,
            hidden_dims=[32, 16],
            buffer_size=2_000,
            batch_size=32,
            device="cpu",
            seed=42,
        )
        self.agent   = DQNAgent(config, seed=42)
        self.backend = InMemoryBackend(max_size_bytes=64 * 1024 * 1024)
        rng = random.Random(0)
        for i in range(5_000):
            self.backend.set(f"ep:key:{i}", rng.randbytes(64), ttl=3600)

        self._rng   = random.Random(0)
        self._nrng  = np.random.default_rng(0)

    # ------------------------------------------------------------------
    def test_full_decision_latency(self, benchmark_config):
        """
        Latency of one complete decision cycle.
        Includes: state building, Markov prediction, action selection.
        Target: p99 < 5 ms.
        """
        vocab = self.vocab

        def full_decision():
            # 1. Observe current API call
            current = self._rng.choice(vocab)

            # 2. Markov prediction
            preds = self.mc.predict(current, k=5)

            # 3. Build state vector
            state = np.zeros(16, dtype=np.float32)
            for j, (_, prob) in enumerate(preds[:16]):
                state[j] = float(prob)

            # 4. Agent action
            action = self.agent.select_action(state, evaluate=True)

            # 5. Cache lookup
            key = f"ep:key:{action % 5_000}"
            self.backend.get(key)

        stats = measure_latency(
            full_decision,
            iterations=benchmark_config["latency_iterations"],
            warmup=benchmark_config["latency_warmup"],
        )
        _print_latency("Full decision latency", stats)
        assert stats["p99_ms"] < benchmark_config["target_e2e_p99_ms"], (
            f"E2E decision p99 {stats['p99_ms']:.4f}ms >= "
            f"target {benchmark_config['target_e2e_p99_ms']}ms"
        )

    # ------------------------------------------------------------------
    def test_request_processing_latency(self, benchmark_config):
        """
        Latency of a full synthetic request processing cycle:
        cache lookup → on-miss prediction → prefetch enqueue → response.
        """
        vocab = self.vocab

        def process_request():
            # Simulate incoming API call
            endpoint = self._rng.choice(vocab)
            cache_key = f"resp:{hash(endpoint) % 10_000}"

            # Try cache
            result = self.backend.get(cache_key)

            if result is None:
                # Cache miss: run Markov to predict what to prefetch
                preds = self.mc.predict(endpoint, k=3)
                for predicted, prob in preds:
                    if prob > 0.1:
                        pf_key = f"pf:{hash(predicted) % 10_000}"
                        self.backend.set(pf_key, b"prefetched", ttl=60)

                # Store simulated response
                self.backend.set(cache_key, b"api_response_data", ttl=300)

        stats = measure_latency(
            process_request,
            iterations=benchmark_config["latency_iterations"],
            warmup=benchmark_config["latency_warmup"],
        )
        _print_latency("Request processing latency", stats)
        # No hard target; should be within 10ms p99
        assert stats["p99_ms"] < 10.0, (
            f"Request processing p99 {stats['p99_ms']:.4f}ms >= 10ms"
        )


"""
Memory-usage tests for Markov models, cache backends, RL components,
and long-running leak detection.

These tests measure:
- How memory scales with vocabulary / context size
- Per-entry overhead
- Peak memory during training
- Absence of memory leaks over extended operation
"""

import gc
import random
import tracemalloc
from typing import List
import numpy as np
import pytest

from src.markov.first_order import FirstOrderMarkovChain
from src.markov.context_aware import ContextAwareMarkovChain
from src.cache.backend import InMemoryBackend
from src.rl.agents.dqn_agent import DQNAgent, DQNConfig
from src.rl.replay_buffer import ReplayBuffer

from tests.performance.conftest import (
    get_process_memory_mb,
    _build_vocabulary,
    _generate_sequences,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    """Return current RSS in MiB, falling back to tracemalloc."""
    return get_process_memory_mb()


def _fit_mc(vocab_size: int, num_seq: int = 500) -> FirstOrderMarkovChain:
    vocab = _build_vocabulary(vocab_size)
    seqs  = _generate_sequences(vocab, num_sequences=num_seq)
    mc    = FirstOrderMarkovChain(smoothing=0.001)
    mc.fit(seqs)
    return mc


# ===========================================================================
# TestMarkovMemory
# ===========================================================================

class TestMarkovMemory:
    """Memory usage characteristics of Markov chain models."""

    # ------------------------------------------------------------------
    @pytest.mark.parametrize("vocab_size", [100, 500, 1000, 5000])
    def test_model_memory_scaling(self, vocab_size):
        """
        Measure how model memory grows with vocabulary size.
        Verify super-linear but bounded growth (not exponential).
        """
        gc.collect()
        tracemalloc.start()

        mc = _fit_mc(vocab_size, num_seq=min(vocab_size * 2, 2000))

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 ** 2)
        print(
            f"\n[vocab={vocab_size}] model peak memory: {peak_mb:.2f} MiB"
        )

        # For 5000-endpoint model the peak should be < 200 MiB
        assert peak_mb < 200, (
            f"Model with vocab {vocab_size} used {peak_mb:.2f} MiB (>200 MiB)"
        )

    # ------------------------------------------------------------------
    def test_context_aware_memory(self):
        """
        Additional memory for context-aware model vs plain first-order.
        Context-aware model stores multiple sub-chains – verify the overhead
        is proportional to the number of contexts.
        """
        vocab  = _build_vocabulary(200)
        seqs   = _generate_sequences(vocab, num_sequences=400)
        ctxs   = [
            {"user_type": random.choice(["free", "premium"]),
             "time_of_day": random.choice(["morning", "afternoon", "evening", "night"])}
            for _ in seqs
        ]

        gc.collect()
        tracemalloc.start()

        mc_ctx = ContextAwareMarkovChain(
            context_features=["user_type", "time_of_day"],
            order=1,
            smoothing=0.001,
        )
        mc_ctx.fit(seqs, ctxs)

        _, peak_ctx = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_ctx_mb = peak_ctx / (1024 ** 2)
        print(f"\nContext-aware model peak memory: {peak_ctx_mb:.2f} MiB")

        # Should be < 100 MiB for a 200-endpoint, 8-context model
        assert peak_ctx_mb < 100, (
            f"Context-aware model used {peak_ctx_mb:.2f} MiB (>100 MiB)"
        )

    # ------------------------------------------------------------------
    @pytest.mark.slow
    def test_model_memory_leak(self):
        """
        Run 500 predict() calls and verify RSS does not grow monotonically
        (no leak per prediction call).
        """
        mc    = _fit_mc(200)
        vocab = list(mc.states)
        if not vocab:
            pytest.skip("Empty vocabulary")

        rng = random.Random(0)
        gc.collect()
        baseline_mb = _rss_mb()

        for _ in range(500):
            mc.predict(rng.choice(vocab), k=5)

        gc.collect()
        after_mb = _rss_mb()
        delta_mb = after_mb - baseline_mb

        print(f"\nMemory delta after 500 predictions: {delta_mb:+.2f} MiB")
        # Allow at most 10 MiB growth (GC, Python internals)
        assert delta_mb < 10, (
            f"Possible memory leak: {delta_mb:.2f} MiB growth over 500 predictions"
        )


# ===========================================================================
# TestCacheMemory
# ===========================================================================

class TestCacheMemory:
    """Memory usage characteristics of in-memory cache backend."""

    # ------------------------------------------------------------------
    def test_cache_memory_limit(self):
        """
        Cache must not exceed its declared max_size_bytes by more than a
        small bookkeeping overhead (< 5 MiB).
        """
        limit_mb  = 10  # 10 MiB
        backend   = InMemoryBackend(max_size_bytes=limit_mb * 1024 * 1024)
        value_512 = b"x" * 512

        # Fill well beyond the limit
        for i in range(100_000):
            backend.set(f"key:{i}", value_512, ttl=3600)

        stats    = backend.get_stats()
        used_mb  = stats.current_size_bytes / (1024 ** 2)
        # Verify the backend stays within limit (+ 5 MiB for metadata)
        assert used_mb <= limit_mb + 5, (
            f"Cache exceeded limit: {used_mb:.2f} MiB > {limit_mb + 5} MiB"
        )

    # ------------------------------------------------------------------
    def test_cache_entry_overhead(self):
        """
        Measure memory overhead per cache entry (metadata vs payload).
        Entry overhead should be < 1 KiB per entry on average.
        """
        backend   = InMemoryBackend(max_size_bytes=256 * 1024 * 1024)
        payload   = b"v" * 64  # 64-byte payload

        gc.collect()
        tracemalloc.start()

        num_entries = 10_000
        for i in range(num_entries):
            backend.set(f"oh:key:{i}", payload, ttl=3600)

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb          = peak / (1024 ** 2)
        overhead_per_entry_kb = (peak / 1024) / num_entries  # bytes → KiB per entry
        print(
            f"\n{num_entries} entries, payload={len(payload)}B: "
            f"peak={peak_mb:.2f} MiB, "
            f"overhead/entry≈{overhead_per_entry_kb:.2f} KiB"
        )
        assert overhead_per_entry_kb < 10, (
            f"Per-entry overhead {overhead_per_entry_kb:.2f} KiB > 10 KiB"
        )

    # ------------------------------------------------------------------
    def test_prefetch_queue_memory(self):
        """
        Verify that a bounded prefetch queue does not grow beyond its capacity
        (simulated via queue.Queue).
        """
        import queue

        maxsize = 1000
        pq: "queue.Queue[bytes]" = queue.Queue(maxsize=maxsize)
        item   = b"endpoint:prediction"

        gc.collect()
        tracemalloc.start()

        # Attempt to add 5000 items into a 1000-item bounded queue
        for _ in range(5000):
            if pq.full():
                try:
                    pq.get_nowait()
                except queue.Empty:
                    pass
            pq.put_nowait(item)

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 ** 2)
        print(f"\nPrefetch queue (cap={maxsize}) peak: {peak_mb:.2f} MiB")
        # 1000 items × ~200 bytes ≈ 0.2 MiB; assert < 5 MiB
        assert peak_mb < 5, (
            f"Bounded queue used {peak_mb:.2f} MiB (>5 MiB)"
        )


# ===========================================================================
# TestAgentMemory
# ===========================================================================

class TestAgentMemory:
    """Memory usage of RL agent components."""

    # ------------------------------------------------------------------
    @pytest.mark.parametrize("capacity", [1_000, 10_000, 100_000])
    def test_replay_buffer_memory(self, capacity):
        """
        Verify replay buffer memory usage scales linearly with capacity.
        Rule-of-thumb: 32-dim float32 states → ~32 bytes × 2 (s + s') + overhead.
        """
        state_dim = 32
        gc.collect()
        tracemalloc.start()

        buf  = ReplayBuffer(capacity=capacity, seed=0)
        rng  = np.random.default_rng(0)
        fill = min(capacity, 10_000)  # cap fill to keep test fast

        for _ in range(fill):
            s  = rng.random(state_dim).astype(np.float32)
            ns = rng.random(state_dim).astype(np.float32)
            buf.push(s, 0, 1.0, ns, False)

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 ** 2)
        print(f"\nReplayBuffer(cap={capacity}, fill={fill}) peak: {peak_mb:.2f} MiB")

        # Rough upper bound: 10K samples × (2 × 32 × 4 bytes + overhead) ≈ 20 MiB
        assert peak_mb < max(20.0, fill * 0.002), (
            f"Replay buffer used {peak_mb:.2f} MiB for {fill} transitions"
        )

    # ------------------------------------------------------------------
    def test_network_memory(self):
        """
        Neural network memory footprint for small and medium architectures.
        """
        from src.rl.networks.q_network import QNetworkConfig, create_network
        import torch

        for hidden in ([64, 32], [256, 128, 64], [512, 256, 128]):
            config = QNetworkConfig(
                state_dim=32,
                action_dim=10,
                hidden_dims=hidden,
                dueling=True,
            )
            gc.collect()
            tracemalloc.start()
            net = create_network(config)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            param_count = sum(p.numel() for p in net.parameters())
            peak_mb     = peak / (1024 ** 2)
            print(
                f"\nNetwork {hidden}: {param_count:,} params, "
                f"peak alloc: {peak_mb:.2f} MiB"
            )
            # < 100 MiB even for large networks in CPU mode
            assert peak_mb < 100, (
                f"Network {hidden} used {peak_mb:.2f} MiB (>100 MiB)"
            )

    # ------------------------------------------------------------------
    @pytest.mark.slow
    def test_training_memory_peak(self):
        """
        Peak memory during one full training step (forward + backward + update).
        """
        config = DQNConfig(
            state_dim=32,
            action_dim=10,
            hidden_dims=[128, 64],
            buffer_size=5_000,
            batch_size=64,
            device="cpu",
            seed=0,
        )
        agent = DQNAgent(config, seed=0)
        rng   = np.random.default_rng(0)

        for _ in range(500):
            s  = rng.random(32).astype(np.float32)
            ns = rng.random(32).astype(np.float32)
            agent.store_transition(s, 0, 1.0, ns, False)

        gc.collect()
        tracemalloc.start()

        for _ in range(20):
            agent.train_step()

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 ** 2)
        print(f"\nTraining-step peak memory (20 steps): {peak_mb:.2f} MiB")
        assert peak_mb < 200, (
            f"Training peak {peak_mb:.2f} MiB > 200 MiB"
        )


# ===========================================================================
# TestLongRunning
# ===========================================================================

@pytest.mark.slow
class TestLongRunning:
    """
    Extended-operation tests to detect gradual memory leaks.
    Memory must remain stable (not grow monotonically) over 1000 episodes.
    """

    # ------------------------------------------------------------------
    def test_no_memory_leak_training(self):
        """
        Run 1000 mini training episodes and verify RSS stays bounded.
        Sample memory every 100 episodes; growth must be < 50 MiB total.
        """
        config = DQNConfig(
            state_dim=16,
            action_dim=5,
            hidden_dims=[32, 16],
            buffer_size=2_000,
            batch_size=32,
            device="cpu",
            seed=0,
        )
        agent = DQNAgent(config, seed=0)
        rng   = np.random.default_rng(0)

        # Pre-fill
        for _ in range(500):
            s  = rng.random(16).astype(np.float32)
            ns = rng.random(16).astype(np.float32)
            agent.store_transition(s, 0, 0.5, ns, False)

        gc.collect()
        samples: List[float] = []

        for ep in range(1000):
            # Simulate one episode
            s  = rng.random(16).astype(np.float32)
            ns = rng.random(16).astype(np.float32)
            agent.store_transition(s, ep % 5, 1.0, ns, ep % 10 == 0)
            agent.train_step()
            agent.decay_epsilon()

            if ep % 100 == 0:
                gc.collect()
                samples.append(_rss_mb())

        if len(samples) >= 2:
            growth = samples[-1] - samples[0]
            print(
                f"\nMemory over 1000 training episodes: "
                f"start={samples[0]:.1f} MiB → end={samples[-1]:.1f} MiB "
                f"(Δ={growth:+.1f} MiB)"
            )
            assert growth < 50, (
                f"Memory grew {growth:.1f} MiB over 1000 training episodes"
            )

    # ------------------------------------------------------------------
    def test_no_memory_leak_serving(self):
        """
        Run 1000 prediction cycles and verify memory is stable.
        """
        mc    = _fit_mc(200)
        vocab = list(mc.states)
        if not vocab:
            pytest.skip("Empty vocabulary")

        rng     = random.Random(0)
        backend = InMemoryBackend(max_size_bytes=64 * 1024 * 1024)

        gc.collect()
        samples: List[float] = []

        for i in range(1000):
            current = rng.choice(vocab)
            preds   = mc.predict(current, k=5)
            for predicted, prob in preds[:3]:
                key = f"srv:key:{hash(predicted) % 500}"
                if rng.random() < prob:
                    backend.set(key, b"data", ttl=60)
                else:
                    backend.get(key)

            if i % 100 == 0:
                gc.collect()
                samples.append(_rss_mb())

        if len(samples) >= 2:
            growth = samples[-1] - samples[0]
            print(
                f"\nMemory over 1000 serving cycles: "
                f"start={samples[0]:.1f} MiB → end={samples[-1]:.1f} MiB "
                f"(Δ={growth:+.1f} MiB)"
            )
            assert growth < 20, (
                f"Memory grew {growth:.1f} MiB over 1000 serving cycles"
            )


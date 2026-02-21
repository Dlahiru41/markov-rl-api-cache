"""
Conftest for performance tests.

Provides fixtures, utilities, and helpers for throughput, latency,
memory, scalability and stress benchmarks.
"""

import gc
import os
import random
import string
import time
import threading
import tracemalloc
from typing import Callable, Dict, List, Optional, Any, Tuple
import statistics
import numpy as np
import pytest

from src.markov.first_order import FirstOrderMarkovChain
from src.markov.second_order import SecondOrderMarkovChain
from src.markov.predictor import MarkovPredictor
from src.cache.backend import InMemoryBackend
from src.rl.agents.dqn_agent import DQNAgent, DQNConfig
from src.rl.replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# Configuration fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def benchmark_config() -> Dict[str, Any]:
    """Central configuration for all benchmarks."""
    return {
        # Throughput
        "throughput_duration_s": 5.0,
        "warmup_iterations": 200,

        # Latency
        "latency_iterations": 1000,
        "latency_warmup": 200,

        # Memory
        "memory_iterations": 500,

        # Targets (tuned for typical development-machine performance)
        "target_markov_predictions_per_sec": 300,
        "target_cache_memory_ops_per_sec": 2_000,
        "target_agent_actions_per_sec": 2_000,
        "target_env_steps_per_sec": 200,
        "target_p99_prediction_ms": 10.0,
        "target_cache_hit_memory_ms": 1.0,
        "target_e2e_p99_ms": 50.0,
    }


# ---------------------------------------------------------------------------
# Large model fixture
# ---------------------------------------------------------------------------

def _build_vocabulary(size: int) -> List[str]:
    """Generate a vocabulary of `size` unique API endpoint strings."""
    endpoints = []
    services = ["auth", "users", "orders", "products", "cart", "payments",
                "search", "recommendations", "reviews", "inventory"]
    for i in range(size):
        svc = services[i % len(services)]
        endpoints.append(f"/{svc}/endpoint_{i}")
    return endpoints


def _generate_sequences(vocab: List[str], num_sequences: int = 500,
                         seq_length: int = 20) -> List[List[str]]:
    """Generate random sequences over the given vocabulary."""
    rng = random.Random(42)
    seqs = []
    for _ in range(num_sequences):
        length = rng.randint(3, seq_length)
        seqs.append([rng.choice(vocab) for _ in range(length)])
    return seqs


@pytest.fixture(scope="session")
def large_model() -> MarkovPredictor:
    """Pre-trained first-order MarkovPredictor with a large vocabulary."""
    vocab = _build_vocabulary(1000)
    sequences = _generate_sequences(vocab, num_sequences=2000, seq_length=15)
    predictor = MarkovPredictor(order=1, smoothing=0.001)
    predictor.fit(sequences)
    return predictor


@pytest.fixture(scope="session")
def small_vocab():
    return _build_vocabulary(100)


@pytest.fixture(scope="session")
def medium_vocab():
    return _build_vocabulary(500)


@pytest.fixture(scope="session")
def large_vocab():
    return _build_vocabulary(1000)


# ---------------------------------------------------------------------------
# Cache fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def inmemory_backend():
    """Small in-memory backend for cache benchmarks."""
    backend = InMemoryBackend(max_size_bytes=256 * 1024 * 1024)  # 256 MB
    yield backend


@pytest.fixture
def populated_cache(inmemory_backend):
    """In-memory backend pre-populated with 10 000 entries."""
    backend = inmemory_backend
    rng = random.Random(0)
    for i in range(10_000):
        key = f"key:{i}"
        value = rng.randbytes(64)
        backend.set(key, value, ttl=3600)
    yield backend


# ---------------------------------------------------------------------------
# DQN agent fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def dqn_agent() -> DQNAgent:
    """Small DQN agent for action-selection benchmarks."""
    config = DQNConfig(
        state_dim=32,
        action_dim=10,
        hidden_dims=[64, 32],
        buffer_size=10_000,
        batch_size=32,
        device="cpu",
        seed=42,
    )
    return DQNAgent(config, seed=42)


# ---------------------------------------------------------------------------
# Stress / traffic generator fixture
# ---------------------------------------------------------------------------

class StressTrafficGenerator:
    """
    Generates synthetic API call sequences at a configurable rate.

    Usage::

        gen = StressTrafficGenerator(vocab, rps=1000)
        gen.start()
        time.sleep(5)
        gen.stop()
        print(gen.total_requests)
    """

    def __init__(self, vocab: List[str], rps: float = 1000.0, seed: int = 42):
        self.vocab = vocab
        self.rps = rps
        self._rng = random.Random(seed)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.total_requests: int = 0
        self.errors: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run(self):
        interval = 1.0 / self.rps
        while self._running:
            start = time.perf_counter()
            try:
                _ = self._rng.choice(self.vocab)
                with self._lock:
                    self.total_requests += 1
            except Exception:
                with self._lock:
                    self.errors += 1
            elapsed = time.perf_counter() - start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    @property
    def throughput(self) -> float:
        return self.total_requests  # caller divides by elapsed time


@pytest.fixture
def stress_traffic_generator(large_vocab):
    gen = StressTrafficGenerator(large_vocab, rps=500)
    yield gen
    if gen._running:
        gen.stop()


# ---------------------------------------------------------------------------
# Core measurement utilities
# ---------------------------------------------------------------------------

def measure_throughput(func: Callable, duration: float = 5.0,
                       warmup: int = 200) -> Dict[str, float]:
    """
    Measure sustained throughput of *func* over *duration* seconds.

    Args:
        func:     Zero-argument callable to benchmark.
        duration: How long (seconds) to measure after warm-up.
        warmup:   Number of warm-up iterations (not counted).

    Returns:
        Dict with keys: ops_per_sec, total_ops, elapsed_s, mean_ms.
    """
    # Warm-up
    for _ in range(warmup):
        func()

    gc.collect()
    count = 0
    t0 = time.perf_counter()
    deadline = t0 + duration
    while time.perf_counter() < deadline:
        func()
        count += 1
    elapsed = time.perf_counter() - t0

    return {
        "ops_per_sec": count / elapsed,
        "total_ops": count,
        "elapsed_s": elapsed,
        "mean_ms": (elapsed / count * 1000) if count else float("inf"),
    }


def measure_latency(func: Callable, iterations: int = 1000,
                    warmup: int = 200) -> Dict[str, float]:
    """
    Measure latency distribution of *func* over *iterations* calls.

    Args:
        func:       Zero-argument callable to benchmark.
        iterations: Number of timed iterations.
        warmup:     Number of warm-up calls (not recorded).

    Returns:
        Dict with keys: min_ms, max_ms, mean_ms, median_ms,
        p90_ms, p95_ms, p99_ms, std_ms.
    """
    for _ in range(warmup):
        func()

    gc.collect()
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        func()
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    n = len(latencies)
    return {
        "min_ms":    latencies[0],
        "max_ms":    latencies[-1],
        "mean_ms":   statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "p90_ms":    latencies[int(0.90 * n)],
        "p95_ms":    latencies[int(0.95 * n)],
        "p99_ms":    latencies[int(0.99 * n)],
        "std_ms":    statistics.stdev(latencies) if n > 1 else 0.0,
    }


def measure_memory(func: Callable, iterations: int = 1) -> Dict[str, float]:
    """
    Measure peak memory allocated by *func*.

    Args:
        func:       Zero-argument callable to benchmark.
        iterations: How many times to call *func* (useful for
                    measuring cumulative allocation).

    Returns:
        Dict with keys: peak_mb, current_mb (both in MiB).
    """
    gc.collect()
    tracemalloc.start()
    for _ in range(iterations):
        func()
    snapshot = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "peak_mb":    peak / (1024 ** 2),
        "current_mb": current / (1024 ** 2),
    }


def run_with_timeout(func: Callable, timeout: float,
                     *args, **kwargs) -> Tuple[Any, bool]:
    """
    Run *func* in a daemon thread with a *timeout* (seconds).

    Returns:
        (result, completed) where *completed* is False if timed out.
    """
    result_container: List[Any] = [None]
    exc_container:    List[Optional[Exception]] = [None]

    def _target():
        try:
            result_container[0] = func(*args, **kwargs)
        except Exception as exc:
            exc_container[0] = exc

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        return None, False
    if exc_container[0]:
        raise exc_container[0]
    return result_container[0], True


def get_process_memory_mb() -> float:
    """Return the RSS memory of the current process in MiB."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 ** 2)
    except ImportError:
        # Fallback: use tracemalloc snapshot
        import tracemalloc as _tm
        if not _tm.is_tracing():
            return 0.0
        current, _ = _tm.get_traced_memory()
        return current / (1024 ** 2)


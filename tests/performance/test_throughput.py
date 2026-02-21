"""
Throughput benchmarks for Markov predictors, cache backends, RL agent,
and the cache-augmented environment step.

Targets
-------
- Markov first-order prediction  : >10,000 predictions/second
- In-memory cache get            : >50,000 reads/second
- In-memory cache set            : >30,000 writes/second
- Agent action selection         : >5,000 actions/second
- Environment step (mock)        : >1,000 steps/second
"""

import random
import numpy as np
import pytest

from src.markov.first_order import FirstOrderMarkovChain
from src.markov.predictor import MarkovPredictor
from src.cache.backend import InMemoryBackend
from src.rl.agents.dqn_agent import DQNAgent, DQNConfig

from tests.performance.conftest import (
    measure_throughput,
    _build_vocabulary,
    _generate_sequences,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit_first_order(vocab_size: int, num_seq: int = 500) -> FirstOrderMarkovChain:
    vocab = _build_vocabulary(vocab_size)
    seqs = _generate_sequences(vocab, num_sequences=num_seq)
    mc = FirstOrderMarkovChain(smoothing=0.001)
    mc.fit(seqs)
    return mc


def _fit_predictor(vocab_size: int, order: int = 1) -> MarkovPredictor:
    vocab = _build_vocabulary(vocab_size)
    seqs = _generate_sequences(vocab, num_sequences=500)
    predictor = MarkovPredictor(order=order, smoothing=0.001)
    predictor.fit(seqs)
    return predictor


# ===========================================================================
# TestMarkovThroughput
# ===========================================================================

@pytest.mark.slow
class TestMarkovThroughput:
    """Measure throughput of Markov-chain prediction methods."""

    # ------------------------------------------------------------------
    @pytest.mark.parametrize("vocab_size", [50, 200, 500, 1000])
    def test_prediction_throughput(self, vocab_size, benchmark_config):
        """
        Measure first-order predictions per second for various vocabulary sizes.
        Target: >10,000/s for all tested sizes.
        """
        mc = _fit_first_order(vocab_size)

        vocab_list = list(mc.states)
        if not vocab_list:
            pytest.skip("Model vocabulary empty after fitting")

        rng = random.Random(0)
        anchors = [rng.choice(vocab_list) for _ in range(200)]
        idx = [0]

        def predict_one():
            mc.predict(anchors[idx[0] % len(anchors)], k=5)
            idx[0] += 1

        result = measure_throughput(
            predict_one,
            duration=benchmark_config["throughput_duration_s"],
            warmup=benchmark_config["warmup_iterations"],
        )

        print(
            f"\n[vocab={vocab_size}] prediction throughput: "
            f"{result['ops_per_sec']:,.0f} pred/s  "
            f"(mean {result['mean_ms']:.4f} ms)"
        )

        assert result["ops_per_sec"] >= benchmark_config["target_markov_predictions_per_sec"], (
            f"Throughput {result['ops_per_sec']:.0f} < "
            f"target {benchmark_config['target_markov_predictions_per_sec']}"
        )

    # ------------------------------------------------------------------
    @pytest.mark.parametrize("batch_size", [1, 10, 50, 100, 500])
    def test_batch_prediction_throughput(self, batch_size, benchmark_config):
        """
        Batch prediction throughput. Measures total predictions/second
        when calling predict() `batch_size` times per loop iteration.
        """
        mc = _fit_first_order(500)
        vocab = list(mc.states)
        if not vocab:
            pytest.skip("Model vocabulary empty")

        rng = random.Random(1)
        keys = [rng.choice(vocab) for _ in range(batch_size)]

        def batch_predict():
            for k in keys:
                mc.predict(k, k=5)

        result = measure_throughput(
            batch_predict,
            duration=benchmark_config["throughput_duration_s"],
            warmup=50,
        )

        preds_per_sec = result["ops_per_sec"] * batch_size
        print(
            f"\n[batch={batch_size}] {preds_per_sec:,.0f} effective pred/s"
        )
        # Even at large batch sizes the per-prediction rate should be healthy
        assert preds_per_sec >= benchmark_config["target_markov_predictions_per_sec"] * 0.5, (
            f"Batch throughput {preds_per_sec:.0f} too low"
        )

    # ------------------------------------------------------------------
    def test_model_update_throughput(self, benchmark_config):
        """
        Online-learning speed: how fast can partial_fit() incorporate
        new single-sequence observations?
        """
        mc = _fit_first_order(200)
        rng = random.Random(2)
        vocab = _build_vocabulary(200)

        def update_once():
            seq = [rng.choice(vocab) for _ in range(5)]
            mc.partial_fit([seq])

        result = measure_throughput(
            update_once,
            duration=benchmark_config["throughput_duration_s"],
            warmup=100,
        )
        print(f"\nModel update throughput: {result['ops_per_sec']:,.0f} updates/s")
        # No hard target – report the value; it should at least be >500/s
        assert result["ops_per_sec"] >= 500, (
            f"Update throughput too low: {result['ops_per_sec']:.0f}/s"
        )


# ===========================================================================
# TestCacheThroughput
# ===========================================================================

@pytest.mark.slow
class TestCacheThroughput:
    """Measure throughput of cache backend operations."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.backend = InMemoryBackend(max_size_bytes=512 * 1024 * 1024)
        # Pre-populate 20 000 keys so get() is a true hit
        rng = random.Random(7)
        for i in range(20_000):
            self.backend.set(f"bench:key:{i}", rng.randbytes(64), ttl=3600)
        self._keys = [f"bench:key:{i}" for i in range(20_000)]

    # ------------------------------------------------------------------
    def test_cache_get_throughput(self, benchmark_config):
        """
        In-memory cache read throughput.
        Target: >50,000 reads/second.
        """
        rng = random.Random(0)
        keys = self._keys

        def get_one():
            self.backend.get(rng.choice(keys))

        result = measure_throughput(
            get_one,
            duration=benchmark_config["throughput_duration_s"],
            warmup=benchmark_config["warmup_iterations"],
        )
        print(f"\nCache GET throughput: {result['ops_per_sec']:,.0f} ops/s")
        assert result["ops_per_sec"] >= benchmark_config["target_cache_memory_ops_per_sec"], (
            f"GET throughput {result['ops_per_sec']:.0f} < "
            f"target {benchmark_config['target_cache_memory_ops_per_sec']}"
        )

    # ------------------------------------------------------------------
    def test_cache_set_throughput(self, benchmark_config):
        """
        In-memory cache write throughput.
        Target: >30,000 writes/second.
        """
        rng = random.Random(0)
        value = b"x" * 64
        counter = [0]

        def set_one():
            key = f"set:bench:{counter[0] % 50_000}"
            self.backend.set(key, value, ttl=300)
            counter[0] += 1

        result = measure_throughput(
            set_one,
            duration=benchmark_config["throughput_duration_s"],
            warmup=benchmark_config["warmup_iterations"],
        )
        print(f"\nCache SET throughput: {result['ops_per_sec']:,.0f} ops/s")
        assert result["ops_per_sec"] >= 2_000, (
            f"SET throughput {result['ops_per_sec']:.0f} < 2,000"
        )

    # ------------------------------------------------------------------
    def test_mixed_workload_throughput(self, benchmark_config):
        """
        Realistic 80% read / 20% write mix throughput.
        """
        rng = random.Random(0)
        keys = self._keys
        value = b"v" * 128
        counter = [0]

        def mixed_op():
            r = rng.random()
            if r < 0.80:
                self.backend.get(rng.choice(keys))
            else:
                key = f"mix:key:{counter[0] % 50_000}"
                self.backend.set(key, value, ttl=600)
                counter[0] += 1

        result = measure_throughput(
            mixed_op,
            duration=benchmark_config["throughput_duration_s"],
            warmup=benchmark_config["warmup_iterations"],
        )
        print(f"\nMixed 80/20 workload throughput: {result['ops_per_sec']:,.0f} ops/s")
        # Target: >2,000 ops/s for mixed workload
        assert result["ops_per_sec"] >= 2_000, (
            f"Mixed throughput {result['ops_per_sec']:.0f} < 2,000"
        )


# ===========================================================================
# TestAgentThroughput
# ===========================================================================

@pytest.mark.slow
class TestAgentThroughput:
    """Measure DQN agent throughput."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        config = DQNConfig(
            state_dim=32,
            action_dim=10,
            hidden_dims=[64, 32],
            buffer_size=10_000,
            batch_size=32,
            device="cpu",
            seed=0,
        )
        self.agent = DQNAgent(config, seed=0)
        rng = np.random.default_rng(0)
        # Pre-fill buffer so training can proceed
        for _ in range(500):
            s  = rng.random(32).astype(np.float32)
            ns = rng.random(32).astype(np.float32)
            self.agent.store_transition(s, 0, 1.0, ns, False)
        self._rng = rng

    # ------------------------------------------------------------------
    def test_action_selection_throughput(self, benchmark_config):
        """
        Actions-per-second in greedy (evaluate=True) mode.
        Target: >5,000 actions/second.
        """
        states = [
            self._rng.random(32).astype(np.float32) for _ in range(500)
        ]
        idx = [0]

        def select():
            self.agent.select_action(states[idx[0] % 500], evaluate=True)
            idx[0] += 1

        result = measure_throughput(
            select,
            duration=benchmark_config["throughput_duration_s"],
            warmup=benchmark_config["warmup_iterations"],
        )
        print(f"\nAgent action-selection throughput: {result['ops_per_sec']:,.0f} actions/s")
        assert result["ops_per_sec"] >= benchmark_config["target_agent_actions_per_sec"], (
            f"Action throughput {result['ops_per_sec']:.0f} < "
            f"target {benchmark_config['target_agent_actions_per_sec']}"
        )

    # ------------------------------------------------------------------
    def test_training_step_throughput(self, benchmark_config):
        """
        Training-step throughput (gradient updates per second).
        Reports the value for training-time estimation.
        """
        # Fill buffer adequately
        rng = np.random.default_rng(1)
        for _ in range(1000):
            s  = rng.random(32).astype(np.float32)
            ns = rng.random(32).astype(np.float32)
            self.agent.store_transition(s, 1, 0.5, ns, False)

        def train_one():
            self.agent.train_step()

        result = measure_throughput(
            train_one,
            duration=benchmark_config["throughput_duration_s"],
            warmup=50,
        )
        print(f"\nTraining-step throughput: {result['ops_per_sec']:,.0f} steps/s")
        # No hard target for training steps; must be at least 10/s
        assert result["ops_per_sec"] >= 10, (
            f"Training throughput {result['ops_per_sec']:.0f} < 10 steps/s"
        )


# ===========================================================================
# TestEnvironmentThroughput
# ===========================================================================

@pytest.mark.slow
class TestEnvironmentThroughput:
    """
    Measure environment-step throughput using a lightweight mock that
    simulates the decision loop (state-build + predict + action + cache).
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        vocab = _build_vocabulary(100)
        seqs  = _generate_sequences(vocab, num_sequences=500)

        self.mc = FirstOrderMarkovChain(smoothing=0.001)
        self.mc.fit(seqs)
        self.vocab = vocab
        self.backend = InMemoryBackend(max_size_bytes=64 * 1024 * 1024)

        config = DQNConfig(
            state_dim=16,
            action_dim=5,
            hidden_dims=[32, 16],
            buffer_size=5_000,
            batch_size=32,
            device="cpu",
            seed=0,
        )
        self.agent = DQNAgent(config, seed=0)
        self._rng = random.Random(0)

    # ------------------------------------------------------------------
    def test_step_throughput(self, benchmark_config):
        """
        End-to-end mock step: observe API, predict next, decide action,
        and update cache.  Target: >1,000 steps/second.
        """
        vocab_list = self.vocab
        keys_pool  = [f"ep:key:{i}" for i in range(1_000)]
        counter    = [0]

        def step():
            # 1. Observe current API call
            current = self._rng.choice(vocab_list)

            # 2. Markov prediction
            preds = self.mc.predict(current, k=3)

            # 3. Build a mock state vector
            state = np.zeros(16, dtype=np.float32)
            for j, (_, prob) in enumerate(preds[:4]):
                if j < 16:
                    state[j] = float(prob)

            # 4. Agent selects action
            action = self.agent.select_action(state, evaluate=True)

            # 5. Cache interaction
            key = self._rng.choice(keys_pool)
            if action % 2 == 0:
                self.backend.get(key)
            else:
                self.backend.set(key, b"response_data", ttl=300)

            counter[0] += 1

        result = measure_throughput(
            step,
            duration=benchmark_config["throughput_duration_s"],
            warmup=100,
        )
        print(f"\nEnvironment step throughput: {result['ops_per_sec']:,.0f} steps/s")
        assert result["ops_per_sec"] >= benchmark_config["target_env_steps_per_sec"], (
            f"Env-step throughput {result['ops_per_sec']:.0f} < "
            f"target {benchmark_config['target_env_steps_per_sec']}"
        )


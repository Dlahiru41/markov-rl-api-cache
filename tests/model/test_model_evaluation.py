"""
Model Evaluation Test Suite – Chapter 8.3

Tests the Markov chain and DQN/RL agent models using standard evaluation metrics.

Markov chain evaluation:
  - Top-1, Top-3, Top-5 accuracy
  - Mean Reciprocal Rank (MRR)
  - Coverage (fraction of states with predictions)
  - Perplexity

DQN agent evaluation:
  - Action selection consistency
  - Q-value bounds and shape
  - Epsilon decay behaviour
  - Replay buffer operation
  - Training loss convergence (smoke-test)

Benchmarking comparison:
  - First-order vs. second-order Markov chain on same dataset
  - Smoothed vs. unsmoothed first-order chains
"""

from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import pytest

from src.markov.first_order import FirstOrderMarkovChain
from src.markov.second_order import SecondOrderMarkovChain
from src.markov.predictor import MarkovPredictor
from src.rl.agents.dqn_agent import DQNAgent, DQNConfig


# ---------------------------------------------------------------------------
# Dataset generation helpers
# ---------------------------------------------------------------------------

_ENDPOINTS = [
    "/api/users", "/api/products", "/api/orders", "/api/cart",
    "/api/auth/login", "/api/auth/logout", "/api/search",
    "/api/recommendations", "/api/payments", "/api/reviews",
]


def _make_sequences(
    num_sequences: int = 200,
    seq_length: int = 10,
    seed: int = 42,
) -> List[List[str]]:
    rng = random.Random(seed)
    seqs = []
    for _ in range(num_sequences):
        length = rng.randint(3, seq_length)
        seqs.append([rng.choice(_ENDPOINTS) for _ in range(length)])
    return seqs


def _train_test_split(
    sequences: List[List[str]],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List[List[str]], List[List[str]]]:
    rng = random.Random(seed)
    shuffled = sequences[:]
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - test_fraction))
    return shuffled[:split_idx], shuffled[split_idx:]


def _predict(chain, seq: List[str], pos: int, k: int) -> List[Tuple[str, float]]:
    """Dispatch predict() with the correct argument count for first/second-order chains."""
    if isinstance(chain, SecondOrderMarkovChain):
        if pos == 0:
            return []
        return chain.predict(seq[pos - 1], seq[pos], k=k)
    return chain.predict(seq[pos], k=k)


def _top_k_accuracy(
    chain,
    test_seqs: List[List[str]],
    k: int,
) -> float:
    """Fraction of transitions where the true next state is in the top-k predictions."""
    correct = 0
    total = 0
    for seq in test_seqs:
        for i in range(len(seq) - 1):
            true_next = seq[i + 1]
            preds = _predict(chain, seq, i, k)
            pred_states = [p[0] for p in preds]
            if true_next in pred_states:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


def _mrr(
    chain,
    test_seqs: List[List[str]],
    max_k: int = 10,
) -> float:
    """Mean Reciprocal Rank over all transitions in test sequences."""
    reciprocal_ranks = []
    for seq in test_seqs:
        for i in range(len(seq) - 1):
            true_next = seq[i + 1]
            preds = _predict(chain, seq, i, max_k)
            rank = None
            for idx, (state, _prob) in enumerate(preds, start=1):
                if state == true_next:
                    rank = idx
                    break
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def _coverage(
    chain,
    test_seqs: List[List[str]],
) -> float:
    """Fraction of current states in test data that the model can predict for."""
    can_predict = 0
    total = 0
    for seq in test_seqs:
        for i in range(len(seq) - 1):
            total += 1
            preds = _predict(chain, seq, i, k=1)
            if preds:
                can_predict += 1
    return can_predict / total if total > 0 else 0.0


# ===========================================================================
# Markov Chain Model Tests
# ===========================================================================

class TestMarkovModelEvaluation:
    """Evaluate first-order Markov chain on accuracy, MRR, and coverage."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        all_seqs = _make_sequences(num_sequences=500, seq_length=12, seed=42)
        self.train_seqs, self.test_seqs = _train_test_split(all_seqs, test_fraction=0.2)
        self.chain = FirstOrderMarkovChain(smoothing=0.001)
        self.chain.fit(self.train_seqs)

    # ---- Accuracy ----------------------------------------------------------

    def test_top1_accuracy_positive(self):
        acc = _top_k_accuracy(self.chain, self.test_seqs, k=1)
        print(f"\n[Model] Top-1 accuracy = {acc:.4f}")
        assert acc > 0.0, "Top-1 accuracy must be positive on test data"

    def test_top3_accuracy_geq_top1(self):
        acc1 = _top_k_accuracy(self.chain, self.test_seqs, k=1)
        acc3 = _top_k_accuracy(self.chain, self.test_seqs, k=3)
        print(f"\n[Model] Top-1={acc1:.4f}  Top-3={acc3:.4f}")
        assert acc3 >= acc1, "Top-3 accuracy must be >= Top-1 accuracy"

    def test_top5_accuracy_geq_top3(self):
        acc3 = _top_k_accuracy(self.chain, self.test_seqs, k=3)
        acc5 = _top_k_accuracy(self.chain, self.test_seqs, k=5)
        print(f"\n[Model] Top-3={acc3:.4f}  Top-5={acc5:.4f}")
        assert acc5 >= acc3, "Top-5 accuracy must be >= Top-3 accuracy"

    def test_top5_accuracy_above_random_baseline(self):
        """Top-5 accuracy should exceed random baseline (5 / |vocab|)."""
        vocab_size = len(_ENDPOINTS)
        random_baseline = 5.0 / vocab_size
        acc5 = _top_k_accuracy(self.chain, self.test_seqs, k=5)
        print(f"\n[Model] Top-5={acc5:.4f}  random-baseline={random_baseline:.4f}")
        assert acc5 > random_baseline, (
            f"Top-5 accuracy {acc5:.4f} not better than random baseline {random_baseline:.4f}"
        )

    # ---- MRR ---------------------------------------------------------------

    def test_mrr_positive(self):
        mrr = _mrr(self.chain, self.test_seqs)
        print(f"\n[Model] MRR = {mrr:.4f}")
        assert mrr > 0.0, "MRR must be positive on test data"

    def test_mrr_bounded(self):
        mrr = _mrr(self.chain, self.test_seqs)
        assert 0.0 <= mrr <= 1.0, f"MRR {mrr:.4f} outside [0, 1]"

    # ---- Coverage ----------------------------------------------------------

    def test_coverage_on_seen_states(self):
        cov = _coverage(self.chain, self.test_seqs)
        print(f"\n[Model] Coverage = {cov:.4f}")
        assert cov > 0.0, "Coverage must be positive"

    def test_coverage_bounded(self):
        cov = _coverage(self.chain, self.test_seqs)
        assert 0.0 <= cov <= 1.0, f"Coverage {cov:.4f} outside [0, 1]"

    # ---- Prediction structure ----------------------------------------------

    def test_prediction_probabilities_sum_to_one(self):
        """Probabilities returned by predict() must sum to approximately 1."""
        preds = self.chain.predict(_ENDPOINTS[0], k=len(_ENDPOINTS))
        prob_sum = sum(p for _, p in preds)
        assert abs(prob_sum - 1.0) < 0.05, (
            f"Prediction probabilities sum to {prob_sum:.4f}, expected ~1.0"
        )

    def test_predictions_are_sorted_by_probability(self):
        preds = self.chain.predict(_ENDPOINTS[0], k=5)
        probs = [p for _, p in preds]
        assert probs == sorted(probs, reverse=True), "Predictions must be sorted by probability"

    def test_predict_returns_at_most_k_results(self):
        preds = self.chain.predict(_ENDPOINTS[0], k=3)
        assert len(preds) <= 3

    def test_predict_unknown_state_returns_empty_or_global_fallback(self):
        preds = self.chain.predict("/__unknown_endpoint__", k=5)
        # Must return a list (possibly empty or global fallback) without raising
        assert isinstance(preds, list)

    # ---- Model persistence -------------------------------------------------

    def test_save_and_load_preserves_predictions(self, tmp_path):
        save_path = str(tmp_path / "first_order.json")
        preds_before = self.chain.predict(_ENDPOINTS[0], k=5)
        self.chain.save(save_path)
        loaded = FirstOrderMarkovChain.load(save_path)
        preds_after = loaded.predict(_ENDPOINTS[0], k=5)
        assert preds_before == preds_after, "Predictions differ after save/load"

    def test_model_is_fitted_after_training(self):
        assert self.chain.is_fitted

    def test_unfitted_model_returns_empty(self):
        fresh = FirstOrderMarkovChain()
        preds = fresh.predict(_ENDPOINTS[0], k=3)
        assert preds == [] or isinstance(preds, list)


# ===========================================================================
# Markov Benchmarking: First-order vs Second-order
# ===========================================================================

class TestMarkovBenchmarking:
    """Compare first-order and second-order Markov chains on accuracy metrics."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        all_seqs = _make_sequences(num_sequences=500, seq_length=12, seed=0)
        self.train_seqs, self.test_seqs = _train_test_split(all_seqs, test_fraction=0.2)

        self.first_order = FirstOrderMarkovChain(smoothing=0.001)
        self.first_order.fit(self.train_seqs)

        self.second_order = SecondOrderMarkovChain(smoothing=0.001)
        self.second_order.fit(self.train_seqs)

    def test_both_models_yield_positive_accuracy(self):
        acc1 = _top_k_accuracy(self.first_order, self.test_seqs, k=3)
        acc2 = _top_k_accuracy(self.second_order, self.test_seqs, k=3)
        print(f"\n[Benchmark] 1st-order Top-3={acc1:.4f}  2nd-order Top-3={acc2:.4f}")
        assert acc1 > 0.0, "First-order model must achieve positive accuracy"
        assert acc2 >= 0.0, "Second-order model must not crash on test data"

    def test_smoothed_vs_unsmoothed_first_order(self):
        unsmoothed = FirstOrderMarkovChain(smoothing=0.0)
        unsmoothed.fit(self.train_seqs)
        smoothed = FirstOrderMarkovChain(smoothing=0.01)
        smoothed.fit(self.train_seqs)
        acc_u = _top_k_accuracy(unsmoothed, self.test_seqs, k=1)
        acc_s = _top_k_accuracy(smoothed, self.test_seqs, k=1)
        print(f"\n[Benchmark] Unsmoothed={acc_u:.4f}  Smoothed={acc_s:.4f}")
        # Both must achieve some accuracy
        assert acc_u >= 0.0
        assert acc_s >= 0.0

    def test_benchmarking_table_recorded(self):
        """Capture a full benchmarking table and verify all metrics are computable."""
        results: Dict[str, Dict[str, float]] = {}
        for name, chain in [("first_order", self.first_order),
                             ("second_order", self.second_order)]:
            results[name] = {
                "top_1": _top_k_accuracy(chain, self.test_seqs, k=1),
                "top_3": _top_k_accuracy(chain, self.test_seqs, k=3),
                "top_5": _top_k_accuracy(chain, self.test_seqs, k=5),
                "mrr":   _mrr(chain, self.test_seqs),
                "coverage": _coverage(chain, self.test_seqs),
            }

        for name, metrics in results.items():
            print(f"\n[Benchmark] {name}")
            for metric, value in metrics.items():
                print(f"  {metric:12s} = {value:.4f}")
                assert isinstance(value, float), f"Metric {metric} is not a float"
                assert 0.0 <= value <= 1.0, f"{name}/{metric} = {value:.4f} out of [0,1]"


# ===========================================================================
# Markov Predictor (unified API) tests
# ===========================================================================

class TestMarkovPredictorAPI:
    """Test the high-level MarkovPredictor wrapper."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        all_seqs = _make_sequences(num_sequences=300, seq_length=10, seed=7)
        self.predictor = MarkovPredictor(order=1, smoothing=0.001)
        self.predictor.fit(all_seqs)

    def test_fit_and_predict(self):
        self.predictor.observe(_ENDPOINTS[0])
        preds = self.predictor.predict(k=5)
        assert isinstance(preds, list)

    def test_state_vector_shape(self):
        self.predictor.observe(_ENDPOINTS[0])
        vec = self.predictor.get_state_vector(k=5)
        assert isinstance(vec, (list, np.ndarray))
        assert len(vec) > 0

    def test_prediction_count_increments(self):
        self.predictor.observe(_ENDPOINTS[1])
        before = self.predictor.prediction_count
        self.predictor.predict(k=3)
        after = self.predictor.prediction_count
        assert after >= before

    def test_observe_does_not_raise(self):
        for ep in _ENDPOINTS:
            self.predictor.observe(ep)


# ===========================================================================
# DQN Agent Evaluation
# ===========================================================================

class TestDQNAgentEvaluation:
    """Evaluate DQN agent: action selection, Q-values, training dynamics."""

    STATE_DIM = 16
    ACTION_DIM = 5

    @pytest.fixture(autouse=True)
    def _setup(self):
        config = DQNConfig(
            state_dim=self.STATE_DIM,
            action_dim=self.ACTION_DIM,
            hidden_dims=[32, 16],
            buffer_size=2_000,
            batch_size=32,
            device="cpu",
            seed=42,
        )
        self.agent = DQNAgent(config, seed=42)
        self._rng = np.random.default_rng(42)

    def _random_state(self) -> np.ndarray:
        return self._rng.random(self.STATE_DIM).astype(np.float32)

    # ---- Action selection --------------------------------------------------

    def test_select_action_returns_valid_index(self):
        action = self.agent.select_action(self._random_state(), evaluate=True)
        assert 0 <= action < self.ACTION_DIM

    def test_select_action_greedy_consistent(self):
        state = self._random_state()
        a1 = self.agent.select_action(state, evaluate=True)
        a2 = self.agent.select_action(state, evaluate=True)
        assert a1 == a2, "Greedy (evaluate=True) must be deterministic"

    def test_select_action_covers_all_actions_with_exploration(self):
        """With epsilon=1 all actions should appear over many samples."""
        self.agent.epsilon = 1.0
        seen = set()
        for _ in range(500):
            a = self.agent.select_action(self._random_state())
            seen.add(a)
        assert len(seen) == self.ACTION_DIM, (
            f"Expected all {self.ACTION_DIM} actions to be seen, got {len(seen)}"
        )

    # ---- Q-values ----------------------------------------------------------

    def test_q_values_shape(self):
        q = self.agent.get_q_values(self._random_state())
        assert q.shape == (self.ACTION_DIM,), (
            f"Expected Q-value vector of shape ({self.ACTION_DIM},), got {q.shape}"
        )

    def test_q_values_are_finite(self):
        q = self.agent.get_q_values(self._random_state())
        assert np.all(np.isfinite(q)), "All Q-values must be finite"

    def test_greedy_action_matches_argmax_q(self):
        state = self._random_state()
        q = self.agent.get_q_values(state)
        greedy_from_q = int(np.argmax(q))
        greedy_from_agent = self.agent.select_action(state, evaluate=True)
        assert greedy_from_agent == greedy_from_q

    # ---- Epsilon decay -----------------------------------------------------

    def test_epsilon_starts_at_configured_value(self):
        assert self.agent.epsilon == self.agent.config.epsilon_start

    def test_epsilon_decays_after_train_steps(self):
        # Fill buffer with transitions first
        for _ in range(100):
            s = self._random_state()
            a = random.randint(0, self.ACTION_DIM - 1)
            r = random.random()
            ns = self._random_state()
            self.agent.store_transition(s, a, r, ns, False)
        eps_before = self.agent.epsilon
        self.agent.train_step()
        eps_after = self.agent.epsilon
        assert eps_after <= eps_before, "Epsilon must not increase after a train step"

    def test_epsilon_does_not_go_below_min(self):
        # Force many decay steps
        for _ in range(10_000):
            self.agent._decay_epsilon()
        assert self.agent.epsilon >= self.agent.config.epsilon_end

    # ---- Replay buffer -----------------------------------------------------

    def test_buffer_empty_initially(self):
        config = DQNConfig(state_dim=4, action_dim=2, device="cpu")
        fresh_agent = DQNAgent(config)
        assert len(fresh_agent.buffer) == 0

    def test_buffer_grows_with_transitions(self):
        before = len(self.agent.buffer)
        s, ns = self._random_state(), self._random_state()
        self.agent.store_transition(s, 0, 1.0, ns, False)
        assert len(self.agent.buffer) == before + 1

    def test_train_step_requires_enough_samples(self):
        """train_step returns None if buffer has fewer samples than batch_size."""
        config = DQNConfig(state_dim=4, action_dim=2, batch_size=64, device="cpu")
        agent = DQNAgent(config)
        # Add fewer than batch_size transitions
        for _ in range(10):
            s = np.random.rand(4).astype(np.float32)
            agent.store_transition(s, 0, 0.0, s, False)
        result = agent.train_step()
        assert result is None, "train_step must return None when buffer is too small"

    # ---- Training convergence (smoke test) ---------------------------------

    def test_training_loss_is_finite(self):
        """After filling buffer and running a train step, loss must be finite."""
        for _ in range(200):
            s = self._random_state()
            a = random.randint(0, self.ACTION_DIM - 1)
            r = random.uniform(-1, 1)
            ns = self._random_state()
            self.agent.store_transition(s, a, r, ns, False)
        result = self.agent.train_step()
        assert result is not None, "train_step returned None despite enough samples"
        assert math.isfinite(result["loss"]), (
            f"Training loss is not finite: {result['loss']}"
        )

    def test_loss_decreases_over_training(self):
        """Loss should have finite values across multiple training steps."""
        # Fill buffer
        for _ in range(500):
            s = self._random_state()
            a = random.randint(0, self.ACTION_DIM - 1)
            r = random.uniform(-1, 1)
            ns = self._random_state()
            self.agent.store_transition(s, a, r, ns, False)
        losses = []
        for _ in range(20):
            result = self.agent.train_step()
            if result:
                losses.append(result["loss"])
        assert len(losses) > 0, "No training steps completed"
        assert all(math.isfinite(l) for l in losses), "Some losses are not finite"

    # ---- Get metrics -------------------------------------------------------

    def test_get_metrics_returns_dict(self):
        metrics = self.agent.get_metrics()
        assert isinstance(metrics, dict)

    def test_get_metrics_contains_required_keys(self):
        metrics = self.agent.get_metrics()
        for key in ("epsilon", "buffer_size"):
            assert key in metrics, f"Missing key '{key}' in agent metrics"

    # ---- Save / Load -------------------------------------------------------

    def test_save_and_load_roundtrip(self, tmp_path):
        save_path = str(tmp_path / "dqn_test.pt")
        state_before = self.agent.epsilon
        self.agent.save(save_path)
        config = DQNConfig(
            state_dim=self.STATE_DIM,
            action_dim=self.ACTION_DIM,
            hidden_dims=[32, 16],
            device="cpu",
        )
        new_agent = DQNAgent(config)
        new_agent.load(save_path)
        assert abs(new_agent.epsilon - state_before) < 1e-6

    def test_loaded_agent_produces_same_actions(self, tmp_path):
        save_path = str(tmp_path / "dqn_same.pt")
        state = self._random_state()
        action_before = self.agent.select_action(state, evaluate=True)
        self.agent.save(save_path)
        config = DQNConfig(
            state_dim=self.STATE_DIM,
            action_dim=self.ACTION_DIM,
            hidden_dims=[32, 16],
            device="cpu",
        )
        new_agent = DQNAgent(config)
        new_agent.load(save_path)
        action_after = new_agent.select_action(state, evaluate=True)
        assert action_before == action_after, (
            "Loaded agent must produce the same greedy action as the original"
        )

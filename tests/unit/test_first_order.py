"""
Comprehensive unit tests for FirstOrderMarkovChain.

Tests cover:
- Training (fit, partial_fit, update)
- Prediction (predict, predict_proba)
- Sequence generation and scoring
- Evaluation metrics
- Persistence (save/load)
- Properties and statistics
"""

import pytest
import tempfile
import json
from pathlib import Path

from src.markov.first_order import FirstOrderMarkovChain


class TestTraining:
    """Test training functionality."""

    def test_fit_basic(self):
        """Test basic fit operation."""
        mc = FirstOrderMarkovChain()
        sequences = [['A', 'B', 'C'], ['A', 'B', 'D']]

        result = mc.fit(sequences)

        assert result is mc  # Method chaining
        assert mc.is_fitted
        assert 'A' in mc.states
        assert 'B' in mc.states

    def test_fit_empty_sequences(self):
        """Test fit with empty sequences."""
        mc = FirstOrderMarkovChain()
        mc.fit([])

        assert mc.is_fitted
        assert len(mc.states) == 0

    def test_fit_single_element_sequences(self):
        """Test fit with sequences of length 1."""
        mc = FirstOrderMarkovChain()
        mc.fit([['A'], ['B']])

        assert mc.is_fitted
        assert len(mc.states) == 0  # No transitions

    def test_fit_overwrites_previous(self):
        """Test that fit overwrites previous training."""
        mc = FirstOrderMarkovChain()
        mc.fit([['A', 'B']])
        mc.fit([['C', 'D']])

        assert 'C' in mc.states
        assert 'A' not in mc.states  # Previous data erased

    def test_partial_fit_accumulates(self):
        """Test that partial_fit accumulates data."""
        mc = FirstOrderMarkovChain()
        mc.fit([['A', 'B']])
        mc.partial_fit([['C', 'D']])

        assert 'A' in mc.states
        assert 'C' in mc.states  # Both present

    def test_partial_fit_updates_counts(self):
        """Test that partial_fit updates transition counts."""
        mc = FirstOrderMarkovChain(smoothing=0.0)
        mc.fit([['A', 'B']])

        prob_before = mc.predict_proba('A', 'B')

        mc.partial_fit([['A', 'B']])

        prob_after = mc.predict_proba('A', 'B')
        assert prob_after == prob_before  # Still 100%

    def test_update_single_transition(self):
        """Test update method for single transitions."""
        mc = FirstOrderMarkovChain(smoothing=0.0)
        mc.update('A', 'B', count=10)
        mc.update('A', 'C', count=5)

        assert mc.is_fitted
        prob_b = mc.predict_proba('A', 'B')
        assert abs(prob_b - 10/15) < 1e-10


class TestPrediction:
    """Test prediction functionality."""

    def test_predict_basic(self):
        """Test basic prediction."""
        mc = FirstOrderMarkovChain()
        mc.fit([['A', 'B'], ['A', 'C']])

        predictions = mc.predict('A', k=2)

        assert len(predictions) == 2
        assert all(isinstance(p[0], str) for p in predictions)
        assert all(isinstance(p[1], float) for p in predictions)
        assert predictions[0][1] >= predictions[1][1]  # Sorted descending

    def test_predict_unseen_state(self):
        """Test prediction for unseen state."""
        mc = FirstOrderMarkovChain()
        mc.fit([['A', 'B']])

        predictions = mc.predict('Z', k=5)

        assert predictions == []

    def test_predict_unfitted(self):
        """Test prediction on unfitted model."""
        mc = FirstOrderMarkovChain()

        predictions = mc.predict('A', k=5)

        assert predictions == []

    def test_predict_proba_basic(self):
        """Test predict_proba for known transition."""
        mc = FirstOrderMarkovChain(smoothing=0.0)
        mc.fit([['A', 'B'], ['A', 'B'], ['A', 'C']])

        prob = mc.predict_proba('A', 'B')

        assert abs(prob - 2/3) < 1e-10

    def test_predict_proba_unseen_transition(self):
        """Test predict_proba for unseen transition without smoothing."""
        mc = FirstOrderMarkovChain(smoothing=0.0)
        mc.fit([['A', 'B']])

        prob = mc.predict_proba('A', 'C')

        assert prob == 0.0

    def test_predict_proba_with_smoothing(self):
        """Test predict_proba with smoothing gives non-zero."""
        mc = FirstOrderMarkovChain(smoothing=0.1)
        mc.fit([['A', 'B'], ['C', 'D']])

        prob = mc.predict_proba('A', 'C')

        assert prob > 0.0  # Smoothed


class TestSequenceGeneration:
    """Test sequence generation functionality."""

    def test_generate_sequence_basic(self):
        """Test basic sequence generation."""
        mc = FirstOrderMarkovChain()
        mc.fit([['A', 'B', 'C'], ['A', 'B', 'D']])

        seq = mc.generate_sequence('A', length=5, seed=42)

        assert len(seq) <= 5
        assert seq[0] == 'A'
        assert all(s in mc.states for s in seq)

    def test_generate_sequence_reproducible(self):
        """Test that seed makes generation reproducible."""
        mc = FirstOrderMarkovChain()
        mc.fit([['A', 'B', 'C'], ['A', 'B', 'D']])

        seq1 = mc.generate_sequence('A', length=10, seed=42)
        seq2 = mc.generate_sequence('A', length=10, seed=42)

        assert seq1 == seq2

    def test_generate_sequence_with_stop_states(self):
        """Test sequence generation with stop states."""
        mc = FirstOrderMarkovChain()
        mc.fit([['A', 'B', 'C'], ['A', 'B', 'D']])

        seq = mc.generate_sequence('A', length=100, stop_states={'C'}, seed=42)

        # Should stop at C or before
        if 'C' in seq:
            assert seq[-1] == 'C' or seq.index('C') == len(seq) - 1

    def test_generate_sequence_dead_end(self):
        """Test generation when reaching state with no outgoing transitions."""
        mc = FirstOrderMarkovChain()
        mc.fit([['A', 'B']])  # B has no outgoing transitions

        seq = mc.generate_sequence('A', length=10, seed=42)

        # Should stop at B
        assert len(seq) == 2
        assert seq == ['A', 'B']

    def test_generate_sequence_unfitted(self):
        """Test generation on unfitted model."""
        mc = FirstOrderMarkovChain()

        seq = mc.generate_sequence('A', length=5)

        assert seq == ['A']


class TestSequenceScoring:
    """Test sequence scoring functionality."""

    def test_score_sequence_basic(self):
        """Test basic sequence scoring."""
        mc = FirstOrderMarkovChain(smoothing=0.0)
        mc.fit([['A', 'B', 'C']])

        score = mc.score_sequence(['A', 'B', 'C'])

        # With no smoothing and deterministic transitions, log(1.0) = 0.0
        assert score <= 0  # Log probabilities are negative or zero
        assert score != float('-inf')

    def test_score_sequence_unseen_transition(self):
        """Test scoring sequence with unseen transition."""
        mc = FirstOrderMarkovChain(smoothing=0.0)
        mc.fit([['A', 'B']])

        score = mc.score_sequence(['A', 'C'])

        assert score == float('-inf')

    def test_score_sequence_with_smoothing(self):
        """Test that smoothing prevents -inf scores."""
        mc = FirstOrderMarkovChain(smoothing=0.1)
        mc.fit([['A', 'B']])

        score = mc.score_sequence(['A', 'C'])

        assert score != float('-inf')
        assert score < 0

    def test_score_sequence_short(self):
        """Test scoring sequence too short."""
        mc = FirstOrderMarkovChain()
        mc.fit([['A', 'B']])

        score = mc.score_sequence(['A'])

        assert score == 0.0


class TestEvaluation:
    """Test evaluation metrics."""

    def test_evaluate_basic(self):
        """Test basic evaluation."""
        mc = FirstOrderMarkovChain()
        sequences = [['A', 'B', 'C'], ['A', 'B', 'D']]
        mc.fit(sequences)

        metrics = mc.evaluate(sequences, k_values=[1, 3])

        assert 'top_1_accuracy' in metrics
        assert 'top_3_accuracy' in metrics
        assert 'mrr' in metrics
        assert 'coverage' in metrics
        assert 'perplexity' in metrics

        assert 0 <= metrics['top_1_accuracy'] <= 1
        assert 0 <= metrics['mrr'] <= 1
        assert 0 <= metrics['coverage'] <= 1

    def test_evaluate_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        mc = FirstOrderMarkovChain()
        sequences = [['A', 'B'], ['A', 'B']]
        mc.fit(sequences)

        metrics = mc.evaluate(sequences, k_values=[1])

        assert metrics['top_1_accuracy'] == 1.0
        assert metrics['mrr'] == 1.0
        assert metrics['coverage'] == 1.0

    def test_evaluate_unfitted(self):
        """Test evaluation on unfitted model."""
        mc = FirstOrderMarkovChain()

        metrics = mc.evaluate([['A', 'B']], k_values=[1, 3])

        assert metrics['top_1_accuracy'] == 0.0
        assert metrics['mrr'] == 0.0

    def test_evaluate_top_k_ordering(self):
        """Test that top_k accuracy increases with k."""
        mc = FirstOrderMarkovChain()
        sequences = [['A', 'B'], ['A', 'C'], ['A', 'D']]
        mc.fit(sequences)

        metrics = mc.evaluate(sequences, k_values=[1, 2, 3])

        assert metrics['top_1_accuracy'] <= metrics['top_2_accuracy']
        assert metrics['top_2_accuracy'] <= metrics['top_3_accuracy']


class TestPersistence:
    """Test save/load functionality."""

    def test_save_and_load(self):
        """Test saving and loading model."""
        mc1 = FirstOrderMarkovChain(smoothing=0.01)
        mc1.fit([['A', 'B', 'C'], ['A', 'B', 'D']])

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            mc1.save(temp_path)
            mc2 = FirstOrderMarkovChain.load(temp_path)

            assert mc2.smoothing == mc1.smoothing
            assert mc2.is_fitted == mc1.is_fitted
            assert mc2.states == mc1.states

            # Test predictions match
            pred1 = mc1.predict('A', k=2)
            pred2 = mc2.predict('A', k=2)
            assert pred1 == pred2
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_unfitted_model(self):
        """Test loading an unfitted model."""
        mc1 = FirstOrderMarkovChain(smoothing=0.05)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            mc1.save(temp_path)
            mc2 = FirstOrderMarkovChain.load(temp_path)

            assert not mc2.is_fitted
            assert mc2.smoothing == 0.05
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestProperties:
    """Test properties and statistics."""

    def test_is_fitted_property(self):
        """Test is_fitted property."""
        mc = FirstOrderMarkovChain()

        assert not mc.is_fitted

        mc.fit([['A', 'B']])

        assert mc.is_fitted

    def test_states_property(self):
        """Test states property."""
        mc = FirstOrderMarkovChain()
        mc.fit([['A', 'B', 'C'], ['D', 'E']])

        states = mc.states

        assert isinstance(states, set)
        assert states == {'A', 'B', 'C', 'D', 'E'}

    def test_states_unfitted(self):
        """Test states property on unfitted model."""
        mc = FirstOrderMarkovChain()

        assert mc.states == set()

    def test_get_statistics(self):
        """Test get_statistics method."""
        mc = FirstOrderMarkovChain()
        mc.fit([['A', 'B', 'C'], ['A', 'B', 'D']])

        stats = mc.get_statistics()

        assert stats['is_fitted']
        assert stats['num_states'] == 4
        assert stats['num_transitions'] > 0
        assert 'sparsity' in stats
        assert 'most_common_transitions' in stats

    def test_get_statistics_unfitted(self):
        """Test statistics on unfitted model."""
        mc = FirstOrderMarkovChain()

        stats = mc.get_statistics()

        assert not stats['is_fitted']
        assert stats['num_states'] == 0


class TestRepresentation:
    """Test string representations."""

    def test_repr_fitted(self):
        """Test __repr__ for fitted model."""
        mc = FirstOrderMarkovChain(smoothing=0.01)
        mc.fit([['A', 'B']])

        repr_str = repr(mc)

        assert 'FirstOrderMarkovChain' in repr_str
        assert 'states=' in repr_str
        assert 'transitions=' in repr_str

    def test_repr_unfitted(self):
        """Test __repr__ for unfitted model."""
        mc = FirstOrderMarkovChain(smoothing=0.02)

        repr_str = repr(mc)

        assert 'FirstOrderMarkovChain' in repr_str
        assert 'fitted=False' in repr_str

    def test_str(self):
        """Test __str__ method."""
        mc = FirstOrderMarkovChain()
        mc.fit([['A', 'B']])

        str_str = str(mc)

        assert 'FirstOrderMarkovChain' in str_str


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_api_endpoint_prediction(self):
        """Test realistic API endpoint prediction."""
        mc = FirstOrderMarkovChain(smoothing=0.001)

        sequences = [
            ['login', 'profile', 'orders'],
            ['login', 'profile', 'settings'],
            ['login', 'browse', 'product', 'cart'],
            ['browse', 'product', 'product', 'cart', 'checkout']
        ]

        mc.fit(sequences)

        # Predict after login
        predictions = mc.predict('login', k=2)
        assert len(predictions) > 0
        assert predictions[0][0] == 'profile'  # Most common after login

        # Evaluate
        metrics = mc.evaluate(sequences, k_values=[1, 3])
        assert metrics['coverage'] == 1.0  # All states predictable

    def test_incremental_learning(self):
        """Test incremental learning scenario."""
        mc = FirstOrderMarkovChain()

        # Initial training
        mc.fit([['A', 'B', 'C']])
        initial_states = len(mc.states)

        # Add more data
        mc.partial_fit([['D', 'E', 'F']])

        assert len(mc.states) > initial_states
        assert 'A' in mc.states
        assert 'D' in mc.states


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


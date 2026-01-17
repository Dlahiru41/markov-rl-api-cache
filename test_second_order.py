"""
Comprehensive tests for SecondOrderMarkovChain.

Tests cover all functionality including training, prediction, evaluation,
persistence, and edge cases.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path

from src.markov.second_order import SecondOrderMarkovChain, START_TOKEN
from src.markov.first_order import FirstOrderMarkovChain


class TestSecondOrderBasics:
    """Test basic second-order Markov chain operations."""

    def test_initialization(self):
        """Test model initialization with different parameters."""
        mc2 = SecondOrderMarkovChain()
        assert mc2.smoothing == 0.0
        assert mc2.fallback_to_first_order is True
        assert not mc2.is_fitted

        mc2_custom = SecondOrderMarkovChain(smoothing=0.01, fallback_to_first_order=False)
        assert mc2_custom.smoothing == 0.01
        assert mc2_custom.fallback_to_first_order is False

    def test_state_key_operations(self):
        """Test state key creation and parsing."""
        key = SecondOrderMarkovChain._make_state_key('A', 'B')
        assert isinstance(key, str)
        assert '|' in key

        prev, curr = SecondOrderMarkovChain._parse_state_key(key)
        assert prev == 'A'
        assert curr == 'B'

    def test_fit_basic(self):
        """Test basic model fitting."""
        sequences = [
            ['A', 'B', 'C'],
            ['A', 'B', 'D']
        ]
        mc2 = SecondOrderMarkovChain()
        result = mc2.fit(sequences)

        assert result is mc2  # Should return self
        assert mc2.is_fitted
        assert len(mc2.states) > 0

    def test_states_property(self):
        """Test that states property returns individual states."""
        sequences = [['A', 'B', 'C', 'D']]
        mc2 = SecondOrderMarkovChain()
        mc2.fit(sequences)

        states = mc2.states
        assert isinstance(states, set)
        assert 'A' in states
        assert 'B' in states
        assert 'C' in states
        assert 'D' in states
        assert START_TOKEN not in states  # Should be excluded

    def test_state_pairs_property(self):
        """Test that state_pairs property returns pairs."""
        sequences = [['A', 'B', 'C', 'D']]
        mc2 = SecondOrderMarkovChain()
        mc2.fit(sequences)

        pairs = mc2.state_pairs
        assert isinstance(pairs, set)
        # Sequence ['A', 'B', 'C', 'D'] creates:
        # (START, A) -> B, (A, B) -> C, (B, C) -> D
        assert (START_TOKEN, 'A') in pairs
        assert ('A', 'B') in pairs
        assert ('B', 'C') in pairs


class TestSecondOrderPredictions:
    """Test prediction functionality."""

    def test_predict_basic(self):
        """Test basic prediction."""
        sequences = [
            ['A', 'B', 'C'],
            ['A', 'B', 'C'],
            ['A', 'B', 'D']
        ]
        mc2 = SecondOrderMarkovChain(smoothing=0.001)
        mc2.fit(sequences)

        predictions = mc2.predict('A', 'B', k=2)
        assert len(predictions) <= 2
        assert all(isinstance(p, tuple) and len(p) == 2 for p in predictions)
        assert all(isinstance(p[0], str) and isinstance(p[1], float) for p in predictions)

        # Most common should be 'C'
        assert predictions[0][0] == 'C'

    def test_predict_unfitted(self):
        """Test prediction on unfitted model."""
        mc2 = SecondOrderMarkovChain()
        predictions = mc2.predict('A', 'B', k=5)
        assert predictions == []

    def test_predict_with_fallback(self):
        """Test fallback to first-order for unseen pairs."""
        sequences = [
            ['A', 'B', 'C'],
            ['X', 'Y', 'Z']
        ]
        mc2 = SecondOrderMarkovChain(smoothing=0.001, fallback_to_first_order=True)
        mc2.fit(sequences)

        # Seen pair
        pred_seen = mc2.predict('A', 'B', k=2)
        assert len(pred_seen) > 0

        # Unseen pair with fallback
        pred_unseen = mc2.predict('Q', 'B', k=2, use_fallback=True)
        # Should fall back to first-order for 'B'

        # Unseen pair without fallback
        pred_no_fallback = mc2.predict('Q', 'R', k=2, use_fallback=False)
        assert len(pred_no_fallback) == 0

    def test_predict_no_fallback_model(self):
        """Test with fallback disabled."""
        sequences = [['A', 'B', 'C']]
        mc2 = SecondOrderMarkovChain(smoothing=0.001, fallback_to_first_order=False)
        mc2.fit(sequences)

        # Unseen pair should return empty
        pred = mc2.predict('X', 'Y', k=2, use_fallback=True)
        assert len(pred) == 0

    def test_predict_proba(self):
        """Test probability prediction."""
        sequences = [
            ['A', 'B', 'C'],
            ['A', 'B', 'C'],
            ['A', 'B', 'D']
        ]
        mc2 = SecondOrderMarkovChain(smoothing=0.001)
        mc2.fit(sequences)

        prob_c = mc2.predict_proba('A', 'B', 'C')
        prob_d = mc2.predict_proba('A', 'B', 'D')

        assert 0 <= prob_c <= 1
        assert 0 <= prob_d <= 1
        assert prob_c > prob_d  # C is more common


class TestSecondOrderTraining:
    """Test training methods."""

    def test_partial_fit(self):
        """Test incremental training."""
        initial = [['A', 'B', 'C']]
        additional = [['A', 'B', 'D']]

        mc2 = SecondOrderMarkovChain(smoothing=0.001)
        mc2.fit(initial)

        # Before partial_fit, only knows C
        pred_before = mc2.predict('A', 'B', k=5)
        states_before = [p[0] for p in pred_before]
        assert 'C' in states_before

        # After partial_fit, should know both C and D
        mc2.partial_fit(additional)
        pred_after = mc2.predict('A', 'B', k=5)
        states_after = [p[0] for p in pred_after]
        assert 'C' in states_after
        assert 'D' in states_after

    def test_update_single_transition(self):
        """Test single transition update."""
        mc2 = SecondOrderMarkovChain(smoothing=0.001)

        mc2.update('A', 'B', 'C', count=10)
        mc2.update('A', 'B', 'D', count=5)

        assert mc2.is_fitted
        pred = mc2.predict('A', 'B', k=2)
        assert len(pred) == 2
        assert pred[0][0] == 'C'  # More common


class TestSecondOrderGeneration:
    """Test sequence generation."""

    def test_generate_sequence(self):
        """Test synthetic sequence generation."""
        sequences = [
            ['A', 'B', 'C', 'D'],
            ['A', 'B', 'D', 'E']
        ]
        mc2 = SecondOrderMarkovChain(smoothing=0.001)
        mc2.fit(sequences)

        seq = mc2.generate_sequence('A', length=5, seed=42)
        assert isinstance(seq, list)
        assert seq[0] == 'A'
        assert len(seq) <= 5
        assert all(isinstance(s, str) for s in seq)

    def test_generate_with_stop_states(self):
        """Test generation with stop states."""
        sequences = [['A', 'B', 'C', 'D', 'E']]
        mc2 = SecondOrderMarkovChain(smoothing=0.001)
        mc2.fit(sequences)

        seq = mc2.generate_sequence('A', length=10, stop_states={'C'}, seed=42)
        # Should stop at or before C
        if 'C' in seq:
            c_index = seq.index('C')
            assert c_index == len(seq) - 1  # C should be last

    def test_score_sequence(self):
        """Test sequence scoring."""
        sequences = [
            ['A', 'B', 'C'],
            ['A', 'B', 'C'],
            ['A', 'B', 'D']
        ]
        mc2 = SecondOrderMarkovChain(smoothing=0.001)
        mc2.fit(sequences)

        score_common = mc2.score_sequence(['A', 'B', 'C'])
        score_rare = mc2.score_sequence(['A', 'B', 'D'])

        assert score_common < 0  # Log probability
        assert score_rare < 0
        assert score_common > score_rare  # More common has higher score


class TestSecondOrderEvaluation:
    """Test evaluation methods."""

    def test_evaluate_basic(self):
        """Test basic evaluation."""
        sequences = [
            ['A', 'B', 'C'],
            ['A', 'B', 'D']
        ]
        mc2 = SecondOrderMarkovChain(smoothing=0.001, fallback_to_first_order=True)
        mc2.fit(sequences)

        metrics = mc2.evaluate(sequences, k_values=[1, 3])

        assert 'top_1_accuracy' in metrics
        assert 'top_3_accuracy' in metrics
        assert 'mrr' in metrics
        assert 'coverage' in metrics
        assert 'perplexity' in metrics
        assert 'fallback_rate' in metrics

        assert 0 <= metrics['top_1_accuracy'] <= 1
        assert 0 <= metrics['mrr'] <= 1
        assert 0 <= metrics['coverage'] <= 1
        assert 0 <= metrics['fallback_rate'] <= 1

    def test_evaluate_unfitted(self):
        """Test evaluation on unfitted model."""
        mc2 = SecondOrderMarkovChain()
        metrics = mc2.evaluate([['A', 'B', 'C']])

        assert metrics['top_1_accuracy'] == 0.0
        assert metrics['coverage'] == 0.0

    def test_compare_with_first_order(self):
        """Test comparison with first-order model."""
        sequences = [
            ['A', 'B', 'C'],
            ['A', 'B', 'D'],
            ['X', 'B', 'E']
        ]
        mc2 = SecondOrderMarkovChain(smoothing=0.001, fallback_to_first_order=True)
        mc2.fit(sequences)

        comparison = mc2.compare_with_first_order(sequences)

        assert 'second_order_metrics' in comparison
        assert 'first_order_metrics' in comparison
        assert 'improvement' in comparison
        assert 'fallback_rate' in comparison

        assert 'top_1_accuracy' in comparison['improvement']


class TestSecondOrderPersistence:
    """Test model saving and loading."""

    def test_save_and_load(self):
        """Test model persistence."""
        sequences = [
            ['A', 'B', 'C'],
            ['A', 'B', 'D']
        ]
        mc2 = SecondOrderMarkovChain(smoothing=0.001, fallback_to_first_order=True)
        mc2.fit(sequences)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            mc2.save(temp_path)
            assert os.path.exists(temp_path)

            # Load and verify
            mc2_loaded = SecondOrderMarkovChain.load(temp_path)
            assert mc2_loaded.is_fitted
            assert mc2_loaded.smoothing == mc2.smoothing
            assert mc2_loaded.fallback_to_first_order == mc2.fallback_to_first_order

            # Verify predictions match
            pred_orig = mc2.predict('A', 'B', k=2)
            pred_loaded = mc2_loaded.predict('A', 'B', k=2)
            assert len(pred_orig) == len(pred_loaded)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_creates_directories(self):
        """Test that save creates parent directories."""
        sequences = [['A', 'B', 'C']]
        mc2 = SecondOrderMarkovChain()
        mc2.fit(sequences)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'subdir', 'model.json')
            mc2.save(save_path)
            assert os.path.exists(save_path)


class TestSecondOrderStatistics:
    """Test statistics methods."""

    def test_get_statistics(self):
        """Test statistics collection."""
        sequences = [
            ['A', 'B', 'C'],
            ['D', 'E', 'F']
        ]
        mc2 = SecondOrderMarkovChain(smoothing=0.001, fallback_to_first_order=True)
        mc2.fit(sequences)

        stats = mc2.get_statistics()

        assert 'is_fitted' in stats
        assert 'num_individual_states' in stats
        assert 'num_state_pairs' in stats
        assert 'num_transitions' in stats
        assert 'fallback_to_first_order' in stats

        assert stats['is_fitted'] is True
        assert stats['num_individual_states'] > 0
        assert stats['num_state_pairs'] > 0

    def test_statistics_unfitted(self):
        """Test statistics on unfitted model."""
        mc2 = SecondOrderMarkovChain()
        stats = mc2.get_statistics()

        assert stats['is_fitted'] is False
        assert stats['num_individual_states'] == 0


class TestSecondOrderEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_sequences(self):
        """Test with empty sequence list."""
        mc2 = SecondOrderMarkovChain()
        mc2.fit([])
        # Should not crash, just not be fitted effectively
        assert len(mc2.states) == 0

    def test_short_sequences(self):
        """Test with sequences too short for second-order."""
        mc2 = SecondOrderMarkovChain()
        mc2.fit([['A'], ['B']])
        # Should handle gracefully

        mc2.fit([['A', 'B']])  # Minimum length for one transition
        assert mc2.is_fitted

    def test_single_long_sequence(self):
        """Test with single long sequence."""
        mc2 = SecondOrderMarkovChain(smoothing=0.001)
        mc2.fit([['A', 'B', 'C', 'D', 'E']])

        pred = mc2.predict('A', 'B', k=1)
        assert len(pred) == 1
        assert pred[0][0] == 'C'

    def test_predict_k_zero(self):
        """Test prediction with k=0."""
        sequences = [['A', 'B', 'C']]
        mc2 = SecondOrderMarkovChain()
        mc2.fit(sequences)

        pred = mc2.predict('A', 'B', k=0)
        assert len(pred) == 0

    def test_predict_k_large(self):
        """Test prediction with k larger than available transitions."""
        sequences = [['A', 'B', 'C']]
        mc2 = SecondOrderMarkovChain()
        mc2.fit(sequences)

        pred = mc2.predict('A', 'B', k=100)
        assert len(pred) <= 10  # Should return available transitions


class TestSecondOrderContextCapture:
    """Test that second-order captures context better than first-order."""

    def test_context_matters(self):
        """Test that different contexts lead to different predictions."""
        sequences = [
            # After A->B, always go to C
            ['A', 'B', 'C'],
            ['A', 'B', 'C'],
            ['A', 'B', 'C'],
            # After X->B, always go to D
            ['X', 'B', 'D'],
            ['X', 'B', 'D'],
            ['X', 'B', 'D'],
        ]

        mc2 = SecondOrderMarkovChain(smoothing=0.001, fallback_to_first_order=True)
        mc2.fit(sequences)

        # Different contexts should give different predictions
        pred_after_a = mc2.predict('A', 'B', k=1)
        pred_after_x = mc2.predict('X', 'B', k=1)

        assert pred_after_a[0][0] == 'C'
        assert pred_after_x[0][0] == 'D'

        # First-order would average both
        mc1 = FirstOrderMarkovChain(smoothing=0.001)
        mc1.fit(sequences)
        pred_fo = mc1.predict('B', k=2)
        # First-order sees both C and D as equally likely

    def test_second_order_outperforms_first_order(self):
        """Test that second-order can outperform first-order on context-dependent data."""
        # Create data where context matters
        sequences = [
            ['login', 'auth', 'profile', 'orders'],
            ['login', 'auth', 'profile', 'orders'],
            ['browse', 'search', 'profile', 'settings'],
            ['browse', 'search', 'profile', 'settings'],
        ]

        mc2 = SecondOrderMarkovChain(smoothing=0.001, fallback_to_first_order=True)
        mc2.fit(sequences)

        comparison = mc2.compare_with_first_order(sequences)

        # Second-order should do better or equal
        so_acc = comparison['second_order_metrics']['top_1_accuracy']
        fo_acc = comparison['first_order_metrics']['top_1_accuracy']

        assert so_acc >= fo_acc


class TestSecondOrderRepr:
    """Test string representations."""

    def test_repr_unfitted(self):
        """Test repr of unfitted model."""
        mc2 = SecondOrderMarkovChain(smoothing=0.01, fallback_to_first_order=False)
        repr_str = repr(mc2)

        assert 'SecondOrderMarkovChain' in repr_str
        assert 'fitted=False' in repr_str
        assert '0.01' in repr_str

    def test_repr_fitted(self):
        """Test repr of fitted model."""
        mc2 = SecondOrderMarkovChain(smoothing=0.001)
        mc2.fit([['A', 'B', 'C']])

        repr_str = repr(mc2)
        assert 'SecondOrderMarkovChain' in repr_str
        assert 'states=' in repr_str
        assert 'pairs=' in repr_str

    def test_str(self):
        """Test str representation."""
        mc2 = SecondOrderMarkovChain()
        str_repr = str(mc2)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


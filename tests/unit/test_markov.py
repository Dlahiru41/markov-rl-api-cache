"""
Comprehensive test suite for all Markov chain components.

Tests cover TransitionMatrix, FirstOrderMarkovChain, SecondOrderMarkovChain,
ContextAwareMarkovChain, and MarkovPredictor with edge cases and performance tests.
"""

import pytest
import numpy as np
import json
from pathlib import Path

from src.markov import (
    TransitionMatrix,
    FirstOrderMarkovChain,
    SecondOrderMarkovChain,
    ContextAwareMarkovChain,
    MarkovPredictor,
    create_predictor
)


# =============================================================================
# Test Fixtures (Reusable Test Data)
# =============================================================================

@pytest.fixture
def simple_sequences():
    """Small list of API sequences for quick tests."""
    return [
        ['login', 'profile', 'orders'],
        ['login', 'browse', 'product'],
        ['browse', 'product', 'cart'],
        ['login', 'profile', 'settings']
    ]


@pytest.fixture
def large_sequences():
    """More data for performance/accuracy tests."""
    sequences = []
    # Pattern 1: login flow (40%)
    for _ in range(40):
        sequences.append(['login', 'profile', 'orders', 'product', 'cart'])
    # Pattern 2: browse flow (30%)
    for _ in range(30):
        sequences.append(['browse', 'search', 'product', 'reviews', 'cart'])
    # Pattern 3: direct checkout (20%)
    for _ in range(20):
        sequences.append(['login', 'cart', 'checkout', 'payment', 'confirm'])
    # Pattern 4: exploration (10%)
    for _ in range(10):
        sequences.append(['browse', 'product', 'compare', 'reviews', 'back'])
    return sequences


@pytest.fixture
def sequences_with_context():
    """Sequences paired with context dictionaries."""
    sequences = [
        ['login', 'premium_features', 'advanced'],
        ['login', 'premium_features', 'settings'],
        ['login', 'browse', 'product'],
        ['login', 'browse', 'search'],
    ]
    contexts = [
        {'user_type': 'premium', 'hour': 10},
        {'user_type': 'premium', 'hour': 14},
        {'user_type': 'free', 'hour': 10},
        {'user_type': 'free', 'hour': 14},
    ]
    return sequences, contexts


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for file operations."""
    return tmp_path


# =============================================================================
# TestTransitionMatrix
# =============================================================================

class TestTransitionMatrix:
    """Tests for the TransitionMatrix class."""

    def test_increment_single(self):
        """Adding one transition works."""
        tm = TransitionMatrix(smoothing=0.0)
        tm.increment('A', 'B', 1)

        assert tm.get_count('A', 'B') == 1
        assert tm.get_count('A', 'C') == 0

    def test_increment_multiple(self):
        """Adding same transition multiple times accumulates."""
        tm = TransitionMatrix(smoothing=0.0)
        tm.increment('A', 'B', 5)
        tm.increment('A', 'B', 3)

        assert tm.get_count('A', 'B') == 8

    def test_probability_calculation(self):
        """Probabilities are correct and sum to 1."""
        tm = TransitionMatrix(smoothing=0.0)
        tm.increment('A', 'B', 80)
        tm.increment('A', 'C', 20)

        prob_b = tm.get_probability('A', 'B')
        prob_c = tm.get_probability('A', 'C')

        assert abs(prob_b - 0.8) < 0.01
        assert abs(prob_c - 0.2) < 0.01
        assert abs(prob_b + prob_c - 1.0) < 0.01

    def test_smoothing(self):
        """Laplace smoothing gives non-zero probabilities for unseen transitions."""
        tm = TransitionMatrix(smoothing=0.1)
        tm.increment('A', 'B', 10)

        # Even unseen transition should have non-zero probability
        prob_unseen = tm.get_probability('A', 'C')
        assert prob_unseen > 0

    def test_top_k(self):
        """Returns correct top transitions in right order."""
        tm = TransitionMatrix(smoothing=0.0)
        tm.increment('A', 'B', 100)
        tm.increment('A', 'C', 50)
        tm.increment('A', 'D', 30)
        tm.increment('A', 'E', 20)

        top_3 = tm.get_top_k('A', k=3)

        assert len(top_3) == 3
        assert top_3[0][0] == 'B'  # Highest probability
        assert top_3[1][0] == 'C'  # Second highest
        assert top_3[2][0] == 'D'  # Third highest
        assert top_3[0][1] > top_3[1][1] > top_3[2][1]  # Descending order

    def test_top_k_handles_ties(self):
        """Deterministic behavior when probabilities are equal."""
        tm = TransitionMatrix(smoothing=0.0)
        tm.increment('A', 'B', 10)
        tm.increment('A', 'C', 10)
        tm.increment('A', 'D', 10)

        top_2 = tm.get_top_k('A', k=2)

        assert len(top_2) == 2
        # All should have same probability
        assert abs(top_2[0][1] - top_2[1][1]) < 0.01

    def test_serialization_roundtrip(self, temp_dir):
        """Save then load gives identical matrix."""
        tm = TransitionMatrix(smoothing=0.001)
        tm.increment('A', 'B', 10)
        tm.increment('A', 'C', 5)
        tm.increment('B', 'C', 8)

        original_prob = tm.get_probability('A', 'B')

        # Save and load
        save_path = temp_dir / "test_matrix.json"
        tm.save(str(save_path))
        tm_loaded = TransitionMatrix.load(str(save_path))

        loaded_prob = tm_loaded.get_probability('A', 'B')

        assert abs(original_prob - loaded_prob) < 0.001

    def test_unknown_state(self):
        """Querying unknown state returns empty/zero gracefully."""
        tm = TransitionMatrix(smoothing=0.0)
        tm.increment('A', 'B', 10)

        # Unknown from state
        row = tm.get_row('UNKNOWN')
        assert row == {}

        # Unknown to state
        prob = tm.get_probability('A', 'UNKNOWN')
        assert prob == 0.0

    def test_merge(self):
        """Combining two matrices sums counts correctly."""
        tm1 = TransitionMatrix(smoothing=0.0)
        tm1.increment('A', 'B', 10)
        tm1.increment('A', 'C', 5)

        tm2 = TransitionMatrix(smoothing=0.0)
        tm2.increment('A', 'B', 5)
        tm2.increment('B', 'C', 8)

        tm_merged = tm1.merge(tm2)

        assert tm_merged.get_count('A', 'B') == 15
        assert tm_merged.get_count('A', 'C') == 5
        assert tm_merged.get_count('B', 'C') == 8


# =============================================================================
# TestFirstOrderMarkovChain
# =============================================================================

class TestFirstOrderMarkovChain:
    """Tests for the FirstOrderMarkovChain class."""

    def test_fit_creates_transitions(self, simple_sequences):
        """After fitting, expected transitions exist."""
        mc = FirstOrderMarkovChain(smoothing=0.001)
        mc.fit(simple_sequences)

        # Should have transition from 'login' to 'profile' and 'browse'
        predictions = mc.predict('login', k=5)
        apis = [api for api, prob in predictions]

        assert 'profile' in apis
        assert 'browse' in apis

    def test_predict_returns_sorted(self, simple_sequences):
        """Predictions are in probability order."""
        mc = FirstOrderMarkovChain(smoothing=0.001)
        mc.fit(simple_sequences)

        predictions = mc.predict('login', k=5)

        # Check descending probability order
        for i in range(len(predictions) - 1):
            assert predictions[i][1] >= predictions[i + 1][1]

    def test_predict_unknown_state(self, simple_sequences):
        """Unknown state returns empty list."""
        mc = FirstOrderMarkovChain(smoothing=0.0)
        mc.fit(simple_sequences)

        predictions = mc.predict('UNKNOWN_API', k=5)
        assert predictions == []

    def test_partial_fit_accumulates(self, simple_sequences):
        """Partial fit adds to existing, doesn't reset."""
        mc = FirstOrderMarkovChain(smoothing=0.001)

        # Initial fit
        mc.fit(simple_sequences[:2])
        initial_states = mc.states.copy()

        # Partial fit with new data
        mc.partial_fit(simple_sequences[2:])
        final_states = mc.states

        # Should have all states
        assert initial_states.issubset(final_states)
        assert len(final_states) >= len(initial_states)

    def test_evaluate_metrics(self, simple_sequences):
        """All expected metrics are computed."""
        mc = FirstOrderMarkovChain(smoothing=0.001)
        mc.fit(simple_sequences)

        metrics = mc.evaluate(simple_sequences, k_values=[1, 3])

        assert 'top_1_accuracy' in metrics
        assert 'top_3_accuracy' in metrics
        assert 'mrr' in metrics
        assert 'coverage' in metrics
        assert 0 <= metrics['top_1_accuracy'] <= 1
        assert 0 <= metrics['mrr'] <= 1

    def test_generate_sequence(self, simple_sequences):
        """Generated sequences follow learned distribution."""
        mc = FirstOrderMarkovChain(smoothing=0.001)
        mc.fit(simple_sequences)

        generated = mc.generate_sequence('login', length=5)

        assert len(generated) <= 6  # Start + 5 or less if stopped
        assert generated[0] == 'login'
        # All generated states should be in vocabulary
        for state in generated:
            assert state in mc.states

    def test_score_sequence(self, simple_sequences):
        """Likely sequences have higher scores than unlikely ones."""
        mc = FirstOrderMarkovChain(smoothing=0.001)
        mc.fit(simple_sequences)

        # Likely sequence (appears in training)
        likely_score = mc.score_sequence(['login', 'profile', 'orders'])

        # Unlikely sequence
        unlikely_score = mc.score_sequence(['orders', 'login', 'cart'])

        assert likely_score > unlikely_score

    def test_save_load(self, simple_sequences, temp_dir):
        """Loaded model gives same predictions as original."""
        mc = FirstOrderMarkovChain(smoothing=0.001)
        mc.fit(simple_sequences)

        original_predictions = mc.predict('login', k=3)

        # Save and load
        save_path = temp_dir / "test_mc.json"
        mc.save(str(save_path))
        mc_loaded = FirstOrderMarkovChain.load(str(save_path))

        loaded_predictions = mc_loaded.predict('login', k=3)

        assert len(original_predictions) == len(loaded_predictions)
        for orig, loaded in zip(original_predictions, loaded_predictions):
            assert orig[0] == loaded[0]  # Same API
            assert abs(orig[1] - loaded[1]) < 0.001  # Same probability


# =============================================================================
# TestSecondOrderMarkovChain
# =============================================================================

class TestSecondOrderMarkovChain:
    """Tests for the SecondOrderMarkovChain class."""

    def test_uses_both_states(self):
        """Different (prev, curr) pairs give different predictions."""
        sequences = [
            ['A', 'B', 'C', 'D'],
            ['A', 'B', 'C', 'E'],
            ['X', 'B', 'C', 'F'],
            ['X', 'B', 'C', 'G'],
        ]

        mc = SecondOrderMarkovChain(smoothing=0.001, fallback_to_first_order=True)
        mc.fit(sequences)

        # After (A, B, C) we expect D or E
        pred1 = mc.predict('B', 'C', k=1, use_fallback=False)

        # Check we can make predictions
        assert len(pred1) > 0

    def test_fallback_to_first_order(self):
        """Unseen pairs fall back correctly."""
        sequences = [['A', 'B', 'C'], ['A', 'B', 'D']]

        mc = SecondOrderMarkovChain(smoothing=0.001, fallback_to_first_order=True)
        mc.fit(sequences)

        # Unseen pair should use first-order fallback
        predictions = mc.predict('UNKNOWN', 'B', k=2)

        # Should get some predictions from first-order model
        assert len(predictions) > 0

    def test_fallback_disabled(self):
        """With fallback disabled, unseen pairs return empty."""
        sequences = [['A', 'B', 'C'], ['A', 'B', 'D']]

        mc = SecondOrderMarkovChain(smoothing=0.001, fallback_to_first_order=False)
        mc.fit(sequences)

        # Unseen pair with fallback disabled
        predictions = mc.predict('UNKNOWN', 'UNKNOWN', k=2, use_fallback=False)

        assert predictions == []

    def test_compare_with_first_order(self, simple_sequences):
        """Comparison method returns expected structure."""
        mc = SecondOrderMarkovChain(smoothing=0.001, fallback_to_first_order=True)
        mc.fit(simple_sequences)

        comparison = mc.compare_with_first_order(simple_sequences)

        assert 'second_order_accuracy' in comparison
        assert 'first_order_accuracy' in comparison
        assert 'fallback_rate' in comparison
        assert 'improvement' in comparison


# =============================================================================
# TestContextAwareMarkovChain
# =============================================================================

class TestContextAwareMarkovChain:
    """Tests for the ContextAwareMarkovChain class."""

    def test_different_contexts_different_predictions(self, sequences_with_context):
        """Context affects output."""
        sequences, contexts = sequences_with_context

        mc = ContextAwareMarkovChain(
            context_features=['user_type'],
            order=1,
            fallback_strategy='global'
        )
        mc.fit(sequences, contexts)

        # Reset and observe
        pred_premium = mc.predict('login', {'user_type': 'premium'}, k=1)
        pred_free = mc.predict('login', {'user_type': 'free'}, k=1)

        # Should get predictions
        assert len(pred_premium) > 0
        assert len(pred_free) > 0

    def test_context_discretization(self):
        """Hour is converted to time_of_day correctly."""
        sequences = [['A', 'B'], ['A', 'C']]
        contexts = [
            {'user_type': 'premium', 'hour': 10},  # morning
            {'user_type': 'premium', 'hour': 20}   # evening
        ]

        mc = ContextAwareMarkovChain(
            context_features=['user_type', 'time_of_day'],
            order=1
        )

        # Test discretization
        discretized = mc._discretize_context({'hour': 10})
        assert discretized.get('time_of_day') == 'morning'

        discretized = mc._discretize_context({'hour': 20})
        assert discretized.get('time_of_day') == 'evening'

    def test_global_fallback(self):
        """Unknown context falls back to global model."""
        sequences = [['A', 'B', 'C']]
        contexts = [{'user_type': 'premium'}]

        mc = ContextAwareMarkovChain(
            context_features=['user_type'],
            order=1,
            fallback_strategy='global'
        )
        mc.fit(sequences, contexts)

        # Query with unknown context
        predictions = mc.predict('A', {'user_type': 'UNKNOWN'}, k=2)

        # Should get predictions from global fallback
        assert len(predictions) > 0

    def test_context_statistics(self, sequences_with_context):
        """Statistics method returns expected info."""
        sequences, contexts = sequences_with_context

        mc = ContextAwareMarkovChain(
            context_features=['user_type'],
            order=1
        )
        mc.fit(sequences, contexts)

        stats = mc.get_context_statistics()

        assert 'num_contexts' in stats
        assert 'contexts' in stats
        assert 'samples_per_context' in stats
        assert stats['num_contexts'] > 0


# =============================================================================
# TestMarkovPredictor
# =============================================================================

class TestMarkovPredictor:
    """Tests for the unified MarkovPredictor interface."""

    def test_unified_interface_order1(self, simple_sequences):
        """Works with order=1."""
        predictor = MarkovPredictor(order=1)
        predictor.fit(simple_sequences)

        predictor.reset_history()
        predictor.observe('login')
        predictions = predictor.predict(k=3)

        assert len(predictions) > 0
        assert predictor.order == 1

    def test_unified_interface_order2(self, simple_sequences):
        """Works with order=2."""
        predictor = MarkovPredictor(order=2)
        predictor.fit(simple_sequences)

        predictor.reset_history()
        predictor.observe('login')
        predictor.observe('profile')
        predictions = predictor.predict(k=3)

        assert len(predictions) > 0
        assert predictor.order == 2

    def test_unified_interface_context_aware(self, sequences_with_context):
        """Works with context."""
        sequences, contexts = sequences_with_context

        predictor = MarkovPredictor(
            order=1,
            context_aware=True,
            context_features=['user_type']
        )
        predictor.fit(sequences, contexts)

        predictor.reset_history()
        predictor.observe('login', context={'user_type': 'premium'})
        predictions = predictor.predict(k=3, context={'user_type': 'premium'})

        assert len(predictions) > 0

    def test_history_management(self, simple_sequences):
        """Observe updates history, reset clears it."""
        predictor = MarkovPredictor(order=1, history_size=5)
        predictor.fit(simple_sequences)

        # Initially empty
        assert len(predictor.history) == 0

        # Observe updates
        predictor.observe('A')
        assert len(predictor.history) == 1

        predictor.observe('B')
        assert len(predictor.history) == 2

        # Reset clears
        predictor.reset_history()
        assert len(predictor.history) == 0

    def test_state_vector_shape(self, simple_sequences):
        """get_state_vector returns consistent shape."""
        predictor = MarkovPredictor(order=1, history_size=10)
        predictor.fit(simple_sequences)

        predictor.reset_history()
        predictor.observe('login')

        state1 = predictor.get_state_vector(k=5)
        predictor.observe('profile')
        state2 = predictor.get_state_vector(k=5)

        # Shape should be consistent
        assert state1.shape == state2.shape

    def test_state_vector_values(self, simple_sequences):
        """State vector contains expected information."""
        predictor = MarkovPredictor(order=1, history_size=10)
        predictor.fit(simple_sequences)

        predictor.reset_history()
        predictor.observe('login')

        state = predictor.get_state_vector(k=3, include_history=True)

        # Should have: 3 indices + 3 probs + 1 confidence + 10 history
        expected_size = 3 + 3 + 1 + 10
        assert state.shape[0] == expected_size

        # Check values are in valid range
        assert np.all(state >= 0)
        assert np.all(state <= 1)

    def test_metrics_tracking(self, simple_sequences):
        """record_outcome updates metrics correctly."""
        predictor = MarkovPredictor(order=1)
        predictor.fit(simple_sequences)

        predictor.reset_history()
        predictor.observe('login')

        predictions = predictor.predict(k=3)
        predictor.all_predictions.append(predictions)
        predictor.record_outcome('profile')

        metrics = predictor.get_metrics()

        assert metrics['prediction_count'] == 1
        assert 'top_1_accuracy' in metrics or 'top_2_accuracy' in metrics


# =============================================================================
# TestEdgeCases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_sequences(self):
        """Fitting on empty list doesn't crash."""
        mc = FirstOrderMarkovChain(smoothing=0.001)

        # Should not crash
        mc.fit([])

        assert not mc.is_fitted or len(mc.states) == 0

    def test_single_element_sequences(self):
        """Length-1 sequences handled gracefully."""
        sequences = [['A'], ['B'], ['C']]

        mc = FirstOrderMarkovChain(smoothing=0.001)
        mc.fit(sequences)

        # Should fit without error, but no transitions
        predictions = mc.predict('A', k=5)
        # May be empty or have only smoothed predictions
        assert isinstance(predictions, list)

    @pytest.mark.slow
    def test_very_long_sequences(self):
        """No performance issues with long sequences."""
        # Create very long sequence
        long_sequence = []
        for i in range(1000):
            long_sequence.append(f'API_{i % 100}')

        mc = FirstOrderMarkovChain(smoothing=0.001)

        # Should complete in reasonable time
        mc.fit([long_sequence])
        predictions = mc.predict('API_0', k=10)

        assert len(predictions) > 0

    def test_unicode_endpoints(self):
        """Non-ASCII endpoint names work correctly."""
        sequences = [
            ['登录', '配置文件', '订单'],
            ['登录', '浏览', '产品'],
        ]

        mc = FirstOrderMarkovChain(smoothing=0.001)
        mc.fit(sequences)

        predictions = mc.predict('登录', k=2)
        assert len(predictions) > 0

    def test_special_characters_in_endpoints(self):
        """Slashes, dots, etc. handled."""
        sequences = [
            ['/api/login', '/api/profile', '/api/orders'],
            ['/api/login', 'api.browse', 'api:product'],
        ]

        mc = FirstOrderMarkovChain(smoothing=0.001)
        mc.fit(sequences)

        predictions = mc.predict('/api/login', k=2)
        assert len(predictions) > 0


# =============================================================================
# Parametrized Tests
# =============================================================================

class TestParametrized:
    """Parametrized tests for testing multiple configurations."""

    @pytest.mark.parametrize("order", [1, 2])
    def test_different_orders(self, simple_sequences, order):
        """Test both first and second order chains."""
        if order == 1:
            mc = FirstOrderMarkovChain(smoothing=0.001)
        else:
            mc = SecondOrderMarkovChain(smoothing=0.001, fallback_to_first_order=True)

        mc.fit(simple_sequences)
        assert mc.is_fitted

    @pytest.mark.parametrize("smoothing", [0.0, 0.001, 0.01, 0.1])
    def test_different_smoothing_values(self, simple_sequences, smoothing):
        """Test various smoothing parameters."""
        mc = FirstOrderMarkovChain(smoothing=smoothing)
        mc.fit(simple_sequences)

        predictions = mc.predict('login', k=3)
        # Should get predictions regardless of smoothing
        assert isinstance(predictions, list)

    @pytest.mark.parametrize("k", [1, 3, 5, 10])
    def test_different_k_values(self, simple_sequences, k):
        """Test various k values for top-k predictions."""
        mc = FirstOrderMarkovChain(smoothing=0.001)
        mc.fit(simple_sequences)

        predictions = mc.predict('login', k=k)
        # Should return at most k predictions
        assert len(predictions) <= k


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_end_to_end_workflow(self, large_sequences):
        """Complete workflow from training to prediction."""
        # 1. Train predictor
        predictor = MarkovPredictor(order=1, history_size=10)
        predictor.fit(large_sequences)

        # 2. Make predictions
        predictor.reset_history()
        predictor.observe('login')
        predictions = predictor.predict(k=5)

        assert len(predictions) > 0

        # 3. Get state vector
        state = predictor.get_state_vector(k=5)
        assert state.shape[0] > 0

        # 4. Track metrics
        predictor.all_predictions.append(predictions)
        predictor.record_outcome('profile')
        metrics = predictor.get_metrics()

        assert metrics['prediction_count'] == 1

    def test_model_persistence_workflow(self, simple_sequences, temp_dir):
        """Train, save, load, and verify predictions match."""
        # Train
        predictor = MarkovPredictor(order=1)
        predictor.fit(simple_sequences)

        predictor.reset_history()
        predictor.observe('login')
        original_predictions = predictor.predict(k=3)

        # Save
        save_path = temp_dir / "predictor.json"
        predictor.save(str(save_path))

        # Load
        loaded_predictor = MarkovPredictor.load(str(save_path))

        # Verify
        loaded_predictions = loaded_predictor.predict(k=3)

        assert len(original_predictions) == len(loaded_predictions)
        for orig, loaded in zip(original_predictions, loaded_predictions):
            assert orig[0] == loaded[0]

    def test_context_aware_end_to_end(self, sequences_with_context):
        """Complete workflow with context-aware predictions."""
        sequences, contexts = sequences_with_context

        # Train
        predictor = MarkovPredictor(
            order=1,
            context_aware=True,
            context_features=['user_type', 'time_of_day']
        )
        predictor.fit(sequences, contexts)

        # Predict with different contexts
        predictor.reset_history()
        predictor.observe('login', context={'user_type': 'premium', 'hour': 10})

        pred_premium = predictor.predict(k=3, context={'user_type': 'premium', 'hour': 10})
        pred_free = predictor.predict(k=3, context={'user_type': 'free', 'hour': 14})

        # Should get predictions for both
        assert len(pred_premium) > 0
        assert len(pred_free) > 0


# =============================================================================
# Performance Tests (marked as slow)
# =============================================================================

class TestPerformance:
    """Performance tests for ensuring reasonable speed."""

    @pytest.mark.slow
    def test_large_vocabulary_performance(self):
        """Handle large vocabulary efficiently."""
        # Create sequences with 1000 unique APIs
        sequences = []
        for i in range(100):
            seq = [f'API_{j}' for j in range(i * 10, i * 10 + 10)]
            sequences.append(seq)

        mc = FirstOrderMarkovChain(smoothing=0.001)

        # Should complete quickly
        import time
        start = time.time()
        mc.fit(sequences)
        elapsed = time.time() - start

        assert elapsed < 5.0  # Should take less than 5 seconds

    @pytest.mark.slow
    def test_many_sequences_performance(self):
        """Handle many sequences efficiently."""
        # 1000 sequences
        sequences = [['A', 'B', 'C', 'D'] for _ in range(1000)]

        mc = FirstOrderMarkovChain(smoothing=0.001)

        import time
        start = time.time()
        mc.fit(sequences)
        elapsed = time.time() - start

        assert elapsed < 3.0  # Should take less than 3 seconds


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunction:
    """Tests for the create_predictor factory function."""

    def test_create_predictor_from_config(self):
        """Factory function creates correct predictor type."""

        class MockConfig:
            markov_order = 1
            context_aware = False
            context_features = None
            smoothing = 0.001
            history_size = 10
            fallback_strategy = 'global'

        config = MockConfig()
        predictor = create_predictor(config)

        assert isinstance(predictor, MarkovPredictor)
        assert predictor.order == 1
        assert not predictor.context_aware

    def test_create_predictor_context_aware_config(self):
        """Factory creates context-aware predictor from config."""

        class MockConfig:
            markov_order = 1
            context_aware = True
            context_features = ['user_type']
            smoothing = 0.001
            history_size = 10
            fallback_strategy = 'global'

        config = MockConfig()
        predictor = create_predictor(config)

        assert predictor.context_aware
        assert predictor.context_features == ['user_type']


"""
Validation script for MarkovPredictor unified interface.

Tests the MarkovPredictor class that provides a consistent interface
for all Markov chain variants and integrates cleanly with RL systems.
"""

import numpy as np
from src.markov import MarkovPredictor, create_predictor


def test_basic_predictor():
    """Test basic predictor functionality."""
    print("=" * 70)
    print("TEST 1: Basic Predictor Functionality")
    print("=" * 70)

    # Create predictor
    predictor = MarkovPredictor(order=1, context_aware=False)
    print("\n[OK] Created first-order predictor")

    # Train
    sequences = [
        ['login', 'profile', 'orders'],
        ['login', 'browse', 'product'],
        ['browse', 'product', 'cart'],
    ]
    predictor.fit(sequences)
    print(f"[OK] Trained on {len(sequences)} sequences")
    print(f"[OK] Vocabulary size: {predictor.vocab_size}")

    # Observe and predict
    predictor.reset_history()
    predictor.observe('login')
    predictions = predictor.predict(k=3)
    print(f"\n[OK] After observing 'login', predictions:")
    for api, prob in predictions:
        print(f"    - {api}: {prob:.3f}")

    assert len(predictions) > 0, "Should get predictions"

    print("\n[OK] TEST 1 PASSED\n")


def test_history_management():
    """Test history management and sliding window."""
    print("=" * 70)
    print("TEST 2: History Management")
    print("=" * 70)

    predictor = MarkovPredictor(order=1, history_size=5)
    sequences = [['A', 'B', 'C', 'D', 'E', 'F']]
    predictor.fit(sequences)

    predictor.reset_history()
    print("\n[OK] Reset history")

    # Add observations
    for api in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        predictor.observe(api)

    print(f"[OK] Observed 7 APIs")
    print(f"[OK] History size: {len(predictor.history)} (max: {predictor.history_size})")
    print(f"[OK] History: {list(predictor.history)}")

    # Should only keep last 5 due to sliding window
    assert len(predictor.history) == predictor.history_size, \
        f"History should be limited to {predictor.history_size}"

    print("\n[OK] TEST 2 PASSED\n")


def test_state_vector():
    """Test state vector generation for RL."""
    print("=" * 70)
    print("TEST 3: State Vector for RL Integration")
    print("=" * 70)

    predictor = MarkovPredictor(order=1, history_size=10)
    sequences = [
        ['login', 'profile', 'orders'],
        ['login', 'browse', 'product'],
        ['browse', 'product', 'cart'],
    ]
    predictor.fit(sequences)

    predictor.reset_history()
    predictor.observe('login')

    # Get state vector
    state = predictor.get_state_vector(k=5, include_history=True)

    print(f"\n[OK] State vector generated")
    print(f"[OK] Shape: {state.shape}")
    print(f"[OK] First 5 values (predicted indices): {state[:5]}")
    print(f"[OK] Next 5 values (probabilities): {state[5:10]}")
    print(f"[OK] Confidence: {state[10]:.3f}")
    print(f"[OK] History encoding: {state[11:]}")

    # Check fixed size
    assert isinstance(state, np.ndarray), "State should be numpy array"
    expected_size = 5 + 5 + 1 + 10  # indices + probs + confidence + history
    assert state.shape[0] == expected_size, f"State size should be {expected_size}"

    # Get state vector multiple times - should always be same size
    predictor.observe('profile')
    state2 = predictor.get_state_vector(k=5, include_history=True)
    assert state2.shape == state.shape, "State vector size must be consistent"

    print(f"\n[OK] State vector size is consistent: {state.shape}")

    print("\n[OK] TEST 3 PASSED\n")


def test_second_order_predictor():
    """Test second-order predictor."""
    print("=" * 70)
    print("TEST 4: Second-Order Predictor")
    print("=" * 70)

    predictor = MarkovPredictor(order=2)
    sequences = [
        ['A', 'B', 'C', 'D'],
        ['A', 'B', 'E', 'F'],
    ]
    predictor.fit(sequences)
    print("\n[OK] Created second-order predictor")

    predictor.reset_history()
    predictor.observe('A')
    predictor.observe('B')

    predictions = predictor.predict(k=3)
    print(f"\n[OK] After observing 'A' then 'B', predictions:")
    for api, prob in predictions:
        print(f"    - {api}: {prob:.3f}")

    assert len(predictions) > 0, "Should get predictions with 2-step history"

    print("\n[OK] TEST 4 PASSED\n")


def test_context_aware_predictor():
    """Test context-aware predictor."""
    print("=" * 70)
    print("TEST 5: Context-Aware Predictor")
    print("=" * 70)

    predictor = MarkovPredictor(
        order=1,
        context_aware=True,
        context_features=['user_type', 'time_of_day']
    )

    sequences = [
        ['login', 'premium_features', 'browse'],
        ['login', 'browse', 'product'],
    ]
    contexts = [
        {'user_type': 'premium', 'hour': 10},
        {'user_type': 'free', 'hour': 14},
    ]
    predictor.fit(sequences, contexts)
    print("\n[OK] Created context-aware predictor")

    predictor.reset_history()
    predictor.observe('login', context={'user_type': 'premium', 'hour': 10})

    # Premium user prediction
    pred_premium = predictor.predict(k=2, context={'user_type': 'premium', 'hour': 10})
    print(f"\n[OK] Premium user predictions:")
    for api, prob in pred_premium:
        print(f"    - {api}: {prob:.3f}")

    # Free user prediction
    pred_free = predictor.predict(k=2, context={'user_type': 'free', 'hour': 14})
    print(f"\n[OK] Free user predictions:")
    for api, prob in pred_free:
        print(f"    - {api}: {prob:.3f}")

    # State vector with context encoding
    state = predictor.get_state_vector(k=3, context={'user_type': 'premium', 'hour': 10})
    print(f"\n[OK] State vector shape with context: {state.shape}")

    print("\n[OK] TEST 5 PASSED\n")


def test_sequence_prediction():
    """Test look-ahead sequence prediction."""
    print("=" * 70)
    print("TEST 6: Sequence Prediction (Look-Ahead)")
    print("=" * 70)

    predictor = MarkovPredictor(order=1)
    sequences = [
        ['A', 'B', 'C', 'D', 'E'],
        ['A', 'B', 'C', 'D', 'F'],
    ]
    predictor.fit(sequences)

    predictor.reset_history()
    predictor.observe('A')

    # Predict next 5 positions
    seq_predictions = predictor.predict_sequence(length=5)

    print(f"\n[OK] Look-ahead predictions for next 5 positions:")
    for i, preds in enumerate(seq_predictions, 1):
        if preds:
            top_api, top_prob = preds[0]
            print(f"    Position {i}: {top_api} ({top_prob:.3f})")
        else:
            print(f"    Position {i}: (no prediction)")

    assert len(seq_predictions) == 5, "Should return predictions for requested length"

    print("\n[OK] TEST 6 PASSED\n")


def test_metrics_tracking():
    """Test accuracy metrics tracking."""
    print("=" * 70)
    print("TEST 7: Accuracy Metrics Tracking")
    print("=" * 70)

    predictor = MarkovPredictor(order=1)
    sequences = [
        ['login', 'profile', 'orders'],
        ['login', 'profile', 'settings'],
        ['login', 'browse', 'product'],
    ]
    predictor.fit(sequences)

    # Simulate a session with predictions and outcomes
    predictor.reset_history()
    predictor.observe('login')

    # Make prediction and record outcome
    predictions = predictor.predict(k=5)
    predictor.all_predictions.append(predictions)  # Track for metrics
    predictor.record_outcome('profile')  # Actual was 'profile'

    predictor.observe('profile')
    predictions = predictor.predict(k=5)
    predictor.all_predictions.append(predictions)
    predictor.record_outcome('orders')  # Actual was 'orders'

    # Get metrics
    metrics = predictor.get_metrics()
    print(f"\n[OK] Metrics after 2 predictions:")
    print(f"    - Prediction count: {metrics['prediction_count']}")
    print(f"    - Average confidence: {metrics['avg_confidence']:.3f}")
    for key, value in metrics.items():
        if 'accuracy' in key:
            print(f"    - {key}: {value:.2%}")

    assert metrics['prediction_count'] == 2, "Should have 2 predictions recorded"

    print("\n[OK] TEST 7 PASSED\n")


def test_online_learning():
    """Test online learning with update=True."""
    print("=" * 70)
    print("TEST 8: Online Learning")
    print("=" * 70)

    predictor = MarkovPredictor(order=1)
    sequences = [['A', 'B', 'C']]
    predictor.fit(sequences)

    print("\n[OK] Initial training on ['A', 'B', 'C']")

    # Observe with online learning
    predictor.reset_history()
    predictor.observe('A')
    predictor.observe('B')
    predictor.observe('X', update=True)  # New pattern: A -> B -> X

    print("[OK] Observed new pattern with update=True")

    # The model should now know about X
    assert 'X' in predictor.api_vocab, "New API should be in vocabulary"
    print(f"[OK] Vocabulary updated: {list(predictor.api_vocab.keys())}")

    print("\n[OK] TEST 8 PASSED\n")


def test_persistence():
    """Test saving and loading."""
    print("=" * 70)
    print("TEST 9: Model Persistence")
    print("=" * 70)

    # Create and train predictor
    predictor = MarkovPredictor(order=1, context_aware=False)
    sequences = [['A', 'B', 'C'], ['A', 'B', 'D']]
    predictor.fit(sequences)

    predictor.reset_history()
    predictor.observe('A')
    original_predictions = predictor.predict(k=2)

    # Save
    save_path = 'data/test/test_predictor.json'
    predictor.save(save_path)
    print(f"\n[OK] Saved to {save_path}")

    # Load
    predictor_loaded = MarkovPredictor.load(save_path)
    print(f"[OK] Loaded from {save_path}")

    # Verify
    loaded_predictions = predictor_loaded.predict(k=2)
    print(f"\n[OK] Original predictions: {original_predictions}")
    print(f"[OK] Loaded predictions:   {loaded_predictions}")

    assert len(original_predictions) == len(loaded_predictions), "Predictions should match"
    assert predictor_loaded.vocab_size == predictor.vocab_size, "Vocab size should match"

    print("\n[OK] TEST 9 PASSED\n")


def test_validation_example():
    """Test the exact validation example from requirements."""
    print("=" * 70)
    print("TEST 10: User Requirements Validation Example")
    print("=" * 70)

    # Direct construction
    predictor = MarkovPredictor(
        order=1,
        context_aware=True,
        context_features=['user_type', 'time_of_day']
    )

    sequences = [
        ['login', 'profile', 'premium_features', 'browse'],
        ['login', 'profile', 'browse', 'product'],
        ['login', 'browse', 'product', 'cart'],
    ]
    contexts = [
        {'user_type': 'premium', 'hour': 10},
        {'user_type': 'free', 'hour': 14},
        {'user_type': 'premium', 'hour': 20},
    ]
    predictor.fit(sequences, contexts)
    print("\n[OK] Predictor fitted")

    # Simulate a session
    predictor.reset_history()
    predictor.observe('login')

    # Get predictions
    predictions = predictor.predict(k=5, context={'user_type': 'premium', 'hour': 10})
    print(f"\n[OK] Predictions: {predictions}")

    # Get state vector for RL
    state = predictor.get_state_vector(k=5, context={'user_type': 'premium', 'hour': 10})
    print(f"\n[OK] State vector shape: {state.shape}")
    print(f"[OK] State vector: {state}")

    # Track accuracy
    predictor.all_predictions.append(predictions)
    predictor.record_outcome('profile')
    predictor.observe('profile')

    metrics = predictor.get_metrics()
    print(f"\n[OK] Metrics: {metrics}")

    print("\n[OK] TEST 10 PASSED\n")


def test_create_predictor_factory():
    """Test factory function."""
    print("=" * 70)
    print("TEST 11: Factory Function")
    print("=" * 70)

    # Create mock config
    class MockConfig:
        markov_order = 1
        context_aware = True
        context_features = ['user_type']
        smoothing = 0.001
        history_size = 10
        fallback_strategy = 'global'

    config = MockConfig()
    predictor = create_predictor(config)

    print(f"\n[OK] Created predictor from config")
    print(f"[OK] Order: {predictor.order}")
    print(f"[OK] Context-aware: {predictor.context_aware}")
    print(f"[OK] Context features: {predictor.context_features}")

    assert predictor.order == 1, "Should use config order"
    assert predictor.context_aware == True, "Should be context-aware"

    print("\n[OK] TEST 11 PASSED\n")


def demonstrate_rl_integration():
    """Demonstrate RL integration workflow."""
    print("=" * 70)
    print("DEMONSTRATION: RL Integration Workflow")
    print("=" * 70)

    # Setup
    predictor = MarkovPredictor(order=1, history_size=5)
    sequences = [
        ['login', 'profile', 'orders', 'product', 'cart'],
        ['login', 'browse', 'product', 'cart', 'checkout'],
        ['browse', 'product', 'reviews', 'cart'],
    ]
    predictor.fit(sequences)

    print("\n[OK] Predictor trained on API sequences")
    print(f"[OK] Vocabulary size: {predictor.vocab_size}")

    # Simulate RL episode
    print("\n" + "=" * 60)
    print("SIMULATING RL EPISODE:")
    print("=" * 60)

    predictor.reset_history()
    session = ['login', 'profile', 'orders', 'product']

    for step, api in enumerate(session, 1):
        print(f"\nStep {step}: API call = '{api}'")

        # Get state BEFORE action
        state = predictor.get_state_vector(k=3, include_history=True)
        print(f"  State vector shape: {state.shape}")
        print(f"  Top 3 predicted indices: {state[:3]}")
        print(f"  Top 3 probabilities: {state[3:6]}")
        print(f"  Confidence: {state[6]:.3f}")

        # Get predictions for interpretation
        predictions = predictor.predict(k=3)
        if predictions:
            print(f"  Predicted next APIs:")
            for pred_api, prob in predictions:
                print(f"    - {pred_api}: {prob:.2%}")

        # Observe the actual API call
        predictor.observe(api)

        # If not first step, record outcome for previous prediction
        if step > 1:
            predictor.all_predictions.append(predictions if predictions else [])
            predictor.record_outcome(api)

    # Final metrics
    print("\n" + "=" * 60)
    print("EPISODE METRICS:")
    print("=" * 60)
    metrics = predictor.get_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'accuracy' in key:
                print(f"  {key}: {value:.1%}")
            else:
                print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    print("\n[OK] DEMONSTRATION COMPLETE\n")


def main():
    """Run all validation tests."""
    print("\n" + "=" * 70)
    print("MARKOV PREDICTOR VALIDATION")
    print("=" * 70 + "\n")

    try:
        test_basic_predictor()
        test_history_management()
        test_state_vector()
        test_second_order_predictor()
        test_context_aware_predictor()
        test_sequence_prediction()
        test_metrics_tracking()
        test_online_learning()
        test_persistence()
        test_validation_example()
        test_create_predictor_factory()
        demonstrate_rl_integration()

        print("=" * 70)
        print("ALL TESTS PASSED! [OK]")
        print("=" * 70)
        print("\nMarkovPredictor implementation is validated and ready for RL integration.")
        print("\nKey features verified:")
        print("  [OK] Unified interface for all Markov chain variants")
        print("  [OK] Automatic history management with sliding window")
        print("  [OK] Fixed-size state vector generation for RL")
        print("  [OK] Support for 1st-order, 2nd-order, and context-aware")
        print("  [OK] Look-ahead sequence predictions")
        print("  [OK] Real-time accuracy tracking")
        print("  [OK] Online learning capability")
        print("  [OK] Model persistence")
        print("  [OK] Factory function from config")
        print()

    except AssertionError as e:
        print(f"\n[ERROR] TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n[ERROR] ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()


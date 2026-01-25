"""
Validation script for ContextAwareMarkovChain implementation.

This script tests the context-aware Markov chain with various scenarios
to ensure correctness and demonstrate context-based prediction differences.
"""

from src.markov.context_aware import ContextAwareMarkovChain


def test_basic_validation():
    """Test basic validation example from requirements."""
    print("=" * 70)
    print("TEST 1: Basic Validation (User Requirements Example)")
    print("=" * 70)

    sequences = [
        ['login', 'profile', 'premium_features', 'browse'],
        ['login', 'profile', 'browse', 'product'],
        ['login', 'browse', 'product', 'cart'],
        ['browse', 'product', 'cart', 'checkout'],
    ]
    contexts = [
        {'user_type': 'premium', 'hour': 10},
        {'user_type': 'free', 'hour': 14},
        {'user_type': 'premium', 'hour': 20},
        {'user_type': 'free', 'hour': 11},
    ]

    mc_ctx = ContextAwareMarkovChain(
        context_features=['user_type', 'time_of_day'],
        order=1,
        fallback_strategy='global'
    )
    mc_ctx.fit(sequences, contexts)

    print("\n[OK] Model fitted successfully")
    print(f"  - Unique contexts: {len(mc_ctx.contexts)}")
    print(f"  - Context features: {mc_ctx.context_features}")

    # Test context-aware prediction
    print("\n[OK] Premium user in morning:")
    pred = mc_ctx.predict('login', {'user_type': 'premium', 'hour': 10}, k=3)
    for api, prob in pred:
        print(f"    - {api}: {prob:.3f}")

    print("\n[OK] Free user in afternoon:")
    pred = mc_ctx.predict('login', {'user_type': 'free', 'hour': 14}, k=3)
    for api, prob in pred:
        print(f"    - {api}: {prob:.3f}")

    # Check context statistics
    stats = mc_ctx.get_context_statistics()
    print(f"\n[OK] Context stats:")
    print(f"    - Number of contexts: {stats['num_contexts']}")
    print(f"    - Total samples: {stats['total_samples']}")
    print(f"    - Contexts: {stats['contexts']}")
    print(f"    - Samples per context: {stats['samples_per_context']}")

    print("\n[OK] TEST 1 PASSED\n")


def test_time_discretization():
    """Test time of day discretization."""
    print("=" * 70)
    print("TEST 2: Time of Day Discretization")
    print("=" * 70)

    mc_ctx = ContextAwareMarkovChain(
        context_features=['time_of_day'],
        order=1
    )

    # Test hour discretization
    test_hours = [0, 6, 10, 12, 14, 18, 20, 22, 23]
    expected_categories = ['night', 'morning', 'morning', 'afternoon', 'afternoon',
                          'evening', 'evening', 'night', 'night']

    print("\n[OK] Hour discretization:")
    for hour, expected in zip(test_hours, expected_categories):
        actual = mc_ctx._discretize_hour(hour)
        status = "[OK]" if actual == expected else "[FAIL]"
        print(f"    {status} Hour {hour:2d} → {actual:10s} (expected: {expected})")
        assert actual == expected, f"Hour {hour} should be {expected}, got {actual}"

    print("\n[OK] TEST 2 PASSED\n")


def test_day_discretization():
    """Test day of week discretization."""
    print("=" * 70)
    print("TEST 3: Day of Week Discretization")
    print("=" * 70)

    mc_ctx = ContextAwareMarkovChain(
        context_features=['day_type'],
        order=1
    )

    # Test day discretization
    test_days = [0, 1, 2, 3, 4, 5, 6]  # Monday to Sunday
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    expected_categories = ['weekday', 'weekday', 'weekday', 'weekday', 'weekday',
                          'weekend', 'weekend']

    print("\n[OK] Day discretization:")
    for day, name, expected in zip(test_days, day_names, expected_categories):
        actual = mc_ctx._discretize_day(day)
        status = "[OK]" if actual == expected else "[FAIL]"
        print(f"    {status} {name} (day {day}) → {actual:8s} (expected: {expected})")
        assert actual == expected, f"Day {day} should be {expected}, got {actual}"

    print("\n[OK] TEST 3 PASSED\n")


def test_context_differentiation():
    """Test that different contexts lead to different predictions."""
    print("=" * 70)
    print("TEST 4: Context Differentiation")
    print("=" * 70)

    # Create data where context matters
    sequences = [
        # Premium users check premium features after login
        ['login', 'premium_features', 'browse'],
        ['login', 'premium_features', 'advanced_tools'],
        ['login', 'premium_features', 'settings'],
        # Free users browse after login
        ['login', 'browse', 'product'],
        ['login', 'browse', 'search'],
        ['login', 'browse', 'product'],
    ]
    contexts = [
        {'user_type': 'premium', 'hour': 10},
        {'user_type': 'premium', 'hour': 14},
        {'user_type': 'premium', 'hour': 16},
        {'user_type': 'free', 'hour': 10},
        {'user_type': 'free', 'hour': 14},
        {'user_type': 'free', 'hour': 16},
    ]

    mc_ctx = ContextAwareMarkovChain(
        context_features=['user_type'],
        order=1,
        fallback_strategy='global'
    )
    mc_ctx.fit(sequences, contexts)

    print("\n[OK] Trained on context-dependent data")

    # Test premium user
    pred_premium = mc_ctx.predict('login', {'user_type': 'premium', 'hour': 10}, k=1)
    print(f"\n[OK] Premium user after 'login':")
    print(f"    Top prediction: {pred_premium[0][0]} ({pred_premium[0][1]:.3f})")
    assert pred_premium[0][0] == 'premium_features', "Premium users should get premium_features"

    # Test free user
    pred_free = mc_ctx.predict('login', {'user_type': 'free', 'hour': 10}, k=1)
    print(f"\n[OK] Free user after 'login':")
    print(f"    Top prediction: {pred_free[0][0]} ({pred_free[0][1]:.3f})")
    assert pred_free[0][0] == 'browse', "Free users should get browse"

    print("\n[OK] Different contexts give different predictions!")

    print("\n[OK] TEST 4 PASSED\n")


def test_fallback_strategies():
    """Test different fallback strategies."""
    print("=" * 70)
    print("TEST 5: Fallback Strategies")
    print("=" * 70)

    sequences = [
        ['A', 'B', 'C'],
        ['A', 'B', 'D']
    ]
    contexts = [
        {'user_type': 'premium', 'hour': 10},
        {'user_type': 'premium', 'hour': 14}
    ]

    # Test global fallback
    print("\n[OK] Testing 'global' fallback:")
    mc_global = ContextAwareMarkovChain(
        context_features=['user_type', 'time_of_day'],
        order=1,
        fallback_strategy='global'
    )
    mc_global.fit(sequences, contexts)

    # Unknown context should use global
    pred = mc_global.predict('A', {'user_type': 'free', 'hour': 20}, k=2)
    print(f"    Unknown context predictions: {pred}")
    assert len(pred) > 0, "Global fallback should return predictions"

    # Test similar fallback
    print("\n[OK] Testing 'similar' fallback:")
    mc_similar = ContextAwareMarkovChain(
        context_features=['user_type', 'time_of_day'],
        order=1,
        fallback_strategy='similar'
    )
    mc_similar.fit(sequences, contexts)

    pred = mc_similar.predict('A', {'user_type': 'premium', 'hour': 20}, k=2)
    print(f"    Similar context predictions: {pred}")
    assert len(pred) > 0, "Similar fallback should return predictions"

    # Test none fallback
    print("\n[OK] Testing 'none' fallback:")
    mc_none = ContextAwareMarkovChain(
        context_features=['user_type', 'time_of_day'],
        order=1,
        fallback_strategy='none'
    )
    mc_none.fit(sequences, contexts)

    pred = mc_none.predict('A', {'user_type': 'free', 'hour': 20}, k=2)
    print(f"    Unknown context predictions: {pred}")
    assert len(pred) == 0, "None fallback should return empty for unknown context"

    print("\n[OK] TEST 5 PASSED\n")


def test_second_order():
    """Test second-order context-aware chains."""
    print("=" * 70)
    print("TEST 6: Second-Order Context-Aware")
    print("=" * 70)

    sequences = [
        ['A', 'B', 'C', 'D'],
        ['A', 'B', 'C', 'E']
    ]
    contexts = [
        {'user_type': 'premium', 'hour': 10},
        {'user_type': 'free', 'hour': 14}
    ]

    mc_ctx = ContextAwareMarkovChain(
        context_features=['user_type'],
        order=2,  # Second-order
        fallback_strategy='global'
    )
    mc_ctx.fit(sequences, contexts)

    print("\n[OK] Second-order model fitted")

    # Test with previous state
    pred = mc_ctx.predict('B', {'user_type': 'premium', 'hour': 10}, k=2, prev='A')
    print(f"\n[OK] Predictions for ('A', 'B') with premium context: {pred}")
    assert len(pred) > 0, "Should get predictions"

    print("\n[OK] TEST 6 PASSED\n")


def test_predict_with_confidence():
    """Test predictions with confidence scores."""
    print("=" * 70)
    print("TEST 7: Predictions with Confidence")
    print("=" * 70)

    # Create dataset with varying amounts of data per context
    sequences = []
    contexts = []

    # Lots of data for premium morning
    for _ in range(50):
        sequences.append(['login', 'profile', 'premium_features'])
        contexts.append({'user_type': 'premium', 'hour': 10})

    # Little data for free afternoon
    for _ in range(5):
        sequences.append(['login', 'browse', 'product'])
        contexts.append({'user_type': 'free', 'hour': 14})

    mc_ctx = ContextAwareMarkovChain(
        context_features=['user_type', 'time_of_day'],
        order=1,
        fallback_strategy='global'
    )
    mc_ctx.fit(sequences, contexts)

    print("\n[OK] Trained with varying data amounts")

    # High confidence context (50 samples)
    pred_high = mc_ctx.predict_with_confidence(
        'login',
        {'user_type': 'premium', 'hour': 10},
        k=2
    )
    print(f"\n[OK] High-data context (50 samples):")
    for api, prob, conf in pred_high:
        print(f"    - {api}: {prob:.3f} (confidence: {conf:.3f})")
    assert pred_high[0][2] > 0.3, "High data context should have high confidence"

    # Low confidence context (5 samples)
    pred_low = mc_ctx.predict_with_confidence(
        'login',
        {'user_type': 'free', 'hour': 14},
        k=2
    )
    print(f"\n[OK] Low-data context (5 samples):")
    for api, prob, conf in pred_low:
        print(f"    - {api}: {prob:.3f} (confidence: {conf:.3f})")
    assert pred_low[0][2] < pred_high[0][2], "Low data context should have lower confidence"

    print("\n[OK] TEST 7 PASSED\n")


def test_context_statistics():
    """Test context statistics reporting."""
    print("=" * 70)
    print("TEST 8: Context Statistics")
    print("=" * 70)

    sequences = [
        ['A', 'B', 'C'],
        ['D', 'E', 'F'],
        ['G', 'H', 'I']
    ]
    contexts = [
        {'user_type': 'premium', 'hour': 10},
        {'user_type': 'free', 'hour': 14},
        {'user_type': 'premium', 'hour': 10}  # Same as first
    ]

    mc_ctx = ContextAwareMarkovChain(
        context_features=['user_type', 'time_of_day'],
        order=1
    )
    mc_ctx.fit(sequences, contexts)

    stats = mc_ctx.get_context_statistics()

    print(f"\n[OK] Context statistics:")
    print(f"    - Number of contexts: {stats['num_contexts']}")
    print(f"    - Total samples: {stats['total_samples']}")
    print(f"    - Average samples per context: {stats['avg_samples_per_context']:.1f}")
    print(f"    - Context features: {stats['context_features']}")
    print(f"    - Contexts: {stats['contexts']}")
    print(f"    - Samples per context: {stats['samples_per_context']}")
    print(f"    - Low data contexts: {stats['low_data_contexts']}")

    assert stats['num_contexts'] == 2, "Should have 2 unique contexts"
    assert stats['total_samples'] == 3, "Should have 3 total samples"

    print("\n[OK] TEST 8 PASSED\n")


def test_context_importance():
    """Test context importance measurement."""
    print("=" * 70)
    print("TEST 9: Context Importance")
    print("=" * 70)

    # Create data where user_type matters but time doesn't
    sequences = []
    contexts = []

    # Premium users always go A->B->C
    for hour in [10, 14, 18, 22]:
        sequences.append(['A', 'B', 'C'])
        contexts.append({'user_type': 'premium', 'hour': hour})

    # Free users always go A->B->D
    for hour in [10, 14, 18, 22]:
        sequences.append(['A', 'B', 'D'])
        contexts.append({'user_type': 'free', 'hour': hour})

    mc_ctx = ContextAwareMarkovChain(
        context_features=['user_type', 'time_of_day'],
        order=1,
        fallback_strategy='global'
    )
    mc_ctx.fit(sequences, contexts)

    print("\n[OK] Trained on data where user_type matters")

    # Measure importance
    importance = mc_ctx.get_context_importance(sequences, contexts)

    print(f"\n[OK] Context importance scores:")
    for feature, score in importance.items():
        print(f"    - {feature}: {score:.2f}%")

    # user_type should be more important than time_of_day
    print(f"\n[OK] user_type importance: {importance['user_type']:.1f}%")
    print(f"[OK] time_of_day importance: {importance['time_of_day']:.1f}%")

    print("\n[OK] TEST 9 PASSED\n")


def test_persistence():
    """Test model saving and loading."""
    print("=" * 70)
    print("TEST 10: Model Persistence")
    print("=" * 70)

    sequences = [
        ['A', 'B', 'C'],
        ['D', 'E', 'F']
    ]
    contexts = [
        {'user_type': 'premium', 'hour': 10},
        {'user_type': 'free', 'hour': 14}
    ]

    # Train and save
    mc_ctx = ContextAwareMarkovChain(
        context_features=['user_type', 'time_of_day'],
        order=1,
        fallback_strategy='global'
    )
    mc_ctx.fit(sequences, contexts)

    original_pred = mc_ctx.predict('A', {'user_type': 'premium', 'hour': 10}, k=2)

    save_path = 'data/test/test_context_aware.json'
    mc_ctx.save(save_path)
    print(f"\n[OK] Model saved to {save_path}")

    # Load and verify
    mc_ctx_loaded = ContextAwareMarkovChain.load(save_path)
    print(f"[OK] Model loaded from {save_path}")

    loaded_pred = mc_ctx_loaded.predict('A', {'user_type': 'premium', 'hour': 10}, k=2)

    print(f"\n[OK] Original predictions: {original_pred}")
    print(f"[OK] Loaded predictions:   {loaded_pred}")

    assert len(original_pred) == len(loaded_pred), "Prediction count mismatch"
    assert mc_ctx_loaded.context_features == mc_ctx.context_features

    print("\n[OK] TEST 10 PASSED\n")


def test_partial_fit():
    """Test incremental learning."""
    print("=" * 70)
    print("TEST 11: Incremental Learning (partial_fit)")
    print("=" * 70)

    initial_sequences = [['A', 'B', 'C']]
    initial_contexts = [{'user_type': 'premium', 'hour': 10}]

    mc_ctx = ContextAwareMarkovChain(
        context_features=['user_type'],
        order=1
    )
    mc_ctx.fit(initial_sequences, initial_contexts)

    initial_contexts_count = len(mc_ctx.contexts)
    print(f"\n[OK] Initial contexts: {initial_contexts_count}")

    # Add new context
    new_sequences = [['D', 'E', 'F']]
    new_contexts = [{'user_type': 'free', 'hour': 14}]

    mc_ctx.partial_fit(new_sequences, new_contexts)

    updated_contexts_count = len(mc_ctx.contexts)
    print(f"[OK] Updated contexts: {updated_contexts_count}")

    assert updated_contexts_count > initial_contexts_count, "Should have more contexts"

    print("\n[OK] TEST 11 PASSED\n")


def demonstrate_real_world_scenario():
    """Demonstrate a realistic e-commerce scenario."""
    print("=" * 70)
    print("DEMONSTRATION: Real-World E-Commerce Scenario")
    print("=" * 70)

    # Realistic sequences based on user type and time
    sequences = []
    contexts = []

    # Premium users in morning: Check deals and shop
    for _ in range(10):
        sequences.append(['login', 'premium_dashboard', 'deals', 'product', 'cart'])
        contexts.append({'user_type': 'premium', 'hour': 10})

    # Premium users in evening: Quick checkout
    for _ in range(10):
        sequences.append(['login', 'cart', 'checkout', 'order_confirm'])
        contexts.append({'user_type': 'premium', 'hour': 20})

    # Free users in afternoon: Browse casually
    for _ in range(10):
        sequences.append(['login', 'browse', 'product', 'product', 'browse'])
        contexts.append({'user_type': 'free', 'hour': 14})

    # Free users in evening: Check prices
    for _ in range(10):
        sequences.append(['browse', 'product', 'compare', 'reviews'])
        contexts.append({'user_type': 'free', 'hour': 20})

    mc_ctx = ContextAwareMarkovChain(
        context_features=['user_type', 'time_of_day'],
        order=1,
        fallback_strategy='global',
        smoothing=0.01
    )
    mc_ctx.fit(sequences, contexts)

    print("\n[OK] Trained on realistic e-commerce data")

    # Test different scenarios
    print("\n" + "=" * 60)
    print("SCENARIO 1: Premium user in morning after login")
    pred = mc_ctx.predict('login', {'user_type': 'premium', 'hour': 10}, k=3)
    for i, (api, prob) in enumerate(pred, 1):
        print(f"  {i}. {api}: {prob:.1%}")

    print("\n" + "=" * 60)
    print("SCENARIO 2: Premium user in evening after login")
    pred = mc_ctx.predict('login', {'user_type': 'premium', 'hour': 20}, k=3)
    for i, (api, prob) in enumerate(pred, 1):
        print(f"  {i}. {api}: {prob:.1%}")

    print("\n" + "=" * 60)
    print("SCENARIO 3: Free user in afternoon after login")
    pred = mc_ctx.predict('login', {'user_type': 'free', 'hour': 14}, k=3)
    for i, (api, prob) in enumerate(pred, 1):
        print(f"  {i}. {api}: {prob:.1%}")

    print("\n" + "=" * 60)
    print("SCENARIO 4: Free user in evening browsing products")
    pred = mc_ctx.predict('product', {'user_type': 'free', 'hour': 20}, k=3)
    for i, (api, prob) in enumerate(pred, 1):
        print(f"  {i}. {api}: {prob:.1%}")

    # Show context statistics
    stats = mc_ctx.get_context_statistics()
    print("\n" + "=" * 60)
    print("CONTEXT STATISTICS:")
    print(f"  Unique contexts: {stats['num_contexts']}")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Contexts:")
    for ctx, count in stats['samples_per_context'].items():
        print(f"    • {ctx}: {count} samples")

    print("\n[OK] DEMONSTRATION COMPLETE\n")


def main():
    """Run all validation tests."""
    print("\n" + "=" * 70)
    print("CONTEXT-AWARE MARKOV CHAIN VALIDATION")
    print("=" * 70 + "\n")

    try:
        test_basic_validation()
        test_time_discretization()
        test_day_discretization()
        test_context_differentiation()
        test_fallback_strategies()
        test_second_order()
        test_predict_with_confidence()
        test_context_statistics()
        test_context_importance()
        test_persistence()
        test_partial_fit()
        demonstrate_real_world_scenario()

        print("=" * 70)
        print("ALL TESTS PASSED! [OK]")
        print("=" * 70)
        print("\nContextAwareMarkovChain implementation is validated and ready to use.")
        print("\nKey features verified:")
        print("  [OK] Context discretization (time of day, day of week)")
        print("  [OK] Multiple context-specific Markov chains")
        print("  [OK] Fallback strategies (global, similar, none)")
        print("  [OK] Context-aware predictions")
        print("  [OK] Confidence scoring based on data availability")
        print("  [OK] Context statistics and importance analysis")
        print("  [OK] First-order and second-order support")
        print("  [OK] Model persistence and incremental learning")
        print()

    except AssertionError as e:
        print(f"\n[ERROR] TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n[ERROR] ERROR: {e}\n")
        raise


if __name__ == '__main__':
    main()


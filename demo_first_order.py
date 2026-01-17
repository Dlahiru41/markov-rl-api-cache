"""
Comprehensive demo of FirstOrderMarkovChain functionality.

Demonstrates all major features including training, prediction,
sequence generation, evaluation, and persistence.
"""

from src.markov.first_order import FirstOrderMarkovChain
import json


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_training():
    """Demonstrate training methods."""
    print_section("1. Training Methods")

    sequences = [
        ['login', 'profile', 'browse', 'product', 'cart'],
        ['login', 'profile', 'orders'],
        ['login', 'browse', 'product', 'product', 'cart', 'checkout'],
        ['browse', 'search', 'product', 'cart'],
    ]

    print("\nTraining sequences:")
    for i, seq in enumerate(sequences, 1):
        print(f"  {i}. {' → '.join(seq)}")

    # Fit model
    print("\n✓ Training model with fit()...")
    mc = FirstOrderMarkovChain(smoothing=0.001)
    mc.fit(sequences)

    print(f"  Fitted: {mc.is_fitted}")
    print(f"  Known states: {len(mc.states)}")
    print(f"  States: {sorted(mc.states)}")

    # Partial fit
    print("\n✓ Adding more data with partial_fit()...")
    new_sequences = [['checkout', 'confirmation']]
    mc.partial_fit(new_sequences)
    print(f"  States after partial_fit: {len(mc.states)}")

    # Update single transition
    print("\n✓ Adding single transition with update()...")
    mc.update('confirmation', 'logout')
    print(f"  Added: confirmation → logout")

    return mc


def demo_prediction(mc):
    """Demonstrate prediction methods."""
    print_section("2. Making Predictions")

    # Top-k predictions
    test_states = ['login', 'product', 'browse']

    for state in test_states:
        print(f"\nPredictions after '{state}':")
        predictions = mc.predict(state, k=3)

        if predictions:
            for i, (next_state, prob) in enumerate(predictions, 1):
                print(f"  {i}. {next_state:15s}: {prob:.2%}")
        else:
            print(f"  No predictions available")

    # Specific probability
    print(f"\nSpecific transition probabilities:")
    print(f"  P(profile | login) = {mc.predict_proba('login', 'profile'):.3f}")
    print(f"  P(cart | product) = {mc.predict_proba('product', 'cart'):.3f}")


def demo_sequence_generation(mc):
    """Demonstrate sequence generation."""
    print_section("3. Sequence Generation")

    print("\nGenerating synthetic sequences:")

    # Generate from different starting states
    for start_state in ['login', 'browse']:
        seq = mc.generate_sequence(start_state, length=8, seed=42)
        print(f"\n  Start: '{start_state}'")
        print(f"  Generated: {' → '.join(seq)}")

    # Generate with stop states
    print(f"\n  With stop state 'checkout':")
    seq = mc.generate_sequence('login', length=20, stop_states={'checkout'}, seed=123)
    print(f"  Generated: {' → '.join(seq)}")
    print(f"  Length: {len(seq)} (stopped at checkout)")


def demo_sequence_scoring(mc):
    """Demonstrate sequence scoring."""
    print_section("4. Sequence Scoring")

    test_sequences = [
        ['login', 'profile', 'orders'],  # Common pattern
        ['login', 'checkout'],  # Unusual (skips steps)
        ['browse', 'product', 'cart'],  # Common pattern
    ]

    print("\nScoring sequences (higher = more likely):")

    scores = []
    for seq in test_sequences:
        score = mc.score_sequence(seq)
        scores.append((seq, score))
        print(f"\n  {' → '.join(seq)}")
        if score == float('-inf'):
            print(f"  Score: -∞ (impossible transition)")
        else:
            print(f"  Score: {score:.3f}")

    # Find most/least likely
    scores.sort(key=lambda x: x[1], reverse=True)
    print(f"\nMost likely pattern: {' → '.join(scores[0][0])}")
    print(f"Least likely pattern: {' → '.join(scores[-1][0])}")


def demo_evaluation(mc):
    """Demonstrate evaluation metrics."""
    print_section("5. Model Evaluation")

    # Test sequences
    test_sequences = [
        ['login', 'profile', 'browse', 'product', 'cart'],
        ['login', 'browse', 'product', 'cart', 'checkout'],
        ['browse', 'search', 'product'],
    ]

    print("\nTest sequences:")
    for i, seq in enumerate(test_sequences, 1):
        print(f"  {i}. {' → '.join(seq)}")

    # Evaluate
    print("\nEvaluation metrics:")
    metrics = mc.evaluate(test_sequences, k_values=[1, 3, 5])

    print(f"\n  Accuracy Metrics:")
    print(f"    Top-1 accuracy: {metrics['top_1_accuracy']:.2%}")
    print(f"    Top-3 accuracy: {metrics['top_3_accuracy']:.2%}")
    print(f"    Top-5 accuracy: {metrics['top_5_accuracy']:.2%}")

    print(f"\n  Ranking Metrics:")
    print(f"    Mean Reciprocal Rank (MRR): {metrics['mrr']:.3f}")

    print(f"\n  Coverage & Uncertainty:")
    print(f"    Coverage: {metrics['coverage']:.2%} (% of states we can predict)")
    print(f"    Perplexity: {metrics['perplexity']:.3f} (lower = more confident)")


def demo_persistence(mc):
    """Demonstrate save/load functionality."""
    print_section("6. Model Persistence")

    # Save model
    model_path = "demo_markov_model.json"
    mc.save(model_path)
    print(f"\n✓ Saved model to '{model_path}'")

    # Load model
    mc_loaded = FirstOrderMarkovChain.load(model_path)
    print(f"✓ Loaded model from '{model_path}'")

    # Verify
    print(f"\nVerification:")
    print(f"  Original states: {len(mc.states)}")
    print(f"  Loaded states: {len(mc_loaded.states)}")
    print(f"  States match: {mc.states == mc_loaded.states}")

    # Test prediction match
    pred_original = mc.predict('login', k=2)
    pred_loaded = mc_loaded.predict('login', k=2)
    print(f"  Predictions match: {pred_original == pred_loaded}")

    # Clean up
    import os
    os.remove(model_path)
    print(f"\n✓ Cleaned up '{model_path}'")


def demo_statistics(mc):
    """Demonstrate statistics retrieval."""
    print_section("7. Model Statistics")

    stats = mc.get_statistics()

    print(f"\nModel Overview:")
    print(f"  Fitted: {stats['is_fitted']}")
    print(f"  Smoothing: {stats['smoothing']}")
    print(f"  Number of states: {stats['num_states']}")
    print(f"  Number of transitions: {stats['num_transitions']}")
    print(f"  Sparsity: {stats['sparsity']:.1%}")
    print(f"  Avg transitions per state: {stats['avg_transitions_per_state']:.2f}")

    print(f"\nMost Common Transitions:")
    for i, trans in enumerate(stats['most_common_transitions'], 1):
        print(f"  {i}. {trans['from']:12s} → {trans['to']:12s}: {trans['count']} times")


def demo_cache_prefetching(mc):
    """Demonstrate practical cache prefetching use case."""
    print_section("8. Practical Use Case: Cache Prefetching")

    print("\nScenario: User accesses an API endpoint")
    print("Question: Which endpoints should we prefetch to cache?")

    current_endpoint = 'browse'
    threshold = 0.10  # 10% probability threshold

    print(f"\nCurrent endpoint: '{current_endpoint}'")
    print(f"Prefetch threshold: {threshold:.0%}")

    # Get predictions
    predictions = mc.predict(current_endpoint, k=5)

    print(f"\nPredictions:")
    to_prefetch = []
    for i, (endpoint, prob) in enumerate(predictions, 1):
        decision = "✓ PREFETCH" if prob >= threshold else "✗ Skip"
        print(f"  {i}. {endpoint:15s}: {prob:.2%}  [{decision}]")

        if prob >= threshold:
            to_prefetch.append(endpoint)

    print(f"\nDecision: Prefetch {len(to_prefetch)} endpoints")
    print(f"Expected cache hit rate: {sum(p for _, p in predictions if p >= threshold):.1%}")

    print(f"\nExample cache integration code:")
    print(f"""
    def prefetch_likely_endpoints(current_endpoint):
        predictions = mc.predict(current_endpoint, k=5)
        for endpoint, prob in predictions:
            if prob >= 0.10:  # threshold
                cache.warm(endpoint)
    """)


def main():
    """Run all demonstrations."""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "FirstOrderMarkovChain Demo" + " " * 27 + "║")
    print("╚" + "═" * 68 + "╝")

    try:
        # Run all demos
        mc = demo_training()
        demo_prediction(mc)
        demo_sequence_generation(mc)
        demo_sequence_scoring(mc)
        demo_evaluation(mc)
        demo_persistence(mc)
        demo_statistics(mc)
        demo_cache_prefetching(mc)

        # Summary
        print_section("Summary")
        print("\n✓ All demonstrations completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  • Training: fit(), partial_fit(), update()")
        print("  • Prediction: predict(), predict_proba()")
        print("  • Generation: generate_sequence()")
        print("  • Scoring: score_sequence()")
        print("  • Evaluation: evaluate() with multiple metrics")
        print("  • Persistence: save() and load()")
        print("  • Statistics: get_statistics()")
        print("  • Real-world use case: cache prefetching")

        print("\nNext Steps:")
        print("  1. Read documentation: src/markov/first_order.py")
        print("  2. Run tests: pytest tests/unit/test_first_order.py")
        print("  3. Try with your own data!")
        print("  4. Integrate with API caching system")

    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())


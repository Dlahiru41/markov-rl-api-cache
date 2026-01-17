"""
Demo script for SecondOrderMarkovChain - matches user's validation example.

This demonstrates the basic usage of the second-order Markov chain
as specified in the user's requirements.
"""

from src.markov.second_order import SecondOrderMarkovChain


def main():
    """Run the demo as specified in requirements."""
    print("=" * 70)
    print("Second-Order Markov Chain Demo")
    print("=" * 70)

    # Define training sequences
    sequences = [
        ['login', 'profile', 'browse', 'product', 'cart'],
        ['login', 'profile', 'orders'],
        ['login', 'browse', 'product', 'product', 'cart', 'checkout'],
        ['browse', 'search', 'product', 'cart'],
    ]

    # Create and train the model
    print("\n1. Creating second-order Markov chain with fallback...")
    mc2 = SecondOrderMarkovChain(smoothing=0.001, fallback_to_first_order=True)

    print("2. Training on sequences...")
    mc2.fit(sequences)

    # Display known states
    print(f"\n3. Known states: {mc2.states}")

    # Test predictions for different contexts
    print(f"\n4. Predictions after 'login' → 'profile':")
    predictions = mc2.predict('login', 'profile', k=3)
    for api, prob in predictions:
        print(f"   - {api}: {prob:.3f}")

    print(f"\n5. Predictions after 'browse' → 'profile':")
    predictions = mc2.predict('browse', 'profile', k=3)
    if predictions:
        for api, prob in predictions:
            print(f"   - {api}: {prob:.3f}")
    else:
        print("   (No predictions - unseen pair, falling back to first-order)")

    # Test fallback with unseen pair
    print(f"\n6. Testing fallback with unseen pair 'xyz' → 'profile':")
    predictions = mc2.predict('xyz', 'profile', k=3)
    if predictions:
        print("   Using first-order fallback:")
        for api, prob in predictions:
            print(f"   - {api}: {prob:.3f}")
    else:
        print("   (No predictions available)")

    # Evaluate on training data
    print(f"\n7. Evaluating on training sequences...")
    metrics = mc2.evaluate(sequences, k_values=[1, 3])
    print(f"   - Top-1 accuracy: {metrics['top_1_accuracy']:.3f}")
    print(f"   - MRR: {metrics['mrr']:.3f}")
    print(f"   - Fallback rate: {metrics['fallback_rate']:.3f}")

    # Compare with first-order
    print(f"\n8. Comparing with first-order baseline...")
    comparison = mc2.compare_with_first_order(sequences)
    print(f"   Second-order accuracy: {comparison['second_order_metrics']['top_1_accuracy']:.3f}")
    print(f"   First-order accuracy: {comparison['first_order_metrics']['top_1_accuracy']:.3f}")
    improvement = comparison['improvement']['top_1_accuracy']
    print(f"   Improvement: {improvement:+.1f}%")
    print(f"   Fallback rate: {comparison['fallback_rate']:.3f}")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Second-order uses (previous, current) to predict next")
    print("  • Falls back to first-order for unseen state pairs")
    print("  • Captures context: 'login→profile' differs from 'browse→profile'")
    print("  • Better predictions when API patterns depend on history")
    print()


if __name__ == '__main__':
    main()


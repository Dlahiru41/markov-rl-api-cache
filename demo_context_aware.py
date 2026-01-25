"""
Demo script for ContextAwareMarkovChain - matches user's validation example.

This demonstrates the basic usage of the context-aware Markov chain
as specified in the user's requirements.
"""

from src.markov.context_aware import ContextAwareMarkovChain


def main():
    """Run the demo as specified in requirements."""
    print("=" * 70)
    print("Context-Aware Markov Chain Demo")
    print("=" * 70)

    # Define training data
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

    print("\n1. Creating context-aware Markov chain...")
    mc_ctx = ContextAwareMarkovChain(
        context_features=['user_type', 'time_of_day'],
        order=1,
        fallback_strategy='global'
    )

    print("2. Training on sequences with contexts...")
    mc_ctx.fit(sequences, contexts)
    print(f"   [OK] Trained on {len(sequences)} sequences")
    print(f"   [OK] Unique contexts: {len(mc_ctx.contexts)}")

    # Test context-aware predictions
    print("\n3. Premium user in morning:")
    pred = mc_ctx.predict('login', {'user_type': 'premium', 'hour': 10}, k=3)
    for api, prob in pred:
        print(f"   - {api}: {prob:.3f}")

    print("\n4. Free user in afternoon:")
    pred = mc_ctx.predict('login', {'user_type': 'free', 'hour': 14}, k=3)
    for api, prob in pred:
        print(f"   - {api}: {prob:.3f}")

    # Check context statistics
    print(f"\n5. Context stats:")
    stats = mc_ctx.get_context_statistics()
    print(f"   - Number of contexts: {stats['num_contexts']}")
    print(f"   - Total samples: {stats['total_samples']}")
    print(f"   - Contexts: {stats['contexts']}")
    print(f"   - Samples per context:")
    for ctx, count in stats['samples_per_context'].items():
        print(f"     • {ctx}: {count}")

    # Show how context affects predictions
    print("\n6. Demonstrating context-awareness:")
    print("\n   After 'browse' with different contexts:")

    pred_premium_morning = mc_ctx.predict('browse', {'user_type': 'premium', 'hour': 10}, k=2)
    print(f"\n   Premium user, morning:")
    for api, prob in pred_premium_morning:
        print(f"     - {api}: {prob:.3f}")

    pred_free_afternoon = mc_ctx.predict('browse', {'user_type': 'free', 'hour': 14}, k=2)
    print(f"\n   Free user, afternoon:")
    for api, prob in pred_free_afternoon:
        print(f"     - {api}: {prob:.3f}")

    # Test confidence scores
    print("\n7. Predictions with confidence:")
    pred_conf = mc_ctx.predict_with_confidence('login', {'user_type': 'premium', 'hour': 10}, k=3)
    for api, prob, conf in pred_conf:
        print(f"   - {api}: {prob:.3f} (confidence: {conf:.3f})")

    # Test fallback
    print("\n8. Testing fallback for unknown context:")
    pred_unknown = mc_ctx.predict('login', {'user_type': 'enterprise', 'hour': 3}, k=2)
    print(f"   Enterprise user at 3 AM (unknown context):")
    for api, prob in pred_unknown:
        print(f"   - {api}: {prob:.3f}")
    print(f"   → Used fallback strategy: '{mc_ctx.fallback_strategy}'")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Different user types have different behavior patterns")
    print("  • Time of day affects predictions (morning vs. evening)")
    print("  • Automatic fallback for unseen contexts")
    print("  • Confidence scores reflect data availability")
    print()


if __name__ == '__main__':
    main()


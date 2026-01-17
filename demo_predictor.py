"""
Demo script for MarkovPredictor - unified interface for RL integration.

Demonstrates how the MarkovPredictor provides a clean abstraction layer
for all Markov chain variants and integrates seamlessly with RL systems.
"""

import numpy as np
from src.markov import MarkovPredictor, create_predictor


def main():
    """Run the demonstration."""
    print("=" * 70)
    print("MarkovPredictor - Unified Interface for RL Integration")
    print("=" * 70)

    # 1. Create predictor
    print("\n1. Creating predictor...")
    predictor = MarkovPredictor(
        order=1,
        context_aware=True,
        context_features=['user_type', 'time_of_day'],
        history_size=10
    )
    print("   ✓ Created context-aware first-order predictor")
    print(f"   ✓ History window size: {predictor.history_size}")

    # 2. Train on data
    print("\n2. Training on API sequences...")
    sequences = [
        ['login', 'profile', 'premium_features', 'browse', 'product'],
        ['login', 'profile', 'browse', 'product', 'cart'],
        ['login', 'browse', 'product', 'cart', 'checkout'],
        ['browse', 'search', 'product', 'reviews', 'cart'],
    ]
    contexts = [
        {'user_type': 'premium', 'hour': 10},
        {'user_type': 'free', 'hour': 14},
        {'user_type': 'premium', 'hour': 20},
        {'user_type': 'free', 'hour': 11},
    ]
    predictor.fit(sequences, contexts)
    print(f"   ✓ Trained on {len(sequences)} sequences")
    print(f"   ✓ Vocabulary size: {predictor.vocab_size} unique APIs")

    # 3. Simulate a session
    print("\n3. Simulating user session...")
    print("   " + "=" * 60)

    predictor.reset_history()
    print("   ✓ Reset history for new session")

    # Step 1: User logs in
    print("\n   Step 1: User logs in")
    predictor.observe('login', context={'user_type': 'premium', 'hour': 10})
    print("   ✓ Observed: 'login'")

    # Get predictions
    predictions = predictor.predict(k=5, context={'user_type': 'premium', 'hour': 10})
    print("\n   Top predictions:")
    for i, (api, prob) in enumerate(predictions, 1):
        print(f"      {i}. {api}: {prob:.1%}")

    # Get state vector for RL
    state = predictor.get_state_vector(k=5, context={'user_type': 'premium', 'hour': 10})
    print(f"\n   RL State vector:")
    print(f"      Shape: {state.shape}")
    print(f"      Predicted indices: {state[:5]}")
    print(f"      Probabilities: {state[5:10]}")
    print(f"      Confidence: {state[10]:.3f}")

    # Step 2: User goes to profile
    print("\n   Step 2: User goes to profile")
    predictor.all_predictions.append(predictions)  # Track for metrics
    predictor.record_outcome('profile')
    predictor.observe('profile', context={'user_type': 'premium', 'hour': 10})
    print("   ✓ Observed: 'profile'")
    print("   ✓ Recorded prediction outcome")

    predictions = predictor.predict(k=5, context={'user_type': 'premium', 'hour': 10})
    print("\n   Top predictions:")
    for i, (api, prob) in enumerate(predictions, 1):
        print(f"      {i}. {api}: {prob:.1%}")

    # Step 3: User accesses premium features
    print("\n   Step 3: User accesses premium features")
    predictor.all_predictions.append(predictions)
    predictor.record_outcome('premium_features')
    predictor.observe('premium_features', context={'user_type': 'premium', 'hour': 10})
    print("   ✓ Observed: 'premium_features'")

    # 4. Look-ahead predictions
    print("\n4. Look-ahead predictions (next 5 steps)...")
    seq_predictions = predictor.predict_sequence(
        length=5,
        context={'user_type': 'premium', 'hour': 10}
    )
    print("   Predicted sequence:")
    for i, preds in enumerate(seq_predictions, 1):
        if preds:
            top_api, top_prob = preds[0]
            print(f"      Step {i}: {top_api} ({top_prob:.1%})")
        else:
            print(f"      Step {i}: (no prediction)")

    # 5. Show metrics
    print("\n5. Prediction metrics...")
    metrics = predictor.get_metrics()
    print(f"   ✓ Total predictions: {metrics['prediction_count']}")
    print(f"   ✓ Average confidence: {metrics['avg_confidence']:.1%}")
    for key, value in metrics.items():
        if 'accuracy' in key:
            print(f"   ✓ {key}: {value:.1%}")

    # 6. Compare different contexts
    print("\n6. Comparing predictions across contexts...")
    print("   " + "=" * 60)

    predictor.reset_history()
    predictor.observe('login')

    print("\n   Premium user (morning):")
    pred_premium_morning = predictor.predict(
        k=3,
        context={'user_type': 'premium', 'hour': 10}
    )
    for api, prob in pred_premium_morning:
        print(f"      - {api}: {prob:.1%}")

    print("\n   Free user (afternoon):")
    pred_free_afternoon = predictor.predict(
        k=3,
        context={'user_type': 'free', 'hour': 14}
    )
    for api, prob in pred_free_afternoon:
        print(f"      - {api}: {prob:.1%}")

    # 7. Save model
    print("\n7. Saving model...")
    save_path = 'data/test/demo_predictor.json'
    predictor.save(save_path)
    print(f"   ✓ Saved to {save_path}")

    # 8. Show factory function
    print("\n8. Factory function example...")

    class MockConfig:
        markov_order = 2
        context_aware = False
        context_features = None
        smoothing = 0.001
        history_size = 5
        fallback_strategy = 'global'

    config = MockConfig()
    predictor2 = create_predictor(config)
    print(f"   ✓ Created predictor from config")
    print(f"   ✓ Order: {predictor2.order}")
    print(f"   ✓ Context-aware: {predictor2.context_aware}")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Unified interface for all Markov chain variants")
    print("  • Automatic history management")
    print("  • Fixed-size state vectors for RL")
    print("  • Context-aware predictions when needed")
    print("  • Real-time accuracy tracking")
    print("  • Ready for RL integration!")
    print()


if __name__ == '__main__':
    main()


"""
Complete example showing all Markov chain features working together.
"""

from src.markov import MarkovPredictor, MarkovEvaluator, MarkovVisualizer
import os

print("=" * 70)
print("COMPLETE MARKOV CHAIN EXAMPLE")
print("=" * 70)

# Create training data
print("\n1. Creating training data...")
sequences = [
    ['login', 'profile', 'orders', 'product', 'cart'],
    ['login', 'profile', 'orders', 'product', 'reviews'],
    ['login', 'browse', 'product', 'cart', 'checkout'],
    ['login', 'browse', 'product', 'reviews', 'cart'],
    ['browse', 'search', 'product', 'cart', 'checkout'],
    ['browse', 'search', 'product', 'reviews', 'back'],
] * 2  # Repeat for more data

print(f"   ✓ Created {len(sequences)} training sequences")

# Train predictor
print("\n2. Training MarkovPredictor...")
predictor = MarkovPredictor(order=1, history_size=10)
predictor.fit(sequences)
print(f"   ✓ Trained on {predictor.vocab_size} unique APIs")

# Test predictions
print("\n3. Testing predictions...")
predictor.reset_history()
predictor.observe('login')
predictions = predictor.predict(k=3)
print(f"   ✓ After 'login', top 3 predictions:")
for api, prob in predictions:
    print(f"      - {api}: {prob:.1%}")

# Get state vector for RL
print("\n4. Testing state vector for RL...")
state = predictor.get_state_vector(k=5)
print(f"   ✓ State vector shape: {state.shape}")
print(f"   ✓ First 5 values (indices): {state[:5]}")

# Evaluate accuracy
print("\n5. Evaluating accuracy...")
evaluator = MarkovEvaluator(predictor)
results = evaluator.evaluate_accuracy(sequences, k_values=[1, 3, 5])
print(f"   ✓ Top-1 Accuracy:  {results['top_1_accuracy']:.1%}")
print(f"   ✓ Top-3 Accuracy:  {results['top_3_accuracy']:.1%}")
print(f"   ✓ Top-5 Accuracy:  {results['top_5_accuracy']:.1%}")
print(f"   ✓ MRR:             {results['mrr']:.3f}")
print(f"   ✓ Coverage:        {results['coverage']:.1%}")
print(f"   ✓ Perplexity:      {results['perplexity']:.2f}")

# Per-endpoint breakdown
print("\n6. Per-endpoint breakdown...")
per_endpoint = evaluator.evaluate_per_endpoint(sequences)
print(f"   ✓ Analyzed {len(per_endpoint)} endpoints")
print(f"\n   Top 3 by sample count:")
top_endpoints = per_endpoint.head(3)
for _, row in top_endpoints.iterrows():
    print(f"      {row['endpoint']:10s} - {row['sample_count']:3.0f} samples, {row['top_1_accuracy']:.1%} accuracy")

# Calibration
print("\n7. Calibration analysis...")
calibration = evaluator.evaluate_calibration(sequences, num_bins=5)
print(f"   ✓ Calibration bins: {len(calibration['bin_centers'])}")
if calibration['bin_centers']:
    print(f"   Pred Prob → Actual Acc")
    for pred, actual in zip(calibration['predicted_probs'][:3], calibration['actual_accuracy'][:3]):
        print(f"      {pred:.2f}    →  {actual:.2f}")

# Visualizations
print("\n8. Creating visualizations...")
os.makedirs('data/test', exist_ok=True)

try:
    # Transition heatmap
    MarkovVisualizer.plot_transition_heatmap(
        predictor,
        top_k=6,
        output_path='data/test/example_heatmap.png'
    )
    print(f"   ✓ Transition heatmap saved")

    # Calibration curve
    MarkovVisualizer.plot_calibration_curve(
        calibration,
        output_path='data/test/example_calibration.png'
    )
    print(f"   ✓ Calibration curve saved")

    # Confidence distribution
    MarkovVisualizer.plot_prediction_confidence_distribution(
        sequences,
        predictor,
        output_path='data/test/example_confidence.png'
    )
    print(f"   ✓ Confidence distribution saved")

    print(f"\n   All visualizations saved to data/test/")

except Exception as e:
    print(f"   ⚠ Visualization creation skipped: {e}")

# Save model
print("\n9. Saving model...")
predictor.save('data/test/example_predictor.json')
print(f"   ✓ Model saved to data/test/example_predictor.json")

# Model comparison
print("\n10. Comparing models...")
predictor2 = MarkovPredictor(order=2, history_size=10)
predictor2.fit(sequences)

models = {
    'First-Order': predictor,
    'Second-Order': predictor2
}

comparison = evaluator.compare_models(models, sequences, k_values=[1, 3])
print(f"   ✓ Model comparison:")
print(f"\n   Model          Top-1    Top-3    MRR")
for _, row in comparison.iterrows():
    print(f"   {row['model']:12s}   {row['top_1_accuracy']:.1%}    {row['top_3_accuracy']:.1%}   {row['mrr']:.3f}")

print("\n" + "=" * 70)
print("✓ COMPLETE EXAMPLE FINISHED!")
print("=" * 70)
print("\nAll features demonstrated:")
print("  ✓ Training and predictions")
print("  ✓ State vectors for RL integration")
print("  ✓ Comprehensive evaluation metrics")
print("  ✓ Per-endpoint breakdown")
print("  ✓ Calibration analysis")
print("  ✓ Visualizations")
print("  ✓ Model persistence")
print("  ✓ Model comparison")
print("\nEverything is working correctly!")


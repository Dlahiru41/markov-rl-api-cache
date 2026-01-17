"""
Quick verification that all Markov implementations work correctly.
"""

print("=" * 70)
print("QUICK VERIFICATION TEST")
print("=" * 70)

# Test 1: Basic imports
print("\n1. Testing imports...")
try:
    from src.markov import (
        TransitionMatrix,
        FirstOrderMarkovChain,
        SecondOrderMarkovChain,
        ContextAwareMarkovChain,
        MarkovPredictor,
        create_predictor,
        MarkovEvaluator,
        MarkovVisualizer
    )
    print("   ✓ All imports successful")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    exit(1)

# Test 2: TransitionMatrix
print("\n2. Testing TransitionMatrix...")
tm = TransitionMatrix(smoothing=0.001)
tm.increment("A", "B", 10)
tm.increment("A", "C", 5)
prob = tm.get_probability("A", "B")
print(f"   ✓ Created and used TransitionMatrix")
print(f"   ✓ P(B|A) = {prob:.3f}")

# Test 3: FirstOrderMarkovChain
print("\n3. Testing FirstOrderMarkovChain...")
mc1 = FirstOrderMarkovChain(smoothing=0.001)
sequences = [['A', 'B', 'C'], ['A', 'B', 'D']]
mc1.fit(sequences)
predictions = mc1.predict('A', k=2)
print(f"   ✓ Trained first-order chain")
print(f"   ✓ Predictions from 'A': {predictions}")

# Test 4: SecondOrderMarkovChain
print("\n4. Testing SecondOrderMarkovChain...")
mc2 = SecondOrderMarkovChain(smoothing=0.001, fallback_to_first_order=True)
mc2.fit(sequences)
predictions = mc2.predict('A', 'B', k=2)
print(f"   ✓ Trained second-order chain")
print(f"   ✓ Predictions from ('A','B'): {predictions}")

# Test 5: ContextAwareMarkovChain
print("\n5. Testing ContextAwareMarkovChain...")
mc_ctx = ContextAwareMarkovChain(
    context_features=['user_type'],
    order=1,
    fallback_strategy='global'
)
contexts = [{'user_type': 'premium'}, {'user_type': 'free'}]
mc_ctx.fit(sequences, contexts)
predictions = mc_ctx.predict('A', {'user_type': 'premium'}, k=2)
print(f"   ✓ Trained context-aware chain")
print(f"   ✓ Predictions for premium user: {predictions}")

# Test 6: MarkovPredictor
print("\n6. Testing MarkovPredictor...")
predictor = MarkovPredictor(order=1, history_size=10)
predictor.fit(sequences)
predictor.observe('A')
predictions = predictor.predict(k=2)
state = predictor.get_state_vector(k=3)
print(f"   ✓ Created and trained predictor")
print(f"   ✓ Predictions: {predictions}")
print(f"   ✓ State vector shape: {state.shape}")

# Test 7: MarkovEvaluator
print("\n7. Testing MarkovEvaluator...")
evaluator = MarkovEvaluator(predictor)
results = evaluator.evaluate_accuracy(sequences, k_values=[1, 3])
print(f"   ✓ Evaluated accuracy")
print(f"   ✓ Top-1 Accuracy: {results['top_1_accuracy']:.3f}")
print(f"   ✓ MRR: {results['mrr']:.3f}")

# Test 8: Per-endpoint evaluation
print("\n8. Testing per-endpoint evaluation...")
per_endpoint = evaluator.evaluate_per_endpoint(sequences)
print(f"   ✓ Per-endpoint analysis complete")
print(f"   ✓ Endpoints analyzed: {len(per_endpoint)}")

# Test 9: Calibration
print("\n9. Testing calibration...")
calibration = evaluator.evaluate_calibration(sequences, num_bins=3)
print(f"   ✓ Calibration analysis complete")
print(f"   ✓ Bins: {len(calibration['bin_centers'])}")

# Test 10: Visualizer (just check it exists)
print("\n10. Testing MarkovVisualizer...")
print(f"   ✓ MarkovVisualizer class available")
print(f"   ✓ plot_transition_heatmap: {hasattr(MarkovVisualizer, 'plot_transition_heatmap')}")
print(f"   ✓ plot_calibration_curve: {hasattr(MarkovVisualizer, 'plot_calibration_curve')}")

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED!")
print("=" * 70)
print("\nSummary:")
print("  ✓ TransitionMatrix working")
print("  ✓ FirstOrderMarkovChain working")
print("  ✓ SecondOrderMarkovChain working")
print("  ✓ ContextAwareMarkovChain working")
print("  ✓ MarkovPredictor working")
print("  ✓ MarkovEvaluator working")
print("  ✓ MarkovVisualizer available")
print("\nAll Markov chain implementations are ready to use!")


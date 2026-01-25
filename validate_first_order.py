"""
Validation script for FirstOrderMarkovChain implementation.
Tests the exact example from the user's request.
"""

from src.markov.first_order import FirstOrderMarkovChain

# Test data from user's request
sequences = [
    ['login', 'profile', 'browse', 'product', 'cart'],
    ['login', 'profile', 'orders'],
    ['login', 'browse', 'product', 'product', 'cart', 'checkout'],
    ['browse', 'search', 'product', 'cart'],
]

print("=" * 70)
print("FirstOrderMarkovChain Validation")
print("=" * 70)

# Create and train model
print("\n1. Training model on sequences...")
mc = FirstOrderMarkovChain(smoothing=0.001)
mc.fit(sequences)
print(f"   [OK] Model trained")

# Known states
print(f"\n2. Known states: {mc.states}")
print(f"   Number of states: {len(mc.states)}")

# Predictions after 'login'
print(f"\n3. Predictions after 'login':")
login_predictions = mc.predict('login', k=3)
for i, (state, prob) in enumerate(login_predictions, 1):
    print(f"   {i}. {state:15s}: {prob:.3f}")

# Predictions after 'product'
print(f"\n4. Predictions after 'product':")
product_predictions = mc.predict('product', k=3)
for i, (state, prob) in enumerate(product_predictions, 1):
    print(f"   {i}. {state:15s}: {prob:.3f}")

# Evaluate metrics
print(f"\n5. Evaluation metrics:")
metrics = mc.evaluate(sequences, k_values=[1, 3])
print(f"   Top-1 accuracy: {metrics['top_1_accuracy']:.3f}")
print(f"   Top-3 accuracy: {metrics['top_3_accuracy']:.3f}")
print(f"   MRR: {metrics['mrr']:.3f}")
print(f"   Coverage: {metrics['coverage']:.3f}")
print(f"   Perplexity: {metrics['perplexity']:.3f}")

print("\n" + "=" * 70)
print("[OK] Validation Complete!")
print("=" * 70)


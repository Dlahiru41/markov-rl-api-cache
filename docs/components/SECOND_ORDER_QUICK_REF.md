# Second-Order Markov Chain Implementation

## Overview

The `SecondOrderMarkovChain` class implements a second-order Markov model that uses the last **two** API calls to predict the next one, modeling `P(next | current, previous)`. This captures more context than first-order chains and can significantly improve prediction accuracy when API access patterns depend on history.

## Key Features

### 1. **Context-Aware Predictions**
Second-order models capture sequential context. For example:
- After `login → profile`, users typically go to `orders`
- After `browse → profile`, users typically go to `settings`

A first-order model only sees `profile` and averages both patterns, while second-order distinguishes them.

### 2. **Automatic Fallback**
When encountering unseen state pairs `(previous, current)`, the model can automatically fall back to first-order predictions using just `current`. This ensures robust predictions even with limited training data.

### 3. **Efficient Sparse Storage**
Uses the same efficient `TransitionMatrix` backend as first-order, with composite state keys like `"login|profile"` to represent state pairs.

### 4. **Seamless Integration**
API matches `FirstOrderMarkovChain` for easy drop-in replacement and comparison.

## When to Use Second-Order vs First-Order

### Use Second-Order When:
- ✅ API access patterns depend on context/history
- ✅ You have sufficient training data (thousands+ of sequences)
- ✅ Sequence patterns are non-Markovian at first-order
- ✅ Prediction accuracy is critical

### Use First-Order When:
- ✅ Limited training data (hundreds of sequences)
- ✅ API access is mostly local/independent
- ✅ Need simple, interpretable model
- ✅ Memory/storage is constrained

### Key Tradeoffs

| Aspect | First-Order | Second-Order |
|--------|-------------|--------------|
| **Context** | Current state only | Previous + current states |
| **Data needed** | Less (100s of sequences) | More (1000s+ of sequences) |
| **State space** | O(n) states | O(n²) state pairs |
| **Sparsity** | Moderate | Higher (more zero entries) |
| **Accuracy** | Good baseline | Better with enough data |
| **Speed** | Slightly faster | Slightly slower |

## Quick Start

```python
from src.markov.second_order import SecondOrderMarkovChain

# Create model with fallback enabled
mc2 = SecondOrderMarkovChain(
    smoothing=0.001,
    fallback_to_first_order=True
)

# Train on sequences
sequences = [
    ['login', 'profile', 'orders'],
    ['login', 'browse', 'product'],
    ['browse', 'profile', 'settings']
]
mc2.fit(sequences)

# Make predictions with context
predictions = mc2.predict('login', 'profile', k=3)
# Returns: [('orders', 0.99), ...]

# Compare different contexts
pred1 = mc2.predict('login', 'profile', k=1)
pred2 = mc2.predict('browse', 'profile', k=1)
# Different top predictions!
```

## API Reference

### Constructor

```python
SecondOrderMarkovChain(smoothing=0.0, fallback_to_first_order=True)
```

**Parameters:**
- `smoothing` (float): Laplace smoothing parameter (default: 0.0)
  - Recommended: 0.001 - 0.01 for robust predictions
- `fallback_to_first_order` (bool): Enable first-order fallback (default: True)
  - When True, trains a first-order model alongside second-order
  - Falls back when encountering unseen state pairs

### Training Methods

#### `fit(sequences)`
Train model on sequences (resets existing model).

```python
mc2.fit([
    ['A', 'B', 'C'],
    ['A', 'B', 'D']
])
```

**Parameters:**
- `sequences`: List of sequences, each a list of API strings

**Returns:** Self (for chaining)

#### `partial_fit(sequences)`
Incrementally update model with new data.

```python
mc2.partial_fit([['A', 'C', 'E']])  # Add new patterns
```

#### `update(previous, current, next_state, count=1)`
Add single transition observation.

```python
mc2.update('login', 'profile', 'orders', count=10)
```

### Prediction Methods

#### `predict(previous, current, k=5, use_fallback=True)`
Get top-k most likely next states.

```python
predictions = mc2.predict('login', 'profile', k=3)
# Returns: [('orders', 0.8), ('settings', 0.15), ...]
```

**Parameters:**
- `previous` (str): Previous API endpoint
- `current` (str): Current API endpoint  
- `k` (int): Number of predictions to return
- `use_fallback` (bool): Use first-order fallback if pair unseen

**Returns:** List of `(api, probability)` tuples, sorted by probability descending

#### `predict_proba(previous, current, target, use_fallback=True)`
Get probability of specific transition.

```python
prob = mc2.predict_proba('login', 'profile', 'orders')
# Returns: 0.8
```

### Generation Methods

#### `generate_sequence(start, length=10, stop_states=None, seed=None, use_fallback=True)`
Generate synthetic sequence.

```python
seq = mc2.generate_sequence('login', length=10, seed=42)
# Returns: ['login', 'profile', 'orders', ...]
```

#### `score_sequence(sequence, use_fallback=True)`
Calculate log-likelihood of sequence.

```python
score = mc2.score_sequence(['login', 'profile', 'orders'])
# Returns: -1.5 (higher = more likely)
```

Useful for anomaly detection - unusual sequences have low scores.

### Evaluation Methods

#### `evaluate(test_sequences, k_values=[1,3,5], track_fallback=True)`
Comprehensive evaluation on test data.

```python
metrics = mc2.evaluate(test_sequences, k_values=[1, 3, 5])
print(metrics)
# {
#     'top_1_accuracy': 0.85,
#     'top_3_accuracy': 0.95,
#     'top_5_accuracy': 0.98,
#     'mrr': 0.89,
#     'coverage': 1.0,
#     'perplexity': 1.5,
#     'fallback_rate': 0.12
# }
```

**Metrics:**
- `top_k_accuracy`: Fraction of correct predictions in top-k
- `mrr`: Mean Reciprocal Rank (1/rank of correct answer)
- `coverage`: Fraction of transitions we could predict
- `perplexity`: Uncertainty measure (lower = better)
- `fallback_rate`: Fraction of predictions using first-order fallback

#### `compare_with_first_order(test_sequences, k_values=[1,3,5])`
Direct comparison with first-order baseline.

```python
comparison = mc2.compare_with_first_order(test_sequences)
print(f"Second-order: {comparison['second_order_metrics']['top_1_accuracy']:.3f}")
print(f"First-order:  {comparison['first_order_metrics']['top_1_accuracy']:.3f}")
print(f"Improvement:  {comparison['improvement']['top_1_accuracy']:.1f}%")
```

### Persistence

#### `save(path)` / `load(path)`
Save and load trained models.

```python
mc2.save('models/second_order.json')
mc2_loaded = SecondOrderMarkovChain.load('models/second_order.json')
```

### Properties

```python
mc2.is_fitted          # True if trained
mc2.states             # Set of individual API endpoints
mc2.state_pairs        # Set of (previous, current) tuples
```

### Statistics

```python
stats = mc2.get_statistics()
# {
#     'num_individual_states': 20,
#     'num_state_pairs': 45,
#     'num_transitions': 150,
#     'sparsity': 0.85,
#     'fallback_to_first_order': True,
#     'first_order_stats': {...}
# }
```

## Implementation Details

### State Representation

States are represented as composite keys:
- Format: `"previous|current"` (using `|` delimiter)
- Start of sequence: `"<START>|first_api"`
- Helper methods: `_make_state_key()`, `_parse_state_key()`

### Fallback Strategy

When `(previous, current)` pair is unseen:
1. Check if second-order transition exists
2. If not, check if fallback is enabled
3. If enabled, use first-order prediction for `current`
4. Track fallback usage in evaluation metrics

### Training Process

For each sequence `[A, B, C, D]`:
1. Extract transitions:
   - `(<START>, A) → B`
   - `(A, B) → C`
   - `(B, C) → D`
2. Increment transition counts in matrix
3. If fallback enabled, also train first-order model

### Memory Usage

- First-order: ~O(n²) for n states (n × avg_transitions)
- Second-order: ~O(n³) for n states (n² pairs × avg_transitions)
- Sparse storage reduces actual memory significantly

Example: 100 APIs, 10 avg transitions
- First-order: ~1,000 non-zero entries
- Second-order: ~10,000 - 100,000 non-zero entries (depends on data)

## Usage Examples

### Example 1: E-Commerce Navigation

```python
# User behavior patterns
sequences = [
    # Authenticated users check orders
    ['login', 'auth', 'profile', 'orders', 'order_detail'],
    ['login', 'auth', 'profile', 'orders', 'reorder'],
    # Guest browsers just window-shop
    ['home', 'browse', 'profile', 'settings', 'logout'],
    ['home', 'browse', 'profile', 'wishlist', 'browse']
]

mc2 = SecondOrderMarkovChain(smoothing=0.001, fallback_to_first_order=True)
mc2.fit(sequences)

# After login flow
print(mc2.predict('auth', 'profile', k=2))
# [('orders', 0.95), ...]  ← High confidence

# After browse flow
print(mc2.predict('browse', 'profile', k=2))
# [('settings', 0.48), ('wishlist', 0.48)]  ← Different pattern!
```

### Example 2: Anomaly Detection

```python
# Train on normal sequences
normal = [
    ['login', 'auth', 'dashboard', 'data'],
    ['login', 'auth', 'dashboard', 'reports'],
]
mc2.fit(normal)

# Score new sequences
normal_score = mc2.score_sequence(['login', 'auth', 'dashboard', 'data'])
anomaly_score = mc2.score_sequence(['login', 'admin', 'users', 'delete'])

print(f"Normal: {normal_score:.2f}")    # -2.5 (higher)
print(f"Anomaly: {anomaly_score:.2f}")  # -15.3 (much lower)

if anomaly_score < normal_score - 5:
    print("⚠️ Unusual API access pattern detected!")
```

### Example 3: Adaptive Prefetching

```python
# Use second-order for intelligent prefetch
mc2 = SecondOrderMarkovChain(smoothing=0.001, fallback_to_first_order=True)
mc2.fit(training_sequences)

def prefetch_recommendations(api_history):
    """Get APIs to prefetch based on recent history."""
    if len(api_history) < 2:
        return []
    
    previous = api_history[-2]
    current = api_history[-1]
    
    # Get top-3 most likely next APIs
    predictions = mc2.predict(previous, current, k=3)
    
    # Prefetch if probability > threshold
    to_prefetch = [api for api, prob in predictions if prob > 0.3]
    return to_prefetch

# Example usage
history = ['login', 'auth', 'profile']
prefetch = prefetch_recommendations(history)
print(f"Prefetch: {prefetch}")  # ['orders', 'settings']
```

### Example 4: Model Comparison

```python
# Compare second-order improvement over first-order
train_sequences = load_training_data()
test_sequences = load_test_data()

mc2 = SecondOrderMarkovChain(smoothing=0.001, fallback_to_first_order=True)
mc2.fit(train_sequences)

comparison = mc2.compare_with_first_order(test_sequences)

print("Performance Comparison:")
print(f"Second-order accuracy: {comparison['second_order_metrics']['top_1_accuracy']:.1%}")
print(f"First-order accuracy:  {comparison['first_order_metrics']['top_1_accuracy']:.1%}")
print(f"Improvement:           {comparison['improvement']['top_1_accuracy']:.1f}%")
print(f"Fallback usage:        {comparison['fallback_rate']:.1%}")

# Decision logic
if comparison['improvement']['top_1_accuracy'] > 5.0:
    print("✓ Use second-order - significant improvement!")
elif comparison['fallback_rate'] > 0.5:
    print("⚠ High fallback rate - need more training data")
else:
    print("→ First-order sufficient for this dataset")
```

## Performance Tips

### 1. Choosing Smoothing
```python
# No smoothing: sharp predictions, may overfit
mc2 = SecondOrderMarkovChain(smoothing=0.0)

# Small smoothing: robust predictions
mc2 = SecondOrderMarkovChain(smoothing=0.001)  # Recommended

# Large smoothing: very smooth, may underfit
mc2 = SecondOrderMarkovChain(smoothing=0.1)
```

### 2. Handling Limited Data
```python
# Enable fallback for robustness
mc2 = SecondOrderMarkovChain(
    smoothing=0.01,           # Higher smoothing
    fallback_to_first_order=True  # Enable fallback
)
```

### 3. Online Learning
```python
# Initial training
mc2.fit(historical_data)

# Periodic updates with new data
for new_batch in streaming_data:
    mc2.partial_fit(new_batch)  # Incremental learning
```

### 4. Memory Optimization
```python
# Save model, free memory
mc2.fit(large_dataset)
mc2.save('model.json')
del mc2

# Reload when needed
mc2 = SecondOrderMarkovChain.load('model.json')
```

## Testing

Run the validation script to verify installation:

```bash
python validate_second_order.py
```

Run the demo:

```bash
python demo_second_order.py
```

## Files

- **Implementation:** `src/markov/second_order.py`
- **Validation:** `validate_second_order.py`
- **Demo:** `demo_second_order.py`
- **Documentation:** `SECOND_ORDER_QUICK_REF.md` (this file)

## Related Components

- **TransitionMatrix:** Sparse storage backend (`src/markov/transition_matrix.py`)
- **FirstOrderMarkovChain:** Baseline model (`src/markov/first_order.py`)

## References

- **Markov Chain Theory:** Standard textbooks on stochastic processes
- **Context in Prediction:** Higher-order Markov models capture sequential dependencies
- **Web Prefetching:** "Prediction of Web Page Accesses by Proxy Server Log" - Pitkow & Pirolli (1999)

---

**Implementation Status:** ✅ Complete and Validated

Last Updated: January 2026


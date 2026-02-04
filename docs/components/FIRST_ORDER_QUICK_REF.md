# FirstOrderMarkovChain Quick Reference

## Import
```python
from src.markov import FirstOrderMarkovChain
```

## Create
```python
# No smoothing
mc = FirstOrderMarkovChain()

# With Laplace smoothing (recommended)
mc = FirstOrderMarkovChain(smoothing=0.001)
```

## Training

### fit() - Train from scratch
```python
sequences = [
    ['login', 'profile', 'orders'],
    ['login', 'browse', 'product'],
    ['browse', 'search', 'product']
]

mc.fit(sequences)  # Replaces any previous training
```

### partial_fit() - Incremental learning
```python
# Add new data without losing existing model
new_sequences = [['checkout', 'confirmation']]
mc.partial_fit(new_sequences)
```

### update() - Single transition
```python
mc.update('current_state', 'next_state', count=10)
```

## Prediction

### predict() - Top-k predictions
```python
predictions = mc.predict('login', k=5)
# Returns: [('profile', 0.6), ('browse', 0.4)]

for next_state, probability in predictions:
    print(f"{next_state}: {probability:.2%}")
```

### predict_proba() - Specific probability
```python
prob = mc.predict_proba('login', 'profile')
# Returns: 0.6
```

## Sequence Generation

### generate_sequence() - Create synthetic data
```python
# Basic generation
seq = mc.generate_sequence('login', length=10, seed=42)

# With stop states
seq = mc.generate_sequence(
    'login',
    length=20,
    stop_states={'checkout', 'logout'},
    seed=42
)
```

## Sequence Scoring

### score_sequence() - Log-likelihood
```python
score = mc.score_sequence(['login', 'profile', 'orders'])
# Returns: -1.234 (higher = more likely)

# Useful for anomaly detection
if score < threshold:
    print("Unusual behavior detected!")
```

## Evaluation

### evaluate() - Compute metrics
```python
test_sequences = [['login', 'profile'], ['browse', 'product']]

metrics = mc.evaluate(test_sequences, k_values=[1, 3, 5])

print(f"Top-1 accuracy: {metrics['top_1_accuracy']:.2%}")
print(f"Top-3 accuracy: {metrics['top_3_accuracy']:.2%}")
print(f"MRR: {metrics['mrr']:.3f}")
print(f"Coverage: {metrics['coverage']:.2%}")
print(f"Perplexity: {metrics['perplexity']:.2f}")
```

## Persistence

### save() and load()
```python
# Save model
mc.save('model.json')

# Load model
mc = FirstOrderMarkovChain.load('model.json')
```

## Properties

```python
mc.is_fitted        # bool: Is model trained?
mc.states           # set: All known states
mc.get_statistics() # dict: Comprehensive stats
```

## Common Patterns

### Pattern 1: Cache Prefetching
```python
def prefetch_for_endpoint(endpoint):
    """Prefetch likely next endpoints."""
    predictions = mc.predict(endpoint, k=3)
    
    for next_ep, prob in predictions:
        if prob > 0.15:  # 15% threshold
            cache.warm(next_ep)
```

### Pattern 2: Anomaly Detection
```python
def is_anomalous_sequence(sequence, threshold=-10.0):
    """Detect unusual API call patterns."""
    score = mc.score_sequence(sequence)
    return score < threshold
```

### Pattern 3: Incremental Learning
```python
# Production system with continuous updates
mc = FirstOrderMarkovChain.load('production.json')

# Process new logs
for user_session in new_sessions:
    mc.partial_fit([user_session])

# Periodically save
if time_to_checkpoint():
    mc.save('production.json')
```

### Pattern 4: A/B Testing
```python
# Train separate models for comparison
mc_control = FirstOrderMarkovChain().fit(control_data)
mc_treatment = FirstOrderMarkovChain().fit(treatment_data)

# Compare metrics
metrics_control = mc_control.evaluate(test_data)
metrics_treatment = mc_treatment.evaluate(test_data)
```

## Metrics Explained

| Metric | Meaning | Range |
|--------|---------|-------|
| **top_k_accuracy** | % of correct predictions in top-k | 0-1 |
| **MRR** | Mean Reciprocal Rank (1/rank of correct answer) | 0-1 |
| **coverage** | % of test states we can predict for | 0-1 |
| **perplexity** | Uncertainty measure (lower = better) | 1-∞ |

## Tips

✅ Use smoothing (0.001-0.01) for robustness  
✅ Use `partial_fit()` for online learning  
✅ Set reasonable k values (3-5) for predictions  
✅ Monitor perplexity to detect model drift  
✅ Use `generate_sequence()` for testing  
✅ Save checkpoints regularly in production  

## Full Example
```python
from src.markov import FirstOrderMarkovChain

# Training
mc = FirstOrderMarkovChain(smoothing=0.001)
sequences = [
    ['login', 'profile', 'orders'],
    ['login', 'browse', 'product', 'cart'],
]
mc.fit(sequences)

# Prediction
predictions = mc.predict('login', k=3)
print(f"After login: {predictions}")

# Evaluation
metrics = mc.evaluate(sequences, k_values=[1, 3])
print(f"Accuracy: {metrics['top_1_accuracy']:.1%}")

# Save
mc.save('model.json')

# Cache integration
for endpoint, prob in mc.predict('browse', k=5):
    if prob > 0.10:
        cache.prefetch(endpoint)
```

---

See full documentation in `src/markov/first_order.py`


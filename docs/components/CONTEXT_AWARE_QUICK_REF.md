# Context-Aware Markov Chain - Quick Reference

## Overview

The `ContextAwareMarkovChain` maintains separate Markov models for different contexts (e.g., user types, times of day). This captures the fact that different users behave differently - premium users might have different browsing patterns than free users, and morning traffic might differ from evening traffic.

## Key Concept

Instead of one global model, we maintain:
- **Context-specific chains**: One chain per unique context (e.g., "premium|morning")
- **Global fallback chain**: Trained on all data regardless of context
- **Automatic fallback**: When a context is unseen, fall back to global or similar context

## Quick Start

```python
from src.markov import ContextAwareMarkovChain

# Create model
mc_ctx = ContextAwareMarkovChain(
    context_features=['user_type', 'time_of_day'],
    order=1,  # 1 or 2
    fallback_strategy='global',  # 'global', 'similar', or 'none'
    smoothing=0.001
)

# Train with contexts
sequences = [
    ['login', 'profile', 'premium_features'],
    ['login', 'profile', 'browse', 'product']
]
contexts = [
    {'user_type': 'premium', 'hour': 10},
    {'user_type': 'free', 'hour': 14}
]
mc_ctx.fit(sequences, contexts)

# Context-aware predictions
predictions = mc_ctx.predict('login', {'user_type': 'premium', 'hour': 10}, k=3)
# Different predictions for different contexts!
```

## Context Features

### Built-in Feature Discretization

#### Time of Day (from 'hour' field)
```python
context_features=['time_of_day']

# Automatically discretizes hours:
hour  6-11 → 'morning'
hour 12-17 → 'afternoon'
hour 18-21 → 'evening'
hour 22-5  → 'night'

# Use:
{'hour': 10} → 'morning'
{'hour': 14} → 'afternoon'
```

#### Day Type (from 'day' field)
```python
context_features=['day_type']

# Discretizes day of week:
day 0-4 (Mon-Fri) → 'weekday'
day 5-6 (Sat-Sun) → 'weekend'

# Use:
{'day': 0} → 'weekday'  # Monday
{'day': 6} → 'weekend'  # Sunday
```

#### Custom Features
```python
context_features=['user_type', 'region', 'device']

# Pass through as-is:
{'user_type': 'premium', 'region': 'US', 'device': 'mobile'}
```

## API Reference

### Constructor

```python
ContextAwareMarkovChain(
    context_features: List[str],
    order: int = 1,
    fallback_strategy: str = 'global',
    smoothing: float = 0.001
)
```

**Parameters:**
- `context_features`: List of feature names (e.g., `['user_type', 'time_of_day']`)
- `order`: Markov chain order (1 or 2)
- `fallback_strategy`: What to do for unseen contexts
  - `'global'`: Use global chain (default, most robust)
  - `'similar'`: Find similar context (e.g., same user_type, different time)
  - `'none'`: Return empty predictions
- `smoothing`: Laplace smoothing parameter

### Training

#### `fit(sequences, contexts)`
Train model (resets existing):
```python
mc_ctx.fit(sequences, contexts)
```

**Parameters:**
- `sequences`: List of API call sequences
- `contexts`: List of context dicts (one per sequence)

**Example:**
```python
sequences = [['A', 'B', 'C'], ['D', 'E', 'F']]
contexts = [
    {'user_type': 'premium', 'hour': 10},
    {'user_type': 'free', 'hour': 14}
]
mc_ctx.fit(sequences, contexts)
```

#### `partial_fit(sequences, contexts)`
Incremental updates:
```python
mc_ctx.partial_fit(new_sequences, new_contexts)
```

### Prediction

#### `predict(current, context, k=5, prev=None)`
Context-aware predictions:
```python
predictions = mc_ctx.predict('login', {'user_type': 'premium', 'hour': 10}, k=3)
# [('premium_features', 0.8), ('profile', 0.15), ...]
```

**Parameters:**
- `current`: Current API endpoint
- `context`: Context dictionary
- `k`: Number of predictions
- `prev`: Previous API (only for order=2)

#### `predict_with_confidence(current, context, k=5, prev=None)`
Predictions with confidence scores:
```python
predictions = mc_ctx.predict_with_confidence('login', {'user_type': 'premium', 'hour': 10})
# [('premium_features', 0.8, 0.62), ...]
#   api, probability, confidence
```

Confidence is based on:
- How much training data for this context (tanh(samples/100))
- Reduced by 50% if using fallback

### Analysis

#### `get_context_statistics()`
Get context information:
```python
stats = mc_ctx.get_context_statistics()
# {
#     'num_contexts': 4,
#     'contexts': ['premium|morning', 'free|afternoon', ...],
#     'samples_per_context': {'premium|morning': 50, ...},
#     'low_data_contexts': [...],  # < 10 samples
#     'total_samples': 200,
#     'avg_samples_per_context': 50.0
# }
```

#### `get_context_importance(test_sequences, test_contexts)`
Measure feature importance:
```python
importance = mc_ctx.get_context_importance(test_sequences, test_contexts)
# {'user_type': 15.3%, 'time_of_day': 5.2%}
```

Returns percentage improvement in accuracy when using each feature.

### Persistence

```python
# Save
mc_ctx.save('models/context_aware.json')

# Load
mc_ctx = ContextAwareMarkovChain.load('models/context_aware.json')
```

### Properties

```python
mc_ctx.is_fitted       # True if trained
mc_ctx.contexts        # Set of context keys
mc_ctx.context_features # List of feature names
```

## Usage Patterns

### Pattern 1: User Type Segmentation

```python
# Different behavior by user type
mc_ctx = ContextAwareMarkovChain(
    context_features=['user_type'],
    order=1,
    fallback_strategy='global'
)

# Premium users might access premium features
# Free users might just browse
```

### Pattern 2: Time-Based Patterns

```python
# Behavior changes throughout the day
mc_ctx = ContextAwareMarkovChain(
    context_features=['time_of_day'],
    order=1
)

# Morning: check emails, read news
# Evening: shop, entertainment
```

### Pattern 3: Multi-Feature Context

```python
# Combine multiple contextual features
mc_ctx = ContextAwareMarkovChain(
    context_features=['user_type', 'time_of_day', 'day_type'],
    order=1,
    fallback_strategy='similar'  # More fallback needed with more features
)

# Premium users on weekday mornings: business tasks
# Free users on weekend evenings: leisure browsing
```

### Pattern 4: Second-Order with Context

```python
# Context + history for maximum accuracy
mc_ctx = ContextAwareMarkovChain(
    context_features=['user_type'],
    order=2,  # Use previous + current state
    fallback_strategy='global'
)

# Predictions use: (previous, current, context) → next
```

## Examples

### Example 1: E-Commerce

```python
sequences = [
    # Premium morning: Quick deals
    ['login', 'premium_deals', 'cart', 'checkout'],
    ['login', 'premium_deals', 'product', 'buy'],
    
    # Free afternoon: Browse casually
    ['browse', 'product', 'product', 'compare'],
    ['search', 'product', 'reviews', 'browse'],
]

contexts = [
    {'user_type': 'premium', 'hour': 10},
    {'user_type': 'premium', 'hour': 9},
    {'user_type': 'free', 'hour': 14},
    {'user_type': 'free', 'hour': 15},
]

mc_ctx = ContextAwareMarkovChain(
    context_features=['user_type', 'time_of_day'],
    order=1,
    fallback_strategy='global'
)
mc_ctx.fit(sequences, contexts)

# Premium user in morning
pred = mc_ctx.predict('login', {'user_type': 'premium', 'hour': 10}, k=2)
# [('premium_deals', high_prob), ...]

# Free user in afternoon
pred = mc_ctx.predict('browse', {'user_type': 'free', 'hour': 14}, k=2)
# [('product', high_prob), ...]
```

### Example 2: Content Platform

```python
sequences = [
    # Weekend evening: Entertainment
    ['home', 'movies', 'watch', 'watch'],
    ['home', 'shows', 'watch', 'continue'],
    
    # Weekday morning: News
    ['home', 'news', 'read', 'notifications'],
    ['home', 'news', 'briefing', 'home'],
]

contexts = [
    {'day': 6, 'hour': 20},  # Sunday evening
    {'day': 5, 'hour': 19},  # Saturday evening
    {'day': 1, 'hour': 8},   # Tuesday morning
    {'day': 3, 'hour': 7},   # Thursday morning
]

mc_ctx = ContextAwareMarkovChain(
    context_features=['day_type', 'time_of_day'],
    order=1
)
mc_ctx.fit(sequences, contexts)

# Weekend evening → entertainment
pred = mc_ctx.predict('home', {'day': 6, 'hour': 20}, k=2)

# Weekday morning → news
pred = mc_ctx.predict('home', {'day': 1, 'hour': 8}, k=2)
```

### Example 3: Analyzing Context Importance

```python
# Train model
mc_ctx.fit(train_sequences, train_contexts)

# Measure which contexts matter most
importance = mc_ctx.get_context_importance(test_sequences, test_contexts)

print("Feature Importance:")
for feature, score in sorted(importance.items(), key=lambda x: -x[1]):
    print(f"  {feature}: {score:.1f}% improvement")

# Output might show:
#   user_type: 15.3% improvement
#   time_of_day: 5.2% improvement
#   day_type: 2.1% improvement

# → Focus on user_type for prediction, maybe drop day_type
```

## Fallback Strategies Explained

### Global Fallback (Default)

```python
fallback_strategy='global'
```

**When:** Context not seen during training  
**Action:** Use global chain (trained on all data)  
**Best for:** Maximum robustness, always returns predictions

**Example:**
```python
# Trained on 'premium' and 'free' users
# Query with 'enterprise' user → uses global chain
```

### Similar Fallback

```python
fallback_strategy='similar'
```

**When:** Context not seen during training  
**Action:** Find most similar context (match on some features)  
**Best for:** When you want context-specific predictions even for novel contexts

**Example:**
```python
# Trained on 'premium|morning' and 'free|afternoon'
# Query 'premium|evening' → uses 'premium|morning' (same user_type)
```

### None Fallback

```python
fallback_strategy='none'
```

**When:** Context not seen during training  
**Action:** Return empty predictions  
**Best for:** When you only want predictions from specific contexts

**Example:**
```python
# Only want predictions for known user-time combinations
# Unknown context → return []
```

## Tips & Best Practices

### 1. Start Simple

```python
# Start with one feature
mc_ctx = ContextAwareMarkovChain(context_features=['user_type'], order=1)

# Add more features if needed
mc_ctx = ContextAwareMarkovChain(
    context_features=['user_type', 'time_of_day'],
    order=1
)
```

### 2. Handle Data Sparsity

```python
# With many features, data gets sparse → use global fallback
mc_ctx = ContextAwareMarkovChain(
    context_features=['user_type', 'time_of_day', 'day_type', 'region'],
    fallback_strategy='global',  # Essential with many features
    smoothing=0.01  # Higher smoothing for sparse data
)
```

### 3. Monitor Low-Data Contexts

```python
stats = mc_ctx.get_context_statistics()
if stats['low_data_contexts']:
    print(f"Warning: {len(stats['low_data_contexts'])} contexts have < 10 samples")
    # Consider collecting more data or merging contexts
```

### 4. Choose Order Based on Data

```python
# Lots of data per context → order=2
# Limited data per context → order=1

samples_per_context = stats['avg_samples_per_context']
if samples_per_context > 100:
    order = 2  # Can afford second-order
else:
    order = 1  # Stick with first-order
```

### 5. Measure Context Value

```python
# Check if context actually helps
importance = mc_ctx.get_context_importance(test_sequences, test_contexts)

# If improvement is small, might not need context
for feature, score in importance.items():
    if score < 5.0:  # Less than 5% improvement
        print(f"Consider removing {feature} - minimal benefit")
```

## Context Key Format

Context keys combine features with '|' delimiter:

```python
# Single feature
{'user_type': 'premium'} → 'premium'

# Two features
{'user_type': 'premium', 'time_of_day': 'morning'} → 'premium|morning'

# Three features
{'user_type': 'premium', 'time_of_day': 'morning', 'day_type': 'weekday'}
→ 'premium|morning|weekday'
```

## Confidence Scores

Confidence formula: `tanh(samples / 100)`

| Samples | Confidence | Meaning |
|---------|-----------|---------|
| 10      | 0.10      | Very low - not much data |
| 50      | 0.46      | Moderate - some data |
| 100     | 0.76      | Good - decent amount |
| 200     | 0.96      | High - lots of data |

If using fallback: confidence × 0.5

## Common Issues

### Issue: Too many contexts, sparse data

**Solution:**
```python
# Use fewer features
context_features=['user_type']  # Instead of ['user_type', 'time', 'day', 'region']

# Or use global fallback
fallback_strategy='global'
```

### Issue: Contexts not differentiating

**Solution:**
```python
# Check importance
importance = mc_ctx.get_context_importance(test, test_ctx)
# If all low, context might not matter for your data
```

### Issue: Unknown context at inference

**Solution:**
```python
# Already handled by fallback strategy!
# Use 'global' for robust predictions
fallback_strategy='global'
```

## Files

- **Implementation:** `src/markov/context_aware.py`
- **Demo:** `demo_context_aware.py`
- **Validation:** `validate_context_aware.py`
- **Quick Ref:** `CONTEXT_AWARE_QUICK_REF.md` (this file)

## Related Components

- **FirstOrderMarkovChain:** Used for order=1 contexts
- **SecondOrderMarkovChain:** Used for order=2 contexts
- **TransitionMatrix:** Underlying storage

---

**Status:** ✅ Complete and Validated  
**Date:** January 17, 2026


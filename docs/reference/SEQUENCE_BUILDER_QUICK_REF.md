# SequenceBuilder Quick Reference

## Import
```python
from preprocessing.sequence_builder import SequenceBuilder
```

## Initialize
```python
# Default (recommended)
builder = SequenceBuilder(normalize_endpoints=True, min_sequence_length=2)

# Custom
builder = SequenceBuilder(normalize_endpoints=False, min_sequence_length=1)
```

## Methods

### Normalization
```python
# Normalize single endpoint
normalized = builder.normalize_endpoint("/API/Users/123/Profile/")
# Returns: "/api/users/{id}/profile"
```

### Basic Sequences
```python
# Extract endpoint sequences
sequences = builder.build_sequences(sessions)
# Returns: [["/login", "/profile"], ["/browse", "/product"]]
```

### Labeled Sequences
```python
# Create (history, next) pairs
labeled = builder.build_labeled_sequences(sessions)
# Returns: [(["/login"], "/profile"), (["/login", "/profile"], "/browse")]
```

### N-grams
```python
# Extract bigrams
bigrams = builder.build_ngrams(sessions, n=2)
# Returns: [("/login", "/profile"), ("/profile", "/browse")]

# Extract trigrams
trigrams = builder.build_ngrams(sessions, n=3)
```

### Contextual Sequences
```python
# Get sequences with metadata
contextual = builder.build_contextual_sequences(sessions)
# Returns: [ContextualSequence(sequence=[...], user_type="premium", ...)]

# Access metadata
for ctx in contextual:
    print(ctx.user_type)           # premium/free/guest
    print(ctx.time_of_day)         # morning/afternoon/evening/night
    print(ctx.day_type)            # weekday/weekend
    print(ctx.session_length_category)  # short/medium/long
    print(ctx.sequence)            # List of endpoints
```

### Transition Analysis
```python
# Get transition counts
counts = builder.get_transition_counts(sessions)
# Returns: {("/login", "/profile"): 100, ...}

# Get transition probabilities
probs = builder.get_transition_probabilities(sessions)
# Returns: {"/login": {"/profile": 0.95, "/logout": 0.05}, ...}

# Use for prediction
current = "/login"
if current in probs:
    next_ep = max(probs[current].items(), key=lambda x: x[1])[0]
```

### Statistics
```python
# Get overall statistics
stats = builder.get_sequence_statistics(sessions)
# Returns: {
#   'total_sequences': 100,
#   'total_calls': 450,
#   'avg_sequence_length': 4.5,
#   'min_sequence_length': 2,
#   'max_sequence_length': 10,
#   'unique_endpoints': 25,
#   'total_transitions': 350
# }
```

### Utilities
```python
# Get unique endpoints
endpoints = builder.get_unique_endpoints(sessions)
# Returns: ["/api/cart", "/api/login", "/api/products/{id}"]

# Split train/test
train, test = builder.split_sequences(sessions, train_ratio=0.8)
```

## Complete Workflow

```python
from preprocessing.models import Dataset
from preprocessing.sequence_builder import SequenceBuilder

# 1. Load data
dataset = Dataset.load_from_parquet('sessions.parquet')

# 2. Initialize builder
builder = SequenceBuilder(normalize_endpoints=True, min_sequence_length=2)

# 3. Split data
train_sessions, test_sessions = builder.split_sequences(
    dataset.sessions, 
    train_ratio=0.8
)

# 4. Train: Calculate probabilities
train_probs = builder.get_transition_probabilities(train_sessions)

# 5. Evaluate: Create test pairs
test_labeled = builder.build_labeled_sequences(test_sessions)

# 6. Predict and evaluate
correct = 0
for history, actual_next in test_labeled:
    current = history[-1]
    if current in train_probs:
        predicted = max(train_probs[current].items(), key=lambda x: x[1])[0]
        if predicted == actual_next:
            correct += 1

accuracy = correct / len(test_labeled)
print(f"Accuracy: {accuracy:.2%}")
```

## Normalization Examples

```python
builder = SequenceBuilder(normalize_endpoints=True)

# Lowercase
builder.normalize_endpoint("/API/Users")         # "/api/users"

# Remove trailing slash
builder.normalize_endpoint("/api/users/")         # "/api/users"

# Strip query params
builder.normalize_endpoint("/search?q=test")      # "/search"

# Replace numeric IDs
builder.normalize_endpoint("/users/123/profile")  # "/users/{id}/profile"

# Replace UUIDs
builder.normalize_endpoint("/orders/550e8400-e29b-41d4-a716-446655440000")
# "/orders/{id}"

# Combined
builder.normalize_endpoint("/API/Users/123/Profile/?ref=email")
# "/api/users/{id}/profile"
```

## Time Categories

```python
from datetime import datetime

# morning: 6:00-11:59
# afternoon: 12:00-17:59
# evening: 18:00-23:59
# night: 0:00-5:59

time = datetime(2026, 1, 11, 14, 30)  # 2:30 PM
category = SequenceBuilder._get_time_of_day(time)
# Returns: "afternoon"
```

## Session Length Categories

- **Short**: < 60 seconds (< 1 minute)
- **Medium**: 60-600 seconds (1-10 minutes)
- **Long**: > 600 seconds (> 10 minutes)

## Common Patterns

### Filter by User Type
```python
contextual = builder.build_contextual_sequences(sessions)
premium = [c.sequence for c in contextual if c.user_type == 'premium']
free = [c.sequence for c in contextual if c.user_type == 'free']
```

### Filter by Time of Day
```python
contextual = builder.build_contextual_sequences(sessions)
morning = [c.sequence for c in contextual if c.time_of_day == 'morning']
```

### Count Endpoint Frequency
```python
counts = builder.get_transition_counts(sessions)
endpoint_freq = {}
for (from_ep, to_ep), count in counts.items():
    endpoint_freq[from_ep] = endpoint_freq.get(from_ep, 0) + count
```

### Top Transitions
```python
counts = builder.get_transition_counts(sessions)
top_10 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
for (from_ep, to_ep), count in top_10:
    print(f"{from_ep} → {to_ep}: {count}x")
```

## Performance Tips

1. **Enable normalization** - Dramatically improves pattern recognition
2. **Set min_sequence_length=2** - Filters out single-call sessions
3. **Cache probabilities** - Calculate once, reuse for predictions
4. **Batch processing** - Process sessions in batches for large datasets

## Error Handling

```python
try:
    trigrams = builder.build_ngrams(sessions, n=1)
except ValueError as e:
    # "n must be at least 2, got 1"
    pass

try:
    train, test = builder.split_sequences(sessions, train_ratio=1.5)
except ValueError as e:
    # "train_ratio must be between 0 and 1, got 1.5"
    pass
```

## See Also

- `preprocessing/SEQUENCE_BUILDER_GUIDE.md` - Detailed documentation
- `demo_sequence_builder.py` - Comprehensive examples
- `test_sequence_builder.py` - Test suite
- `preprocessing/models.py` - Data model definitions


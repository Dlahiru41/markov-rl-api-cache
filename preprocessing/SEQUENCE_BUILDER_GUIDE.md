# SequenceBuilder Module Documentation

## Overview

The `SequenceBuilder` module is a critical component of the Markov chain training pipeline. It converts Session objects into various sequence formats needed for learning transition probabilities between API endpoints.

## Purpose

When training a Markov chain to predict API calls, we need to learn: **"Given that a user just called endpoint A, what's the probability they'll call endpoint B next?"**

The SequenceBuilder transforms raw session data into the structured formats needed for this learning process.

## Key Features

### 1. **Endpoint Normalization** ⭐
The most critical feature! Without normalization, `/users/1/profile` and `/users/999/profile` would be treated as completely different endpoints.

Normalization steps:
- Convert to lowercase
- Remove trailing slashes
- Strip query parameters
- Replace numeric IDs with `{id}` placeholders
- Replace UUIDs with `{id}` placeholders

**Example:**
```python
builder = SequenceBuilder(normalize_endpoints=True)
builder.normalize_endpoint("/API/Users/123/Profile/")
# Output: "/api/users/{id}/profile"
```

### 2. **Basic Sequence Extraction**
Extracts just the endpoint sequences from sessions, optionally filtering by minimum length.

**Example:**
```python
sessions = [session1, session2, session3]
sequences = builder.build_sequences(sessions, min_sequence_length=2)
# Output: [["/login", "/profile", "/browse"], ["/login", "/search", "/product"]]
```

### 3. **Labeled Sequences**
Creates (history, next_endpoint) pairs for evaluating prediction accuracy.

For sequence `[A, B, C, D]`, generates:
- `([A], B)`
- `([A, B], C)`
- `([A, B, C], D)`

**Example:**
```python
labeled = builder.build_labeled_sequences(sessions)
# Output: [(['/login'], '/profile'), (['/login', '/profile'], '/browse'), ...]
```

### 4. **N-gram Extraction**
Extracts overlapping tuples of N consecutive endpoints for pattern analysis.

**Bigrams (N=2):** For `[A, B, C, D]` → `[(A,B), (B,C), (C,D)]`
**Trigrams (N=3):** For `[A, B, C, D]` → `[(A,B,C), (B,C,D)]`

**Example:**
```python
bigrams = builder.build_ngrams(sessions, n=2)
# Output: [('/login', '/profile'), ('/profile', '/browse'), ...]

trigrams = builder.build_ngrams(sessions, n=3)
# Output: [('/login', '/profile', '/browse'), ...]
```

### 5. **Contextual Sequences**
Includes metadata alongside each sequence for context-aware modeling:
- **User type:** premium/free/guest
- **Time of day:** morning/afternoon/evening/night
- **Day type:** weekday/weekend
- **Session length:** short/medium/long

**Example:**
```python
contextual = builder.build_contextual_sequences(sessions)
for ctx_seq in contextual:
    print(f"User: {ctx_seq.user_type}, Time: {ctx_seq.time_of_day}")
    print(f"Sequence: {ctx_seq.sequence}")
```

### 6. **Transition Analysis**
Calculate transition counts and probabilities between endpoints.

**Example:**
```python
# Get raw counts
counts = builder.get_transition_counts(sessions)
# Output: {('/login', '/profile'): 100, ('/profile', '/browse'): 75, ...}

# Get probabilities
probs = builder.get_transition_probabilities(sessions)
# Output: {'/login': {'/profile': 0.95, '/logout': 0.05}, ...}
```

## Usage Examples

### Basic Usage
```python
from preprocessing.sequence_builder import SequenceBuilder
from preprocessing.models import Session

# Initialize builder
builder = SequenceBuilder(
    normalize_endpoints=True,  # Enable normalization
    min_sequence_length=2      # Filter out single-call sessions
)

# Extract sequences
sequences = builder.build_sequences(sessions)
print(f"Extracted {len(sequences)} sequences")
```

### Training a Markov Chain
```python
# Get transition probabilities for training
probabilities = builder.get_transition_probabilities(sessions)

# For each state (endpoint), we now know the probability of transitioning
# to each possible next state
for from_endpoint, transitions in probabilities.items():
    print(f"From {from_endpoint}:")
    for to_endpoint, prob in transitions.items():
        print(f"  -> {to_endpoint}: {prob:.2%}")
```

### Evaluating Predictions
```python
# Create labeled test data
labeled_sequences = builder.build_labeled_sequences(test_sessions)

# For each (history, next) pair, try to predict next
for history, actual_next in labeled_sequences:
    predicted_next = markov_model.predict(history)
    correct = (predicted_next == actual_next)
    print(f"History: {history} | Predicted: {predicted_next} | Actual: {actual_next} | {'✓' if correct else '✗'}")
```

### Context-Aware Analysis
```python
# Group sequences by context
contextual = builder.build_contextual_sequences(sessions)

# Analyze patterns by user type
premium_sequences = [c.sequence for c in contextual if c.user_type == 'premium']
free_sequences = [c.sequence for c in contextual if c.user_type == 'free']

print(f"Premium users: {len(premium_sequences)} sequences")
print(f"Free users: {len(free_sequences)} sequences")
```

## Class Reference

### SequenceBuilder

**Constructor:**
```python
SequenceBuilder(normalize_endpoints=True, min_sequence_length=1)
```

**Parameters:**
- `normalize_endpoints` (bool): Enable endpoint normalization
- `min_sequence_length` (int): Minimum sequence length to include

**Methods:**

#### `normalize_endpoint(endpoint: str) -> str`
Normalize a single endpoint path.

#### `build_sequences(sessions: List[Session]) -> List[List[str]]`
Extract endpoint sequences from sessions.

#### `build_labeled_sequences(sessions: List[Session]) -> List[Tuple[List[str], str]]`
Create (history, next_endpoint) pairs.

#### `build_ngrams(sessions: List[Session], n: int = 2) -> List[Tuple[str, ...]]`
Extract n-grams from sequences.

#### `build_contextual_sequences(sessions: List[Session]) -> List[ContextualSequence]`
Extract sequences with contextual metadata.

#### `get_transition_counts(sessions: List[Session]) -> Dict[Tuple[str, str], int]`
Count endpoint transitions.

#### `get_transition_probabilities(sessions: List[Session]) -> Dict[str, Dict[str, float]]`
Calculate transition probabilities.

#### `get_unique_endpoints(sessions: List[Session]) -> List[str]`
Get all unique endpoints.

#### `get_sequence_statistics(sessions: List[Session]) -> Dict[str, Any]`
Get statistical information about sequences.

#### `split_sequences(sessions: List[Session], train_ratio: float = 0.8) -> Tuple[List[Session], List[Session]]`
Split sessions into train/test sets.

### ContextualSequence

**Attributes:**
- `sequence: List[str]` - The endpoint sequence
- `user_type: str` - premium/free/guest
- `time_of_day: str` - morning/afternoon/evening/night
- `day_type: str` - weekday/weekend
- `session_length_category: str` - short/medium/long

## Time of Day Categories

The module categorizes hours into four time periods:
- **Night:** 0-5 (midnight to 6am)
- **Morning:** 6-11 (6am to noon)
- **Afternoon:** 12-17 (noon to 6pm)
- **Evening:** 18-23 (6pm to midnight)

## Session Length Categories

Sessions are categorized by duration:
- **Short:** < 60 seconds
- **Medium:** 60-600 seconds (1-10 minutes)
- **Long:** > 600 seconds (> 10 minutes)

## Integration with Markov Chain

The SequenceBuilder output integrates directly with Markov chain training:

1. **Extract sequences:** Use `build_sequences()` to get training data
2. **Calculate probabilities:** Use `get_transition_probabilities()` to build the transition matrix
3. **Evaluate model:** Use `build_labeled_sequences()` to test predictions
4. **Context-aware modeling:** Use `build_contextual_sequences()` to train separate models per context

## Best Practices

1. **Always enable normalization:** Set `normalize_endpoints=True` unless you have a specific reason not to
2. **Filter short sequences:** Use `min_sequence_length=2` to exclude sessions with single calls
3. **Split data properly:** Use 80/20 train/test split for evaluation
4. **Consider context:** Premium users may have different patterns than free users
5. **Handle edge cases:** Check for empty sequences before processing

## Example Workflow

```python
from preprocessing.sequence_builder import SequenceBuilder
from preprocessing.models import Dataset

# Load dataset
dataset = Dataset.load_from_parquet('data/processed/sessions.parquet')

# Split into train/test
train_dataset, test_dataset = dataset.split(train_ratio=0.8)

# Initialize builder
builder = SequenceBuilder(normalize_endpoints=True, min_sequence_length=2)

# Train: Calculate transition probabilities
train_probs = builder.get_transition_probabilities(train_dataset.sessions)

# Test: Evaluate predictions
test_labeled = builder.build_labeled_sequences(test_dataset.sessions)

# Analyze statistics
stats = builder.get_sequence_statistics(train_dataset.sessions)
print(f"Training data: {stats['total_sequences']} sequences")
print(f"Average length: {stats['avg_sequence_length']:.2f}")
print(f"Unique endpoints: {stats['unique_endpoints']}")
```

## Performance Considerations

- Normalization adds minimal overhead (~ms per endpoint)
- Transition probability calculation is O(n) where n is total number of calls
- N-gram extraction is O(n*m) where m is the n-gram size
- Memory usage is proportional to number of unique endpoints and transitions

## Error Handling

The module validates inputs and raises appropriate exceptions:
- `ValueError`: For invalid n-gram sizes (n < 2)
- `ValueError`: For invalid train_ratio (not between 0 and 1)

## Related Modules

- **preprocessing.models:** Defines Session, APICall, and Dataset classes
- **src.markov:** Uses SequenceBuilder output for Markov chain training
- **evaluation:** Uses labeled sequences for model evaluation


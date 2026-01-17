# TransitionMatrix: Efficient Markov Chain Transition Storage

## Overview

The `TransitionMatrix` class provides an efficient sparse representation for storing and querying Markov chain transition probabilities. It's optimized for real-world scenarios where most state pairs never transition to each other (sparse data).

## Key Features

✅ **Sparse Storage**: Dictionary-based structure for O(1) lookups and updates  
✅ **Laplace Smoothing**: Optional smoothing to prevent zero probabilities  
✅ **Top-K Queries**: Efficient heap-based algorithm for finding most likely transitions  
✅ **Matrix Operations**: Merge matrices, compute statistics, analyze sparsity  
✅ **Serialization**: Save/load to JSON, convert to pandas DataFrame  
✅ **Well-Documented**: Comprehensive docstrings with time complexity analysis  

## Installation

The TransitionMatrix is part of the `src.markov` package:

```python
from src.markov import TransitionMatrix
```

No additional dependencies required for core functionality. Optional:
- `pandas` for DataFrame conversion

## Quick Start

```python
from src.markov import TransitionMatrix

# Create a transition matrix
tm = TransitionMatrix(smoothing=0.001)

# Add observations of state transitions
tm.increment("login", "profile", count=80)
tm.increment("login", "browse", count=20)
tm.increment("profile", "orders", count=50)

# Query transition probabilities
prob = tm.get_probability("login", "profile")
print(f"P(profile|login) = {prob:.2f}")  # 0.80

# Get most likely next states
top_states = tm.get_top_k("login", k=2)
print(f"Top transitions: {top_states}")
# [('profile', 0.8), ('browse', 0.2)]

# Save for later use
tm.save("transitions.json")
```

## Core Operations

### 1. Building the Matrix

**Increment transition counts**:
```python
tm = TransitionMatrix()

# Add single observation
tm.increment("A", "B")

# Add multiple observations
tm.increment("A", "C", count=10)

# Accumulate over time
tm.increment("A", "B", count=5)  # Total now 6
```

**Time complexity**: O(1) average case

### 2. Querying Counts and Probabilities

**Get raw counts**:
```python
count = tm.get_count("A", "B")  # Returns 6
unseen = tm.get_count("X", "Y")  # Returns 0
```

**Get transition probabilities**:
```python
prob = tm.get_probability("A", "B")  # P(B|A)
```

With Laplace smoothing:
```python
tm = TransitionMatrix(smoothing=0.1)
tm.increment("A", "B", 10)

# Even unseen transitions have small probability
prob = tm.get_probability("A", "C")  # Small but non-zero
```

**Time complexity**: 
- O(1) without smoothing
- O(n) with smoothing (where n = vocabulary size)

### 3. Row Operations

**Get all transitions from a state**:
```python
row = tm.get_row("A")
# Returns: {"B": 0.6, "C": 0.4}

# Probabilities sum to 1.0
assert abs(sum(row.values()) - 1.0) < 1e-10
```

**Time complexity**: O(k) where k is the number of outgoing transitions

### 4. Top-K Queries

**Find most likely next states**:
```python
# Get top 3 most likely transitions
top = tm.get_top_k("login", k=3)
# [('profile', 0.5), ('browse', 0.3), ('search', 0.2)]

# Results are sorted by probability (descending)
for state, prob in top:
    print(f"{state}: {prob:.2%}")
```

Uses a min-heap for efficiency:
- **Time complexity**: O(n log k) where n is total transitions from state
- More efficient than full sorting when k << n

### 5. Matrix Properties

```python
# Number of unique states
num_states = tm.num_states

# Number of non-zero transitions
num_transitions = tm.num_transitions

# Fraction of zero entries
sparsity = tm.sparsity  # 0.0 to 1.0

print(f"Matrix: {num_states} states, {num_transitions} transitions")
print(f"Sparsity: {sparsity:.1%}")
```

**Time complexity**: O(n) where n is the number of source states

### 6. Statistics

**Get comprehensive statistics**:
```python
stats = tm.get_statistics()

print(f"States: {stats['num_states']}")
print(f"Transitions: {stats['num_transitions']}")
print(f"Sparsity: {stats['sparsity']:.2%}")
print(f"Avg transitions per state: {stats['avg_transitions_per_state']:.2f}")

# Top 10 most common transitions
for trans in stats['most_common_transitions']:
    print(f"{trans['from']} -> {trans['to']}: {trans['count']}")
```

**Time complexity**: O(n log k) where n is total transitions

### 7. Merging Matrices

**Combine counts from multiple matrices**:
```python
# Matrix from first time period
tm1 = TransitionMatrix()
tm1.increment("A", "B", 10)

# Matrix from second time period
tm2 = TransitionMatrix()
tm2.increment("A", "B", 5)
tm2.increment("A", "C", 3)

# Merge them
merged = tm1.merge(tm2)

# Counts are summed
assert merged.get_count("A", "B") == 15
assert merged.get_count("A", "C") == 3
```

**Time complexity**: O(n + m) where n, m are the number of non-zero entries

### 8. Serialization

**Save to JSON**:
```python
tm.save("transitions.json")
```

**Load from JSON**:
```python
tm = TransitionMatrix.load("transitions.json")
```

**Convert to/from dictionary**:
```python
# To dictionary
data = tm.to_dict()

# From dictionary
tm2 = TransitionMatrix.from_dict(data)
```

**Convert to pandas DataFrame**:
```python
df = tm.to_dataframe()
# Columns: from_state, to_state, count, probability

# Useful for analysis
print(df.head())
print(df.describe())

# Export to CSV
df.to_csv("transitions.csv", index=False)
```

**Time complexity**: O(n) where n is the number of non-zero transitions

## Advanced Usage

### Laplace Smoothing

Smoothing prevents zero probabilities for unseen transitions:

```python
# Without smoothing
tm_no_smooth = TransitionMatrix(smoothing=0.0)
tm_no_smooth.increment("A", "B", 10)
prob = tm_no_smooth.get_probability("A", "C")  # 0.0

# With smoothing
tm_smooth = TransitionMatrix(smoothing=0.1)
tm_smooth.increment("A", "B", 10)
prob = tm_smooth.get_probability("A", "C")  # Small but non-zero
```

Formula: `P(to|from) = (count + α) / (total + α * |V|)`

Where:
- α = smoothing parameter
- |V| = vocabulary size (number of unique states)

### Incremental Learning

Build the matrix incrementally as data arrives:

```python
tm = TransitionMatrix()

# Process API request logs
for log_entry in api_logs:
    current_endpoint = log_entry['endpoint']
    next_endpoint = log_entry['next_endpoint']
    tm.increment(current_endpoint, next_endpoint)

# Periodically save
if num_processed % 1000 == 0:
    tm.save(f"checkpoint_{num_processed}.json")
```

### Combining Multiple Sources

Merge transition matrices from different data sources:

```python
# User group 1
tm_group1 = TransitionMatrix()
# ... build from group1 data ...

# User group 2
tm_group2 = TransitionMatrix()
# ... build from group2 data ...

# Combined model
tm_combined = tm_group1.merge(tm_group2)
```

### Predictive Caching

Use top-k queries for cache prefetching:

```python
def prefetch_cache(current_endpoint, cache, k=3):
    """Prefetch likely next endpoints."""
    likely_next = tm.get_top_k(current_endpoint, k=k)
    
    for endpoint, probability in likely_next:
        if probability > 0.1:  # Threshold
            cache.warm(endpoint)
```

### Visualization

Convert to DataFrame for visualization:

```python
import matplotlib.pyplot as plt
import seaborn as sns

df = tm.to_dataframe()

# Plot top transitions
top_df = df.nlargest(10, 'count')
plt.figure(figsize=(10, 6))
sns.barplot(data=top_df, x='count', y='from_state')
plt.title('Top 10 Transitions by Count')
plt.show()

# Heatmap of transition probabilities (for small matrices)
pivot = df.pivot(index='from_state', columns='to_state', values='probability')
plt.figure(figsize=(12, 10))
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd')
plt.title('Transition Probability Heatmap')
plt.show()
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `increment()` | O(1) | Average case |
| `get_count()` | O(1) | Dictionary lookup |
| `get_probability()` | O(1) or O(n) | O(n) with smoothing |
| `get_row()` | O(k) | k = outgoing transitions |
| `get_top_k()` | O(n log k) | n = total transitions |
| `merge()` | O(n + m) | n, m = entry counts |
| `num_states` | O(n) | n = source states |
| `to_dict()` | O(n) | n = non-zero entries |
| `save()` | O(n) | Plus I/O time |

### Space Complexity

**Sparse storage**: O(e) where e is the number of non-zero transitions

For typical API endpoint data:
- Dense matrix: O(|V|²) where |V| is vocabulary size
- Sparse matrix: O(|V| + e) where e << |V|²

Example: 1000 endpoints with 5 transitions each:
- Dense: 1,000,000 cells
- Sparse: ~5,000 cells (99.5% savings!)

## Best Practices

### 1. Choose Appropriate Smoothing

```python
# No smoothing: Use when you want hard zeros
tm = TransitionMatrix(smoothing=0.0)

# Light smoothing: Typical choice for most applications
tm = TransitionMatrix(smoothing=0.001)

# Heavy smoothing: Use when data is very sparse
tm = TransitionMatrix(smoothing=0.1)
```

### 2. Periodic Checkpointing

```python
# Save periodically during long training
for i, batch in enumerate(data_batches):
    process_batch(tm, batch)
    
    if i % 100 == 0:
        tm.save(f"checkpoint_batch_{i}.json")
```

### 3. Memory Management for Large Matrices

```python
# For very large matrices, process in chunks
def process_large_dataset(data_path, output_path, chunk_size=10000):
    tm = TransitionMatrix()
    
    for chunk in read_chunks(data_path, chunk_size):
        for from_state, to_state in chunk:
            tm.increment(from_state, to_state)
        
        # Optional: Clear memory if needed
        if tm.num_transitions > 1_000_000:
            tm.save(output_path)
            tm = TransitionMatrix.load(output_path)
```

### 4. Validation

```python
# Check that probabilities sum to 1.0
for state in unique_states:
    row = tm.get_row(state)
    total_prob = sum(row.values())
    assert abs(total_prob - 1.0) < 1e-6, f"Invalid probabilities for {state}"
```

## Common Patterns

### Pattern 1: Sequential Learning

```python
tm = TransitionMatrix()

sequences = [
    ["login", "profile", "orders", "checkout"],
    ["login", "browse", "product", "cart"],
    # ...
]

for sequence in sequences:
    for i in range(len(sequence) - 1):
        tm.increment(sequence[i], sequence[i + 1])
```

### Pattern 2: Weighted Transitions

```python
# Weight by response time (faster = more likely)
for transition in transitions:
    weight = 1.0 / transition['response_time']
    tm.increment(transition['from'], transition['to'], int(weight * 100))
```

### Pattern 3: Time-Windowed Analysis

```python
from datetime import datetime, timedelta

# Recent transitions (last 7 days)
tm_recent = TransitionMatrix()
cutoff = datetime.now() - timedelta(days=7)

for transition in all_transitions:
    if transition['timestamp'] >= cutoff:
        tm_recent.increment(transition['from'], transition['to'])
```

## Testing

Run the comprehensive test suite:

```bash
# All tests
pytest tests/unit/test_transition_matrix.py -v

# Specific test class
pytest tests/unit/test_transition_matrix.py::TestTopK -v

# With coverage
pytest tests/unit/test_transition_matrix.py --cov=src.markov
```

Quick validation script:

```bash
python test_transition_matrix_validation.py
```

## FAQ

**Q: When should I use smoothing?**  
A: Use smoothing when you need to handle unseen transitions gracefully. For cache prefetching, light smoothing (0.001-0.01) works well.

**Q: How do I handle very large vocabularies?**  
A: The sparse representation scales well. For 10K+ states, consider:
- Saving checkpoints regularly
- Using chunked processing
- Filtering low-frequency transitions

**Q: Can I update a saved matrix?**  
A: Yes! Load it, add more transitions, and save again:
```python
tm = TransitionMatrix.load("matrix.json")
tm.increment("new", "transition", 10)
tm.save("matrix.json")
```

**Q: How do I compare two transition matrices?**  
A: Convert to DataFrames and compare:
```python
df1 = tm1.to_dataframe()
df2 = tm2.to_dataframe()
merged_df = df1.merge(df2, on=['from_state', 'to_state'], suffixes=('_1', '_2'))
```

**Q: What's the difference between `get_top_k` and sorting `get_row`?**  
A: `get_top_k` is more efficient for k << n:
- `get_top_k`: O(n log k)
- Sorting: O(n log n)

## Related Components

- **MarkovChain**: Higher-level Markov chain model (to be implemented)
- **SequenceBuilder**: Extract sequences from raw data (see `preprocessing/sequence_builder.py`)
- **FeatureEngineer**: Feature extraction for RL models (see `preprocessing/feature_engineer.py`)

## Contributing

When modifying TransitionMatrix:

1. Update docstrings with time complexity
2. Add tests to `tests/unit/test_transition_matrix.py`
3. Update this README with new features
4. Run full test suite before committing

## License

Part of the Markov-RL API Cache project.

---

**Need help?** Check the test files for more examples or open an issue on GitHub.


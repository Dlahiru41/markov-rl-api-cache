# TransitionMatrix Quick Reference

## Import
```python
from src.markov import TransitionMatrix
```

## Create
```python
# No smoothing
tm = TransitionMatrix()

# With Laplace smoothing
tm = TransitionMatrix(smoothing=0.001)
```

## Build Matrix
```python
# Add single transition
tm.increment("A", "B")

# Add multiple observations
tm.increment("A", "B", count=10)

# Build from sequences
sequence = ["login", "profile", "browse", "checkout"]
for i in range(len(sequence) - 1):
    tm.increment(sequence[i], sequence[i + 1])
```

## Query
```python
# Get raw count
count = tm.get_count("A", "B")  # Returns int

# Get probability
prob = tm.get_probability("A", "B")  # Returns float [0.0, 1.0]

# Get all transitions from state
row = tm.get_row("A")  # Returns Dict[str, float]

# Get top k most likely
top = tm.get_top_k("A", k=3)  # Returns List[Tuple[str, float]]
```

## Properties
```python
tm.num_states          # Number of unique states
tm.num_transitions     # Number of non-zero transitions  
tm.sparsity           # Fraction of zeros (0.0 to 1.0)
tm.smoothing          # Smoothing parameter
```

## Statistics
```python
stats = tm.get_statistics()
# Returns dict with:
#   - num_states
#   - num_transitions
#   - sparsity
#   - avg_transitions_per_state
#   - most_common_transitions (top 10)
```

## Matrix Operations
```python
# Merge two matrices
merged = tm1.merge(tm2)
```

## Serialization
```python
# Save/Load JSON
tm.save("matrix.json")
tm = TransitionMatrix.load("matrix.json")

# Dict conversion
data = tm.to_dict()
tm = TransitionMatrix.from_dict(data)

# DataFrame (requires pandas)
df = tm.to_dataframe()
```

## Common Patterns

### Pattern: Build from API logs
```python
tm = TransitionMatrix(smoothing=0.001)

for user_session in sessions:
    for i in range(len(user_session) - 1):
        current = user_session[i]['endpoint']
        next_ep = user_session[i + 1]['endpoint']
        tm.increment(current, next_ep)
```

### Pattern: Predictive caching
```python
def get_prefetch_candidates(endpoint, k=3, threshold=0.1):
    """Get endpoints to prefetch."""
    candidates = tm.get_top_k(endpoint, k=k)
    return [ep for ep, prob in candidates if prob > threshold]
```

### Pattern: Incremental updates
```python
# Load existing matrix
tm = TransitionMatrix.load("production.json")

# Add new observations
for transition in new_data:
    tm.increment(transition.from_state, transition.to_state)

# Save updated matrix
tm.save("production.json")
```

## Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `increment()` | O(1) | Average |
| `get_count()` | O(1) | Dictionary lookup |
| `get_probability()` | O(1) or O(n) | O(n) with smoothing |
| `get_row()` | O(k) | k = outgoing transitions |
| `get_top_k()` | O(n log k) | Uses heap |
| `merge()` | O(n + m) | Sum of entries |

## Tips

✅ Use smoothing (0.001 - 0.01) for robustness  
✅ Save checkpoints for long training runs  
✅ Use `get_top_k()` instead of sorting `get_row()` for efficiency  
✅ Convert to DataFrame for visualization  
✅ Validate probabilities sum to 1.0  

## Full Example
```python
from src.markov import TransitionMatrix

# Create matrix with smoothing
tm = TransitionMatrix(smoothing=0.001)

# Build from data
tm.increment("login", "profile", 80)
tm.increment("login", "browse", 20)
tm.increment("profile", "orders", 50)
tm.increment("profile", "browse", 30)

# Query
print(f"P(profile|login) = {tm.get_probability('login', 'profile'):.2f}")

# Get likely next states
top = tm.get_top_k("login", k=2)
for state, prob in top:
    print(f"  {state}: {prob:.2%}")

# Save
tm.save("transitions.json")

# Statistics
stats = tm.get_statistics()
print(f"States: {stats['num_states']}, Sparsity: {stats['sparsity']:.1%}")
```

---
See `src/markov/README.md` for detailed documentation.


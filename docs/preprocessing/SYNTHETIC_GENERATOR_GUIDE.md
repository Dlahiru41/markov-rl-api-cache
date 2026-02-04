# Synthetic Data Generator Guide

## Overview

The `SyntheticGenerator` module creates realistic API call traces with **known ground truth patterns**. This is crucial for validating that your Markov chain and RL models correctly learn the underlying transition probabilities.

## Why Synthetic Data?

### Problems with Real Data:
- **Privacy concerns**: Real API logs contain sensitive user information
- **Unknown patterns**: You don't know the "true" behavior model
- **Validation difficulty**: Can't verify if your model learned correctly
- **Limited scenarios**: May not have cascade failures or edge cases

### Benefits of Synthetic Data:
- **Known ground truth**: You define the exact transition probabilities
- **Controllable**: Generate specific scenarios (normal, failures, etc.)
- **Reproducible**: Same seed → same data every time
- **Unlimited**: Generate as much data as needed
- **Safe**: No privacy issues

## Core Concepts

### WorkflowDefinition

A workflow defines user behavior as a **Markov chain**:

```python
workflow = WorkflowDefinition(
    name="ecommerce",
    entry_points={
        "/api/login": 0.6,        # 60% start here
        "/api/browse": 0.4        # 40% start here
    },
    transitions={
        "/api/login": {
            "/api/profile": 0.8,  # 80% go to profile
            "/api/browse": 0.2    # 20% go to browse
        },
        "/api/profile": {
            "/api/browse": 0.7,
            "/api/logout": 0.3
        }
    },
    exit_points={"/api/logout"},
    avg_response_times={
        "/api/login": 120,
        "/api/profile": 80,
        "/api/browse": 200
    }
)
```

**Key Components:**
- **Entry points**: Where users start (probabilities must sum to 1.0)
- **Transitions**: Where users go next (probabilities must sum to 1.0 per endpoint)
- **Exit points**: Endpoints that typically end sessions
- **Response times**: Realistic latencies (optional)

### SyntheticGenerator

Generates sessions by random walking through a workflow:

```python
gen = SyntheticGenerator(seed=42)  # Reproducible!

# Generate single session
session = gen.generate_session(
    workflow=workflow,
    user_id="user_123",
    start_time=datetime.now()
)

# Generate complete dataset
dataset = gen.generate_dataset(
    num_users=1000,
    sessions_per_user=(3, 2),  # mean=3, std=2
    date_range_days=30
)
```

## Pre-Built Workflows

### 1. E-commerce Workflow

Realistic e-commerce user journeys:

```python
workflow = SyntheticGenerator.ECOMMERCE_WORKFLOW
```

**Entry Points:**
- 60%: Login
- 30%: Browse products
- 10%: Search

**Key Paths:**
1. **Purchase path**: login → profile → browse → details → cart → checkout → payment → confirmation
2. **Browse path**: browse → details → browse (loop)
3. **Orders path**: login → profile → orders → order details

**Realistic Features:**
- Cart abandonment (50% checkout rate)
- Window shopping (browsing without buying)
- Order tracking
- Profile updates

### 2. Simple Workflow

For testing:

```python
workflow = create_simple_workflow()
```

### 3. Microservices Workflow

Simulates microservices architecture:

```python
workflow = create_microservices_workflow()
```

## Usage Examples

### Example 1: Generate Training Data

```python
from preprocessing.synthetic_generator import SyntheticGenerator
from datetime import datetime

# Create generator with seed for reproducibility
gen = SyntheticGenerator(seed=42)

# Generate dataset
dataset = gen.generate_dataset(
    num_users=500,
    sessions_per_user=(4, 2),  # Average 4 sessions per user
    date_range_days=30,
    show_progress=True
)

print(f"Generated {dataset.total_calls} API calls")
print(f"Unique users: {dataset.num_unique_users}")

# Save for later use
dataset.save_to_parquet('data/synthetic/train.parquet')
```

### Example 2: Validate Markov Chain Learning

```python
from preprocessing.synthetic_generator import SyntheticGenerator
from preprocessing.sequence_builder import SequenceBuilder

# Generate data with known probabilities
gen = SyntheticGenerator(seed=42)
workflow = gen.ECOMMERCE_WORKFLOW
dataset = gen.generate_dataset(num_users=200)

# Train your Markov chain
builder = SequenceBuilder()
learned_probs = builder.get_transition_probabilities(dataset.sessions)

# Compare with ground truth
true_prob = workflow.transitions["/api/login"]["/api/users/{id}/profile"]
learned_prob = learned_probs.get("/api/login", {}).get("/api/users/{id}/profile", 0)

print(f"True probability: {true_prob:.1%}")
print(f"Learned probability: {learned_prob:.1%}")
print(f"Error: {abs(true_prob - learned_prob):.1%}")
```

### Example 3: Generate Cascade Failures

```python
# Generate dataset with 20% cascade failures
dataset = gen.generate_dataset(
    num_users=100,
    cascade_failure_rate=0.20,  # 20% of sessions have failures
    show_progress=True
)

# Analyze failure patterns
failed_sessions = [s for s in dataset.sessions if any(c.status_code != 200 for c in s.calls)]
print(f"Sessions with errors: {len(failed_sessions)}")
```

### Example 4: Custom Workflow

```python
# Define your own workflow
my_workflow = WorkflowDefinition(
    name="api_gateway",
    entry_points={
        "/health": 0.10,
        "/api/v1/login": 0.90
    },
    transitions={
        "/health": {
            "/metrics": 1.0
        },
        "/api/v1/login": {
            "/api/v1/users/me": 0.8,
            "/api/v1/logout": 0.2
        },
        "/api/v1/users/me": {
            "/api/v1/resources": 0.6,
            "/api/v1/logout": 0.4
        },
        "/api/v1/resources": {
            "/api/v1/resources/{id}": 0.9,
            "/api/v1/logout": 0.1
        },
        "/api/v1/resources/{id}": {
            "/api/v1/resources": 0.5,
            "/api/v1/logout": 0.5
        },
        "/metrics": {
            "/health": 1.0
        }
    },
    exit_points={"/api/v1/logout", "/metrics"}
)

# Validate workflow
errors = SyntheticGenerator.validate_workflow(my_workflow)
if errors:
    print("Errors:", errors)
else:
    print("Workflow is valid!")

# Generate data
gen = SyntheticGenerator(seed=42)
dataset = gen.generate_dataset(num_users=50, workflow=my_workflow)
```

### Example 5: Save/Load Workflows

```python
from pathlib import Path

# Save workflow to YAML
workflow = gen.ECOMMERCE_WORKFLOW
workflow.to_yaml(Path("workflows/ecommerce.yaml"))

# Load workflow from YAML
loaded = WorkflowDefinition.from_yaml(Path("workflows/ecommerce.yaml"))

# Use loaded workflow
dataset = gen.generate_dataset(num_users=100, workflow=loaded)
```

## Features

### 1. Reproducibility

Same seed → same data:

```python
gen1 = SyntheticGenerator(seed=42)
session1 = gen1.generate_session(workflow, "user1", datetime.now())

gen2 = SyntheticGenerator(seed=42)
session2 = gen2.generate_session(workflow, "user1", datetime.now())

# session1.endpoint_sequence == session2.endpoint_sequence ✓
```

### 2. Realistic Timestamps

- Sessions spread over configurable date range
- Random hours and minutes
- Delays between calls (1-5s normal, 10-30s for thinking)

### 3. User Type Distribution

Configurable user type proportions:

```python
workflow.user_type_distribution = {
    'premium': 0.3,   # 30%
    'free': 0.5,      # 50%
    'guest': 0.2      # 20%
}
```

### 4. Realistic Response Times

- Based on endpoint type (payments slow, logout fast)
- Random variation (normal distribution)
- Cascade failures multiply times by 3-10x

### 5. HTTP Methods

Automatically determined:
- POST: login, add, remove, payment, checkout
- PUT: update, settings
- GET: everything else

### 6. Status Codes

- 95% success (200)
- 5% errors (400/500)
- Cascade failures add timeouts (504) and unavailable (503)

### 7. Request Parameters

Realistic parameters for search and browse:

```python
# Search parameters
{'q': 'laptop', 'sort': 'price'}

# Browse parameters
{'category': 'electronics', 'page': '2'}
```

## Cascade Failure Injection

Simulates cascading failures in microservices:

```python
# Generate normal session
session = gen.generate_session(workflow, "user1", datetime.now())

# Inject cascade failure
failed_session = gen.inject_cascade(session)
```

**Cascade Effects:**
- **Slow responses**: 3-10x slower (60% of calls)
- **Timeouts**: 504 errors with 30s response time (20%)
- **Service errors**: 503 errors (15%)
- **Retries**: Duplicate calls (30%)
- **Starts mid-session**: Affects second half of session

## Workflow Validation

Built-in validation checks:

```python
errors = SyntheticGenerator.validate_workflow(workflow)

if errors:
    for error in errors:
        print(f"❌ {error}")
else:
    print("✓ Workflow is valid")
```

**Checks:**
- Entry points sum to 1.0
- Transitions sum to 1.0 per endpoint
- No dead ends (unreachable states)
- Exit points properly defined

## Progress Tracking

For large datasets:

```python
dataset = gen.generate_dataset(
    num_users=10000,
    show_progress=True  # Shows tqdm progress bar
)
```

Output:
```
Generating users: 100%|████████████| 10000/10000 [00:45<00:00, 220.45it/s]
```

## Ground Truth Validation

The key benefit: **verify your models learn correctly!**

```python
# 1. Generate data with known probabilities
gen = SyntheticGenerator(seed=42)
workflow = gen.ECOMMERCE_WORKFLOW
dataset = gen.generate_dataset(num_users=1000)

# 2. Train your model
from preprocessing.sequence_builder import SequenceBuilder
builder = SequenceBuilder()
learned_probs = builder.get_transition_probabilities(dataset.sessions)

# 3. Compare with ground truth
def validate_learning(workflow, learned_probs):
    errors = []
    for from_ep, transitions in workflow.transitions.items():
        for to_ep, true_prob in transitions.items():
            learned_prob = learned_probs.get(from_ep, {}).get(to_ep, 0)
            error = abs(true_prob - learned_prob)
            if error > 0.10:  # More than 10% error
                errors.append({
                    'transition': f"{from_ep} → {to_ep}",
                    'true': true_prob,
                    'learned': learned_prob,
                    'error': error
                })
    return errors

errors = validate_learning(workflow, learned_probs)
if errors:
    print("Learning errors:")
    for err in errors:
        print(f"  {err['transition']}: {err['error']:.1%} error")
else:
    print("✓ Model learned correctly!")
```

## Best Practices

### 1. Always Use Seeds

For reproducible experiments:

```python
gen = SyntheticGenerator(seed=42)  # Always specify seed
```

### 2. Validate Workflows

Before generating large datasets:

```python
errors = SyntheticGenerator.validate_workflow(workflow)
assert not errors, f"Workflow invalid: {errors}"
```

### 3. Start Small

Test with small datasets first:

```python
# Test
test_dataset = gen.generate_dataset(num_users=10, show_progress=False)

# Production
full_dataset = gen.generate_dataset(num_users=10000, show_progress=True)
```

### 4. Version Workflows

Save workflows to YAML for reproducibility:

```python
workflow.to_yaml(Path(f"workflows/{workflow.name}_v1.yaml"))
```

### 5. Separate Train/Test/Validation

```python
# Train
train_gen = SyntheticGenerator(seed=42)
train_dataset = train_gen.generate_dataset(num_users=1000)

# Test
test_gen = SyntheticGenerator(seed=123)
test_dataset = test_gen.generate_dataset(num_users=200)

# Validation
val_gen = SyntheticGenerator(seed=456)
val_dataset = val_gen.generate_dataset(num_users=200)
```

### 6. Mix Normal and Failure Data

```python
# 80% normal, 20% with failures
dataset = gen.generate_dataset(
    num_users=1000,
    cascade_failure_rate=0.20
)
```

## Common Patterns

### Pattern 1: A/B Testing Validation

```python
# Workflow A
workflow_a = WorkflowDefinition(...)

# Workflow B (variant)
workflow_b = WorkflowDefinition(...)

# Generate data for each
data_a = gen.generate_dataset(num_users=500, workflow=workflow_a)
data_b = gen.generate_dataset(num_users=500, workflow=workflow_b)

# Train models and compare performance
```

### Pattern 2: Time-Series Data

```python
# Generate month of data
dataset = gen.generate_dataset(
    num_users=1000,
    date_range_days=30,
    start_date=datetime(2026, 1, 1)
)

# Split by week for time-series analysis
```

### Pattern 3: Failure Detection Training

```python
# Generate normal data
normal = gen.generate_dataset(num_users=800, cascade_failure_rate=0.0)

# Generate failure data
failures = gen.generate_dataset(num_users=200, cascade_failure_rate=1.0)

# Train anomaly detector
```

## Troubleshooting

### Issue: Probabilities Don't Match

**Problem**: Learned probabilities differ significantly from workflow

**Solutions:**
- Generate more data (1000+ users)
- Check for bugs in learning algorithm
- Verify workflow is valid
- Ensure random seed is set

### Issue: Sessions Too Short/Long

**Problem**: Generated sessions not realistic length

**Solutions:**
- Adjust `max_length` parameter
- Add more exit points
- Adjust transition probabilities to exits

### Issue: Workflow Invalid

**Problem**: Validation errors

**Solutions:**
- Check probabilities sum to 1.0
- Ensure all paths lead somewhere
- Add exit points for dead ends

## Related Modules

- **preprocessing.models**: APICall, Session, Dataset classes
- **preprocessing.sequence_builder**: Sequence extraction
- **preprocessing.feature_engineer**: Feature extraction for RL

## Performance

- **Generation speed**: ~100-200 users/second
- **Memory**: O(num_sessions * avg_calls_per_session)
- **Disk**: ~1KB per call in Parquet format

## References

- Markov chains: [Wikipedia](https://en.wikipedia.org/wiki/Markov_chain)
- Synthetic data generation: [Research paper]
- Workflow modeling: [Process mining]


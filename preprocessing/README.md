# Preprocessing Data Models

## Overview

This document describes the data models created for representing API call traces in the preprocessing module. These models are designed to support the Markov-RL API caching system by capturing API call patterns, user sessions, and training datasets.

## Models

### 1. APICall

Represents a single API request with all relevant timing, user context, and response characteristics.

**Fields:**
- `call_id`: Unique identifier for the call
- `endpoint`: API endpoint path (e.g., "/api/users/123")
- `method`: HTTP method (GET, POST, etc.)
- `params`: Dictionary of request parameters
- `user_id`: User identifier
- `session_id`: Session identifier
- `timestamp`: When the call was made
- `response_time_ms`: Response time in milliseconds
- `status_code`: HTTP status code
- `response_size_bytes`: Response size in bytes
- `user_type`: User tier (premium/free/guest)

**Methods:**
- `to_dict()`: Convert to dictionary (with timestamp as ISO string)
- `from_dict(data)`: Create from dictionary
- `get_service_name()`: Extract service name from endpoint
  - Example: "/api/users/123" → "users"
- `is_successful()`: Check if call succeeded (2xx status)

**Validation:**
- Response time and size must be non-negative
- User type must be 'premium', 'free', or 'guest'
- Endpoint must start with '/'

### 2. Session

Groups related API calls from one user visit, providing analysis methods for Markov chain training.

**Fields:**
- `session_id`: Unique session identifier
- `user_id`: User identifier
- `user_type`: User tier
- `start_timestamp`: Session start time
- `end_timestamp`: Session end time (optional)
- `calls`: List of APICall objects

**Properties:**
- `duration_seconds`: Total session duration
- `num_calls`: Number of calls in session
- `unique_endpoints`: List of unique endpoints visited
- `endpoint_sequence`: Chronological list of endpoints (for Markov training)

**Methods:**
- `append_call(call)`: Add a new call to the session
- `get_endpoint_transitions()`: Get consecutive endpoint pairs
  - Returns list of (from_endpoint, to_endpoint) tuples
  - Used to train the Markov chain
- `to_dict()` / `from_dict(data)`: Serialization methods

**Validation:**
- End timestamp must be after start timestamp
- All calls must belong to this session

### 3. Dataset

Holds a collection of sessions for analysis, training, and persistence.

**Fields:**
- `name`: Dataset name
- `sessions`: List of Session objects
- `metadata`: Dictionary for storing statistics and info

**Properties:**
- `total_calls`: Total number of API calls across all sessions
- `num_unique_users`: Number of unique users
- `unique_endpoints`: Set of all unique endpoints
- `date_range`: Tuple of (earliest, latest) timestamps

**Methods:**
- `get_all_sequences()`: Get all endpoint sequences for Markov training
- `count_endpoint_occurrences()`: Count frequency of each endpoint
- `split(train_ratio)`: Split into train/test datasets (default 80/20)
- `save_to_parquet(filepath)`: Save to parquet file (requires pyarrow)
- `load_from_parquet(filepath)`: Load from parquet file (requires pyarrow)

## Usage Example

```python
from preprocessing.models import APICall, Session, Dataset
from datetime import datetime

# Create an API call
call = APICall(
    call_id="1",
    endpoint="/api/users/123",
    method="GET",
    params={},
    user_id="user1",
    session_id="sess1",
    timestamp=datetime.now(),
    response_time_ms=100,
    status_code=200,
    response_size_bytes=1024,
    user_type="free"
)

# Extract service name
service = call.get_service_name()  # Returns "users"

# Check if successful
is_ok = call.is_successful()  # Returns True for 2xx status

# Create a session
session = Session(
    session_id="sess1",
    user_id="user1",
    user_type="free",
    start_timestamp=datetime.now()
)

# Add calls to session
session.append_call(call)
# ... add more calls ...

# Get transitions for Markov chain training
transitions = session.get_endpoint_transitions()
# Returns: [('/api/users/123', '/api/products/456'), ...]

# Create a dataset
dataset = Dataset(
    name="training_data",
    sessions=[session],
    metadata={"version": "1.0"}
)

# Split for training
train_dataset, test_dataset = dataset.split(train_ratio=0.8)

# Get sequences for Markov training
sequences = dataset.get_all_sequences()
# Returns: [['/api/home', '/api/products', ...], ...]

# Count endpoint frequencies
counts = dataset.count_endpoint_occurrences()
# Returns: {'/api/users': 150, '/api/products': 200, ...}
```

## Serialization

All models support serialization to/from dictionaries:

```python
# Serialize
call_dict = call.to_dict()
session_dict = session.to_dict()

# Deserialize
restored_call = APICall.from_dict(call_dict)
restored_session = Session.from_dict(session_dict)
```

## Parquet Support

Datasets can be saved to and loaded from Parquet files for efficient storage:

```python
from pathlib import Path

# Save
dataset.save_to_parquet(Path("data/training.parquet"))

# Load
loaded_dataset = Dataset.load_from_parquet(Path("data/training.parquet"))
```

**Note:** Parquet functionality requires `pyarrow` to be installed:
```bash
pip install pyarrow
```

## Markov Chain Training Integration

The models are specifically designed to support Markov chain training:

1. **Endpoint Sequences**: `Session.endpoint_sequence` provides the ordered list of endpoints
2. **Transitions**: `Session.get_endpoint_transitions()` extracts state transitions
3. **Batch Processing**: `Dataset.get_all_sequences()` collects sequences from all sessions
4. **Service Extraction**: `APICall.get_service_name()` groups by microservice

## Testing

Comprehensive tests are provided in:
- `test_models_validation.py` - Basic validation
- `test_comprehensive_models.py` - Full feature tests
- `test_parquet_functionality.py` - Parquet I/O tests

Run tests:
```bash
python test_models_validation.py
python test_comprehensive_models.py
python test_parquet_functionality.py  # Requires pyarrow
```

## File Location

`preprocessing/models.py`

## Dependencies

- Python 3.10+
- dataclasses (built-in)
- datetime (built-in)
- typing (built-in)
- json (built-in)
- pathlib (built-in)
- pyarrow (optional, for Parquet support)


# SessionExtractor Module

## Overview

The `SessionExtractor` module provides functionality to group raw API call logs into meaningful user sessions based on temporal patterns and inactivity timeouts. This is essential for training Markov chains that predict user behavior.

## Problem Statement

Raw API logs are just a stream of individual requests from many users. To train a Markov chain that predicts user behavior, we need to identify which calls belong to the same user "session" - a continuous sequence of activity before the user goes idle or leaves.

## Features

### 1. Configuration
- **Inactivity timeout**: Configurable timeout period (default 30 minutes) - if a user is inactive for longer than this, their next request starts a new session
- **Session length filtering**: Minimum and maximum session length parameters to filter out sessions that are too short (not useful for learning) or suspiciously long (might be bots or errors)
- **Progress tracking**: Optional progress bars for large datasets

### 2. Session Extraction
- Groups `APICall` objects into `Session` objects
- Automatically sorts calls by user ID and timestamp
- Detects session boundaries based on inactivity timeouts
- Assigns unique session IDs
- Updates all calls with their session ID

### 3. Multiple Input Formats
- **List of APICall objects**: Direct extraction from in-memory objects
- **Pandas DataFrame**: Extract from DataFrame with configurable column mapping
- **File formats**: Automatic detection and loading from CSV, JSON, or Parquet files

### 4. Robust Error Handling
- Empty input returns empty list
- Missing timestamps are tracked and skipped
- Malformed data is tracked and skipped
- Multiple users' data can be interleaved

### 5. Statistics Tracking
- Total calls processed
- Total sessions created
- Sessions kept vs filtered
- Unique users
- Average session length and duration
- Min/max session metrics
- Error counts

## Usage

### Basic Usage

```python
from preprocessing.session_extractor import SessionExtractor
from preprocessing.models import APICall
from datetime import datetime

# Create test calls
calls = [
    APICall("1", "/login", "POST", {}, "user1", "", datetime(2024,1,1,10,0), 100, 200, 500, "free"),
    APICall("2", "/profile", "GET", {}, "user1", "", datetime(2024,1,1,10,5), 50, 200, 200, "free"),
    APICall("3", "/browse", "GET", {}, "user1", "", datetime(2024,1,1,10,10), 80, 200, 1000, "free"),
    # Gap of 1 hour - should be new session
    APICall("4", "/login", "POST", {}, "user1", "", datetime(2024,1,1,11,30), 100, 200, 500, "free"),
]

# Create extractor
extractor = SessionExtractor(
    inactivity_timeout_minutes=30,
    min_session_length=2
)

# Extract sessions
sessions = extractor.extract_sessions(calls)

print(f"Found {len(sessions)} sessions")
print(extractor.get_statistics())
```

### Configuration Options

```python
extractor = SessionExtractor(
    inactivity_timeout_minutes=30,  # Time gap to trigger new session
    min_session_length=2,           # Minimum calls per session
    max_session_length=100,         # Maximum calls per session (None = no limit)
    show_progress=True              # Show progress bar
)
```

### Extract from DataFrame

```python
import pandas as pd

# Load data
df = pd.read_csv('api_logs.csv')

# Extract with custom column mapping
sessions = extractor.extract_from_dataframe(
    df,
    column_mapping={
        'id': 'call_id',
        'path': 'endpoint',
        'user': 'user_id',
        'time': 'timestamp',
        # ... other mappings
    }
)
```

### Extract from File

```python
from pathlib import Path

# Automatically detect format (CSV, JSON, or Parquet)
sessions = extractor.extract_from_file(Path('api_logs.json'))

# Or
sessions = extractor.extract_from_file(Path('api_logs.csv'))

# Or
sessions = extractor.extract_from_file(Path('api_logs.parquet'))
```

### Get Statistics

```python
sessions = extractor.extract_sessions(calls)
stats = extractor.get_statistics()

print(f"Processed: {stats['total_calls_processed']} calls")
print(f"Created: {stats['total_sessions_created']} sessions")
print(f"Kept: {stats['sessions_kept']} sessions")
print(f"Filtered (too short): {stats['sessions_filtered_too_short']}")
print(f"Filtered (too long): {stats['sessions_filtered_too_long']}")
print(f"Unique users: {stats['unique_users']}")
print(f"Avg session length: {stats['average_session_length']:.2f} calls")
print(f"Avg session duration: {stats['average_session_duration_seconds']:.2f}s")
```

## Algorithm

The session extraction algorithm works as follows:

1. **Filter invalid calls**: Remove calls with missing or malformed timestamps
2. **Group by user**: Organize calls by `user_id`
3. **Sort by timestamp**: Sort each user's calls chronologically
4. **Identify session boundaries**: For each user:
   - Start with the first call in a new session
   - For each subsequent call:
     - If time gap from previous call > timeout → start new session
     - Otherwise → continue current session
5. **Filter sessions**: Apply min/max length constraints
6. **Assign session IDs**: Generate unique IDs like `{user_id}_session_{number}`
7. **Update calls**: Set `session_id` field for all calls
8. **Calculate statistics**: Track all metrics

## Session ID Format

Session IDs follow the format: `{user_id}_session_{session_number}`

Example:
- `user1_session_0` - First session for user1
- `user1_session_1` - Second session for user1
- `user2_session_0` - First session for user2

## Statistics Dictionary

The `get_statistics()` method returns a dictionary with the following keys:

- `total_calls_processed` (int): Total number of valid API calls processed
- `total_sessions_created` (int): Total sessions created before filtering
- `sessions_kept` (int): Sessions that passed filtering
- `sessions_filtered_too_short` (int): Sessions removed (< min_session_length)
- `sessions_filtered_too_long` (int): Sessions removed (> max_session_length)
- `unique_users` (int): Number of unique users in the data
- `average_session_length` (float): Average number of calls per session
- `average_session_duration_seconds` (float): Average session duration
- `min_session_length` (int): Shortest session (in calls)
- `max_session_length` (int): Longest session (in calls)
- `min_session_duration_seconds` (float): Shortest session duration
- `max_session_duration_seconds` (float): Longest session duration
- `calls_with_missing_timestamps` (int): Calls skipped due to missing timestamps
- `calls_with_malformed_data` (int): Calls skipped due to malformed data

## Examples

### Example 1: Single User, Multiple Sessions

```python
calls = [
    # Session 1
    APICall("1", "/home", "GET", {}, "user1", "", datetime(2024,1,1,10,0), 100, 200, 500, "free"),
    APICall("2", "/products", "GET", {}, "user1", "", datetime(2024,1,1,10,5), 100, 200, 500, "free"),
    
    # 1 hour gap - new session
    APICall("3", "/home", "GET", {}, "user1", "", datetime(2024,1,1,11,10), 100, 200, 500, "free"),
    APICall("4", "/cart", "POST", {}, "user1", "", datetime(2024,1,1,11,15), 100, 200, 500, "free"),
]

extractor = SessionExtractor(inactivity_timeout_minutes=30, min_session_length=2)
sessions = extractor.extract_sessions(calls)

# Result: 2 sessions
# - user1_session_0: calls 1-2
# - user1_session_1: calls 3-4
```

### Example 2: Multiple Users Interleaved

```python
calls = [
    APICall("1", "/a", "GET", {}, "alice", "", datetime(2024,1,1,10,0), 100, 200, 500, "free"),
    APICall("2", "/b", "GET", {}, "bob", "", datetime(2024,1,1,10,1), 100, 200, 500, "premium"),
    APICall("3", "/c", "GET", {}, "alice", "", datetime(2024,1,1,10,2), 100, 200, 500, "free"),
    APICall("4", "/d", "GET", {}, "bob", "", datetime(2024,1,1,10,3), 100, 200, 500, "premium"),
]

extractor = SessionExtractor(min_session_length=2)
sessions = extractor.extract_sessions(calls)

# Result: 2 sessions (one per user)
# - alice_session_0: calls 1, 3
# - bob_session_0: calls 2, 4
```

### Example 3: Filtering Short Sessions

```python
calls = [
    # User 1: Single call - will be filtered
    APICall("1", "/a", "GET", {}, "user1", "", datetime(2024,1,1,10,0), 100, 200, 500, "free"),
    
    # User 2: Two calls - will be kept
    APICall("2", "/b", "GET", {}, "user2", "", datetime(2024,1,1,10,0), 100, 200, 500, "premium"),
    APICall("3", "/c", "GET", {}, "user2", "", datetime(2024,1,1,10,5), 100, 200, 500, "premium"),
]

extractor = SessionExtractor(min_session_length=2)
sessions = extractor.extract_sessions(calls)
stats = extractor.get_statistics()

# Result: 1 session kept (user2)
# stats['sessions_filtered_too_short'] == 1
```

## Integration with Markov Training

Once sessions are extracted, you can use them to train Markov chains:

```python
from preprocessing.session_extractor import SessionExtractor
from preprocessing.models import Dataset

# Extract sessions
extractor = SessionExtractor(inactivity_timeout_minutes=30, min_session_length=3)
sessions = extractor.extract_sessions(api_calls)

# Create dataset
dataset = Dataset(
    name="user_sessions",
    sessions=sessions,
    metadata={"extraction_config": extractor.get_statistics()}
)

# Get sequences for Markov training
sequences = dataset.get_all_sequences()
transitions = []
for session in dataset.sessions:
    transitions.extend(session.get_endpoint_transitions())

# Train Markov model
# markov_model.train(transitions)
```

## File Location

`preprocessing/session_extractor.py`

## Dependencies

- **Required**: Python 3.10+, datetime, dataclasses, typing, json, pathlib
- **Optional**: 
  - `pandas` (for DataFrame and CSV/Parquet support)
  - `tqdm` (for progress bars)
  - `pyarrow` (for Parquet format)

## Testing

Run the validation tests:

```bash
python test_session_extractor_validation.py
python test_session_extractor_comprehensive.py
```

All tests should pass with detailed output showing:
- Session extraction from various input formats
- Filtering based on session length
- Inactivity timeout detection
- Multi-user handling
- Statistics accuracy
- Error handling


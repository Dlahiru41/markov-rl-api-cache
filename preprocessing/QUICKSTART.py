"""
Quick Reference Guide for Preprocessing Models
==============================================

BASIC USAGE
-----------

from preprocessing.models import APICall, Session, Dataset
from datetime import datetime

# 1. Create API Call
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
    user_type="free"  # or "premium" or "guest"
)

# 2. Extract Information
service = call.get_service_name()      # "users"
success = call.is_successful()        # True (2xx status)
call_dict = call.to_dict()           # Serialize

# 3. Create Session
session = Session(
    session_id="sess1",
    user_id="user1",
    user_type="free",
    start_timestamp=datetime.now()
)

# 4. Add Calls to Session
session.append_call(call)

# 5. Analyze Session
print(session.num_calls)              # Number of calls
print(session.duration_seconds)       # Duration
print(session.endpoint_sequence)      # ['/api/users/123', ...]
print(session.get_endpoint_transitions())  # [('/api/users', '/api/products'), ...]

# 6. Create Dataset
dataset = Dataset(
    name="my_dataset",
    sessions=[session],
    metadata={"version": "1.0"}
)

# 7. Dataset Operations
print(dataset.total_calls)            # Total calls
print(dataset.num_unique_users)       # Unique users
print(dataset.unique_endpoints)       # Set of endpoints

# 8. Get Training Data
sequences = dataset.get_all_sequences()  # For Markov chain training
counts = dataset.count_endpoint_occurrences()  # Endpoint frequencies

# 9. Split Dataset
train, test = dataset.split(train_ratio=0.8)

# 10. Save/Load (requires pyarrow)
from pathlib import Path
dataset.save_to_parquet(Path("data.parquet"))
loaded = Dataset.load_from_parquet(Path("data.parquet"))


MARKOV CHAIN INTEGRATION
-------------------------

# Get all endpoint transitions for training
all_transitions = []
for session in dataset.sessions:
    transitions = session.get_endpoint_transitions()
    all_transitions.extend(transitions)

# transitions = [
#   ('/api/home', '/api/products'),
#   ('/api/products', '/api/cart'),
#   ('/api/cart', '/api/checkout'),
#   ...
# ]


VALIDATION RULES
----------------

APICall:
  - response_time_ms >= 0
  - response_size_bytes >= 0
  - user_type in ['premium', 'free', 'guest']
  - endpoint must start with '/'

Session:
  - user_type in ['premium', 'free', 'guest']
  - end_timestamp >= start_timestamp (if set)
  - all calls must have matching session_id

Dataset:
  - train_ratio must be between 0 and 1 (for split)


EXAMPLE WORKFLOW
----------------

# 1. Load raw API logs
raw_logs = [...]  # Your API logs

# 2. Convert to APICall objects
calls = []
for log in raw_logs:
    call = APICall(
        call_id=log['id'],
        endpoint=log['path'],
        method=log['method'],
        params=log.get('params', {}),
        user_id=log['user_id'],
        session_id=log['session_id'],
        timestamp=datetime.fromisoformat(log['timestamp']),
        response_time_ms=log['response_time'],
        status_code=log['status'],
        response_size_bytes=log['size'],
        user_type=log['user_type']
    )
    calls.append(call)

# 3. Group by session
from collections import defaultdict
sessions_dict = defaultdict(list)
for call in calls:
    sessions_dict[call.session_id].append(call)

# 4. Create Session objects
sessions = []
for session_id, session_calls in sessions_dict.items():
    # Sort by timestamp
    session_calls.sort(key=lambda c: c.timestamp)

    session = Session(
        session_id=session_id,
        user_id=session_calls[0].user_id,
        user_type=session_calls[0].user_type,
        start_timestamp=session_calls[0].timestamp,
        end_timestamp=session_calls[-1].timestamp,
        calls=session_calls
    )
    sessions.append(session)

# 5. Create Dataset
dataset = Dataset(
    name="production_data",
    sessions=sessions,
    metadata={
        "source": "production_logs",
        "date": str(datetime.now().date())
    }
)

# 6. Split and save
train, test = dataset.split(0.8)
train.save_to_parquet(Path("data/train.parquet"))
test.save_to_parquet(Path("data/test.parquet"))

# 7. Train Markov model
sequences = train.get_all_sequences()
# Feed sequences to your Markov chain implementation
"""

if __name__ == "__main__":
    print(__doc__)


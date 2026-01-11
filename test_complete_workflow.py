"""
Complete Workflow Demonstration
================================
This script demonstrates the complete workflow from raw API logs
to dataset preparation for Markov chain training.
"""

from preprocessing.models import APICall, Session, Dataset
from datetime import datetime, timedelta
from collections import defaultdict

print("=" * 70)
print("COMPLETE WORKFLOW: API LOGS → MARKOV TRAINING DATA")
print("=" * 70)
print()

# Step 1: Simulate raw API logs
print("Step 1: Simulating raw API logs...")
print("-" * 70)

raw_logs = [
    # User 1, Session 1: Browse → Add to Cart → Checkout
    {"id": "1", "path": "/api/home", "method": "GET", "user_id": "user_alice",
     "session_id": "sess_alice_1", "timestamp": "2026-01-11T10:00:00",
     "response_time": 50, "status": 200, "size": 1024, "user_type": "premium"},
    {"id": "2", "path": "/api/products", "method": "GET", "user_id": "user_alice",
     "session_id": "sess_alice_1", "timestamp": "2026-01-11T10:00:05",
     "response_time": 75, "status": 200, "size": 2048, "user_type": "premium"},
    {"id": "3", "path": "/api/products/123", "method": "GET", "user_id": "user_alice",
     "session_id": "sess_alice_1", "timestamp": "2026-01-11T10:00:10",
     "response_time": 60, "status": 200, "size": 1536, "user_type": "premium"},
    {"id": "4", "path": "/api/cart/add", "method": "POST", "user_id": "user_alice",
     "session_id": "sess_alice_1", "timestamp": "2026-01-11T10:00:15",
     "response_time": 100, "status": 201, "size": 512, "user_type": "premium"},
    {"id": "5", "path": "/api/checkout", "method": "POST", "user_id": "user_alice",
     "session_id": "sess_alice_1", "timestamp": "2026-01-11T10:00:20",
     "response_time": 150, "status": 200, "size": 768, "user_type": "premium"},

    # User 2, Session 1: Browse → Leave
    {"id": "6", "path": "/api/home", "method": "GET", "user_id": "user_bob",
     "session_id": "sess_bob_1", "timestamp": "2026-01-11T10:05:00",
     "response_time": 45, "status": 200, "size": 1024, "user_type": "free"},
    {"id": "7", "path": "/api/products", "method": "GET", "user_id": "user_bob",
     "session_id": "sess_bob_1", "timestamp": "2026-01-11T10:05:03",
     "response_time": 80, "status": 200, "size": 2048, "user_type": "free"},

    # User 1, Session 2: Quick check
    {"id": "8", "path": "/api/orders", "method": "GET", "user_id": "user_alice",
     "session_id": "sess_alice_2", "timestamp": "2026-01-11T11:00:00",
     "response_time": 55, "status": 200, "size": 1200, "user_type": "premium"},
    {"id": "9", "path": "/api/orders/456", "method": "GET", "user_id": "user_alice",
     "session_id": "sess_alice_2", "timestamp": "2026-01-11T11:00:05",
     "response_time": 65, "status": 200, "size": 1800, "user_type": "premium"},

    # User 3, Session 1: Guest browsing
    {"id": "10", "path": "/api/home", "method": "GET", "user_id": "guest_001",
     "session_id": "sess_guest_1", "timestamp": "2026-01-11T10:10:00",
     "response_time": 40, "status": 200, "size": 1024, "user_type": "guest"},
    {"id": "11", "path": "/api/products", "method": "GET", "user_id": "guest_001",
     "session_id": "sess_guest_1", "timestamp": "2026-01-11T10:10:05",
     "response_time": 70, "status": 200, "size": 2048, "user_type": "guest"},
    {"id": "12", "path": "/api/products/789", "method": "GET", "user_id": "guest_001",
     "session_id": "sess_guest_1", "timestamp": "2026-01-11T10:10:10",
     "response_time": 55, "status": 200, "size": 1536, "user_type": "guest"},
]

print(f"Loaded {len(raw_logs)} raw API log entries")
print()

# Step 2: Convert to APICall objects
print("Step 2: Converting to APICall objects...")
print("-" * 70)

calls = []
for log in raw_logs:
    call = APICall(
        call_id=log['id'],
        endpoint=log['path'],
        method=log['method'],
        params={},
        user_id=log['user_id'],
        session_id=log['session_id'],
        timestamp=datetime.fromisoformat(log['timestamp']),
        response_time_ms=log['response_time'],
        status_code=log['status'],
        response_size_bytes=log['size'],
        user_type=log['user_type']
    )
    calls.append(call)

print(f"Created {len(calls)} APICall objects")
print(f"Sample call: {calls[0]}")
print(f"Service name extracted: {calls[0].get_service_name()}")
print()

# Step 3: Group by session
print("Step 3: Grouping calls by session...")
print("-" * 70)

sessions_dict = defaultdict(list)
for call in calls:
    sessions_dict[call.session_id].append(call)

print(f"Found {len(sessions_dict)} unique sessions")
for session_id, session_calls in sessions_dict.items():
    print(f"  {session_id}: {len(session_calls)} calls")
print()

# Step 4: Create Session objects
print("Step 4: Creating Session objects...")
print("-" * 70)

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

print(f"Created {len(sessions)} Session objects")
for session in sessions:
    print(f"  {session}")
    print(f"    Endpoint sequence: {session.endpoint_sequence}")
    print(f"    Transitions: {session.get_endpoint_transitions()}")
print()

# Step 5: Create Dataset
print("Step 5: Creating Dataset...")
print("-" * 70)

dataset = Dataset(
    name="ecommerce_api_logs",
    sessions=sessions,
    metadata={
        "source": "production",
        "date": "2026-01-11",
        "description": "E-commerce API call traces"
    }
)

print(f"Dataset: {dataset}")
print(f"  Total calls: {dataset.total_calls}")
print(f"  Unique users: {dataset.num_unique_users}")
print(f"  Unique endpoints: {len(dataset.unique_endpoints)}")
print(f"  Date range: {dataset.date_range}")
print()

# Step 6: Analyze endpoint patterns
print("Step 6: Analyzing endpoint patterns...")
print("-" * 70)

endpoint_counts = dataset.count_endpoint_occurrences()
print("Endpoint frequencies:")
for endpoint, count in sorted(endpoint_counts.items(), key=lambda x: -x[1]):
    print(f"  {endpoint}: {count} times")
print()

# Step 7: Extract sequences for Markov training
print("Step 7: Extracting sequences for Markov chain training...")
print("-" * 70)

sequences = dataset.get_all_sequences()
print(f"Extracted {len(sequences)} sequences:")
for i, seq in enumerate(sequences, 1):
    print(f"  Sequence {i}: {' → '.join(seq)}")
print()

# Step 8: Extract all transitions
print("Step 8: Extracting state transitions...")
print("-" * 70)

all_transitions = []
for session in dataset.sessions:
    transitions = session.get_endpoint_transitions()
    all_transitions.extend(transitions)

print(f"Extracted {len(all_transitions)} transitions:")
transition_counts = {}
for from_state, to_state in all_transitions:
    key = f"{from_state} → {to_state}"
    transition_counts[key] = transition_counts.get(key, 0) + 1

for transition, count in sorted(transition_counts.items(), key=lambda x: -x[1]):
    print(f"  {transition}: {count} times")
print()

# Step 9: Split dataset
print("Step 9: Splitting into train/test sets...")
print("-" * 70)

train_dataset, test_dataset = dataset.split(train_ratio=0.75)
print(f"Train dataset: {train_dataset}")
print(f"Test dataset: {test_dataset}")
print()

# Step 10: Summary statistics
print("Step 10: Summary Statistics")
print("-" * 70)

print("User Type Distribution:")
user_types = {}
for session in dataset.sessions:
    user_types[session.user_type] = user_types.get(session.user_type, 0) + 1
for user_type, count in sorted(user_types.items()):
    print(f"  {user_type}: {count} sessions")
print()

print("Response Time Statistics:")
response_times = [call.response_time_ms for session in dataset.sessions for call in session.calls]
avg_response_time = sum(response_times) / len(response_times)
min_response_time = min(response_times)
max_response_time = max(response_times)
print(f"  Average: {avg_response_time:.2f}ms")
print(f"  Min: {min_response_time}ms")
print(f"  Max: {max_response_time}ms")
print()

print("Success Rate:")
successful_calls = sum(1 for session in dataset.sessions for call in session.calls if call.is_successful())
total_calls = dataset.total_calls
success_rate = (successful_calls / total_calls) * 100
print(f"  {successful_calls}/{total_calls} = {success_rate:.1f}%")
print()

print("=" * 70)
print("✓ WORKFLOW COMPLETE!")
print("=" * 70)
print()
print("Next steps:")
print("  1. Use the sequences for Markov chain training")
print("  2. Use the transitions to build transition probability matrix")
print("  3. Save datasets to parquet for later use (requires pyarrow)")
print("  4. Integrate with your caching prediction system")


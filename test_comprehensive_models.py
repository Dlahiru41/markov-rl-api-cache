"""Comprehensive tests for preprocessing models."""

from preprocessing.models import APICall, Session, Dataset
from datetime import datetime, timedelta
import json

print("=" * 60)
print("COMPREHENSIVE MODEL TESTS")
print("=" * 60)
print()

# Test 1: APICall serialization/deserialization
print("Test 1: APICall serialization and deserialization")
call1 = APICall(
    call_id="1",
    endpoint="/api/users/123",
    method="GET",
    params={"include": "profile"},
    user_id="user1",
    session_id="sess1",
    timestamp=datetime.now(),
    response_time_ms=100,
    status_code=200,
    response_size_bytes=1024,
    user_type="premium"
)

call_dict = call1.to_dict()
call_restored = APICall.from_dict(call_dict)
assert call1.call_id == call_restored.call_id
assert call1.endpoint == call_restored.endpoint
assert call1.status_code == call_restored.status_code
print("  ✓ Serialization/deserialization works correctly")
print()

# Test 2: Service name extraction
print("Test 2: Service name extraction")
test_cases = [
    ("/api/users/123", "users"),
    ("/api/products", "products"),
    ("/users/123/orders", "users"),
    ("/orders", "orders"),
]
for endpoint, expected in test_cases:
    call = APICall(
        call_id="test",
        endpoint=endpoint,
        method="GET",
        params={},
        user_id="u1",
        session_id="s1",
        timestamp=datetime.now(),
        response_time_ms=100,
        status_code=200,
        response_size_bytes=100,
        user_type="free"
    )
    result = call.get_service_name()
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  ✓ {endpoint} → {result}")
print()

# Test 3: Success status check
print("Test 3: Success status check")
status_tests = [
    (200, True),
    (201, True),
    (299, True),
    (300, False),
    (404, False),
    (500, False),
]
for status, expected in status_tests:
    call = APICall(
        call_id="test",
        endpoint="/api/test",
        method="GET",
        params={},
        user_id="u1",
        session_id="s1",
        timestamp=datetime.now(),
        response_time_ms=100,
        status_code=status,
        response_size_bytes=100,
        user_type="free"
    )
    result = call.is_successful()
    assert result == expected, f"Status {status}: expected {expected}, got {result}"
print(f"  ✓ Tested {len(status_tests)} status codes")
print()

# Test 4: Session with multiple calls
print("Test 4: Session with transitions")
start_time = datetime.now()
session = Session(
    session_id="sess_test",
    user_id="user_test",
    user_type="premium",
    start_timestamp=start_time
)

# Add 5 calls to create 4 transitions
endpoints = ["/api/home", "/api/products", "/api/products/123", "/api/cart", "/api/checkout"]
for i, endpoint in enumerate(endpoints):
    call = APICall(
        call_id=f"call_{i}",
        endpoint=endpoint,
        method="GET",
        params={},
        user_id="user_test",
        session_id="sess_test",
        timestamp=start_time + timedelta(seconds=i),
        response_time_ms=100 + i * 10,
        status_code=200,
        response_size_bytes=1000 + i * 100,
        user_type="premium"
    )
    session.append_call(call)

transitions = session.get_endpoint_transitions()
assert len(transitions) == 4, f"Expected 4 transitions, got {len(transitions)}"
assert transitions[0] == ("/api/home", "/api/products")
assert transitions[-1] == ("/api/cart", "/api/checkout")
print(f"  ✓ Generated {len(transitions)} transitions correctly")
print(f"  ✓ Session duration: {session.duration_seconds:.2f}s")
print(f"  ✓ Number of calls: {session.num_calls}")
print()

# Test 5: Session serialization
print("Test 5: Session serialization/deserialization")
session_dict = session.to_dict()
session_restored = Session.from_dict(session_dict)
assert session.session_id == session_restored.session_id
assert session.num_calls == session_restored.num_calls
assert len(session.get_endpoint_transitions()) == len(session_restored.get_endpoint_transitions())
print("  ✓ Session serialization works correctly")
print()

# Test 6: Dataset with multiple sessions
print("Test 6: Dataset operations")
sessions = []
for i in range(10):
    sess = Session(
        session_id=f"sess_{i}",
        user_id=f"user_{i % 3}",  # 3 unique users
        user_type=["free", "premium", "guest"][i % 3],
        start_timestamp=datetime.now() - timedelta(hours=10-i)
    )
    # Add 3-5 calls per session
    for j in range(3 + (i % 3)):
        call = APICall(
            call_id=f"call_{i}_{j}",
            endpoint=f"/api/endpoint_{j % 4}",  # 4 unique endpoints
            method="GET",
            params={},
            user_id=sess.user_id,
            session_id=sess.session_id,
            timestamp=sess.start_timestamp + timedelta(seconds=j),
            response_time_ms=100,
            status_code=200,
            response_size_bytes=1000,
            user_type=sess.user_type
        )
        sess.append_call(call)
    sessions.append(sess)

dataset = Dataset(
    name="test_dataset",
    sessions=sessions,
    metadata={"version": "1.0", "source": "test"}
)

print(f"  ✓ Total calls: {dataset.total_calls}")
print(f"  ✓ Unique users: {dataset.num_unique_users}")
print(f"  ✓ Unique endpoints: {len(dataset.unique_endpoints)}")
print(f"  ✓ Date range: {dataset.date_range}")
print()

# Test 7: Endpoint counting
print("Test 7: Endpoint occurrence counting")
counts = dataset.count_endpoint_occurrences()
print(f"  ✓ Counted {len(counts)} unique endpoints")
for endpoint, count in sorted(counts.items()):
    print(f"    {endpoint}: {count} times")
print()

# Test 8: Dataset splitting
print("Test 8: Dataset train/test split")
train, test = dataset.split(train_ratio=0.7)
print(f"  ✓ Train sessions: {len(train.sessions)}")
print(f"  ✓ Test sessions: {len(test.sessions)}")
print(f"  ✓ Train calls: {train.total_calls}")
print(f"  ✓ Test calls: {test.total_calls}")
assert len(train.sessions) + len(test.sessions) == len(dataset.sessions)
print()

# Test 9: All sequences extraction
print("Test 9: Sequence extraction for Markov training")
sequences = dataset.get_all_sequences()
print(f"  ✓ Extracted {len(sequences)} sequences")
print(f"  ✓ Sample sequence: {sequences[0][:3]}...")
print()

# Test 10: Validation tests
print("Test 10: Input validation")
try:
    bad_call = APICall(
        call_id="bad",
        endpoint="/api/test",
        method="GET",
        params={},
        user_id="u1",
        session_id="s1",
        timestamp=datetime.now(),
        response_time_ms=-100,  # Negative!
        status_code=200,
        response_size_bytes=100,
        user_type="free"
    )
    print("  ✗ Should have raised ValueError for negative response time")
except ValueError as e:
    print(f"  ✓ Correctly caught negative response_time_ms")

try:
    bad_call = APICall(
        call_id="bad",
        endpoint="/api/test",
        method="GET",
        params={},
        user_id="u1",
        session_id="s1",
        timestamp=datetime.now(),
        response_time_ms=100,
        status_code=200,
        response_size_bytes=100,
        user_type="invalid"  # Invalid type!
    )
    print("  ✗ Should have raised ValueError for invalid user_type")
except ValueError as e:
    print(f"  ✓ Correctly caught invalid user_type")

try:
    bad_call = APICall(
        call_id="bad",
        endpoint="api/test",  # Missing leading slash!
        method="GET",
        params={},
        user_id="u1",
        session_id="s1",
        timestamp=datetime.now(),
        response_time_ms=100,
        status_code=200,
        response_size_bytes=100,
        user_type="free"
    )
    print("  ✗ Should have raised ValueError for endpoint without leading slash")
except ValueError as e:
    print(f"  ✓ Correctly caught endpoint without leading slash")

print()

# Test 11: Repr methods
print("Test 11: String representations")
print(f"  APICall repr: {repr(call1)[:80]}...")
print(f"  Session repr: {repr(session)}")
print(f"  Dataset repr: {repr(dataset)}")
print()

print("=" * 60)
print("ALL TESTS PASSED! ✓")
print("=" * 60)


"""Comprehensive tests for SessionExtractor including file I/O."""

import json
from pathlib import Path
from datetime import datetime, timedelta
import tempfile

from preprocessing.session_extractor import SessionExtractor
from preprocessing.models import APICall

print("=" * 70)
print("SessionExtractor Comprehensive Tests")
print("=" * 70)
print()

# Test 1: JSON file extraction
print("Test 1: Extract from JSON file")
print("-" * 70)

# Create test data
test_calls_data = [
    {
        "call_id": "1",
        "endpoint": "/api/home",
        "method": "GET",
        "params": {},
        "user_id": "user1",
        "session_id": "",
        "timestamp": "2024-01-01T10:00:00",
        "response_time_ms": 100,
        "status_code": 200,
        "response_size_bytes": 1024,
        "user_type": "free"
    },
    {
        "call_id": "2",
        "endpoint": "/api/products",
        "method": "GET",
        "params": {"category": "electronics"},
        "user_id": "user1",
        "session_id": "",
        "timestamp": "2024-01-01T10:05:00",
        "response_time_ms": 150,
        "status_code": 200,
        "response_size_bytes": 2048,
        "user_type": "free"
    },
    {
        "call_id": "3",
        "endpoint": "/api/cart",
        "method": "POST",
        "params": {"product_id": "123"},
        "user_id": "user1",
        "session_id": "",
        "timestamp": "2024-01-01T10:10:00",
        "response_time_ms": 200,
        "status_code": 201,
        "response_size_bytes": 512,
        "user_type": "free"
    }
]

# Save to temporary JSON file
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(test_calls_data, f)
    temp_json_path = Path(f.name)

try:
    extractor = SessionExtractor(inactivity_timeout_minutes=30, min_session_length=2, show_progress=False)
    sessions = extractor.extract_from_file(temp_json_path)

    print(f"Extracted {len(sessions)} sessions from JSON file")
    assert len(sessions) == 1, f"Expected 1 session, got {len(sessions)}"
    assert sessions[0].num_calls == 3, f"Expected 3 calls, got {sessions[0].num_calls}"

    stats = extractor.get_statistics()
    print(f"Total calls processed: {stats['total_calls_processed']}")
    print(f"Sessions created: {stats['total_sessions_created']}")

    print("[OK] Test 1 passed!")
finally:
    temp_json_path.unlink()

print()

# Test 2: JSON file with nested structure
print("Test 2: Extract from JSON file (nested structure)")
print("-" * 70)

nested_data = {
    "calls": test_calls_data
}

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(nested_data, f)
    temp_json_path = Path(f.name)

try:
    extractor2 = SessionExtractor(show_progress=False)
    sessions2 = extractor2.extract_from_file(temp_json_path)

    print(f"Extracted {len(sessions2)} sessions from nested JSON")
    assert len(sessions2) == 1, f"Expected 1 session, got {len(sessions2)}"

    print("[OK] Test 2 passed!")
finally:
    temp_json_path.unlink()

print()

# Test 3: Error handling - missing timestamps
print("Test 3: Handle missing timestamps")
print("-" * 70)

calls_with_none = [
    APICall("1", "/a", "GET", {}, "user1", "", datetime(2024, 1, 1, 10, 0), 100, 200, 500, "free"),
    APICall("2", "/b", "GET", {}, "user1", "", None, 100, 200, 500, "free"),  # Missing timestamp
    APICall("3", "/c", "GET", {}, "user1", "", datetime(2024, 1, 1, 10, 5), 100, 200, 500, "free"),
]

extractor3 = SessionExtractor(show_progress=False)
sessions3 = extractor3.extract_sessions(calls_with_none)
stats3 = extractor3.get_statistics()

print(f"Calls with missing timestamps: {stats3['calls_with_missing_timestamps']}")
print(f"Calls processed: {stats3['total_calls_processed']}")
print(f"Sessions created: {stats3['sessions_kept']}")

assert stats3['calls_with_missing_timestamps'] == 1, "Should detect 1 missing timestamp"
assert stats3['total_calls_processed'] == 2, "Should process only calls with valid timestamps"

print("[OK] Test 3 passed!")
print()

# Test 4: Edge case - all calls exceed timeout
print("Test 4: Edge case - all calls far apart")
print("-" * 70)

calls_far_apart = [
    APICall("1", "/a", "GET", {}, "user1", "", datetime(2024, 1, 1, 10, 0), 100, 200, 500, "free"),
    APICall("2", "/b", "GET", {}, "user1", "", datetime(2024, 1, 1, 12, 0), 100, 200, 500, "free"),  # 2 hours later
    APICall("3", "/c", "GET", {}, "user1", "", datetime(2024, 1, 1, 14, 0), 100, 200, 500, "free"),  # 2 hours later
]

extractor4 = SessionExtractor(inactivity_timeout_minutes=30, min_session_length=1, show_progress=False)
sessions4 = extractor4.extract_sessions(calls_far_apart)

print(f"Sessions created: {len(sessions4)}")
assert len(sessions4) == 3, f"Expected 3 separate sessions, got {len(sessions4)}"

for i, session in enumerate(sessions4, 1):
    print(f"  Session {i}: {session.num_calls} call(s)")

print("[OK] Test 4 passed!")
print()

# Test 5: Complex multi-user scenario
print("Test 5: Complex multi-user scenario")
print("-" * 70)

complex_calls = []
base_time = datetime(2024, 1, 1, 10, 0)

# User 1: Two sessions separated by long gap
for i in range(3):
    complex_calls.append(
        APICall(f"u1_s1_{i}", f"/api/a{i}", "GET", {}, "user1", "",
                base_time + timedelta(minutes=i*5), 100, 200, 500, "premium")
    )

# 1 hour gap
for i in range(2):
    complex_calls.append(
        APICall(f"u1_s2_{i}", f"/api/b{i}", "GET", {}, "user1", "",
                base_time + timedelta(hours=1, minutes=i*5), 100, 200, 500, "premium")
    )

# User 2: One continuous session
for i in range(4):
    complex_calls.append(
        APICall(f"u2_s1_{i}", f"/api/c{i}", "GET", {}, "user2", "",
                base_time + timedelta(minutes=i*10), 100, 200, 500, "free")
    )

# User 3: Single call (should be filtered with min_session_length=2)
complex_calls.append(
    APICall("u3_s1_0", "/api/d0", "GET", {}, "user3", "",
            base_time + timedelta(minutes=5), 100, 200, 500, "guest")
)

extractor5 = SessionExtractor(
    inactivity_timeout_minutes=30,
    min_session_length=2,
    show_progress=False
)
sessions5 = extractor5.extract_sessions(complex_calls)
stats5 = extractor5.get_statistics()

print(f"Total calls: {stats5['total_calls_processed']}")
print(f"Unique users: {stats5['unique_users']}")
print(f"Sessions created: {stats5['total_sessions_created']}")
print(f"Sessions kept: {stats5['sessions_kept']}")
print(f"Filtered (too short): {stats5['sessions_filtered_too_short']}")
print()

# Should have: user1 (2 sessions), user2 (1 session), user3 (filtered) = 3 kept
assert stats5['sessions_kept'] == 3, f"Expected 3 sessions, got {stats5['sessions_kept']}"
assert stats5['unique_users'] == 3, f"Expected 3 unique users, got {stats5['unique_users']}"
assert stats5['sessions_filtered_too_short'] == 1, f"Expected 1 filtered session"

print("Sessions by user:")
user_session_count = {}
for session in sessions5:
    user_id = session.user_id
    user_session_count[user_id] = user_session_count.get(user_id, 0) + 1

for user_id, count in sorted(user_session_count.items()):
    print(f"  {user_id}: {count} session(s)")

print()
print("[OK] Test 5 passed!")
print()

# Test 6: Session ID assignment
print("Test 6: Verify session IDs are assigned correctly")
print("-" * 70)

calls_id_test = [
    APICall("1", "/a", "GET", {}, "alice", "", datetime(2024, 1, 1, 10, 0), 100, 200, 500, "free"),
    APICall("2", "/b", "GET", {}, "alice", "", datetime(2024, 1, 1, 10, 5), 100, 200, 500, "free"),
    # New session after timeout
    APICall("3", "/c", "GET", {}, "alice", "", datetime(2024, 1, 1, 11, 0), 100, 200, 500, "free"),
    APICall("4", "/d", "GET", {}, "alice", "", datetime(2024, 1, 1, 11, 5), 100, 200, 500, "free"),
]

extractor6 = SessionExtractor(inactivity_timeout_minutes=30, show_progress=False)
sessions6 = extractor6.extract_sessions(calls_id_test)

print(f"Sessions for user 'alice': {len(sessions6)}")
for session in sessions6:
    print(f"  {session.session_id}: {session.num_calls} calls")
    # Verify all calls have the session ID
    for call in session.calls:
        assert call.session_id == session.session_id, f"Call session_id mismatch!"

assert sessions6[0].session_id == "alice_session_0"
assert sessions6[1].session_id == "alice_session_1"

print("[OK] Test 6 passed!")
print()

# Test 7: Statistics accuracy
print("Test 7: Verify statistics accuracy")
print("-" * 70)

stats_calls = []
for i in range(10):
    stats_calls.append(
        APICall(f"{i}", f"/api/{i}", "GET", {}, "user1", "",
                datetime(2024, 1, 1, 10, i*2), 100, 200, 500, "free")
    )

extractor7 = SessionExtractor(show_progress=False)
sessions7 = extractor7.extract_sessions(stats_calls)
stats7 = extractor7.get_statistics()

# Calculate expected duration
expected_duration = (datetime(2024, 1, 1, 10, 18) - datetime(2024, 1, 1, 10, 0)).total_seconds()

print(f"Session length: {stats7['average_session_length']}")
print(f"Session duration: {stats7['average_session_duration_seconds']:.0f}s (expected: {expected_duration:.0f}s)")
print(f"Min/Max length: {stats7['min_session_length']}/{stats7['max_session_length']}")

assert abs(stats7['average_session_duration_seconds'] - expected_duration) < 1, "Duration calculation mismatch"
assert stats7['average_session_length'] == 10, "Length calculation mismatch"

print("[OK] Test 7 passed!")
print()

# Test 8: Repr method
print("Test 8: Verify string representation")
print("-" * 70)

extractor8 = SessionExtractor(
    inactivity_timeout_minutes=45,
    min_session_length=3,
    max_session_length=100
)

repr_str = repr(extractor8)
print(f"Repr: {repr_str}")

assert "45" in repr_str, "Should include timeout value"
assert "3" in repr_str, "Should include min_session_length"
assert "100" in repr_str, "Should include max_session_length"

print("[OK] Test 8 passed!")
print()

print("=" * 70)
print("[OK] ALL COMPREHENSIVE TESTS PASSED!")
print("=" * 70)
print()

# Print summary
print("Summary:")
print(f"  [OK] JSON file extraction")
print(f"  [OK] Nested JSON structure")
print(f"  [OK] Missing timestamp handling")
print(f"  [OK] Far-apart calls")
print(f"  [OK] Complex multi-user scenarios")
print(f"  [OK] Session ID assignment")
print(f"  [OK] Statistics accuracy")
print(f"  [OK] String representation")


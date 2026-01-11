"""Validation test for SessionExtractor module."""

from preprocessing.session_extractor import SessionExtractor
from preprocessing.models import APICall
from datetime import datetime, timedelta

print("=" * 70)
print("SessionExtractor Validation Test")
print("=" * 70)
print()

# Create some test calls with the exact structure from requirements
calls = [
    APICall("1", "/login", "POST", {}, "user1", "", datetime(2024, 1, 1, 10, 0), 100, 200, 500, "free"),
    APICall("2", "/profile", "GET", {}, "user1", "", datetime(2024, 1, 1, 10, 5), 50, 200, 200, "free"),
    APICall("3", "/browse", "GET", {}, "user1", "", datetime(2024, 1, 1, 10, 10), 80, 200, 1000, "free"),
    # Gap of 1 hour - should be new session
    APICall("4", "/login", "POST", {}, "user1", "", datetime(2024, 1, 1, 11, 30), 100, 200, 500, "free"),
]

print("Test 1: Basic session extraction")
print("-" * 70)
extractor = SessionExtractor(inactivity_timeout_minutes=30, min_session_length=2)
sessions = extractor.extract_sessions(calls)

print(f"Found {len(sessions)} sessions")
print()

# Expected: 2 sessions (first 3 calls, then call 4 filtered out for being too short)
assert len(sessions) == 1, f"Expected 1 session (4th call filtered), got {len(sessions)}"

print("Statistics:")
stats = extractor.get_statistics()
for key, value in stats.items():
    print(f"  {key}: {value}")
print()

# Verify session details
print("Session details:")
for i, session in enumerate(sessions, 1):
    print(f"  Session {i}: {session.session_id}")
    print(f"    User: {session.user_id}")
    print(f"    Calls: {session.num_calls}")
    print(f"    Duration: {session.duration_seconds:.1f} seconds")
    print(f"    Endpoints: {session.endpoint_sequence}")
    print()

print("✓ Test 1 passed!")
print()

# Test 2: Multiple users
print("Test 2: Multiple users interleaved")
print("-" * 70)

calls_multi = [
    # User 1
    APICall("1", "/home", "GET", {}, "user1", "", datetime(2024, 1, 1, 10, 0), 100, 200, 500, "free"),
    # User 2
    APICall("2", "/home", "GET", {}, "user2", "", datetime(2024, 1, 1, 10, 1), 100, 200, 500, "premium"),
    # User 1 continues
    APICall("3", "/products", "GET", {}, "user1", "", datetime(2024, 1, 1, 10, 2), 100, 200, 500, "free"),
    # User 2 continues
    APICall("4", "/products", "GET", {}, "user2", "", datetime(2024, 1, 1, 10, 3), 100, 200, 500, "premium"),
    # User 1
    APICall("5", "/cart", "POST", {}, "user1", "", datetime(2024, 1, 1, 10, 4), 100, 200, 500, "free"),
    # User 2
    APICall("6", "/cart", "POST", {}, "user2", "", datetime(2024, 1, 1, 10, 5), 100, 200, 500, "premium"),
]

extractor2 = SessionExtractor(inactivity_timeout_minutes=30, min_session_length=2)
sessions2 = extractor2.extract_sessions(calls_multi)

print(f"Found {len(sessions2)} sessions")
stats2 = extractor2.get_statistics()
print(f"Unique users: {stats2['unique_users']}")
print()

assert len(sessions2) == 2, f"Expected 2 sessions (one per user), got {len(sessions2)}"
assert stats2['unique_users'] == 2, f"Expected 2 unique users, got {stats2['unique_users']}"

for session in sessions2:
    print(f"  {session.session_id}: {session.num_calls} calls")

print()
print("✓ Test 2 passed!")
print()

# Test 3: Empty input
print("Test 3: Empty input handling")
print("-" * 70)

extractor3 = SessionExtractor()
sessions3 = extractor3.extract_sessions([])

print(f"Found {len(sessions3)} sessions")
assert len(sessions3) == 0, f"Expected 0 sessions for empty input, got {len(sessions3)}"

print("✓ Test 3 passed!")
print()

# Test 4: Session length filtering
print("Test 4: Session length filtering")
print("-" * 70)

calls_filter = [
    # User 1: Single call (should be filtered if min_length=2)
    APICall("1", "/home", "GET", {}, "user1", "", datetime(2024, 1, 1, 10, 0), 100, 200, 500, "free"),

    # User 2: Two calls (should be kept)
    APICall("2", "/home", "GET", {}, "user2", "", datetime(2024, 1, 1, 10, 0), 100, 200, 500, "premium"),
    APICall("3", "/products", "GET", {}, "user2", "", datetime(2024, 1, 1, 10, 5), 100, 200, 500, "premium"),

    # User 3: Five calls (should be filtered if max_length=4)
    APICall("4", "/a", "GET", {}, "user3", "", datetime(2024, 1, 1, 10, 0), 100, 200, 500, "guest"),
    APICall("5", "/b", "GET", {}, "user3", "", datetime(2024, 1, 1, 10, 1), 100, 200, 500, "guest"),
    APICall("6", "/c", "GET", {}, "user3", "", datetime(2024, 1, 1, 10, 2), 100, 200, 500, "guest"),
    APICall("7", "/d", "GET", {}, "user3", "", datetime(2024, 1, 1, 10, 3), 100, 200, 500, "guest"),
    APICall("8", "/e", "GET", {}, "user3", "", datetime(2024, 1, 1, 10, 4), 100, 200, 500, "guest"),
]

extractor4 = SessionExtractor(min_session_length=2, max_session_length=4, show_progress=False)
sessions4 = extractor4.extract_sessions(calls_filter)

stats4 = extractor4.get_statistics()
print(f"Total sessions created: {stats4['total_sessions_created']}")
print(f"Sessions kept: {stats4['sessions_kept']}")
print(f"Filtered (too short): {stats4['sessions_filtered_too_short']}")
print(f"Filtered (too long): {stats4['sessions_filtered_too_long']}")
print()

assert sessions4[0].num_calls == 2, "Should keep user2 with 2 calls"
assert len(sessions4) == 1, f"Expected 1 session after filtering, got {len(sessions4)}"
assert stats4['sessions_filtered_too_short'] == 1, "Should filter user1 (too short)"
assert stats4['sessions_filtered_too_long'] == 1, "Should filter user3 (too long)"

print("✓ Test 4 passed!")
print()

# Test 5: Inactivity timeout creates multiple sessions
print("Test 5: Inactivity timeout creating multiple sessions")
print("-" * 70)

calls_timeout = [
    APICall("1", "/a", "GET", {}, "user1", "", datetime(2024, 1, 1, 10, 0), 100, 200, 500, "free"),
    APICall("2", "/b", "GET", {}, "user1", "", datetime(2024, 1, 1, 10, 5), 100, 200, 500, "free"),
    # 1 hour gap (exceeds 30 min timeout)
    APICall("3", "/c", "GET", {}, "user1", "", datetime(2024, 1, 1, 11, 10), 100, 200, 500, "free"),
    APICall("4", "/d", "GET", {}, "user1", "", datetime(2024, 1, 1, 11, 15), 100, 200, 500, "free"),
]

extractor5 = SessionExtractor(inactivity_timeout_minutes=30, min_session_length=2, show_progress=False)
sessions5 = extractor5.extract_sessions(calls_timeout)

print(f"Found {len(sessions5)} sessions for single user")
assert len(sessions5) == 2, f"Expected 2 sessions (split by timeout), got {len(sessions5)}"

print(f"  Session 1: {sessions5[0].num_calls} calls, duration: {sessions5[0].duration_seconds:.0f}s")
print(f"  Session 2: {sessions5[1].num_calls} calls, duration: {sessions5[1].duration_seconds:.0f}s")

print()
print("✓ Test 5 passed!")
print()

# Test 6: Statistics calculation
print("Test 6: Statistics calculation")
print("-" * 70)

calls_stats = []
base_time = datetime(2024, 1, 1, 10, 0)

# Create 3 users with varying session patterns
for user_num in range(3):
    user_id = f"user{user_num}"
    for call_num in range(5):
        call = APICall(
            call_id=f"{user_num}_{call_num}",
            endpoint=f"/endpoint_{call_num}",
            method="GET",
            params={},
            user_id=user_id,
            session_id="",
            timestamp=base_time + timedelta(minutes=call_num * 5),
            response_time_ms=100,
            status_code=200,
            response_size_bytes=1000,
            user_type="free"
        )
        calls_stats.append(call)

extractor6 = SessionExtractor(inactivity_timeout_minutes=30, min_session_length=1, show_progress=False)
sessions6 = extractor6.extract_sessions(calls_stats)
stats6 = extractor6.get_statistics()

print(f"Total calls processed: {stats6['total_calls_processed']}")
print(f"Total sessions created: {stats6['total_sessions_created']}")
print(f"Unique users: {stats6['unique_users']}")
print(f"Average session length: {stats6['average_session_length']:.2f} calls")
print(f"Average session duration: {stats6['average_session_duration_seconds']:.2f} seconds")
print(f"Min/Max session length: {stats6['min_session_length']}/{stats6['max_session_length']}")

assert stats6['total_calls_processed'] == 15, "Should process all 15 calls"
assert stats6['unique_users'] == 3, "Should have 3 unique users"
assert stats6['total_sessions_created'] == 3, "Should create 3 sessions (one per user)"

print()
print("✓ Test 6 passed!")
print()

print("=" * 70)
print("✓ ALL VALIDATION TESTS PASSED!")
print("=" * 70)


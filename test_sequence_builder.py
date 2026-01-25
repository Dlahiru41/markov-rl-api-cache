"""Test script to validate the SequenceBuilder functionality."""

from preprocessing.sequence_builder import SequenceBuilder
from preprocessing.models import APICall, Session
from datetime import datetime, timedelta

# Create test data
def create_test_sessions():
    """Create sample sessions for testing."""
    sessions = []

    # Session 1
    base_time = datetime.now()
    calls1 = [
        APICall(
            call_id="1",
            endpoint="/api/login",
            method="POST",
            params={},
            user_id="user1",
            session_id="sess1",
            timestamp=base_time,
            response_time_ms=100,
            status_code=200,
            response_size_bytes=1024,
            user_type="free"
        ),
        APICall(
            call_id="2",
            endpoint="/API/Users/123/Profile/",
            method="GET",
            params={},
            user_id="user1",
            session_id="sess1",
            timestamp=base_time + timedelta(seconds=5),
            response_time_ms=150,
            status_code=200,
            response_size_bytes=2048,
            user_type="free"
        ),
        APICall(
            call_id="3",
            endpoint="/api/products?category=electronics",
            method="GET",
            params={"category": "electronics"},
            user_id="user1",
            session_id="sess1",
            timestamp=base_time + timedelta(seconds=10),
            response_time_ms=200,
            status_code=200,
            response_size_bytes=4096,
            user_type="free"
        ),
        APICall(
            call_id="4",
            endpoint="/api/products/456/details",
            method="GET",
            params={},
            user_id="user1",
            session_id="sess1",
            timestamp=base_time + timedelta(seconds=15),
            response_time_ms=120,
            status_code=200,
            response_size_bytes=3072,
            user_type="free"
        ),
    ]

    session1 = Session(
        session_id="sess1",
        user_id="user1",
        user_type="free",
        start_timestamp=base_time,
        end_timestamp=base_time + timedelta(seconds=15),
        calls=calls1
    )
    sessions.append(session1)

    # Session 2
    base_time2 = datetime.now() + timedelta(hours=1)
    calls2 = [
        APICall(
            call_id="5",
            endpoint="/api/login",
            method="POST",
            params={},
            user_id="user2",
            session_id="sess2",
            timestamp=base_time2,
            response_time_ms=90,
            status_code=200,
            response_size_bytes=1024,
            user_type="premium"
        ),
        APICall(
            call_id="6",
            endpoint="/api/users/789/profile",
            method="GET",
            params={},
            user_id="user2",
            session_id="sess2",
            timestamp=base_time2 + timedelta(seconds=3),
            response_time_ms=100,
            status_code=200,
            response_size_bytes=2048,
            user_type="premium"
        ),
        APICall(
            call_id="7",
            endpoint="/api/orders/999/status",
            method="GET",
            params={},
            user_id="user2",
            session_id="sess2",
            timestamp=base_time2 + timedelta(seconds=8),
            response_time_ms=180,
            status_code=200,
            response_size_bytes=1536,
            user_type="premium"
        ),
    ]

    session2 = Session(
        session_id="sess2",
        user_id="user2",
        user_type="premium",
        start_timestamp=base_time2,
        end_timestamp=base_time2 + timedelta(seconds=8),
        calls=calls2
    )
    sessions.append(session2)

    return sessions


def test_normalization():
    """Test endpoint normalization."""
    print("=" * 80)
    print("TEST 1: Endpoint Normalization")
    print("=" * 80)

    builder = SequenceBuilder(normalize_endpoints=True, min_sequence_length=2)

    test_cases = [
        "/API/Users/123/Profile/",
        "/api/users/456/orders",
        "/search?q=shoes&category=fashion",
        "/api/products/789",
        "/users/abc123def456/settings",
        "/api/items/550e8400-e29b-41d4-a716-446655440000/details"
    ]

    print("\nNormalization results:")
    for endpoint in test_cases:
        normalized = builder.normalize_endpoint(endpoint)
        print(f"  {endpoint}")
        print(f"    -> {normalized}")
    print()


def test_sequence_building():
    """Test basic sequence extraction."""
    print("=" * 80)
    print("TEST 2: Basic Sequence Building")
    print("=" * 80)

    sessions = create_test_sessions()
    builder = SequenceBuilder(normalize_endpoints=True, min_sequence_length=2)

    sequences = builder.build_sequences(sessions)

    print(f"\nBuilt {len(sequences)} sequences")
    for i, seq in enumerate(sequences, 1):
        print(f"\nSequence {i}:")
        for endpoint in seq:
            print(f"  - {endpoint}")
    print()


def test_labeled_sequences():
    """Test labeled sequence generation."""
    print("=" * 80)
    print("TEST 3: Labeled Sequences (History, Next)")
    print("=" * 80)

    sessions = create_test_sessions()
    builder = SequenceBuilder(normalize_endpoints=True, min_sequence_length=2)

    labeled = builder.build_labeled_sequences(sessions)

    print(f"\nGenerated {len(labeled)} labeled pairs:\n")
    for history, next_endpoint in labeled[:10]:  # Show first 10
        print(f"  History: {history}")
        print(f"  Next: {next_endpoint}")
        print()


def test_ngrams():
    """Test n-gram extraction."""
    print("=" * 80)
    print("TEST 4: N-gram Extraction")
    print("=" * 80)

    sessions = create_test_sessions()
    builder = SequenceBuilder(normalize_endpoints=True, min_sequence_length=2)

    # Test bigrams
    bigrams = builder.build_ngrams(sessions, n=2)
    print(f"\nExtracted {len(bigrams)} bigrams:")
    print(f"First few bigrams: {bigrams[:5]}")

    # Test trigrams
    trigrams = builder.build_ngrams(sessions, n=3)
    print(f"\nExtracted {len(trigrams)} trigrams:")
    print(f"First few trigrams: {trigrams[:5]}")
    print()


def test_contextual_sequences():
    """Test contextual sequence extraction."""
    print("=" * 80)
    print("TEST 5: Contextual Sequences")
    print("=" * 80)

    sessions = create_test_sessions()
    builder = SequenceBuilder(normalize_endpoints=True, min_sequence_length=2)

    contextual = builder.build_contextual_sequences(sessions)

    print(f"\nGenerated {len(contextual)} contextual sequences:\n")
    for i, ctx_seq in enumerate(contextual, 1):
        print(f"Sequence {i}:")
        print(f"  User Type: {ctx_seq.user_type}")
        print(f"  Time of Day: {ctx_seq.time_of_day}")
        print(f"  Day Type: {ctx_seq.day_type}")
        print(f"  Session Length: {ctx_seq.session_length_category}")
        print(f"  Endpoints: {ctx_seq.sequence}")
        print()


def test_transition_probabilities():
    """Test transition probability calculation."""
    print("=" * 80)
    print("TEST 6: Transition Probabilities")
    print("=" * 80)

    sessions = create_test_sessions()
    builder = SequenceBuilder(normalize_endpoints=True, min_sequence_length=2)

    # Get transition counts
    transition_counts = builder.get_transition_counts(sessions)
    print("\nTransition counts:")
    for (from_ep, to_ep), count in transition_counts.items():
        print(f"  {from_ep} -> {to_ep}: {count}")

    # Get transition probabilities
    probabilities = builder.get_transition_probabilities(sessions)
    print("\nTransition probabilities:")
    for from_ep, transitions in probabilities.items():
        print(f"\n  From {from_ep}:")
        for to_ep, prob in transitions.items():
            print(f"    -> {to_ep}: {prob:.2%}")
    print()


def test_statistics():
    """Test sequence statistics."""
    print("=" * 80)
    print("TEST 7: Sequence Statistics")
    print("=" * 80)

    sessions = create_test_sessions()
    builder = SequenceBuilder(normalize_endpoints=True, min_sequence_length=2)

    stats = builder.get_sequence_statistics(sessions)

    print("\nSequence Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    print()


def test_unique_endpoints():
    """Test unique endpoint extraction."""
    print("=" * 80)
    print("TEST 8: Unique Endpoints")
    print("=" * 80)

    sessions = create_test_sessions()
    builder = SequenceBuilder(normalize_endpoints=True, min_sequence_length=2)

    unique = builder.get_unique_endpoints(sessions)

    print(f"\nFound {len(unique)} unique endpoints:")
    for endpoint in unique:
        print(f"  - {endpoint}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SEQUENCE BUILDER VALIDATION TEST SUITE")
    print("=" * 80 + "\n")

    try:
        test_normalization()
        test_sequence_building()
        test_labeled_sequences()
        test_ngrams()
        test_contextual_sequences()
        test_transition_probabilities()
        test_statistics()
        test_unique_endpoints()

        print("=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)

    except Exception as e:
        print(f"\n[ERROR] ERROR: {e}")
        import traceback
        traceback.print_exc()


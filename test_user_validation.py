"""User's validation code for the SequenceBuilder module."""

from preprocessing.sequence_builder import SequenceBuilder
from preprocessing.models import APICall, Session
from datetime import datetime

# Create some test sessions
def create_sample_sessions():
    """Create sample sessions for testing."""
    base_time = datetime.now()

    # Create calls for session 1
    call1 = APICall(
        call_id="1",
        endpoint="/api/users/123",
        method="GET",
        params={},
        user_id="user1",
        session_id="sess1",
        timestamp=base_time,
        response_time_ms=100,
        status_code=200,
        response_size_bytes=1024,
        user_type="free"
    )

    call2 = APICall(
        call_id="2",
        endpoint="/api/products/456",
        method="GET",
        params={},
        user_id="user1",
        session_id="sess1",
        timestamp=base_time,
        response_time_ms=150,
        status_code=200,
        response_size_bytes=2048,
        user_type="free"
    )

    session1 = Session(
        session_id="sess1",
        user_id="user1",
        user_type="free",
        start_timestamp=base_time,
        calls=[call1, call2]
    )

    return [session1]


print("\n" + "="*60)
print("USER VALIDATION CODE")
print("="*60 + "\n")

# Test basic functionality
builder = SequenceBuilder(normalize_endpoints=True, min_sequence_length=2)

# Test normalization
test_endpoint = "/API/Users/123/Profile/"
normalized = builder.normalize_endpoint(test_endpoint)
print(f"Normalization test:")
print(f"  Input: {test_endpoint}")
print(f"  Output: {normalized}")
print(f"  Expected: /api/users/{{id}}/profile")
print(f"  ✓ PASS" if normalized == "/api/users/{id}/profile" else "  ✗ FAIL")

# Create test sessions
sessions = create_sample_sessions()

# Test sequence building
sequences = builder.build_sequences(sessions)
print(f"\nSequence building test:")
print(f"  Built {len(sequences)} sequences")
if sequences:
    print(f"  Sample: {sequences[0]}")

# Test n-grams
bigrams = builder.build_ngrams(sessions, n=2)
print(f"\nN-gram test:")
print(f"  Extracted {len(bigrams)} bigrams")
print(f"  First few bigrams: {bigrams[:5]}")

print("\n" + "="*60)
print("VALIDATION COMPLETE!")
print("="*60 + "\n")


"""
Integration Example: Using SequenceBuilder with Existing Models

This example demonstrates how to integrate the SequenceBuilder with
the existing preprocessing.models (APICall, Session, Dataset) to
prepare data for Markov chain training.
"""

from preprocessing.models import APICall, Session, Dataset
from preprocessing.sequence_builder import SequenceBuilder
from datetime import datetime, timedelta


def example_1_basic_integration():
    """Example 1: Basic integration with existing models."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Integration")
    print("="*70)

    # Create some API calls using the APICall model
    base_time = datetime.now()

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
        timestamp=base_time + timedelta(seconds=5),
        response_time_ms=150,
        status_code=200,
        response_size_bytes=2048,
        user_type="free"
    )

    # Create a session
    session = Session(
        session_id="sess1",
        user_id="user1",
        user_type="free",
        start_timestamp=base_time,
        calls=[call1, call2]
    )

    # Use SequenceBuilder to process the session
    builder = SequenceBuilder(normalize_endpoints=True)

    sequences = builder.build_sequences([session])
    print(f"\n[OK] Extracted {len(sequences)} sequence(s)")
    print(f"  Sequence: {sequences[0]}")

    # Show normalization effect
    print(f"\n[OK] Normalization:")
    print(f"  Original: {call1.endpoint} → {builder.normalize_endpoint(call1.endpoint)}")
    print(f"  Original: {call2.endpoint} → {builder.normalize_endpoint(call2.endpoint)}")


def example_2_dataset_workflow():
    """Example 2: Complete workflow with Dataset."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Complete Dataset Workflow")
    print("="*70)

    # Create multiple sessions
    sessions = []
    base_time = datetime.now()

    for i in range(3):
        calls = []
        session_start = base_time + timedelta(hours=i)

        for j in range(4):
            call = APICall(
                call_id=f"c{i}_{j}",
                endpoint=f"/api/endpoint{j % 2}/item{i*10 + j}",
                method="GET",
                params={},
                user_id=f"user{i}",
                session_id=f"sess{i}",
                timestamp=session_start + timedelta(seconds=j*5),
                response_time_ms=100 + j*10,
                status_code=200,
                response_size_bytes=1024,
                user_type="premium" if i % 2 == 0 else "free"
            )
            calls.append(call)

        session = Session(
            session_id=f"sess{i}",
            user_id=f"user{i}",
            user_type="premium" if i % 2 == 0 else "free",
            start_timestamp=session_start,
            calls=calls
        )
        sessions.append(session)

    # Create dataset
    dataset = Dataset(name="test_dataset", sessions=sessions)

    print(f"\n[OK] Created dataset:")
    print(f"  Name: {dataset.name}")
    print(f"  Sessions: {len(dataset.sessions)}")
    print(f"  Total calls: {dataset.total_calls}")
    print(f"  Unique users: {dataset.num_unique_users}")
    print(f"  Unique endpoints: {len(dataset.unique_endpoints)}")

    # Process with SequenceBuilder
    builder = SequenceBuilder(normalize_endpoints=True, min_sequence_length=2)

    # Get sequences
    sequences = builder.build_sequences(dataset.sessions)
    print(f"\n[OK] Extracted {len(sequences)} sequences")

    # Get statistics
    stats = builder.get_sequence_statistics(dataset.sessions)
    print(f"\n[OK] Statistics:")
    print(f"  Avg sequence length: {stats['avg_sequence_length']:.2f}")
    print(f"  Unique endpoints: {stats['unique_endpoints']}")
    print(f"  Total transitions: {stats['total_transitions']}")


def example_3_markov_training():
    """Example 3: Prepare data for Markov chain training."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Markov Chain Training Preparation")
    print("="*70)

    # Create realistic e-commerce sessions
    sessions = []
    base_time = datetime(2026, 1, 11, 10, 0, 0)

    # Common user journey patterns
    patterns = [
        ["/login", "/profile", "/products", "/product/123", "/cart", "/checkout"],
        ["/login", "/profile", "/orders", "/order/456"],
        ["/products", "/product/789", "/cart"],
        ["/login", "/profile", "/products", "/product/123"],
    ]

    for i, pattern in enumerate(patterns):
        calls = []
        session_start = base_time + timedelta(hours=i)

        for j, endpoint in enumerate(pattern):
            call = APICall(
                call_id=f"c{i}_{j}",
                endpoint=endpoint,
                method="GET" if endpoint != "/login" else "POST",
                params={},
                user_id=f"user{i}",
                session_id=f"sess{i}",
                timestamp=session_start + timedelta(seconds=j*10),
                response_time_ms=100,
                status_code=200,
                response_size_bytes=1024,
                user_type="premium" if i % 2 == 0 else "free"
            )
            calls.append(call)

        session = Session(
            session_id=f"sess{i}",
            user_id=f"user{i}",
            user_type="premium" if i % 2 == 0 else "free",
            start_timestamp=session_start,
            calls=calls
        )
        sessions.append(session)

    # Initialize builder
    builder = SequenceBuilder(normalize_endpoints=True, min_sequence_length=2)

    # Split train/test
    train_sessions, test_sessions = builder.split_sequences(sessions, train_ratio=0.75)
    print(f"\n[OK] Split data:")
    print(f"  Training: {len(train_sessions)} sessions")
    print(f"  Testing: {len(test_sessions)} sessions")

    # Calculate transition probabilities for Markov chain
    train_probs = builder.get_transition_probabilities(train_sessions)
    print(f"\n[OK] Trained Markov model:")
    print(f"  States (endpoints): {len(train_probs)}")

    # Show some probabilities
    print(f"\n[OK] Sample transition probabilities:")
    for state in list(train_probs.keys())[:3]:
        print(f"\n  From '{state}':")
        for next_state, prob in train_probs[state].items():
            print(f"    → {next_state}: {prob:.1%}")


def example_4_context_aware():
    """Example 4: Context-aware sequence analysis."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Context-Aware Analysis")
    print("="*70)

    # Create sessions with different user types and times
    sessions = []

    # Premium user morning session
    morning_time = datetime(2026, 1, 11, 9, 0, 0)  # Weekend morning
    calls1 = [
        APICall("c1", "/login", "POST", {}, "premium_user", "s1",
                morning_time, 80, 200, 512, "premium"),
        APICall("c2", "/dashboard", "GET", {}, "premium_user", "s1",
                morning_time + timedelta(seconds=5), 100, 200, 2048, "premium"),
        APICall("c3", "/analytics", "GET", {}, "premium_user", "s1",
                morning_time + timedelta(seconds=10), 150, 200, 4096, "premium"),
    ]
    session1 = Session("s1", "premium_user", "premium", morning_time, calls=calls1)
    sessions.append(session1)

    # Free user afternoon session
    afternoon_time = datetime(2026, 1, 11, 14, 0, 0)  # Weekend afternoon
    calls2 = [
        APICall("c4", "/login", "POST", {}, "free_user", "s2",
                afternoon_time, 90, 200, 512, "free"),
        APICall("c5", "/browse", "GET", {}, "free_user", "s2",
                afternoon_time + timedelta(seconds=5), 110, 200, 2048, "free"),
    ]
    session2 = Session("s2", "free_user", "free", afternoon_time, calls=calls2)
    sessions.append(session2)

    # Extract contextual sequences
    builder = SequenceBuilder(normalize_endpoints=True)
    contextual = builder.build_contextual_sequences(sessions)

    print(f"\n[OK] Extracted {len(contextual)} contextual sequences:\n")

    for i, ctx in enumerate(contextual, 1):
        print(f"  Session {i}:")
        print(f"    User Type: {ctx.user_type}")
        print(f"    Time: {ctx.time_of_day} ({ctx.day_type})")
        print(f"    Duration: {ctx.session_length_category}")
        print(f"    Sequence: {ctx.sequence}")
        print()

    # Analyze by user type
    premium_seqs = [c.sequence for c in contextual if c.user_type == 'premium']
    free_seqs = [c.sequence for c in contextual if c.user_type == 'free']

    print(f"[OK] User type analysis:")
    print(f"  Premium users: {len(premium_seqs)} sessions")
    print(f"  Free users: {len(free_seqs)} sessions")


def main():
    """Run all integration examples."""
    print("\n" + "="*70)
    print("SEQUENCEBUILDER + MODELS INTEGRATION EXAMPLES")
    print("="*70)
    print("\nDemonstrating how to use SequenceBuilder with existing models")
    print("(APICall, Session, Dataset) for Markov chain training.\n")

    example_1_basic_integration()
    example_2_dataset_workflow()
    example_3_markov_training()
    example_4_context_aware()

    print("\n" + "="*70)
    print("[SUCCESS] ALL INTEGRATION EXAMPLES COMPLETE")
    print("="*70)
    print("\nThe SequenceBuilder seamlessly integrates with existing models")
    print("to provide comprehensive data preparation for Markov chain training.\n")


if __name__ == "__main__":
    main()


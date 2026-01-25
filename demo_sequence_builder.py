"""Comprehensive example demonstrating all SequenceBuilder features.

This script showcases how to use the SequenceBuilder module for Markov chain training.
"""

from preprocessing.sequence_builder import SequenceBuilder
from preprocessing.models import APICall, Session, Dataset
from datetime import datetime, timedelta


def create_realistic_dataset():
    """Create a realistic dataset with multiple sessions."""
    sessions = []
    base_time = datetime(2026, 1, 11, 9, 0, 0)  # Saturday morning

    # Session 1: Premium user shopping journey
    user1_calls = [
        APICall("c1", "/api/login", "POST", {}, "u1", "s1", base_time, 80, 200, 512, "premium"),
        APICall("c2", "/api/users/100/profile", "GET", {}, "u1", "s1", base_time + timedelta(seconds=2), 100, 200, 2048, "premium"),
        APICall("c3", "/api/products/search?q=laptop", "GET", {"q": "laptop"}, "u1", "s1", base_time + timedelta(seconds=10), 250, 200, 8192, "premium"),
        APICall("c4", "/api/products/501/details", "GET", {}, "u1", "s1", base_time + timedelta(seconds=20), 150, 200, 4096, "premium"),
        APICall("c5", "/api/cart/add", "POST", {"product_id": 501}, "u1", "s1", base_time + timedelta(seconds=30), 120, 200, 256, "premium"),
        APICall("c6", "/api/cart", "GET", {}, "u1", "s1", base_time + timedelta(seconds=35), 100, 200, 1024, "premium"),
        APICall("c7", "/api/checkout", "POST", {}, "u1", "s1", base_time + timedelta(seconds=45), 300, 200, 512, "premium"),
    ]
    session1 = Session("s1", "u1", "premium", base_time, base_time + timedelta(seconds=45), user1_calls)
    sessions.append(session1)

    # Session 2: Free user browsing
    base_time2 = datetime(2026, 1, 11, 14, 30, 0)  # Saturday afternoon
    user2_calls = [
        APICall("c8", "/api/login", "POST", {}, "u2", "s2", base_time2, 90, 200, 512, "free"),
        APICall("c9", "/api/users/200/profile", "GET", {}, "u2", "s2", base_time2 + timedelta(seconds=3), 110, 200, 2048, "free"),
        APICall("c10", "/api/products/browse", "GET", {}, "u2", "s2", base_time2 + timedelta(seconds=8), 200, 200, 4096, "free"),
        APICall("c11", "/api/products/301/details", "GET", {}, "u2", "s2", base_time2 + timedelta(seconds=15), 180, 200, 4096, "free"),
        APICall("c12", "/api/products/browse", "GET", {}, "u2", "s2", base_time2 + timedelta(seconds=25), 190, 200, 4096, "free"),
    ]
    session2 = Session("s2", "u2", "free", base_time2, base_time2 + timedelta(seconds=25), user2_calls)
    sessions.append(session2)

    # Session 3: Guest user quick look
    base_time3 = datetime(2026, 1, 13, 20, 15, 0)  # Monday evening (weekday)
    user3_calls = [
        APICall("c13", "/api/products/browse", "GET", {}, "guest1", "s3", base_time3, 150, 200, 4096, "guest"),
        APICall("c14", "/api/products/401/details", "GET", {}, "guest1", "s3", base_time3 + timedelta(seconds=5), 160, 200, 4096, "guest"),
    ]
    session3 = Session("s3", "guest1", "guest", base_time3, base_time3 + timedelta(seconds=5), user3_calls)
    sessions.append(session3)

    # Session 4: Another premium user
    base_time4 = datetime(2026, 1, 13, 10, 0, 0)  # Monday morning (weekday)
    user4_calls = [
        APICall("c15", "/api/login", "POST", {}, "u3", "s4", base_time4, 85, 200, 512, "premium"),
        APICall("c16", "/api/users/300/profile", "GET", {}, "u3", "s4", base_time4 + timedelta(seconds=2), 105, 200, 2048, "premium"),
        APICall("c17", "/api/orders/history", "GET", {}, "u3", "s4", base_time4 + timedelta(seconds=8), 250, 200, 8192, "premium"),
        APICall("c18", "/api/orders/1001/details", "GET", {}, "u3", "s4", base_time4 + timedelta(seconds=15), 200, 200, 4096, "premium"),
    ]
    session4 = Session("s4", "u3", "premium", base_time4, base_time4 + timedelta(seconds=15), user4_calls)
    sessions.append(session4)

    return Dataset("ecommerce_sample", sessions)


def demonstrate_normalization():
    """Demonstrate endpoint normalization."""
    print("\n" + "="*80)
    print("1. ENDPOINT NORMALIZATION")
    print("="*80)

    builder = SequenceBuilder(normalize_endpoints=True)

    test_cases = [
        "/API/Users/123/Profile/",
        "/api/products/456",
        "/search?q=shoes&category=fashion",
        "/api/orders/550e8400-e29b-41d4-a716-446655440000/status",
        "/Users/789/Settings",
    ]

    print("\n🔧 Normalization transforms endpoint variations into consistent patterns:\n")
    for endpoint in test_cases:
        normalized = builder.normalize_endpoint(endpoint)
        print(f"  {endpoint:50} → {normalized}")


def demonstrate_basic_sequences(dataset):
    """Demonstrate basic sequence extraction."""
    print("\n" + "="*80)
    print("2. BASIC SEQUENCE EXTRACTION")
    print("="*80)

    builder = SequenceBuilder(normalize_endpoints=True, min_sequence_length=2)
    sequences = builder.build_sequences(dataset.sessions)

    print(f"\n📊 Extracted {len(sequences)} sequences:\n")
    for i, seq in enumerate(sequences, 1):
        print(f"  Session {i}: {len(seq)} endpoints")
        for endpoint in seq:
            print(f"    → {endpoint}")
        print()


def demonstrate_labeled_sequences(dataset):
    """Demonstrate labeled sequence generation."""
    print("\n" + "="*80)
    print("3. LABELED SEQUENCES FOR PREDICTION")
    print("="*80)

    builder = SequenceBuilder(normalize_endpoints=True)
    labeled = builder.build_labeled_sequences(dataset.sessions)

    print(f"\n🎯 Generated {len(labeled)} (history, next) pairs for evaluation:\n")
    print("  These pairs let us test prediction accuracy.\n")

    # Show first 5 examples
    for i, (history, next_endpoint) in enumerate(labeled[:5], 1):
        print(f"  Example {i}:")
        print(f"    Given history: {history}")
        print(f"    Predict next:  {next_endpoint}\n")


def demonstrate_ngrams(dataset):
    """Demonstrate n-gram extraction."""
    print("\n" + "="*80)
    print("4. N-GRAM EXTRACTION")
    print("="*80)

    builder = SequenceBuilder(normalize_endpoints=True)

    # Bigrams
    bigrams = builder.build_ngrams(dataset.sessions, n=2)
    print(f"\n📌 Extracted {len(bigrams)} bigrams (pairs of consecutive endpoints):\n")
    bigram_counts = {}
    for bigram in bigrams:
        bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

    print("  Most common bigrams:")
    for bigram, count in sorted(bigram_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    {bigram[0]} → {bigram[1]}: {count} times")

    # Trigrams
    trigrams = builder.build_ngrams(dataset.sessions, n=3)
    print(f"\n📌 Extracted {len(trigrams)} trigrams (triples of consecutive endpoints)")


def demonstrate_contextual_sequences(dataset):
    """Demonstrate contextual sequence extraction."""
    print("\n" + "="*80)
    print("5. CONTEXTUAL SEQUENCES")
    print("="*80)

    builder = SequenceBuilder(normalize_endpoints=True, min_sequence_length=2)
    contextual = builder.build_contextual_sequences(dataset.sessions)

    print(f"\n🌐 Extracted {len(contextual)} sequences with context metadata:\n")

    for i, ctx_seq in enumerate(contextual, 1):
        print(f"  Sequence {i}:")
        print(f"    User Type:      {ctx_seq.user_type}")
        print(f"    Time of Day:    {ctx_seq.time_of_day}")
        print(f"    Day Type:       {ctx_seq.day_type}")
        print(f"    Session Length: {ctx_seq.session_length_category}")
        print(f"    Endpoints:      {len(ctx_seq.sequence)} calls")
        print()


def demonstrate_transition_analysis(dataset):
    """Demonstrate transition probability calculation."""
    print("\n" + "="*80)
    print("6. TRANSITION PROBABILITY ANALYSIS")
    print("="*80)

    builder = SequenceBuilder(normalize_endpoints=True)

    # Get transition counts
    counts = builder.get_transition_counts(dataset.sessions)
    print(f"\n📈 Found {len(counts)} unique transitions:\n")

    print("  Transition counts:")
    for (from_ep, to_ep), count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {from_ep:30} → {to_ep:30} : {count}x")

    # Get transition probabilities
    probs = builder.get_transition_probabilities(dataset.sessions)
    print("\n📊 Transition probabilities (what comes next?):\n")

    for from_ep in sorted(probs.keys())[:5]:  # Show first 5 endpoints
        print(f"  From '{from_ep}':")
        for to_ep, prob in sorted(probs[from_ep].items(), key=lambda x: x[1], reverse=True):
            print(f"    → {to_ep:30} : {prob:6.1%}")
        print()


def demonstrate_statistics(dataset):
    """Demonstrate sequence statistics."""
    print("\n" + "="*80)
    print("7. SEQUENCE STATISTICS")
    print("="*80)

    builder = SequenceBuilder(normalize_endpoints=True, min_sequence_length=2)
    stats = builder.get_sequence_statistics(dataset.sessions)

    print("\n📊 Overall statistics:\n")
    print(f"  Total sequences:     {stats['total_sequences']}")
    print(f"  Total API calls:     {stats['total_calls']}")
    print(f"  Avg sequence length: {stats['avg_sequence_length']:.2f} calls")
    print(f"  Min sequence length: {stats['min_sequence_length']} calls")
    print(f"  Max sequence length: {stats['max_sequence_length']} calls")
    print(f"  Unique endpoints:    {stats['unique_endpoints']}")
    print(f"  Total transitions:   {stats['total_transitions']}")


def demonstrate_train_test_split(dataset):
    """Demonstrate train/test splitting."""
    print("\n" + "="*80)
    print("8. TRAIN/TEST SPLIT")
    print("="*80)

    builder = SequenceBuilder(normalize_endpoints=True, min_sequence_length=2)

    train_sessions, test_sessions = builder.split_sequences(dataset.sessions, train_ratio=0.75)

    print(f"\n✂️  Split into train/test sets:\n")
    print(f"  Training sessions:   {len(train_sessions)} ({len(train_sessions)/len(dataset.sessions)*100:.0f}%)")
    print(f"  Testing sessions:    {len(test_sessions)} ({len(test_sessions)/len(dataset.sessions)*100:.0f}%)")

    train_stats = builder.get_sequence_statistics(train_sessions)
    test_stats = builder.get_sequence_statistics(test_sessions)

    print(f"\n  Training set:        {train_stats['total_calls']} calls, {train_stats['unique_endpoints']} unique endpoints")
    print(f"  Testing set:         {test_stats['total_calls']} calls, {test_stats['unique_endpoints']} unique endpoints")


def demonstrate_markov_training(dataset):
    """Demonstrate how to use sequences for Markov chain training."""
    print("\n" + "="*80)
    print("9. MARKOV CHAIN TRAINING EXAMPLE")
    print("="*80)

    builder = SequenceBuilder(normalize_endpoints=True, min_sequence_length=2)

    # Split data
    train_sessions, test_sessions = builder.split_sequences(dataset.sessions, train_ratio=0.75)

    # Train: Calculate transition probabilities
    train_probs = builder.get_transition_probabilities(train_sessions)

    print("\n🎓 Training Markov Chain:\n")
    print(f"  Built transition matrix from {len(train_sessions)} training sessions")
    print(f"  Matrix size: {len(train_probs)} states (endpoints)")

    # Test: Evaluate on labeled sequences
    test_labeled = builder.build_labeled_sequences(test_sessions)

    print(f"\n🧪 Evaluating on {len(test_labeled)} test predictions:\n")

    correct = 0
    for history, actual_next in test_labeled[:5]:  # Show first 5
        current = history[-1]  # Last endpoint in history

        if current in train_probs:
            # Get most likely next endpoint
            next_endpoints = train_probs[current]
            predicted_next = max(next_endpoints.items(), key=lambda x: x[1])[0]
            is_correct = (predicted_next == actual_next)
            correct += is_correct

            status = "[OK] CORRECT" if is_correct else "[FAIL] WRONG"
            print(f"  Current: {current}")
            print(f"    Predicted: {predicted_next} | Actual: {actual_next} | {status}\n")
        else:
            print(f"  Current: {current}")
            print(f"    No training data for this endpoint\n")


def main():
    """Run all demonstrations."""
    print("\n" + "="*80)
    print("SEQUENCE BUILDER - COMPREHENSIVE DEMONSTRATION")
    print("="*80)
    print("\nThis demonstrates all features of the SequenceBuilder module")
    print("for converting sessions into Markov chain training data.\n")

    # Create dataset
    print("📦 Creating sample e-commerce dataset...")
    dataset = create_realistic_dataset()
    print(f"   Created dataset with {len(dataset.sessions)} sessions")
    print(f"   Total API calls: {dataset.total_calls}")
    print(f"   Unique users: {dataset.num_unique_users}")

    # Run demonstrations
    demonstrate_normalization()
    demonstrate_basic_sequences(dataset)
    demonstrate_labeled_sequences(dataset)
    demonstrate_ngrams(dataset)
    demonstrate_contextual_sequences(dataset)
    demonstrate_transition_analysis(dataset)
    demonstrate_statistics(dataset)
    demonstrate_train_test_split(dataset)
    demonstrate_markov_training(dataset)

    print("\n" + "="*80)
    print("[SUCCESS] DEMONSTRATION COMPLETE!")
    print("="*80)
    print("\nThe SequenceBuilder module is ready for Markov chain training.")
    print("See SEQUENCE_BUILDER_GUIDE.md for detailed documentation.\n")


if __name__ == "__main__":
    main()


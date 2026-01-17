"""Validation script for TransitionMatrix implementation."""

from src.markov.transition_matrix import TransitionMatrix
import os

def test_basic_operations():
    """Test basic increment and probability calculations."""
    print("=" * 60)
    print("Testing basic operations...")
    print("=" * 60)

    tm = TransitionMatrix(smoothing=0.001)
    tm.increment("login", "profile", 80)
    tm.increment("login", "browse", 20)
    tm.increment("profile", "browse", 50)
    tm.increment("profile", "orders", 30)

    # Test probability calculation
    prob_profile = tm.get_probability('login', 'profile')
    prob_browse = tm.get_probability('login', 'browse')

    print(f"P(profile|login) = {prob_profile:.3f}  (expected ~0.800)")
    print(f"P(browse|login) = {prob_browse:.3f}  (expected ~0.200)")

    assert abs(prob_profile - 0.8) < 0.01, f"Expected ~0.8, got {prob_profile}"
    assert abs(prob_browse - 0.2) < 0.01, f"Expected ~0.2, got {prob_browse}"

    print("✓ Basic probability calculations correct\n")


def test_top_k():
    """Test top-k transitions."""
    print("=" * 60)
    print("Testing top-k transitions...")
    print("=" * 60)

    tm = TransitionMatrix(smoothing=0.001)
    tm.increment("login", "profile", 80)
    tm.increment("login", "browse", 20)
    tm.increment("profile", "browse", 50)
    tm.increment("profile", "orders", 30)

    top_from_profile = tm.get_top_k('profile', k=2)
    print(f"Top transitions from profile: {top_from_profile}")

    # Should have exactly 2 transitions
    assert len(top_from_profile) == 2, f"Expected 2 transitions, got {len(top_from_profile)}"

    # First should be browse (higher count)
    assert top_from_profile[0][0] == "browse", f"Expected 'browse' as top transition"

    # Check probabilities sum to approximately 1.0
    total_prob = sum(prob for _, prob in top_from_profile)
    assert abs(total_prob - 1.0) < 0.01, f"Probabilities should sum to ~1.0, got {total_prob}"

    print("✓ Top-k functionality working correctly\n")


def test_serialization():
    """Test save/load functionality."""
    print("=" * 60)
    print("Testing serialization...")
    print("=" * 60)

    tm = TransitionMatrix(smoothing=0.001)
    tm.increment("login", "profile", 80)
    tm.increment("login", "browse", 20)
    tm.increment("profile", "browse", 50)
    tm.increment("profile", "orders", 30)

    # Save to file
    test_file = "test_matrix.json"
    tm.save(test_file)
    print(f"✓ Saved matrix to {test_file}")

    # Load from file
    tm2 = TransitionMatrix.load(test_file)
    print(f"✓ Loaded matrix from {test_file}")

    # Verify probabilities match
    prob1 = tm.get_probability("login", "profile")
    prob2 = tm2.get_probability("login", "profile")

    assert prob1 == prob2, f"Probabilities don't match: {prob1} != {prob2}"
    print(f"✓ Probabilities match: {prob1:.3f} == {prob2:.3f}")

    # Verify counts match
    count1 = tm.get_count("profile", "browse")
    count2 = tm2.get_count("profile", "browse")

    assert count1 == count2, f"Counts don't match: {count1} != {count2}"
    print(f"✓ Counts match: {count1} == {count2}")

    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"✓ Cleaned up {test_file}\n")


def test_statistics():
    """Test statistics generation."""
    print("=" * 60)
    print("Testing statistics...")
    print("=" * 60)

    tm = TransitionMatrix(smoothing=0.001)
    tm.increment("login", "profile", 80)
    tm.increment("login", "browse", 20)
    tm.increment("profile", "browse", 50)
    tm.increment("profile", "orders", 30)
    tm.increment("browse", "product", 40)
    tm.increment("orders", "checkout", 25)

    stats = tm.get_statistics()

    print(f"Number of states: {stats['num_states']}")
    print(f"Number of transitions: {stats['num_transitions']}")
    print(f"Sparsity: {stats['sparsity']:.2%}")
    print(f"Avg transitions per state: {stats['avg_transitions_per_state']:.2f}")
    print(f"\nTop transitions:")

    for i, trans in enumerate(stats['most_common_transitions'][:5], 1):
        print(f"  {i}. {trans['from']} -> {trans['to']}: {trans['count']}")

    assert stats['num_states'] == 6, f"Expected 6 states, got {stats['num_states']}"
    assert stats['num_transitions'] == 6, f"Expected 6 transitions, got {stats['num_transitions']}"

    print("\n✓ Statistics generation working correctly\n")


def test_merge():
    """Test merging two matrices."""
    print("=" * 60)
    print("Testing matrix merge...")
    print("=" * 60)

    tm1 = TransitionMatrix()
    tm1.increment("A", "B", 10)
    tm1.increment("A", "C", 5)

    tm2 = TransitionMatrix()
    tm2.increment("A", "B", 5)
    tm2.increment("B", "C", 3)

    merged = tm1.merge(tm2)

    # Check merged counts
    count_ab = merged.get_count("A", "B")
    count_ac = merged.get_count("A", "C")
    count_bc = merged.get_count("B", "C")

    print(f"Merged count A->B: {count_ab} (expected 15)")
    print(f"Merged count A->C: {count_ac} (expected 5)")
    print(f"Merged count B->C: {count_bc} (expected 3)")

    assert count_ab == 15, f"Expected 15, got {count_ab}"
    assert count_ac == 5, f"Expected 5, got {count_ac}"
    assert count_bc == 3, f"Expected 3, got {count_bc}"

    print("✓ Matrix merge working correctly\n")


def test_smoothing():
    """Test Laplace smoothing."""
    print("=" * 60)
    print("Testing Laplace smoothing...")
    print("=" * 60)

    # Without smoothing
    tm_no_smooth = TransitionMatrix(smoothing=0.0)
    tm_no_smooth.increment("A", "B", 10)

    prob_unseen = tm_no_smooth.get_probability("A", "C")
    print(f"P(C|A) without smoothing: {prob_unseen:.3f} (expected 0.000)")
    assert prob_unseen == 0.0, "Unseen transition should have 0 probability"

    # With smoothing
    tm_smooth = TransitionMatrix(smoothing=0.1)
    tm_smooth.increment("A", "B", 10)
    tm_smooth.increment("B", "C", 5)  # Add another state to increase vocab

    prob_seen = tm_smooth.get_probability("A", "B")
    prob_unseen_smooth = tm_smooth.get_probability("A", "C")

    print(f"P(B|A) with smoothing: {prob_seen:.3f}")
    print(f"P(C|A) with smoothing: {prob_unseen_smooth:.3f} (non-zero)")

    assert prob_unseen_smooth > 0, "Smoothed probability should be non-zero"
    assert prob_seen > prob_unseen_smooth, "Seen transitions should be more likely"

    print("✓ Smoothing working correctly\n")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("=" * 60)
    print("Testing edge cases...")
    print("=" * 60)

    tm = TransitionMatrix()

    # Empty matrix
    print(f"Empty matrix states: {tm.num_states} (expected 0)")
    assert tm.num_states == 0

    # Unseen state
    prob = tm.get_probability("unknown", "state")
    print(f"Probability for unseen state: {prob:.3f} (expected 0.000)")
    assert prob == 0.0

    # Empty row
    row = tm.get_row("unknown")
    print(f"Empty row: {row} (expected {{}})")
    assert row == {}

    # Top-k on empty state
    top = tm.get_top_k("unknown", k=5)
    print(f"Top-k on empty state: {top} (expected [])")
    assert top == []

    print("✓ Edge cases handled correctly\n")


def test_dataframe_conversion():
    """Test DataFrame conversion if pandas is available."""
    print("=" * 60)
    print("Testing DataFrame conversion...")
    print("=" * 60)

    try:
        import pandas as pd

        tm = TransitionMatrix()
        tm.increment("A", "B", 10)
        tm.increment("A", "C", 5)
        tm.increment("B", "C", 3)

        df = tm.to_dataframe()

        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print("\nDataFrame preview:")
        print(df.to_string(index=False))

        assert len(df) == 3, f"Expected 3 rows, got {len(df)}"
        assert list(df.columns) == ['from_state', 'to_state', 'count', 'probability']

        print("\n✓ DataFrame conversion working correctly\n")

    except ImportError:
        print("⚠ Pandas not installed, skipping DataFrame test\n")


def main():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("TRANSITION MATRIX VALIDATION TESTS")
    print("=" * 60 + "\n")

    try:
        test_basic_operations()
        test_top_k()
        test_serialization()
        test_statistics()
        test_merge()
        test_smoothing()
        test_edge_cases()
        test_dataframe_conversion()

        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}\n")
        raise


if __name__ == "__main__":
    main()


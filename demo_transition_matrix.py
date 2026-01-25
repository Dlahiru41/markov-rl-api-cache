"""
Comprehensive demo of TransitionMatrix functionality.

This script demonstrates all major features of the TransitionMatrix class
including building, querying, merging, and serialization.
"""

from src.markov import TransitionMatrix
import json
from pathlib import Path


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_basic_operations():
    """Demonstrate basic increment, count, and probability operations."""
    print_section("1. Basic Operations")

    tm = TransitionMatrix(smoothing=0.001)

    # Simulate API endpoint transitions
    print("\nBuilding transition matrix from API logs...")
    tm.increment("login", "profile", 80)
    tm.increment("login", "browse", 20)
    tm.increment("profile", "orders", 50)
    tm.increment("profile", "browse", 30)
    tm.increment("browse", "product", 60)
    tm.increment("browse", "search", 40)
    tm.increment("product", "cart", 70)
    tm.increment("product", "details", 30)

    print("[OK] Added 8 transition patterns")

    # Query counts and probabilities
    print("\nQuerying transition probabilities:")
    print(f"  P(profile|login) = {tm.get_probability('login', 'profile'):.3f}")
    print(f"  P(browse|login)  = {tm.get_probability('login', 'browse'):.3f}")
    print(f"  P(cart|product)  = {tm.get_probability('product', 'cart'):.3f}")

    # Get full row
    print("\nAll transitions from 'browse':")
    row = tm.get_row("browse")
    for state, prob in sorted(row.items(), key=lambda x: x[1], reverse=True):
        print(f"  {state}: {prob:.2%}")

    return tm


def demo_top_k_queries(tm):
    """Demonstrate top-k query functionality."""
    print_section("2. Top-K Queries")

    print("\nFinding most likely next endpoints (for cache prefetching):")

    endpoints = ["login", "profile", "browse", "product"]
    for endpoint in endpoints:
        top = tm.get_top_k(endpoint, k=2)
        print(f"\n  From '{endpoint}':")
        for i, (state, prob) in enumerate(top, 1):
            print(f"    {i}. {state}: {prob:.2%}")


def demo_statistics(tm):
    """Demonstrate statistics generation."""
    print_section("3. Matrix Statistics")

    stats = tm.get_statistics()

    print(f"\nMatrix Overview:")
    print(f"  Number of unique states:      {stats['num_states']}")
    print(f"  Number of transitions:        {stats['num_transitions']}")
    print(f"  Sparsity:                     {stats['sparsity']:.2%}")
    print(f"  Avg transitions per state:    {stats['avg_transitions_per_state']:.2f}")

    print(f"\nTop 5 Most Common Transitions:")
    for i, trans in enumerate(stats['most_common_transitions'][:5], 1):
        print(f"  {i}. {trans['from']:8s} → {trans['to']:8s}: {trans['count']:3d} occurrences")


def demo_smoothing():
    """Demonstrate Laplace smoothing."""
    print_section("4. Laplace Smoothing")

    # Without smoothing
    tm_no_smooth = TransitionMatrix(smoothing=0.0)
    tm_no_smooth.increment("A", "B", 10)
    tm_no_smooth.increment("C", "D", 5)  # Add another state

    prob_seen = tm_no_smooth.get_probability("A", "B")
    prob_unseen = tm_no_smooth.get_probability("A", "C")

    print("\nWithout Smoothing (smoothing=0.0):")
    print(f"  P(B|A) [seen]:     {prob_seen:.3f}")
    print(f"  P(C|A) [unseen]:   {prob_unseen:.3f}")

    # With smoothing
    tm_smooth = TransitionMatrix(smoothing=0.1)
    tm_smooth.increment("A", "B", 10)
    tm_smooth.increment("C", "D", 5)

    prob_seen = tm_smooth.get_probability("A", "B")
    prob_unseen = tm_smooth.get_probability("A", "C")

    print("\nWith Smoothing (smoothing=0.1):")
    print(f"  P(B|A) [seen]:     {prob_seen:.3f}")
    print(f"  P(C|A) [unseen]:   {prob_unseen:.3f} (non-zero!)")

    print("\n  → Smoothing prevents zero probabilities for unseen transitions")


def demo_merge():
    """Demonstrate matrix merging."""
    print_section("5. Matrix Merging")

    # Matrix from first time period
    print("\nTime Period 1 (morning traffic):")
    tm1 = TransitionMatrix()
    tm1.increment("login", "profile", 100)
    tm1.increment("login", "browse", 50)
    tm1.increment("profile", "orders", 80)
    print("  login → profile:  100 occurrences")
    print("  login → browse:    50 occurrences")
    print("  profile → orders:  80 occurrences")

    # Matrix from second time period
    print("\nTime Period 2 (evening traffic):")
    tm2 = TransitionMatrix()
    tm2.increment("login", "profile", 80)
    tm2.increment("login", "browse", 70)
    tm2.increment("browse", "search", 60)
    print("  login → profile:   80 occurrences")
    print("  login → browse:    70 occurrences")
    print("  browse → search:   60 occurrences")

    # Merge
    print("\nMerged (combined traffic):")
    merged = tm1.merge(tm2)
    print(f"  login → profile:  {merged.get_count('login', 'profile')} occurrences")
    print(f"  login → browse:   {merged.get_count('login', 'browse')} occurrences")
    print(f"  profile → orders: {merged.get_count('profile', 'orders')} occurrences")
    print(f"  browse → search:  {merged.get_count('browse', 'search')} occurrences")

    print("\n  → Counts are summed from both matrices")


def demo_serialization(tm):
    """Demonstrate save/load functionality."""
    print_section("6. Serialization")

    # Save to JSON
    output_file = "demo_matrix.json"
    tm.save(output_file)
    print(f"\n[OK] Saved matrix to '{output_file}'")

    # Check file size
    file_size = Path(output_file).stat().st_size
    print(f"  File size: {file_size:,} bytes")

    # Load from JSON
    tm_loaded = TransitionMatrix.load(output_file)
    print(f"\n[OK] Loaded matrix from '{output_file}'")

    # Verify data integrity
    original_prob = tm.get_probability("login", "profile")
    loaded_prob = tm_loaded.get_probability("login", "profile")

    print(f"\nData Integrity Check:")
    print(f"  Original P(profile|login): {original_prob:.6f}")
    print(f"  Loaded P(profile|login):   {loaded_prob:.6f}")
    print(f"  Match: {original_prob == loaded_prob} [OK]")

    # Show structure
    print(f"\nJSON Structure Preview:")
    with open(output_file, 'r') as f:
        data = json.load(f)

    print(f"  smoothing: {data['smoothing']}")
    print(f"  transitions: {len(data['transitions'])} states")
    print(f"  total_from: {len(data['total_from'])} states")

    # Clean up
    Path(output_file).unlink()
    print(f"\n[OK] Cleaned up '{output_file}'")


def demo_dataframe_conversion(tm):
    """Demonstrate DataFrame conversion (if pandas available)."""
    print_section("7. DataFrame Conversion")

    try:
        df = tm.to_dataframe()

        print("\n[OK] Converted to pandas DataFrame")
        print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"Columns: {', '.join(df.columns)}")

        print("\nFirst 5 rows:")
        print(df.head().to_string(index=False))

        print("\nSummary Statistics:")
        print(df[['count', 'probability']].describe().to_string())

    except ImportError:
        print("\n⚠ Pandas not installed - skipping DataFrame demo")
        print("  Install with: pip install pandas")


def demo_cache_prefetching(tm):
    """Demonstrate practical use case: cache prefetching."""
    print_section("8. Practical Use Case: Cache Prefetching")

    print("\nScenario: User just accessed 'browse' endpoint")
    print("Question: Which endpoints should we prefetch to cache?")

    # Get top candidates
    candidates = tm.get_top_k("browse", k=3)
    threshold = 0.15  # Only prefetch if probability > 15%

    print(f"\nTop {len(candidates)} most likely next endpoints:")
    to_prefetch = []
    for state, prob in candidates:
        status = "[OK] PREFETCH" if prob > threshold else "[FAIL] Skip"
        print(f"  {state:12s}: {prob:.2%}  [{status}]")
        if prob > threshold:
            to_prefetch.append(state)

    print(f"\nDecision: Prefetch {to_prefetch}")
    print(f"Expected cache hit rate: {sum(prob for _, prob in candidates if prob > threshold):.1%}")


def demo_incremental_learning():
    """Demonstrate incremental learning pattern."""
    print_section("9. Incremental Learning")

    print("\nSimulating real-time learning from API request stream...")

    tm = TransitionMatrix(smoothing=0.001)

    # Simulate incoming requests
    request_stream = [
        ("login", "profile"),
        ("login", "browse"),
        ("profile", "orders"),
        ("login", "profile"),
        ("browse", "search"),
        ("login", "profile"),
        ("profile", "browse"),
    ]

    print("\nProcessing requests:")
    for i, (from_ep, to_ep) in enumerate(request_stream, 1):
        tm.increment(from_ep, to_ep)

        if i % 3 == 0:  # Show progress every 3 requests
            prob = tm.get_probability("login", "profile")
            print(f"  After {i:2d} requests: P(profile|login) = {prob:.3f}")

    print(f"\nFinal transition counts:")
    stats = tm.get_statistics()
    for trans in stats['most_common_transitions'][:5]:
        print(f"  {trans['from']:8s} → {trans['to']:8s}: {trans['count']}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" + "=" * 68 + "=")
    print("|" + " " * 15 + "TransitionMatrix Comprehensive Demo" + " " * 18 + "|")
    print("=" + "=" * 68 + "╝")

    try:
        # Run all demos
        tm = demo_basic_operations()
        demo_top_k_queries(tm)
        demo_statistics(tm)
        demo_smoothing()
        demo_merge()
        demo_serialization(tm)
        demo_dataframe_conversion(tm)
        demo_cache_prefetching(tm)
        demo_incremental_learning()

        # Final summary
        print_section("Summary")
        print("\n[OK] All demonstrations completed successfully!")
        print("\nKey Takeaways:")
        print("  • Sparse storage is efficient for real-world transition data")
        print("  • Top-k queries enable efficient cache prefetching decisions")
        print("  • Smoothing prevents zero probabilities for unseen transitions")
        print("  • Matrices can be merged, saved, and loaded for production use")
        print("  • Incremental learning supports real-time updates")

        print("\nNext Steps:")
        print("  1. Read full documentation: src/markov/README.md")
        print("  2. See quick reference: TRANSITION_MATRIX_QUICK_REF.md")
        print("  3. Run tests: pytest tests/unit/test_transition_matrix.py")
        print("  4. Integrate with your Markov chain model")

    except Exception as e:
        print(f"\n[FAIL] Error during demo: {e}")
        raise


if __name__ == "__main__":
    main()


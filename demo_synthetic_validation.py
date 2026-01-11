"""Demonstration of using SyntheticGenerator to validate Markov chain learning.

This shows the key benefit: we know the true transition probabilities, so we can
verify that our Markov chain correctly learns them.
"""

import numpy as np
from datetime import datetime
from collections import Counter

from preprocessing.synthetic_generator import SyntheticGenerator, WorkflowDefinition
from preprocessing.sequence_builder import SequenceBuilder


def demo_ground_truth_validation():
    """Demonstrate validating Markov chain learning against ground truth."""
    print("\n" + "="*70)
    print("DEMO: Ground Truth Validation")
    print("="*70)
    print("\nShowing how synthetic data lets us validate that our Markov chain")
    print("correctly learns the underlying transition probabilities.")

    # 1. Get workflow (this is our ground truth)
    workflow = SyntheticGenerator.ECOMMERCE_WORKFLOW

    print(f"\n📊 Ground Truth Workflow: {workflow.name}")
    print(f"  Entry points: {len(workflow.entry_points)}")
    print(f"  Transitions: {len(workflow.transitions)} endpoints")

    # Show a few key transitions
    print(f"\n  Sample ground truth probabilities:")
    sample_transitions = [
        ("/api/login", "/api/users/{id}/profile"),
        ("/api/products/browse", "/api/products/{id}/details"),
        ("/api/cart", "/api/checkout"),
        ("/api/checkout", "/api/payment"),
    ]

    for from_ep, to_ep in sample_transitions:
        if from_ep in workflow.transitions and to_ep in workflow.transitions[from_ep]:
            prob = workflow.transitions[from_ep][to_ep]
            print(f"    {from_ep}")
            print(f"      → {to_ep}: {prob:.1%}")

    # 2. Generate synthetic data
    print(f"\n🔄 Generating synthetic data...")
    gen = SyntheticGenerator(seed=42)
    dataset = gen.generate_dataset(
        num_users=500,
        sessions_per_user=(4, 2),
        show_progress=False
    )

    print(f"  ✓ Generated {len(dataset.sessions)} sessions")
    print(f"  ✓ Total calls: {dataset.total_calls}")

    # 3. Train Markov chain (learn transitions)
    print(f"\n🎓 Training Markov chain...")
    builder = SequenceBuilder(normalize_endpoints=True)
    learned_probs = builder.get_transition_probabilities(dataset.sessions)

    print(f"  ✓ Learned {len(learned_probs)} states")

    # 4. Compare learned vs ground truth
    print(f"\n✅ Validation Results:")
    print(f"\n  {'Transition':<50} {'True':<8} {'Learned':<8} {'Error':<8}")
    print(f"  {'-'*50} {'-'*8} {'-'*8} {'-'*8}")

    errors = []
    for from_ep, to_ep in sample_transitions:
        true_prob = workflow.transitions.get(from_ep, {}).get(to_ep, 0)
        learned_prob = learned_probs.get(from_ep, {}).get(to_ep, 0)
        error = abs(true_prob - learned_prob)
        errors.append(error)

        status = "✓" if error < 0.10 else "✗"
        print(f"  {status} {from_ep:<48}")
        print(f"     → {to_ep:<44} {true_prob:>6.1%} {learned_prob:>8.1%} {error:>8.1%}")

    avg_error = np.mean(errors)
    print(f"\n  Average error: {avg_error:.1%}")

    if avg_error < 0.10:
        print(f"  ✓ Model learned correctly! (< 10% error)")
    else:
        print(f"  ⚠ Model has significant errors (> 10%)")


def demo_sample_size_effect():
    """Show how sample size affects learning accuracy."""
    print("\n" + "="*70)
    print("DEMO: Sample Size Effect on Learning")
    print("="*70)
    print("\nShowing how more data improves learning accuracy.")

    workflow = SyntheticGenerator.ECOMMERCE_WORKFLOW
    gen = SyntheticGenerator(seed=42)
    builder = SequenceBuilder(normalize_endpoints=True)

    # Test with different sample sizes
    sample_sizes = [50, 100, 200, 500, 1000]

    # Pick a transition to track
    test_from = "/api/login"
    test_to = "/api/users/{id}/profile"
    true_prob = workflow.transitions[test_from][test_to]

    print(f"\n  Testing transition: {test_from} → {test_to}")
    print(f"  True probability: {true_prob:.1%}")

    print(f"\n  {'Users':<8} {'Learned':<10} {'Error':<10} {'Status'}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*6}")

    for n_users in sample_sizes:
        dataset = gen.generate_dataset(
            num_users=n_users,
            sessions_per_user=(3, 1),
            show_progress=False
        )

        learned_probs = builder.get_transition_probabilities(dataset.sessions)
        learned_prob = learned_probs.get(test_from, {}).get(test_to, 0)
        error = abs(true_prob - learned_prob)

        status = "✓" if error < 0.05 else "→"
        print(f"  {n_users:<8} {learned_prob:>8.1%} {error:>10.1%} {status:>6}")


def demo_cascade_failure_detection():
    """Show how to detect cascade failures in synthetic data."""
    print("\n" + "="*70)
    print("DEMO: Cascade Failure Detection")
    print("="*70)
    print("\nShowing cascade failure patterns in synthetic data.")

    gen = SyntheticGenerator(seed=42)

    # Generate normal data
    print(f"\n  Generating normal sessions...")
    normal_dataset = gen.generate_dataset(
        num_users=50,
        cascade_failure_rate=0.0,
        show_progress=False
    )

    # Generate data with failures
    print(f"  Generating sessions with cascade failures...")
    failure_dataset = gen.generate_dataset(
        num_users=50,
        cascade_failure_rate=0.30,  # 30% have failures
        show_progress=False
    )

    # Analyze differences
    def analyze_dataset(dataset, name):
        total_calls = dataset.total_calls
        error_calls = sum(1 for s in dataset.sessions for c in s.calls if c.status_code != 200)
        avg_response = np.mean([c.response_time_ms for s in dataset.sessions for c in s.calls])

        print(f"\n  {name}:")
        print(f"    Total calls: {total_calls}")
        print(f"    Error calls: {error_calls} ({error_calls/total_calls*100:.1f}%)")
        print(f"    Avg response time: {avg_response:.0f}ms")

    analyze_dataset(normal_dataset, "Normal Data")
    analyze_dataset(failure_dataset, "With Cascade Failures")


def demo_workflow_customization():
    """Show how to create and use custom workflows."""
    print("\n" + "="*70)
    print("DEMO: Custom Workflow Creation")
    print("="*70)
    print("\nShowing how to define custom workflows for specific use cases.")

    # Create a simple API workflow
    api_workflow = WorkflowDefinition(
        name="simple_api",
        entry_points={
            "/health": 0.20,
            "/api/login": 0.80
        },
        transitions={
            "/health": {
                "/metrics": 1.0
            },
            "/api/login": {
                "/api/users": 0.7,
                "/api/logout": 0.3
            },
            "/api/users": {
                "/api/users/{id}": 0.8,
                "/api/logout": 0.2
            },
            "/api/users/{id}": {
                "/api/logout": 1.0
            },
            "/metrics": {
                "/health": 1.0
            }
        },
        exit_points={"/api/logout", "/metrics"},
        avg_response_times={
            "/health": 50,
            "/metrics": 80,
            "/api/login": 120,
            "/api/users": 100,
            "/api/users/{id}": 90,
            "/api/logout": 60
        }
    )

    print(f"\n✓ Created custom workflow: {api_workflow.name}")

    # Validate
    errors = SyntheticGenerator.validate_workflow(api_workflow)
    if errors:
        print(f"  ❌ Validation errors: {errors}")
    else:
        print(f"  ✓ Workflow is valid")

    # Generate data
    gen = SyntheticGenerator(seed=42)
    dataset = gen.generate_dataset(
        num_users=30,
        workflow=api_workflow,
        show_progress=False
    )

    print(f"\n✓ Generated data from custom workflow:")
    print(f"  Sessions: {len(dataset.sessions)}")
    print(f"  Total calls: {dataset.total_calls}")
    print(f"  Unique endpoints: {len(dataset.unique_endpoints)}")

    # Show sample session
    sample_session = dataset.sessions[0]
    print(f"\n  Sample session path:")
    print(f"    {' → '.join(sample_session.endpoint_sequence)}")


def demo_reproducibility():
    """Demonstrate reproducibility with seeds."""
    print("\n" + "="*70)
    print("DEMO: Reproducibility")
    print("="*70)
    print("\nShowing that same seed produces identical results.")

    workflow = SyntheticGenerator.ECOMMERCE_WORKFLOW
    start_time = datetime(2026, 1, 11, 10, 0, 0)

    # Generate with seed 42
    gen1 = SyntheticGenerator(seed=42)
    session1 = gen1.generate_session(workflow, "user1", start_time)

    # Generate again with same seed
    gen2 = SyntheticGenerator(seed=42)
    session2 = gen2.generate_session(workflow, "user1", start_time)

    # Generate with different seed
    gen3 = SyntheticGenerator(seed=999)
    session3 = gen3.generate_session(workflow, "user1", start_time)

    print(f"\n  Session 1 (seed=42):")
    print(f"    Length: {session1.num_calls} calls")
    print(f"    Path: {' → '.join(session1.endpoint_sequence[:5])}...")

    print(f"\n  Session 2 (seed=42, repeated):")
    print(f"    Length: {session2.num_calls} calls")
    print(f"    Path: {' → '.join(session2.endpoint_sequence[:5])}...")

    print(f"\n  Session 3 (seed=999):")
    print(f"    Length: {session3.num_calls} calls")
    print(f"    Path: {' → '.join(session3.endpoint_sequence[:5])}...")

    if session1.endpoint_sequence == session2.endpoint_sequence:
        print(f"\n  ✓ Sessions 1 and 2 are identical (same seed)")
    else:
        print(f"\n  ❌ Sessions 1 and 2 differ (should be same!)")

    if session1.endpoint_sequence != session3.endpoint_sequence:
        print(f"  ✓ Session 3 is different (different seed)")
    else:
        print(f"  Note: Sessions may be same by chance for simple workflows")


def demo_user_type_distribution():
    """Show user type distribution in generated data."""
    print("\n" + "="*70)
    print("DEMO: User Type Distribution")
    print("="*70)
    print("\nShowing how user types are distributed in generated data.")

    gen = SyntheticGenerator(seed=42)
    dataset = gen.generate_dataset(
        num_users=200,
        sessions_per_user=(3, 1),
        show_progress=False
    )

    # Count user types
    user_type_counts = Counter(s.user_type for s in dataset.sessions)
    total_sessions = len(dataset.sessions)

    print(f"\n  Total sessions: {total_sessions}")
    print(f"\n  User type distribution:")

    expected = SyntheticGenerator.ECOMMERCE_WORKFLOW.user_type_distribution

    for user_type in ['premium', 'free', 'guest']:
        count = user_type_counts[user_type]
        observed = count / total_sessions
        expected_prob = expected[user_type]

        print(f"    {user_type:8s}: {count:4d} ({observed:5.1%}) | Expected: {expected_prob:5.1%}")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("SYNTHETIC GENERATOR VALIDATION DEMONSTRATIONS")
    print("="*70)
    print("\nDemonstrating how synthetic data enables validation of")
    print("Markov chain learning with known ground truth.")

    demo_ground_truth_validation()
    demo_sample_size_effect()
    demo_cascade_failure_detection()
    demo_workflow_customization()
    demo_reproducibility()
    demo_user_type_distribution()

    print("\n" + "="*70)
    print("✅ ALL DEMONSTRATIONS COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. Synthetic data has known ground truth probabilities")
    print("  2. We can validate that models learn correctly")
    print("  3. More data improves learning accuracy")
    print("  4. Cascade failures can be injected for testing")
    print("  5. Custom workflows enable specific scenarios")
    print("  6. Reproducibility ensures consistent experiments")
    print("\nThe SyntheticGenerator is ready for Markov chain validation!")


if __name__ == "__main__":
    main()


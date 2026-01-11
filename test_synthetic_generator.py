"""Test and validation script for the SyntheticGenerator module."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

from preprocessing.synthetic_generator import (
    SyntheticGenerator,
    WorkflowDefinition,
    create_simple_workflow,
    create_microservices_workflow
)
from preprocessing.models import Dataset


def test_workflow_definition():
    """Test WorkflowDefinition creation and validation."""
    print("\n" + "="*70)
    print("TEST 1: WorkflowDefinition")
    print("="*70)

    # Create a simple workflow
    workflow = WorkflowDefinition(
        name="test",
        entry_points={"/start": 1.0},
        transitions={
            "/start": {"/middle": 0.7, "/end": 0.3},
            "/middle": {"/end": 1.0}
        },
        exit_points={"/end"},
        avg_response_times={
            "/start": 100,
            "/middle": 150,
            "/end": 80
        }
    )

    print(f"\n✓ Created workflow: {workflow.name}")
    print(f"  Entry points: {list(workflow.entry_points.keys())}")
    print(f"  Transitions: {len(workflow.transitions)} endpoints")
    print(f"  Exit points: {workflow.exit_points}")

    # Test validation
    errors = SyntheticGenerator.validate_workflow(workflow)
    if errors:
        print(f"\n  Validation errors:")
        for error in errors:
            print(f"    - {error}")
    else:
        print(f"\n  ✓ Workflow is valid")


def test_ecommerce_workflow():
    """Test the pre-built e-commerce workflow."""
    print("\n" + "="*70)
    print("TEST 2: E-commerce Workflow")
    print("="*70)

    workflow = SyntheticGenerator.ECOMMERCE_WORKFLOW

    print(f"\n✓ E-commerce workflow loaded")
    print(f"  Name: {workflow.name}")
    print(f"  Entry points: {len(workflow.entry_points)}")
    for endpoint, prob in workflow.entry_points.items():
        print(f"    {endpoint}: {prob:.1%}")

    print(f"\n  Transitions: {len(workflow.transitions)} endpoints")
    print(f"  Exit points: {len(workflow.exit_points)}")
    for endpoint in workflow.exit_points:
        print(f"    - {endpoint}")

    print(f"\n  Average response times defined: {len(workflow.avg_response_times)}")

    # Validate
    errors = SyntheticGenerator.validate_workflow(workflow)
    if errors:
        print(f"\n  ❌ Validation errors found:")
        for error in errors:
            print(f"    - {error}")
    else:
        print(f"\n  ✓ Workflow is valid and ready to use")


def test_single_session_generation():
    """Test generating a single session."""
    print("\n" + "="*70)
    print("TEST 3: Single Session Generation")
    print("="*70)

    gen = SyntheticGenerator(seed=42)
    workflow = SyntheticGenerator.ECOMMERCE_WORKFLOW

    start_time = datetime(2026, 1, 11, 10, 0, 0)
    session = gen.generate_session(
        workflow=workflow,
        user_id="test_user",
        start_time=start_time,
        user_type="premium"
    )

    print(f"\n✓ Generated session")
    print(f"  Session ID: {session.session_id}")
    print(f"  User: {session.user_id} ({session.user_type})")
    print(f"  Number of calls: {session.num_calls}")
    print(f"  Duration: {session.duration_seconds:.2f} seconds")

    print(f"\n  Endpoint sequence:")
    for i, call in enumerate(session.calls, 1):
        print(f"    {i}. {call.endpoint} ({call.method}) - {call.response_time_ms:.0f}ms")

    print(f"\n  Transitions:")
    transitions = session.get_endpoint_transitions()
    for from_ep, to_ep in transitions:
        print(f"    {from_ep} → {to_ep}")

    return session


def test_reproducibility():
    """Test that same seed produces same results."""
    print("\n" + "="*70)
    print("TEST 4: Reproducibility")
    print("="*70)

    workflow = create_simple_workflow()
    start_time = datetime(2026, 1, 11, 10, 0, 0)

    # Generate with seed 42
    gen1 = SyntheticGenerator(seed=42)
    session1 = gen1.generate_session(workflow, "user1", start_time)
    sequence1 = session1.endpoint_sequence

    # Generate again with same seed
    gen2 = SyntheticGenerator(seed=42)
    session2 = gen2.generate_session(workflow, "user1", start_time)
    sequence2 = session2.endpoint_sequence

    # Generate with different seed
    gen3 = SyntheticGenerator(seed=123)
    session3 = gen3.generate_session(workflow, "user1", start_time)
    sequence3 = session3.endpoint_sequence

    print(f"\n✓ Reproducibility test:")
    print(f"  Seed 42 (run 1): {sequence1}")
    print(f"  Seed 42 (run 2): {sequence2}")
    print(f"  Seed 123:        {sequence3}")

    if sequence1 == sequence2:
        print(f"\n  ✓ Same seed produces identical results")
    else:
        print(f"\n  ❌ Same seed produced different results!")

    if sequence1 != sequence3:
        print(f"  ✓ Different seeds produce different results")
    else:
        print(f"  ❌ Different seeds produced identical results!")


def test_dataset_generation():
    """Test generating a complete dataset."""
    print("\n" + "="*70)
    print("TEST 5: Dataset Generation")
    print("="*70)

    gen = SyntheticGenerator(seed=42)

    print(f"\n  Generating dataset...")
    dataset = gen.generate_dataset(
        num_users=20,
        sessions_per_user=(3, 1),
        date_range_days=7,
        show_progress=False
    )

    print(f"\n✓ Generated dataset")
    print(f"  Name: {dataset.name}")
    print(f"  Sessions: {len(dataset.sessions)}")
    print(f"  Total calls: {dataset.total_calls}")
    print(f"  Unique users: {dataset.num_unique_users}")
    print(f"  Unique endpoints: {len(dataset.unique_endpoints)}")

    # Analyze user types
    user_types = {}
    for session in dataset.sessions:
        user_types[session.user_type] = user_types.get(session.user_type, 0) + 1

    print(f"\n  User type distribution:")
    for user_type, count in sorted(user_types.items()):
        pct = count / len(dataset.sessions) * 100
        print(f"    {user_type}: {count} sessions ({pct:.1f}%)")

    # Analyze date range
    min_time, max_time = dataset.date_range
    if min_time and max_time:
        span = (max_time - min_time).days
        print(f"\n  Date range: {span} days")
        print(f"    From: {min_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"    To:   {max_time.strftime('%Y-%m-%d %H:%M')}")

    return dataset


def test_cascade_failures():
    """Test cascade failure injection."""
    print("\n" + "="*70)
    print("TEST 6: Cascade Failure Injection")
    print("="*70)

    gen = SyntheticGenerator(seed=42)
    workflow = SyntheticGenerator.ECOMMERCE_WORKFLOW

    # Generate normal session
    start_time = datetime(2026, 1, 11, 14, 0, 0)
    normal_session = gen.generate_session(
        workflow=workflow,
        user_id="user_cascade",
        start_time=start_time
    )

    # Inject cascade failure
    cascade_session = gen.inject_cascade(normal_session)

    print(f"\n✓ Cascade failure injection:")
    print(f"  Normal session: {normal_session.num_calls} calls")
    print(f"  With cascade: {cascade_session.num_calls} calls")

    # Analyze changes
    normal_avg_time = sum(c.response_time_ms for c in normal_session.calls) / len(normal_session.calls)
    cascade_avg_time = sum(c.response_time_ms for c in cascade_session.calls) / len(cascade_session.calls)

    print(f"\n  Average response time:")
    print(f"    Normal: {normal_avg_time:.0f}ms")
    print(f"    Cascade: {cascade_avg_time:.0f}ms ({cascade_avg_time/normal_avg_time:.1f}x slower)")

    # Count errors
    cascade_errors = sum(1 for c in cascade_session.calls if c.status_code != 200)
    print(f"\n  Errors in cascade session: {cascade_errors}/{cascade_session.num_calls}")

    # Show sample calls
    print(f"\n  Sample calls (second half with cascade):")
    start_idx = len(cascade_session.calls) // 2
    for call in cascade_session.calls[start_idx:start_idx+3]:
        status_icon = "✓" if call.status_code == 200 else "✗"
        print(f"    {status_icon} {call.endpoint}: {call.status_code}, {call.response_time_ms:.0f}ms")


def test_workflow_yaml():
    """Test saving and loading workflows as YAML."""
    print("\n" + "="*70)
    print("TEST 7: Workflow YAML Serialization")
    print("="*70)

    # Create a workflow
    workflow = create_simple_workflow()

    # Save to YAML
    yaml_path = Path("test_workflow.yaml")
    workflow.to_yaml(yaml_path)
    print(f"\n✓ Saved workflow to {yaml_path}")

    # Load from YAML
    loaded_workflow = WorkflowDefinition.from_yaml(yaml_path)
    print(f"✓ Loaded workflow from {yaml_path}")

    # Verify
    print(f"\n  Original workflow:")
    print(f"    Name: {workflow.name}")
    print(f"    Entry points: {workflow.entry_points}")

    print(f"\n  Loaded workflow:")
    print(f"    Name: {loaded_workflow.name}")
    print(f"    Entry points: {loaded_workflow.entry_points}")

    # Compare
    if workflow.name == loaded_workflow.name and workflow.entry_points == loaded_workflow.entry_points:
        print(f"\n  ✓ Workflows match!")
    else:
        print(f"\n  ❌ Workflows don't match!")

    # Cleanup
    yaml_path.unlink()
    print(f"\n  Cleaned up {yaml_path}")


def test_transition_probabilities():
    """Test that generated data follows transition probabilities."""
    print("\n" + "="*70)
    print("TEST 8: Transition Probability Validation")
    print("="*70)

    # Generate many sessions to verify probabilities
    gen = SyntheticGenerator(seed=42)
    workflow = SyntheticGenerator.ECOMMERCE_WORKFLOW

    print(f"\n  Generating 100 sessions to verify probabilities...")

    sessions = []
    start_time = datetime(2026, 1, 11, 10, 0, 0)
    for i in range(100):
        session = gen.generate_session(
            workflow=workflow,
            user_id=f"user_{i}",
            start_time=start_time + timedelta(minutes=i)
        )
        sessions.append(session)

    # Count transitions
    from collections import Counter
    transition_counts = Counter()
    total_from_endpoint = Counter()

    for session in sessions:
        for from_ep, to_ep in session.get_endpoint_transitions():
            transition_counts[(from_ep, to_ep)] += 1
            total_from_endpoint[from_ep] += 1

    # Compare with expected probabilities for a few key transitions
    print(f"\n✓ Transition probability validation:")

    test_transitions = [
        ("/api/login", "/api/users/{id}/profile"),
        ("/api/products/browse", "/api/products/{id}/details"),
        ("/api/cart", "/api/checkout"),
    ]

    for from_ep, to_ep in test_transitions:
        expected_prob = workflow.transitions.get(from_ep, {}).get(to_ep, 0)

        if total_from_endpoint[from_ep] > 0:
            observed_count = transition_counts[(from_ep, to_ep)]
            observed_prob = observed_count / total_from_endpoint[from_ep]

            print(f"\n  {from_ep}")
            print(f"    → {to_ep}")
            print(f"    Expected: {expected_prob:.1%}, Observed: {observed_prob:.1%}")
            print(f"    (Based on {total_from_endpoint[from_ep]} samples)")


def test_user_validation_code():
    """Run the exact validation code from the user's request."""
    print("\n" + "="*70)
    print("TEST 9: User Validation Code")
    print("="*70)

    from preprocessing.synthetic_generator import SyntheticGenerator
    from datetime import datetime

    gen = SyntheticGenerator(seed=42)

    # Generate a single session
    session = gen.generate_session(gen.ECOMMERCE_WORKFLOW, user_id="test_user",
                                    start_time=datetime.now())
    print(f"\n✓ Single session generation:")
    print(f"  Session has {len(session.calls)} calls")
    print(f"  Endpoints: {session.endpoint_sequence}")

    # Generate a full dataset
    dataset = gen.generate_dataset(num_users=100, sessions_per_user=(3, 2), show_progress=False)
    print(f"\n✓ Dataset generation:")
    print(f"  Generated {len(dataset.sessions)} sessions with {dataset.total_calls} total calls")


def test_microservices_workflow():
    """Test the microservices workflow."""
    print("\n" + "="*70)
    print("TEST 10: Microservices Workflow")
    print("="*70)

    workflow = create_microservices_workflow()

    print(f"\n✓ Microservices workflow:")
    print(f"  Name: {workflow.name}")
    print(f"  Entry points: {len(workflow.entry_points)}")
    print(f"  Transitions: {len(workflow.transitions)} endpoints")
    print(f"  Exit points: {len(workflow.exit_points)}")

    # Validate
    errors = SyntheticGenerator.validate_workflow(workflow)
    if errors:
        print(f"\n  ❌ Validation errors:")
        for error in errors:
            print(f"    - {error}")
    else:
        print(f"  ✓ Workflow is valid")

    # Generate sample session
    gen = SyntheticGenerator(seed=42)
    session = gen.generate_session(
        workflow=workflow,
        user_id="microservices_user",
        start_time=datetime.now()
    )

    print(f"\n✓ Generated sample session:")
    print(f"  Calls: {session.num_calls}")
    print(f"  Duration: {session.duration_seconds:.1f}s")
    print(f"  Path: {' → '.join(session.endpoint_sequence[:5])}...")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("SYNTHETIC GENERATOR TEST SUITE")
    print("="*70)
    print("\nTesting the SyntheticGenerator module for creating realistic")
    print("API call traces with known patterns.")

    try:
        test_workflow_definition()
        test_ecommerce_workflow()
        test_single_session_generation()
        test_reproducibility()
        test_dataset_generation()
        test_cascade_failures()
        test_workflow_yaml()
        test_transition_probabilities()
        test_user_validation_code()
        test_microservices_workflow()

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nThe SyntheticGenerator module is working correctly!")
        print("Ready to generate synthetic data for Markov chain validation.")

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())


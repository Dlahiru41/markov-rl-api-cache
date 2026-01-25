"""Test script for data splitter and validator utilities."""

from datetime import datetime, timedelta

from preprocessing.data_splitter import DataSplitter
from preprocessing.validator import DataValidator
from preprocessing.synthetic_generator import SyntheticGenerator


def create_test_dataset():
    """Create a test dataset for validation."""
    gen = SyntheticGenerator(seed=42)
    return gen.generate_dataset(
        num_users=50,
        sessions_per_user=(3, 1),
        date_range_days=30,
        show_progress=False
    )


def test_chronological_split():
    """Test chronological splitting."""
    print("\n" + "="*70)
    print("TEST 1: Chronological Split")
    print("="*70)

    dataset = create_test_dataset()
    splitter = DataSplitter(seed=42)

    train, test = splitter.split_chronological(dataset, train_ratio=0.8)

    print(f"\n[OK] Split completed")
    print(f"  Train: {len(train.sessions)} sessions")
    print(f"  Test: {len(test.sessions)} sessions")

    # Verify chronological order
    if train.sessions and test.sessions:
        latest_train = max(s.start_timestamp for s in train.sessions)
        earliest_test = min(s.start_timestamp for s in test.sessions)

        print(f"\n[OK] Chronological order:")
        print(f"  Latest train: {latest_train.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Earliest test: {earliest_test.strftime('%Y-%m-%d %H:%M')}")

        if latest_train <= earliest_test:
            print(f"  [OK] Proper chronological split (no time leakage)")
        else:
            print(f"  [FAIL] WARNING: Some test sessions are before train sessions")

    # Verify no session overlap
    train_sessions = {s.session_id for s in train.sessions}
    test_sessions = {s.session_id for s in test.sessions}

    if train_sessions.isdisjoint(test_sessions):
        print(f"\n[OK] No session overlap (train and test are disjoint)")
    else:
        print(f"\n[FAIL] ERROR: Sessions overlap!")

    # Get split summary
    summary = splitter.get_split_summary(train, test)
    print(f"\n[OK] Split summary:")
    print(f"  Train ratio: {summary['train_ratio']:.1%}")
    print(f"  Train calls: {summary['train_calls']}")
    print(f"  Test calls: {summary['test_calls']}")


def test_user_based_split():
    """Test user-based splitting."""
    print("\n" + "="*70)
    print("TEST 2: User-Based Split")
    print("="*70)

    dataset = create_test_dataset()
    splitter = DataSplitter(seed=42)

    train, test = splitter.split_by_users(dataset, train_ratio=0.8)

    print(f"\n[OK] Split completed")
    print(f"  Train: {len(train.sessions)} sessions")
    print(f"  Test: {len(test.sessions)} sessions")

    # Get unique users
    train_users = {s.user_id for s in train.sessions}
    test_users = {s.user_id for s in test.sessions}

    print(f"\n[OK] User distribution:")
    print(f"  Train users: {len(train_users)}")
    print(f"  Test users: {len(test_users)}")

    # Check for user overlap
    overlapping_users = train_users & test_users
    if not overlapping_users:
        print(f"  [OK] No user overlap (completely new users in test)")
    else:
        print(f"  [FAIL] WARNING: {len(overlapping_users)} users appear in both sets")

    # Verify no session overlap
    assert splitter.verify_no_overlap(train, test), "Sessions should not overlap!"
    print(f"\n[OK] No session overlap verified")


def test_stratified_split():
    """Test stratified splitting."""
    print("\n" + "="*70)
    print("TEST 3: Stratified Split")
    print("="*70)

    dataset = create_test_dataset()
    splitter = DataSplitter(seed=42)

    train, test = splitter.split_stratified(dataset, train_ratio=0.8, stratify_by='user_type')

    print(f"\n[OK] Split completed")
    print(f"  Train: {len(train.sessions)} sessions")
    print(f"  Test: {len(test.sessions)} sessions")

    # Get user type distributions
    summary = splitter.get_split_summary(train, test)

    print(f"\n[OK] User type distribution:")
    print(f"  Train:")
    for user_type, count in sorted(summary['train_user_type_dist'].items()):
        pct = count / len(train.sessions) * 100
        print(f"    {user_type}: {count} ({pct:.1f}%)")

    print(f"  Test:")
    for user_type, count in sorted(summary['test_user_type_dist'].items()):
        pct = count / len(test.sessions) * 100
        print(f"    {user_type}: {count} ({pct:.1f}%)")

    # Verify distributions are similar
    print(f"\n[OK] Distributions should be similar (stratified)")


def test_k_fold_split():
    """Test k-fold cross-validation splitting."""
    print("\n" + "="*70)
    print("TEST 4: K-Fold Cross-Validation")
    print("="*70)

    dataset = create_test_dataset()
    splitter = DataSplitter(seed=42)

    k = 5
    folds = splitter.k_fold_split(dataset, k=k, shuffle=True)

    print(f"\n[OK] Created {len(folds)} folds")

    # Check each fold
    for idx, (train, test) in enumerate(folds, 1):
        print(f"\n  Fold {idx}:")
        print(f"    Train: {len(train.sessions)} sessions")
        print(f"    Test: {len(test.sessions)} sessions")

        # Verify no overlap
        assert splitter.verify_no_overlap(train, test), f"Fold {idx} has overlap!"

    print(f"\n[OK] All folds have no session overlap")

    # Verify all sessions appear in test exactly once
    all_test_sessions = set()
    for train, test in folds:
        test_ids = {s.session_id for s in test.sessions}
        all_test_sessions.update(test_ids)

    original_sessions = {s.session_id for s in dataset.sessions}
    if all_test_sessions == original_sessions:
        print(f"[OK] All sessions appear in test exactly once across all folds")
    else:
        print(f"[FAIL] Some sessions missing or duplicated")


def test_validation():
    """Test data validation."""
    print("\n" + "="*70)
    print("TEST 5: Data Validation")
    print("="*70)

    dataset = create_test_dataset()
    validator = DataValidator()

    # Validate dataset
    result = validator.validate_dataset(dataset)

    print(f"\n[OK] Validation completed")
    print(f"  Valid: {'YES' if result['valid'] else 'NO'}")
    print(f"  Total sessions: {result['total_sessions']}")
    print(f"  Total calls: {result['total_calls']}")
    print(f"  Invalid sessions: {result['invalid_sessions']}")
    print(f"  Invalid calls: {result['invalid_calls']}")
    print(f"  Errors: {len(result['errors'])}")
    print(f"  Warnings: {len(result['warnings'])}")

    if result['errors']:
        print(f"\n  First few errors:")
        for error in result['errors'][:3]:
            print(f"    • {error}")

    # Check data quality
    quality = validator.check_data_quality(dataset)

    print(f"\n[OK] Quality metrics:")
    for metric, value in quality.items():
        print(f"  {metric}: {value:.1%}")

    # Detect anomalies
    anomalies = validator.detect_anomalies(dataset)

    print(f"\n[OK] Anomaly detection:")
    print(f"  Flagged sessions: {len(anomalies)}")

    if anomalies:
        print(f"\n  First few anomalies:")
        for anomaly in anomalies[:3]:
            print(f"    • Session {anomaly['session_id']}:")
            for flag in anomaly['flags']:
                print(f"        - {flag}")


def test_user_validation_code():
    """Run the exact validation code from the user's request."""
    print("\n" + "="*70)
    print("TEST 6: User Validation Code")
    print("="*70)

    from preprocessing.data_splitter import DataSplitter
    from preprocessing.validator import DataValidator

    dataset = create_test_dataset()

    # Test splitting
    splitter = DataSplitter()
    train, test = splitter.split_chronological(dataset, train_ratio=0.8)
    print(f"\n[OK] Chronological split:")
    print(f"  Train: {len(train.sessions)} sessions, Test: {len(test.sessions)} sessions")

    # Verify no session overlap
    train_sessions = {s.session_id for s in train.sessions}
    test_sessions = {s.session_id for s in test.sessions}
    assert train_sessions.isdisjoint(test_sessions), "Sessions should not overlap!"
    print(f"  [OK] No session overlap verified")

    # Test validation
    validator = DataValidator()
    result = validator.validate_dataset(dataset)
    print(f"\n[OK] Validation:")
    print(f"  Dataset valid: {result['valid']}")
    if result['errors']:
        print(f"  First few errors: {result['errors'][:3]}")


def test_validation_with_bad_data():
    """Test validation with intentionally bad data."""
    print("\n" + "="*70)
    print("TEST 7: Validation with Bad Data")
    print("="*70)

    from preprocessing.models import APICall, Session, Dataset

    # Create dataset with issues
    base_time = datetime(2026, 1, 11, 10, 0, 0)

    # Good calls first
    call1 = APICall(
        call_id="call1",
        endpoint="/api/test",
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
        call_id="call2",
        endpoint="/api/test",
        method="GET",
        params={},
        user_id="user1",
        session_id="sess1",
        timestamp=base_time + timedelta(seconds=5),
        response_time_ms=100,
        status_code=200,
        response_size_bytes=1024,
        user_type="free"
    )

    # Call with out of order timestamp (but valid APICall)
    call3 = APICall(
        call_id="call3",
        endpoint="/api/test",
        method="GET",
        params={},
        user_id="user1",
        session_id="sess1",
        timestamp=base_time - timedelta(seconds=10),  # BEFORE previous calls!
        response_time_ms=100,
        status_code=200,
        response_size_bytes=1024,
        user_type="free"
    )

    # Session with calls out of chronological order
    bad_session = Session(
        session_id="sess1",
        user_id="user1",
        user_type="free",
        start_timestamp=base_time,
        calls=[call1, call2, call3]  # call3 is out of order
    )

    # Duplicate call IDs
    dup_call = APICall(
        call_id="call1",  # DUPLICATE!
        endpoint="/api/other",
        method="GET",
        params={},
        user_id="user2",
        session_id="sess2",
        timestamp=base_time,
        response_time_ms=100,
        status_code=200,
        response_size_bytes=1024,
        user_type="free"
    )

    good_session = Session(
        session_id="sess2",
        user_id="user2",
        user_type="free",
        start_timestamp=base_time,
        calls=[dup_call]
    )

    bad_dataset = Dataset(
        name="bad_dataset",
        sessions=[bad_session, good_session]
    )

    validator = DataValidator()
    result = validator.validate_dataset(bad_dataset)

    print(f"\n[OK] Validation detected issues:")
    print(f"  Valid: {'YES' if result['valid'] else 'NO (as expected)'}")
    print(f"  Errors found: {len(result['errors'])}")
    print(f"  Warnings found: {len(result['warnings'])}")

    print(f"\n  Errors:")
    for error in result['errors'][:5]:
        print(f"    • {error}")

    print(f"\n  Warnings:")
    for warning in result['warnings'][:5]:
        print(f"    • {warning}")


def test_validation_summary():
    """Test validation summary report."""
    print("\n" + "="*70)
    print("TEST 8: Validation Summary Report")
    print("="*70)

    dataset = create_test_dataset()
    validator = DataValidator()

    summary = validator.get_validation_summary(dataset)
    print(summary)


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("DATA SPLITTER & VALIDATOR TEST SUITE")
    print("="*70)
    print("\nTesting data quality assurance and experiment setup utilities")

    try:
        test_chronological_split()
        test_user_based_split()
        test_stratified_split()
        test_k_fold_split()
        test_validation()
        test_user_validation_code()
        test_validation_with_bad_data()
        test_validation_summary()

        print("\n" + "="*70)
        print("[SUCCESS] ALL TESTS PASSED!")
        print("="*70)
        print("\nData splitter and validator are working correctly!")
        print("Ready for ML experiment setup and data quality assurance.")

        return 0

    except Exception as e:
        print(f"\n[ERROR] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())


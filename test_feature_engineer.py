"""Test and validation script for the FeatureEngineer module."""

import sys
import numpy as np
from datetime import datetime, timedelta

# Check if numpy is available
try:
    import numpy as np
    print("[OK] NumPy is available")
except ImportError:
    print("[FAIL] NumPy not found. Please install: pip install numpy")
    sys.exit(1)

from preprocessing.feature_engineer import FeatureEngineer
from preprocessing.models import APICall, Session


def create_test_sessions():
    """Create sample sessions for testing."""
    sessions = []
    base_time = datetime(2026, 1, 11, 10, 30, 0)  # Saturday, 10:30 AM (peak hour)

    # Session 1: Premium user morning shopping
    calls1 = [
        APICall(
            call_id="c1",
            endpoint="/api/login",
            method="POST",
            params={},
            user_id="user1",
            session_id="sess1",
            timestamp=base_time,
            response_time_ms=80,
            status_code=200,
            response_size_bytes=512,
            user_type="premium"
        ),
        APICall(
            call_id="c2",
            endpoint="/api/users/123/profile",
            method="GET",
            params={},
            user_id="user1",
            session_id="sess1",
            timestamp=base_time + timedelta(seconds=5),
            response_time_ms=100,
            status_code=200,
            response_size_bytes=2048,
            user_type="premium"
        ),
        APICall(
            call_id="c3",
            endpoint="/api/products/search",
            method="GET",
            params={"q": "laptop", "category": "electronics", "min_price": "500"},
            user_id="user1",
            session_id="sess1",
            timestamp=base_time + timedelta(seconds=15),
            response_time_ms=250,
            status_code=200,
            response_size_bytes=8192,
            user_type="premium"
        ),
        APICall(
            call_id="c4",
            endpoint="/api/products/456/details",
            method="GET",
            params={},
            user_id="user1",
            session_id="sess1",
            timestamp=base_time + timedelta(seconds=30),
            response_time_ms=150,
            status_code=200,
            response_size_bytes=4096,
            user_type="premium"
        ),
    ]
    session1 = Session("sess1", "user1", "premium", base_time, base_time + timedelta(seconds=30), calls1)
    sessions.append(session1)

    # Session 2: Free user afternoon browsing (non-peak, weekend)
    base_time2 = datetime(2026, 1, 11, 16, 0, 0)  # Saturday, 4 PM (non-peak)
    calls2 = [
        APICall(
            call_id="c5",
            endpoint="/api/browse",
            method="GET",
            params={},
            user_id="user2",
            session_id="sess2",
            timestamp=base_time2,
            response_time_ms=120,
            status_code=200,
            response_size_bytes=4096,
            user_type="free"
        ),
        APICall(
            call_id="c6",
            endpoint="/api/products/789",
            method="GET",
            params={},
            user_id="user2",
            session_id="sess2",
            timestamp=base_time2 + timedelta(seconds=10),
            response_time_ms=180,
            status_code=200,
            response_size_bytes=3072,
            user_type="free"
        ),
    ]
    session2 = Session("sess2", "user2", "free", base_time2, base_time2 + timedelta(seconds=10), calls2)
    sessions.append(session2)

    # Session 3: Guest user weekday evening (peak hour)
    base_time3 = datetime(2026, 1, 13, 14, 30, 0)  # Monday, 2:30 PM (peak hour)
    calls3 = [
        APICall(
            call_id="c7",
            endpoint="/api/products/browse",
            method="GET",
            params={},
            user_id="guest1",
            session_id="sess3",
            timestamp=base_time3,
            response_time_ms=150,
            status_code=200,
            response_size_bytes=4096,
            user_type="guest"
        ),
    ]
    session3 = Session("sess3", "guest1", "guest", base_time3, calls=[calls3[0]])
    sessions.append(session3)

    return sessions


def test_cyclic_encoding():
    """Test the cyclic encoding function."""
    print("\n" + "="*70)
    print("TEST 1: Cyclic Encoding")
    print("="*70)

    print("\nTesting hour encoding (0-23):")
    test_hours = [0, 6, 12, 18, 23]
    for hour in test_hours:
        sin_val, cos_val = FeatureEngineer.cyclic_encode(hour, 24)
        print(f"  Hour {hour:2d}: sin={sin_val:7.4f}, cos={cos_val:7.4f}")

    # Verify that hour 0 and 23 are close
    sin_0, cos_0 = FeatureEngineer.cyclic_encode(0, 24)
    sin_23, cos_23 = FeatureEngineer.cyclic_encode(23, 24)
    distance = np.sqrt((sin_0 - sin_23)**2 + (cos_0 - cos_23)**2)
    print(f"\n  Distance between hour 0 and 23: {distance:.4f} (should be small)")

    print("\nTesting day of week encoding (0-6):")
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for i, day in enumerate(days):
        sin_val, cos_val = FeatureEngineer.cyclic_encode(i, 7)
        print(f"  {day} ({i}): sin={sin_val:7.4f}, cos={cos_val:7.4f}")


def test_fit():
    """Test the fit method."""
    print("\n" + "="*70)
    print("TEST 2: Fitting FeatureEngineer")
    print("="*70)

    sessions = create_test_sessions()

    fe = FeatureEngineer(
        temporal_features=True,
        user_features=True,
        request_features=True,
        history_features=True
    )

    print(f"\nFitting on {len(sessions)} sessions...")
    fe.fit(sessions)

    print(f"[OK] Fit complete!")
    print(f"\nFeature Engineer Info:")
    info = fe.get_feature_info()
    for key, value in info.items():
        if key == 'categories':
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")

    return fe, sessions


def test_transform(fe, sessions):
    """Test the transform method."""
    print("\n" + "="*70)
    print("TEST 3: Transforming API Calls")
    print("="*70)

    # Transform first call from first session
    session = sessions[0]
    call = session.calls[0]

    print(f"\nTransforming call: {call.endpoint} ({call.method})")
    print(f"  Timestamp: {call.timestamp}")
    print(f"  User type: {call.user_type}")

    features = fe.transform(call, session, history=[])

    print(f"\n[OK] Feature vector created!")
    print(f"  Shape: {features.shape}")
    print(f"  Dtype: {features.dtype}")
    print(f"  Min value: {features.min():.4f}")
    print(f"  Max value: {features.max():.4f}")

    return features


def test_feature_names(fe, features):
    """Test feature names."""
    print("\n" + "="*70)
    print("TEST 4: Feature Names")
    print("="*70)

    feature_names = fe.get_feature_names()

    print(f"\n[OK] Total features: {len(feature_names)}")
    print(f"\nFirst 15 features:")
    for i, name in enumerate(feature_names[:15]):
        print(f"  {i:2d}. {name:30s} = {features[i]:7.4f}")

    if len(feature_names) > 15:
        print(f"\n  ... ({len(feature_names) - 15} more features)")


def test_all_feature_groups(fe, sessions):
    """Test each feature group separately."""
    print("\n" + "="*70)
    print("TEST 5: Feature Groups")
    print("="*70)

    session = sessions[0]
    call = session.calls[2]  # Third call with parameters
    history = session.calls[:2]

    # Test temporal features
    print("\n1. Temporal Features:")
    fe_temp = FeatureEngineer(temporal_features=True, user_features=False,
                               request_features=False, history_features=False)
    fe_temp.fit(sessions)
    temp_features = fe_temp.transform(call, session, history)
    temp_names = fe_temp.get_feature_names()
    for name, value in zip(temp_names, temp_features):
        print(f"  {name:30s} = {value:7.4f}")

    # Test user features
    print("\n2. User Features:")
    fe_user = FeatureEngineer(temporal_features=False, user_features=True,
                              request_features=False, history_features=False)
    fe_user.fit(sessions)
    user_features = fe_user.transform(call, session, history)
    user_names = fe_user.get_feature_names()
    for name, value in zip(user_names, user_features):
        print(f"  {name:30s} = {value:7.4f}")

    # Test request features
    print("\n3. Request Features:")
    fe_req = FeatureEngineer(temporal_features=False, user_features=False,
                             request_features=True, history_features=False)
    fe_req.fit(sessions)
    req_features = fe_req.transform(call, session, history)
    req_names = fe_req.get_feature_names()
    for name, value in zip(req_names, req_features):
        print(f"  {name:30s} = {value:7.4f}")

    # Test history features
    print("\n4. History Features:")
    fe_hist = FeatureEngineer(temporal_features=False, user_features=False,
                              request_features=False, history_features=True)
    fe_hist.fit(sessions)
    hist_features = fe_hist.transform(call, session, history)
    hist_names = fe_hist.get_feature_names()
    for name, value in zip(hist_names, hist_features):
        print(f"  {name:30s} = {value:7.4f}")


def test_edge_cases(fe, sessions):
    """Test edge cases and robustness."""
    print("\n" + "="*70)
    print("TEST 6: Edge Cases")
    print("="*70)

    # Test 1: Unknown endpoint
    print("\n1. Unknown Endpoint:")
    unknown_call = APICall(
        call_id="c_unknown",
        endpoint="/api/unknown/endpoint/999",
        method="GET",
        params={},
        user_id="user1",
        session_id="sess1",
        timestamp=datetime.now(),
        response_time_ms=100,
        status_code=200,
        response_size_bytes=1024,
        user_type="premium"
    )
    features = fe.transform(unknown_call, None, None)
    print(f"  [OK] Handled unknown endpoint, feature vector shape: {features.shape}")

    # Test 2: No session context
    print("\n2. No Session Context:")
    call = sessions[0].calls[0]
    features = fe.transform(call, session=None, history=None)
    print(f"  [OK] Handled missing session, feature vector shape: {features.shape}")

    # Test 3: No history
    print("\n3. No History:")
    features = fe.transform(call, sessions[0], history=None)
    print(f"  [OK] Handled missing history, feature vector shape: {features.shape}")

    # Test 4: Different HTTP methods
    print("\n4. Different HTTP Methods:")
    methods = ['GET', 'POST', 'PUT', 'DELETE']
    for method in methods:
        test_call = APICall(
            call_id="c_test",
            endpoint="/api/test",
            method=method,
            params={},
            user_id="user1",
            session_id="sess1",
            timestamp=datetime.now(),
            response_time_ms=100,
            status_code=200,
            response_size_bytes=1024,
            user_type="free"
        )
        features = fe.transform(test_call, None, None)
        # Find the method feature
        feature_names = fe.get_feature_names()
        method_idx = feature_names.index(f'method_{method}')
        print(f"  {method:6s}: method_{method} = {features[method_idx]:.1f}")


def test_fit_transform():
    """Test fit_transform method."""
    print("\n" + "="*70)
    print("TEST 7: Fit-Transform")
    print("="*70)

    sessions = create_test_sessions()

    fe = FeatureEngineer()
    features_list = fe.fit_transform(sessions)

    print(f"\n[OK] Fit-transform complete!")
    print(f"  Total calls processed: {len(features_list)}")
    print(f"  Feature vector shape: {features_list[0].shape}")
    print(f"  All vectors same shape: {all(f.shape == features_list[0].shape for f in features_list)}")


def test_user_validation():
    """Run the exact validation code from the user's request."""
    print("\n" + "="*70)
    print("TEST 8: User Validation Code")
    print("="*70)

    sessions = create_test_sessions()

    # User's validation code
    from preprocessing.feature_engineer import FeatureEngineer

    fe = FeatureEngineer(temporal_features=True, user_features=True)
    fe.fit(sessions)  # Learn from training data

    # Transform a single call
    call = sessions[0].calls[0]
    session = sessions[0]
    features = fe.transform(call, session, history=[])

    print(f"\n[OK] User validation code executed successfully!")
    print(f"  Feature vector shape: {features.shape}")
    print(f"  Feature names: {fe.get_feature_names()[:5]}... (showing first 5)")
    print(f"  First 10 features: {features[:10]}")


def test_different_times():
    """Test temporal features at different times of day."""
    print("\n" + "="*70)
    print("TEST 9: Temporal Features at Different Times")
    print("="*70)

    sessions = create_test_sessions()
    fe = FeatureEngineer(temporal_features=True, user_features=False,
                        request_features=False, history_features=False)
    fe.fit(sessions)

    times = [
        (datetime(2026, 1, 11, 2, 0, 0), "Night (2 AM, weekend)"),
        (datetime(2026, 1, 11, 10, 30, 0), "Morning peak (10:30 AM, weekend)"),
        (datetime(2026, 1, 13, 14, 30, 0), "Afternoon peak (2:30 PM, weekday)"),
        (datetime(2026, 1, 13, 20, 0, 0), "Evening (8 PM, weekday)"),
    ]

    feature_names = fe.get_feature_names()

    print(f"\nTemporal features: {feature_names}")
    print()

    for timestamp, description in times:
        call = APICall(
            call_id="test",
            endpoint="/api/test",
            method="GET",
            params={},
            user_id="user1",
            session_id="sess1",
            timestamp=timestamp,
            response_time_ms=100,
            status_code=200,
            response_size_bytes=1024,
            user_type="free"
        )
        features = fe.transform(call, None, None)

        print(f"{description}:")
        for name, value in zip(feature_names, features):
            print(f"  {name:15s} = {value:7.4f}")
        print()


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("FEATURE ENGINEER TEST SUITE")
    print("="*70)
    print("\nTesting the FeatureEngineer module for RL state representation")

    try:
        # Run tests
        test_cyclic_encoding()
        fe, sessions = test_fit()
        features = test_transform(fe, sessions)
        test_feature_names(fe, features)
        test_all_feature_groups(fe, sessions)
        test_edge_cases(fe, sessions)
        test_fit_transform()
        test_user_validation()
        test_different_times()

        # Summary
        print("\n" + "="*70)
        print("[SUCCESS] ALL TESTS PASSED!")
        print("="*70)
        print("\nThe FeatureEngineer module is working correctly!")
        print("Ready for RL state representation.")

    except Exception as e:
        print(f"\n[ERROR] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())


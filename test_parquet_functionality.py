"""Test parquet save/load functionality."""

from preprocessing.models import APICall, Session, Dataset
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

print("Testing Parquet Save/Load Functionality")
print("=" * 60)

# Create a dataset with multiple sessions
sessions = []
for i in range(5):
    sess = Session(
        session_id=f"sess_{i}",
        user_id=f"user_{i % 2}",  # 2 unique users
        user_type=["free", "premium"][i % 2],
        start_timestamp=datetime.now() - timedelta(hours=5-i)
    )

    # Add calls to each session
    for j in range(3):
        call = APICall(
            call_id=f"call_{i}_{j}",
            endpoint=f"/api/service_{j}",
            method=["GET", "POST"][j % 2],
            params={"param": f"value_{j}"},
            user_id=sess.user_id,
            session_id=sess.session_id,
            timestamp=sess.start_timestamp + timedelta(seconds=j*10),
            response_time_ms=50.0 + j * 10,
            status_code=200,
            response_size_bytes=1000 + j * 100,
            user_type=sess.user_type
        )
        sess.append_call(call)

    sessions.append(sess)

dataset = Dataset(
    name="parquet_test_dataset",
    sessions=sessions,
    metadata={"version": "1.0", "test": True}
)

print(f"Original dataset: {dataset}")
print(f"  Sessions: {len(dataset.sessions)}")
print(f"  Total calls: {dataset.total_calls}")
print(f"  Unique users: {dataset.num_unique_users}")
print()

# Save to temporary parquet file
with tempfile.TemporaryDirectory() as tmpdir:
    filepath = Path(tmpdir) / "test_dataset.parquet"

    print(f"Saving to: {filepath}")
    dataset.save_to_parquet(filepath)
    print(f"  [OK] File saved successfully")
    print(f"  [OK] File size: {filepath.stat().st_size} bytes")
    print()

    # Load from parquet
    print(f"Loading from: {filepath}")
    loaded_dataset = Dataset.load_from_parquet(filepath)
    print(f"  [OK] Dataset loaded successfully")
    print()

    # Verify loaded data
    print("Verification:")
    print(f"  Original sessions: {len(dataset.sessions)}")
    print(f"  Loaded sessions: {len(loaded_dataset.sessions)}")
    assert len(dataset.sessions) == len(loaded_dataset.sessions), "Session count mismatch!"

    print(f"  Original calls: {dataset.total_calls}")
    print(f"  Loaded calls: {loaded_dataset.total_calls}")
    assert dataset.total_calls == loaded_dataset.total_calls, "Call count mismatch!"

    print(f"  Original users: {dataset.num_unique_users}")
    print(f"  Loaded users: {loaded_dataset.num_unique_users}")
    assert dataset.num_unique_users == loaded_dataset.num_unique_users, "User count mismatch!"

    print(f"  Original endpoints: {dataset.unique_endpoints}")
    print(f"  Loaded endpoints: {loaded_dataset.unique_endpoints}")
    assert dataset.unique_endpoints == loaded_dataset.unique_endpoints, "Endpoint mismatch!"

    # Check a specific session
    orig_session = dataset.sessions[0]
    loaded_session = loaded_dataset.sessions[0]

    print(f"\n  Checking first session:")
    print(f"    Original: {orig_session.session_id}, {orig_session.num_calls} calls")
    print(f"    Loaded: {loaded_session.session_id}, {loaded_session.num_calls} calls")
    assert orig_session.session_id == loaded_session.session_id, "Session ID mismatch!"
    assert orig_session.num_calls == loaded_session.num_calls, "Call count in session mismatch!"

    # Check a specific call
    orig_call = orig_session.calls[0]
    loaded_call = loaded_session.calls[0]

    print(f"\n  Checking first call:")
    print(f"    Original: {orig_call.call_id}, {orig_call.endpoint}")
    print(f"    Loaded: {loaded_call.call_id}, {loaded_call.endpoint}")
    assert orig_call.call_id == loaded_call.call_id, "Call ID mismatch!"
    assert orig_call.endpoint == loaded_call.endpoint, "Endpoint mismatch!"
    assert orig_call.method == loaded_call.method, "Method mismatch!"
    assert orig_call.status_code == loaded_call.status_code, "Status code mismatch!"

    print()

print("=" * 60)
print("[OK] ALL PARQUET TESTS PASSED!")
print("=" * 60)


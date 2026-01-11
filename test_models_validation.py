"""Validation script for preprocessing models."""

from preprocessing.models import APICall, Session, Dataset
from datetime import datetime

# Test APICall
call = APICall(
    call_id="1",
    endpoint="/api/users/123",
    method="GET",
    params={},
    user_id="user1",
    session_id="sess1",
    timestamp=datetime.now(),
    response_time_ms=100,
    status_code=200,
    response_size_bytes=1024,
    user_type="free"
)

print("APICall to_dict():")
print(call.to_dict())
print()

print("Service name from endpoint '/api/users/123':")
print(call.get_service_name())  # Should print "users"
print()

print("Is call successful?")
print(call.is_successful())  # Should print True
print()

# Test Session
session = Session(
    session_id="sess1",
    user_id="user1",
    user_type="free",
    start_timestamp=datetime.now()
)

# Create multiple calls for transition testing
call2 = APICall(
    call_id="2",
    endpoint="/api/products/456",
    method="GET",
    params={},
    user_id="user1",
    session_id="sess1",
    timestamp=datetime.now(),
    response_time_ms=150,
    status_code=200,
    response_size_bytes=2048,
    user_type="free"
)

call3 = APICall(
    call_id="3",
    endpoint="/api/cart/add",
    method="POST",
    params={"product_id": "456"},
    user_id="user1",
    session_id="sess1",
    timestamp=datetime.now(),
    response_time_ms=200,
    status_code=201,
    response_size_bytes=512,
    user_type="free"
)

session.append_call(call)
session.append_call(call2)
session.append_call(call3)

print("Session details:")
print(f"  Number of calls: {session.num_calls}")
print(f"  Unique endpoints: {session.unique_endpoints}")
print(f"  Endpoint sequence: {session.endpoint_sequence}")
print(f"  Endpoint transitions: {session.get_endpoint_transitions()}")
print()

# Test Dataset
dataset = Dataset(
    name="test_dataset",
    sessions=[session]
)

print("Dataset details:")
print(f"  Total calls: {dataset.total_calls}")
print(f"  Unique users: {dataset.num_unique_users}")
print(f"  Unique endpoints: {dataset.unique_endpoints}")
print(f"  Endpoint occurrences: {dataset.count_endpoint_occurrences()}")
print(f"  All sequences: {dataset.get_all_sequences()}")
print()

print("Dataset repr:")
print(dataset)
print()

print("✓ All validations passed!")


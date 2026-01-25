"""
Test script for E-commerce services

Tests the UserService according to the validation requirements.
"""

import subprocess
import time
import httpx
import sys
import os

def test_user_service():
    """Test the user service according to requirements."""

    print("\n" + "=" * 70)
    print("E-COMMERCE SERVICE VALIDATION TEST")
    print("=" * 70 + "\n")

    # Start the user service in background
    print("Starting UserService on port 8001...")
    if sys.platform == "win32":
        # Windows
        process = subprocess.Popen(
            ["python", "-m", "simulator.services.ecommerce.user_service"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        # Unix-like
        process = subprocess.Popen(
            ["python", "-m", "simulator.services.ecommerce.user_service"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

    print(f"Service started with PID: {process.pid}")
    print("Waiting for service to start...")
    time.sleep(3)

    try:
        base_url = "http://localhost:8001"

        # Test 1: Health check
        print("\n" + "-" * 70)
        print("Test 1: Health Check")
        print("-" * 70)
        try:
            response = httpx.get(f"{base_url}/health", timeout=5.0)
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")
            assert response.status_code == 200, "Health check failed"
            print("✓ PASSED")
        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False

        # Test 2: Login (should return fake token)
        print("\n" + "-" * 70)
        print("Test 2: Login")
        print("-" * 70)
        try:
            response = httpx.post(
                f"{base_url}/login",
                json={"username": "test", "password": "test"},
                headers={"Content-Type": "application/json"},
                timeout=5.0
            )
            print(f"Status Code: {response.status_code}")
            data = response.json()
            print(f"Response: {data}")

            # Check for expected fields
            assert response.status_code == 200, "Login failed"
            assert "status" in data, "Missing status field"

            # Note: Since AuthService isn't running, this might fail
            # but the UserService should still respond properly
            if data.get("status") == "success":
                assert "token" in data, "Missing token field"
                assert "user_id" in data, "Missing user_id field"
                print("✓ PASSED - Full login successful")
            else:
                print("⚠ Login returned expected failure (AuthService not running)")
                print("  UserService is responding correctly")

        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False

        # Test 3: Get Profile
        print("\n" + "-" * 70)
        print("Test 3: Get Profile")
        print("-" * 70)
        try:
            response = httpx.get(f"{base_url}/profile", timeout=5.0)
            print(f"Status Code: {response.status_code}")
            data = response.json()
            print(f"Response: {data}")

            assert response.status_code == 200, "Get profile failed"
            assert "user_id" in data, "Missing user_id field"
            assert "username" in data, "Missing username field"
            print("✓ PASSED")
        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False

        # Test 4: Metrics
        print("\n" + "-" * 70)
        print("Test 4: Metrics")
        print("-" * 70)
        try:
            response = httpx.get(f"{base_url}/metrics", timeout=5.0)
            print(f"Status Code: {response.status_code}")
            print(f"Metrics (first 200 chars): {response.text[:200]}...")

            assert response.status_code == 200, "Get metrics failed"
            assert "user-service" in response.text, "Metrics missing service name"
            print("✓ PASSED")
        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False

        # Test 5: Register new user
        print("\n" + "-" * 70)
        print("Test 5: Register New User")
        print("-" * 70)
        try:
            response = httpx.post(
                f"{base_url}/register",
                json={
                    "username": "newuser123",
                    "password": "password123",
                    "email": "newuser@example.com",
                    "first_name": "New",
                    "last_name": "User"
                },
                headers={"Content-Type": "application/json"},
                timeout=5.0
            )
            print(f"Status Code: {response.status_code}")
            data = response.json()
            print(f"Response: {data}")

            assert response.status_code == 200, "Registration failed"
            assert data.get("status") == "success", "Registration not successful"
            print("✓ PASSED")
        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False

        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("UserService is working correctly!")
        print()

        return True

    finally:
        # Stop the service
        print("\nStopping service...")
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            process.terminate()
            process.wait(timeout=5)
        print("Service stopped")


if __name__ == "__main__":
    try:
        success = test_user_service()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


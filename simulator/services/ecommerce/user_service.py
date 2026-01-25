"""
User Service - User management and authentication

Handles user registration, login, profile management.
Characteristics: Medium latency (~100ms), session management
Dependencies: AuthService for token validation
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from simulator.services.base_service import BaseService, ServiceConfig, EndpointConfig
from faker import Faker
from fastapi import Request, HTTPException
import hashlib
from typing import Dict, Optional

fake = Faker()


class UserService(BaseService):
    """User service for authentication and profile management.

    Medium latency service with session management capabilities.
    """

    def __init__(self):
        """Initialize the user service."""
        config = ServiceConfig(
            name="user-service",
            port=8001,
            base_latency_ms=100,
            latency_std_ms=20,
            failure_rate=0.01,
            timeout_rate=0.002,
            dependencies=["auth-service"],
            endpoints=[
                EndpointConfig(
                    path="/login",
                    method="POST",
                    response_size_bytes=512,
                    latency_multiplier=1.2,
                    dependencies=["auth-service:/internal/create-token"],
                    description="Authenticate user"
                ),
                EndpointConfig(
                    path="/profile",
                    method="GET",
                    response_size_bytes=1024,
                    latency_multiplier=0.8,
                    description="Get user profile"
                ),
                EndpointConfig(
                    path="/profile",
                    method="PUT",
                    response_size_bytes=512,
                    latency_multiplier=1.0,
                    description="Update user profile"
                ),
                EndpointConfig(
                    path="/register",
                    method="POST",
                    response_size_bytes=768,
                    latency_multiplier=1.5,
                    description="Create new user"
                ),
            ]
        )

        super().__init__(config)

        # In-memory user database
        self.users: Dict[str, dict] = {}

        # Create some fake users for testing
        self._create_fake_users()

        # Register custom endpoints
        self._register_user_endpoints()

        # Register with auth service
        self.register_service("auth-service", "http://localhost:8002")

        self.logger.info(f"UserService initialized with {len(self.users)} fake users")

    def _create_fake_users(self):
        """Create some fake users for testing."""
        # Create test user
        self.users["test"] = {
            "user_id": "test_user_001",
            "username": "test",
            "password_hash": self._hash_password("test"),
            "email": "test@example.com",
            "first_name": "Test",
            "last_name": "User",
            "address": "123 Test St, Test City, TC 12345",
            "phone": "+1-555-TEST",
            "created_at": "2025-01-01T00:00:00"
        }

        # Create additional fake users
        for i in range(10):
            username = fake.user_name()
            self.users[username] = {
                "user_id": f"user_{i:03d}",
                "username": username,
                "password_hash": self._hash_password("password"),
                "email": fake.email(),
                "first_name": fake.first_name(),
                "last_name": fake.last_name(),
                "address": fake.address().replace('\n', ', '),
                "phone": fake.phone_number(),
                "created_at": fake.date_time_this_year().isoformat()
            }

    def _hash_password(self, password: str) -> str:
        """Hash a password using SHA-256.

        Args:
            password: Plain text password

        Returns:
            Hashed password
        """
        return hashlib.sha256(password.encode()).hexdigest()

    def _register_user_endpoints(self):
        """Register user-specific endpoints."""

        @self.app.post("/login")
        async def login(request: Request):
            """Authenticate a user and return a token.

            Request body:
                {
                    "username": "test",
                    "password": "test"
                }

            Response:
                {
                    "status": "success",
                    "user_id": "user123",
                    "username": "test",
                    "token": "abc123...",
                    "refresh_token": "refresh_xyz...",
                    "expires_at": "2026-01-26T10:00:00"
                }
            """
            try:
                data = await request.json()
                username = data.get("username", "")
                password = data.get("password", "")

                if not username or not password:
                    return {
                        "status": "failed",
                        "error": "Missing username or password"
                    }

                # Check if user exists
                user = self.users.get(username)
                if not user:
                    self.logger.warning(f"Login attempt for non-existent user: {username}")
                    return {
                        "status": "failed",
                        "error": "Invalid credentials"
                    }

                # Verify password
                password_hash = self._hash_password(password)
                if user["password_hash"] != password_hash:
                    self.logger.warning(f"Invalid password for user: {username}")
                    return {
                        "status": "failed",
                        "error": "Invalid credentials"
                    }

                # Call auth service to create token
                try:
                    token_response = await self.call_service(
                        "auth-service",
                        "/internal/create-token",
                        method="POST",
                        json_data={
                            "user_id": user["user_id"],
                            "username": user["username"]
                        }
                    )

                    if token_response.get("status") == "success":
                        self.logger.info(f"User {username} logged in successfully")
                        return {
                            "status": "success",
                            "user_id": user["user_id"],
                            "username": user["username"],
                            "token": token_response["token"],
                            "refresh_token": token_response["refresh_token"],
                            "expires_at": token_response["expires_at"]
                        }
                    else:
                        return {
                            "status": "failed",
                            "error": "Token generation failed"
                        }

                except Exception as e:
                    self.logger.error(f"Error calling auth service: {e}")
                    return {
                        "status": "failed",
                        "error": "Authentication service unavailable"
                    }

            except Exception as e:
                self.logger.error(f"Error during login: {e}")
                return {
                    "status": "failed",
                    "error": "Login failed"
                }

        @self.app.get("/profile")
        async def get_profile(user_id: Optional[str] = None):
            """Get user profile information.

            Query params:
                user_id: User ID (optional, defaults to test user)

            Response:
                {
                    "user_id": "user123",
                    "username": "john_doe",
                    "email": "john@example.com",
                    "first_name": "John",
                    "last_name": "Doe",
                    "address": "123 Main St",
                    "phone": "+1-555-0123",
                    "created_at": "2025-01-01T00:00:00"
                }
            """
            try:
                # If no user_id provided, use test user
                if not user_id:
                    user_id = "test_user_001"

                # Find user by user_id
                user = None
                for u in self.users.values():
                    if u["user_id"] == user_id:
                        user = u
                        break

                if not user:
                    return {
                        "error": "User not found",
                        "status": "failed"
                    }

                # Return profile (exclude password hash)
                profile = {k: v for k, v in user.items() if k != "password_hash"}
                return profile

            except Exception as e:
                self.logger.error(f"Error getting profile: {e}")
                return {
                    "error": "Failed to get profile",
                    "status": "failed"
                }

        @self.app.put("/profile")
        async def update_profile(request: Request):
            """Update user profile information.

            Request body:
                {
                    "user_id": "user123",
                    "email": "newemail@example.com",
                    "first_name": "John",
                    "last_name": "Doe",
                    "address": "456 New St",
                    "phone": "+1-555-9999"
                }

            Response:
                {
                    "status": "success",
                    "message": "Profile updated",
                    "user_id": "user123"
                }
            """
            try:
                data = await request.json()
                user_id = data.get("user_id")

                if not user_id:
                    return {
                        "status": "failed",
                        "error": "Missing user_id"
                    }

                # Find user
                user = None
                for u in self.users.values():
                    if u["user_id"] == user_id:
                        user = u
                        break

                if not user:
                    return {
                        "status": "failed",
                        "error": "User not found"
                    }

                # Update fields
                updatable_fields = ["email", "first_name", "last_name", "address", "phone"]
                for field in updatable_fields:
                    if field in data:
                        user[field] = data[field]

                self.logger.info(f"Profile updated for user {user_id}")

                return {
                    "status": "success",
                    "message": "Profile updated",
                    "user_id": user_id
                }

            except Exception as e:
                self.logger.error(f"Error updating profile: {e}")
                return {
                    "status": "failed",
                    "error": "Update failed"
                }

        @self.app.post("/register")
        async def register(request: Request):
            """Register a new user.

            Request body:
                {
                    "username": "newuser",
                    "password": "password123",
                    "email": "user@example.com",
                    "first_name": "John",
                    "last_name": "Doe"
                }

            Response:
                {
                    "status": "success",
                    "user_id": "user123",
                    "username": "newuser",
                    "message": "User registered successfully"
                }
            """
            try:
                data = await request.json()
                username = data.get("username")
                password = data.get("password")
                email = data.get("email")

                if not username or not password or not email:
                    return {
                        "status": "failed",
                        "error": "Missing required fields"
                    }

                # Check if username already exists
                if username in self.users:
                    return {
                        "status": "failed",
                        "error": "Username already exists"
                    }

                # Create new user
                user_id = f"user_{len(self.users):03d}"
                self.users[username] = {
                    "user_id": user_id,
                    "username": username,
                    "password_hash": self._hash_password(password),
                    "email": email,
                    "first_name": data.get("first_name", ""),
                    "last_name": data.get("last_name", ""),
                    "address": data.get("address", ""),
                    "phone": data.get("phone", ""),
                    "created_at": fake.date_time_this_year().isoformat()
                }

                self.logger.info(f"New user registered: {username}")

                return {
                    "status": "success",
                    "user_id": user_id,
                    "username": username,
                    "message": "User registered successfully"
                }

            except Exception as e:
                self.logger.error(f"Error during registration: {e}")
                return {
                    "status": "failed",
                    "error": "Registration failed"
                }


def main():
    """Run the user service."""
    service = UserService()
    service.run()


if __name__ == "__main__":
    main()


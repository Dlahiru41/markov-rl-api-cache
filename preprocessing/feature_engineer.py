"""Feature engineering module for extracting numerical features from API calls.

This module converts API calls and their context into fixed-size feature vectors
suitable for reinforcement learning state representation. It follows the sklearn
fit/transform pattern for consistent encoding between training and inference.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import math

from preprocessing.models import APICall, Session


class FeatureEngineer:
    """Extracts and encodes features from API calls for RL state representation.

    Follows the sklearn fit/transform pattern:
    - fit() learns parameters from training data (e.g., normalization stats, vocabularies)
    - transform() converts API calls into feature vectors using learned parameters

    This ensures consistent encoding between training and inference.

    Features extracted:
    - Temporal: hour (cyclic), day of week (cyclic), weekend flag, peak hour flag
    - User: user type (one-hot), session progress, session duration
    - Request: HTTP method (one-hot), endpoint category, number of parameters
    - History: number of previous calls, time since session start

    All features are normalized to roughly [0, 1] or [-1, 1] range for neural networks.
    """

    # Peak hours definition (10-12 and 14-16)
    PEAK_HOURS = {10, 11, 14, 15}

    # Supported HTTP methods
    HTTP_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']

    # User types
    USER_TYPES = ['premium', 'free', 'guest']

    # Maximum values for normalization
    MAX_SESSION_DURATION_SECONDS = 1800  # 30 minutes
    MAX_PARAMS = 20
    MAX_HISTORY_LENGTH = 50

    def __init__(
        self,
        temporal_features: bool = True,
        user_features: bool = True,
        request_features: bool = True,
        history_features: bool = True,
        normalize_endpoints: bool = True
    ):
        """Initialize the FeatureEngineer.

        Args:
            temporal_features: Include time-based features
            user_features: Include user-related features
            request_features: Include request-specific features
            history_features: Include session history features
            normalize_endpoints: Use normalized endpoint representation
        """
        self.temporal_features = temporal_features
        self.user_features = user_features
        self.request_features = request_features
        self.history_features = history_features
        self.normalize_endpoints = normalize_endpoints

        # Learned parameters (populated during fit)
        self.endpoint_vocab: Dict[str, int] = {}
        self.endpoint_categories: Dict[str, str] = {}
        self.category_vocab: Dict[str, int] = {}
        self.is_fitted = False

        # Feature names (populated during fit)
        self._feature_names: List[str] = []

    def fit(self, sessions: List[Session]) -> 'FeatureEngineer':
        """Learn parameters from training data.

        This method:
        - Builds vocabulary of endpoints seen during training
        - Extracts endpoint categories (service names)
        - Builds vocabulary of categories
        - Computes any normalization statistics needed

        Args:
            sessions: List of Session objects to learn from

        Returns:
            self (for method chaining)
        """
        # Extract all endpoints and their categories
        endpoints = set()
        categories = set()

        for session in sessions:
            for call in session.calls:
                endpoint = self._normalize_endpoint(call.endpoint) if self.normalize_endpoints else call.endpoint
                endpoints.add(endpoint)

                # Extract category (service name)
                category = self._extract_category(endpoint)
                categories.add(category)
                self.endpoint_categories[endpoint] = category

        # Build vocabularies (sorted for consistency)
        self.endpoint_vocab = {endpoint: idx for idx, endpoint in enumerate(sorted(endpoints))}
        self.category_vocab = {category: idx for idx, category in enumerate(sorted(categories))}

        # Build feature names
        self._build_feature_names()

        self.is_fitted = True
        return self

    def transform(
        self,
        call: APICall,
        session: Optional[Session] = None,
        history: Optional[List[APICall]] = None
    ) -> np.ndarray:
        """Transform an API call into a feature vector.

        Args:
            call: The API call to extract features from
            session: The session this call belongs to (optional but recommended)
            history: List of previous calls in the session (optional)

        Returns:
            Feature vector as numpy array

        Raises:
            ValueError: If called before fit()
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform()")

        features = []

        # Temporal features
        if self.temporal_features:
            features.extend(self._extract_temporal_features(call))

        # User features
        if self.user_features:
            features.extend(self._extract_user_features(call, session, history))

        # Request features
        if self.request_features:
            features.extend(self._extract_request_features(call))

        # History features
        if self.history_features:
            features.extend(self._extract_history_features(call, session, history))

        return np.array(features, dtype=np.float32)

    def fit_transform(
        self,
        sessions: List[Session]
    ) -> List[np.ndarray]:
        """Fit on sessions and transform all calls.

        Args:
            sessions: List of sessions to fit and transform

        Returns:
            List of feature vectors for all calls in all sessions
        """
        self.fit(sessions)

        features = []
        for session in sessions:
            for i, call in enumerate(session.calls):
                history = session.calls[:i] if i > 0 else []
                feature_vector = self.transform(call, session, history)
                features.append(feature_vector)

        return features

    def get_feature_names(self) -> List[str]:
        """Get the names of all features in order.

        Returns:
            List of feature names

        Raises:
            ValueError: If called before fit()
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before get_feature_names()")

        return self._feature_names

    def get_feature_dim(self) -> int:
        """Get the dimensionality of the feature vector.

        Returns:
            Number of features in the vector

        Raises:
            ValueError: If called before fit()
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before get_feature_dim()")

        return len(self._feature_names)

    # ============================================================================
    # Feature Extraction Methods
    # ============================================================================

    def _extract_temporal_features(self, call: APICall) -> List[float]:
        """Extract time-based features.

        Features:
        - Hour of day (cyclic sin/cos encoding)
        - Day of week (cyclic sin/cos encoding)
        - Is weekend (binary)
        - Is peak hour (binary)

        Args:
            call: API call to extract features from

        Returns:
            List of temporal features
        """
        timestamp = call.timestamp

        # Hour of day (0-23) - cyclic encoding
        hour = timestamp.hour
        hour_sin, hour_cos = self.cyclic_encode(hour, 24)

        # Day of week (0-6, Monday=0) - cyclic encoding
        day_of_week = timestamp.weekday()
        day_sin, day_cos = self.cyclic_encode(day_of_week, 7)

        # Is weekend (Saturday=5, Sunday=6)
        is_weekend = 1.0 if day_of_week >= 5 else 0.0

        # Is peak hour
        is_peak = 1.0 if hour in self.PEAK_HOURS else 0.0

        return [hour_sin, hour_cos, day_sin, day_cos, is_weekend, is_peak]

    def _extract_user_features(
        self,
        call: APICall,
        session: Optional[Session],
        history: Optional[List[APICall]]
    ) -> List[float]:
        """Extract user-related features.

        Features:
        - User type (one-hot: premium, free, guest)
        - Session progress (0-1, how far into session)
        - Session duration so far (normalized)

        Args:
            call: API call to extract features from
            session: Session context (optional)
            history: Previous calls in session (optional)

        Returns:
            List of user features
        """
        features = []

        # User type one-hot encoding
        for user_type in self.USER_TYPES:
            features.append(1.0 if call.user_type == user_type else 0.0)

        # Session progress (how far into the session)
        if session and session.num_calls > 0:
            if history is not None:
                progress = len(history) / session.num_calls
            else:
                progress = 0.5  # Default to middle if history not provided
        else:
            progress = 0.0
        features.append(progress)

        # Session duration so far (normalized)
        if session:
            duration = (call.timestamp - session.start_timestamp).total_seconds()
            normalized_duration = min(duration / self.MAX_SESSION_DURATION_SECONDS, 1.0)
        else:
            normalized_duration = 0.0
        features.append(normalized_duration)

        return features

    def _extract_request_features(self, call: APICall) -> List[float]:
        """Extract request-specific features.

        Features:
        - HTTP method (one-hot encoding)
        - Endpoint category (one-hot encoding)
        - Number of parameters (normalized)

        Args:
            call: API call to extract features from

        Returns:
            List of request features
        """
        features = []

        # HTTP method one-hot encoding
        for method in self.HTTP_METHODS:
            features.append(1.0 if call.method == method else 0.0)

        # Endpoint category one-hot encoding
        endpoint = self._normalize_endpoint(call.endpoint) if self.normalize_endpoints else call.endpoint

        # Get category for this endpoint
        if endpoint in self.endpoint_categories:
            category = self.endpoint_categories[endpoint]
        else:
            # Unknown endpoint - use default category
            category = "unknown"

        # One-hot encode category
        for cat, idx in self.category_vocab.items():
            features.append(1.0 if cat == category else 0.0)

        # Number of parameters (normalized)
        num_params = len(call.params) if call.params else 0
        normalized_params = min(num_params / self.MAX_PARAMS, 1.0)
        features.append(normalized_params)

        return features

    def _extract_history_features(
        self,
        call: APICall,
        session: Optional[Session],
        history: Optional[List[APICall]]
    ) -> List[float]:
        """Extract session history features.

        Features:
        - Number of previous calls (normalized)
        - Time since session start (normalized)
        - Average response time so far (normalized)

        Args:
            call: Current API call
            session: Session context (optional)
            history: Previous calls in session (optional)

        Returns:
            List of history features
        """
        features = []

        # Number of previous calls (normalized)
        if history:
            num_previous = len(history)
        else:
            num_previous = 0
        normalized_count = min(num_previous / self.MAX_HISTORY_LENGTH, 1.0)
        features.append(normalized_count)

        # Time since session start (normalized)
        if session:
            time_since_start = (call.timestamp - session.start_timestamp).total_seconds()
            normalized_time = min(time_since_start / self.MAX_SESSION_DURATION_SECONDS, 1.0)
        else:
            normalized_time = 0.0
        features.append(normalized_time)

        # Average response time so far (normalized to 0-1, assuming max 1000ms)
        if history and len(history) > 0:
            avg_response_time = np.mean([c.response_time_ms for c in history])
            normalized_response_time = min(avg_response_time / 1000.0, 1.0)
        else:
            normalized_response_time = 0.0
        features.append(normalized_response_time)

        return features

    # ============================================================================
    # Helper Methods
    # ============================================================================

    @staticmethod
    def cyclic_encode(value: float, max_value: float) -> Tuple[float, float]:
        """Encode a cyclic value (like hour or day) as sine and cosine.

        This ensures that values at the boundaries (e.g., hour 23 and hour 0)
        are close together in the feature space.

        Args:
            value: The value to encode (e.g., hour=14)
            max_value: The maximum value in the cycle (e.g., max=24 for hours)

        Returns:
            Tuple of (sin_value, cos_value) both in range [-1, 1]
        """
        # Normalize to [0, 2π]
        normalized = 2 * math.pi * value / max_value

        # Return sin and cos
        return math.sin(normalized), math.cos(normalized)

    def _normalize_endpoint(self, endpoint: str) -> str:
        """Normalize an endpoint path.

        Similar to SequenceBuilder normalization:
        - Lowercase
        - Remove trailing slashes
        - Strip query parameters
        - Replace numeric IDs with {id}

        Args:
            endpoint: Raw endpoint path

        Returns:
            Normalized endpoint
        """
        import re

        if not endpoint:
            return endpoint

        # Convert to lowercase
        normalized = endpoint.lower()

        # Strip query parameters
        normalized = normalized.split('?')[0]

        # Remove trailing slashes
        normalized = normalized.rstrip('/')

        # Replace numeric IDs with {id}
        normalized = re.sub(r'/\d+(?=/|$)', '/{id}', normalized)

        # Replace UUIDs with {id}
        normalized = re.sub(
            r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?=/|$)',
            '/{id}',
            normalized
        )

        return normalized

    def _extract_category(self, endpoint: str) -> str:
        """Extract the service/category name from an endpoint.

        Examples:
        - "/api/users/123" → "users"
        - "/products" → "products"
        - "/api/orders/456/items" → "orders"

        Args:
            endpoint: Endpoint path

        Returns:
            Category/service name
        """
        parts = endpoint.lstrip('/').split('/')

        # Skip 'api' prefix if present
        if parts and parts[0] == 'api' and len(parts) > 1:
            return parts[1]
        elif parts:
            return parts[0]

        return "unknown"

    def _build_feature_names(self) -> None:
        """Build the list of feature names based on enabled features."""
        names = []

        # Temporal features
        if self.temporal_features:
            names.extend([
                'hour_sin',
                'hour_cos',
                'day_sin',
                'day_cos',
                'is_weekend',
                'is_peak_hour'
            ])

        # User features
        if self.user_features:
            for user_type in self.USER_TYPES:
                names.append(f'user_{user_type}')
            names.extend([
                'session_progress',
                'session_duration_normalized'
            ])

        # Request features
        if self.request_features:
            for method in self.HTTP_METHODS:
                names.append(f'method_{method}')
            for category in sorted(self.category_vocab.keys()):
                names.append(f'category_{category}')
            names.append('num_params_normalized')

        # History features
        if self.history_features:
            names.extend([
                'num_previous_calls_normalized',
                'time_since_start_normalized',
                'avg_response_time_normalized'
            ])

        self._feature_names = names

    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about the fitted feature engineer.

        Returns:
            Dictionary with feature engineer statistics and configuration

        Raises:
            ValueError: If called before fit()
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before get_feature_info()")

        return {
            'feature_dim': len(self._feature_names),
            'num_endpoints': len(self.endpoint_vocab),
            'num_categories': len(self.category_vocab),
            'categories': sorted(self.category_vocab.keys()),
            'temporal_features_enabled': self.temporal_features,
            'user_features_enabled': self.user_features,
            'request_features_enabled': self.request_features,
            'history_features_enabled': self.history_features,
            'normalize_endpoints': self.normalize_endpoints
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        if self.is_fitted:
            return (f"FeatureEngineer(fitted=True, feature_dim={len(self._feature_names)}, "
                    f"endpoints={len(self.endpoint_vocab)}, categories={len(self.category_vocab)})")
        else:
            return "FeatureEngineer(fitted=False)"


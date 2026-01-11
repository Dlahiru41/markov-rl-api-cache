"""Data models for representing API call traces and sessions."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
import json


@dataclass
class APICall:
    """Represents a single API request.

    Captures all relevant information about an API call including timing,
    user context, and response characteristics.
    """
    call_id: str
    endpoint: str
    method: str
    params: Dict[str, Any]
    user_id: str
    session_id: str
    timestamp: datetime
    response_time_ms: float
    status_code: int
    response_size_bytes: int
    user_type: str  # premium/free/guest

    def __post_init__(self):
        """Validate APICall data."""
        if self.response_time_ms < 0:
            raise ValueError(f"response_time_ms must be non-negative, got {self.response_time_ms}")

        if self.response_size_bytes < 0:
            raise ValueError(f"response_size_bytes must be non-negative, got {self.response_size_bytes}")

        if self.user_type not in ['premium', 'free', 'guest']:
            raise ValueError(f"user_type must be 'premium', 'free', or 'guest', got {self.user_type}")

        if not self.endpoint.startswith('/'):
            raise ValueError(f"endpoint must start with '/', got {self.endpoint}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the APICall object to a dictionary.

        Returns:
            Dictionary representation of the APICall with timestamp as ISO format string.
        """
        return {
            'call_id': self.call_id,
            'endpoint': self.endpoint,
            'method': self.method,
            'params': self.params,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat(),
            'response_time_ms': self.response_time_ms,
            'status_code': self.status_code,
            'response_size_bytes': self.response_size_bytes,
            'user_type': self.user_type
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'APICall':
        """Create an APICall object from a dictionary.

        Args:
            data: Dictionary containing APICall fields.

        Returns:
            APICall object created from the dictionary.
        """
        data_copy = data.copy()
        # Convert timestamp string to datetime object
        if isinstance(data_copy['timestamp'], str):
            data_copy['timestamp'] = datetime.fromisoformat(data_copy['timestamp'])
        return cls(**data_copy)

    def get_service_name(self) -> str:
        """Extract the service name from the endpoint.

        Example:
            "/api/users/123" → "users"
            "/api/products" → "products"
            "/users/123/orders" → "users"

        Returns:
            Service name extracted from the endpoint path.
        """
        # Remove leading slash and split by '/'
        parts = self.endpoint.lstrip('/').split('/')

        # Skip 'api' prefix if present and get the next part
        if parts and parts[0].lower() == 'api' and len(parts) > 1:
            return parts[1]
        elif parts:
            return parts[0]
        return ""

    def is_successful(self) -> bool:
        """Check if the API call was successful (status code 2xx).

        Returns:
            True if status code is in the 200-299 range, False otherwise.
        """
        return 200 <= self.status_code < 300

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"APICall(call_id={self.call_id!r}, endpoint={self.endpoint!r}, "
                f"method={self.method!r}, timestamp={self.timestamp.isoformat()}, "
                f"status_code={self.status_code}, response_time_ms={self.response_time_ms})")


@dataclass
class Session:
    """Groups related API calls from one user visit.

    Represents a user session containing multiple API calls with
    timing information and analysis methods for Markov chain training.
    """
    session_id: str
    user_id: str
    user_type: str
    start_timestamp: datetime
    end_timestamp: Optional[datetime] = None
    calls: List[APICall] = field(default_factory=list)

    def __post_init__(self):
        """Validate Session data."""
        if self.user_type not in ['premium', 'free', 'guest']:
            raise ValueError(f"user_type must be 'premium', 'free', or 'guest', got {self.user_type}")

        if self.end_timestamp and self.end_timestamp < self.start_timestamp:
            raise ValueError(f"end_timestamp ({self.end_timestamp}) must be after start_timestamp ({self.start_timestamp})")

        # Validate that all calls belong to this session
        for call in self.calls:
            if call.session_id != self.session_id:
                raise ValueError(f"Call {call.call_id} has session_id {call.session_id} but session has {self.session_id}")

    @property
    def duration_seconds(self) -> float:
        """Calculate total duration of the session in seconds.

        Returns:
            Duration in seconds, or 0 if end_timestamp is not set.
        """
        if self.end_timestamp is None:
            return 0.0
        return (self.end_timestamp - self.start_timestamp).total_seconds()

    @property
    def num_calls(self) -> int:
        """Get the number of API calls in this session.

        Returns:
            Number of calls in the session.
        """
        return len(self.calls)

    @property
    def unique_endpoints(self) -> List[str]:
        """Get list of unique endpoints visited in this session.

        Returns:
            Sorted list of unique endpoint paths.
        """
        return sorted(set(call.endpoint for call in self.calls))

    @property
    def endpoint_sequence(self) -> List[str]:
        """Get the sequence of endpoint names for Markov training.

        Returns:
            List of endpoint paths in chronological order.
        """
        return [call.endpoint for call in self.calls]

    def append_call(self, call: APICall) -> None:
        """Append a new API call to the session.

        Args:
            call: APICall object to add to the session.

        Raises:
            ValueError: If the call's session_id doesn't match this session.
        """
        if call.session_id != self.session_id:
            raise ValueError(f"Call session_id {call.session_id} doesn't match session {self.session_id}")

        self.calls.append(call)

        # Update end_timestamp if this call is later
        if self.end_timestamp is None or call.timestamp > self.end_timestamp:
            self.end_timestamp = call.timestamp

    def get_endpoint_transitions(self) -> List[Tuple[str, str]]:
        """Get all consecutive pairs of endpoints (transitions) for Markov chain training.

        Returns:
            List of tuples representing (from_endpoint, to_endpoint) transitions.
        """
        if len(self.calls) < 2:
            return []

        transitions = []
        for i in range(len(self.calls) - 1):
            from_endpoint = self.calls[i].endpoint
            to_endpoint = self.calls[i + 1].endpoint
            transitions.append((from_endpoint, to_endpoint))

        return transitions

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Session object to a dictionary.

        Returns:
            Dictionary representation of the Session.
        """
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'user_type': self.user_type,
            'start_timestamp': self.start_timestamp.isoformat(),
            'end_timestamp': self.end_timestamp.isoformat() if self.end_timestamp else None,
            'calls': [call.to_dict() for call in self.calls]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        """Create a Session object from a dictionary.

        Args:
            data: Dictionary containing Session fields.

        Returns:
            Session object created from the dictionary.
        """
        data_copy = data.copy()

        # Convert timestamp strings to datetime objects
        if isinstance(data_copy['start_timestamp'], str):
            data_copy['start_timestamp'] = datetime.fromisoformat(data_copy['start_timestamp'])

        if data_copy.get('end_timestamp') and isinstance(data_copy['end_timestamp'], str):
            data_copy['end_timestamp'] = datetime.fromisoformat(data_copy['end_timestamp'])

        # Convert calls from dicts to APICall objects
        if 'calls' in data_copy:
            data_copy['calls'] = [APICall.from_dict(call_data) for call_data in data_copy['calls']]

        return cls(**data_copy)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"Session(session_id={self.session_id!r}, user_id={self.user_id!r}, "
                f"user_type={self.user_type!r}, num_calls={self.num_calls}, "
                f"duration={self.duration_seconds:.2f}s)")


@dataclass
class Dataset:
    """Holds a collection of sessions for analysis and training.

    Represents a complete dataset of user sessions with methods for
    statistical analysis, Markov chain training data extraction, and
    train/test splitting.
    """
    name: str
    sessions: List[Session] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_calls(self) -> int:
        """Get total number of API calls across all sessions.

        Returns:
            Total count of API calls in the dataset.
        """
        return sum(session.num_calls for session in self.sessions)

    @property
    def num_unique_users(self) -> int:
        """Get number of unique users in the dataset.

        Returns:
            Count of unique user IDs.
        """
        return len(set(session.user_id for session in self.sessions))

    @property
    def unique_endpoints(self) -> Set[str]:
        """Get set of all unique endpoints in the dataset.

        Returns:
            Set of unique endpoint paths across all sessions.
        """
        endpoints = set()
        for session in self.sessions:
            endpoints.update(session.unique_endpoints)
        return endpoints

    @property
    def date_range(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get the date range covered by the dataset.

        Returns:
            Tuple of (earliest_timestamp, latest_timestamp) or (None, None) if no sessions.
        """
        if not self.sessions:
            return (None, None)

        min_timestamp = min(session.start_timestamp for session in self.sessions)
        max_timestamp = max(
            session.end_timestamp for session in self.sessions
            if session.end_timestamp is not None
        )

        return (min_timestamp, max_timestamp)

    def get_all_sequences(self) -> List[List[str]]:
        """Get all API sequences (list of endpoint lists) for Markov training.

        Returns:
            List where each element is a list of endpoints from one session.
        """
        return [session.endpoint_sequence for session in self.sessions if session.num_calls > 0]

    def count_endpoint_occurrences(self) -> Dict[str, int]:
        """Count how often each endpoint appears across all sessions.

        Returns:
            Dictionary mapping endpoint paths to their occurrence counts.
        """
        endpoint_counts = {}

        for session in self.sessions:
            for call in session.calls:
                endpoint = call.endpoint
                endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1

        return endpoint_counts

    def split(self, train_ratio: float = 0.8) -> Tuple['Dataset', 'Dataset']:
        """Split the dataset into train and test sets.

        Args:
            train_ratio: Proportion of sessions to include in training set (0.0 to 1.0).

        Returns:
            Tuple of (train_dataset, test_dataset).

        Raises:
            ValueError: If train_ratio is not between 0 and 1.
        """
        if not 0 < train_ratio < 1:
            raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

        # Calculate split index
        split_index = int(len(self.sessions) * train_ratio)

        # Create train dataset
        train_dataset = Dataset(
            name=f"{self.name}_train",
            sessions=self.sessions[:split_index],
            metadata={
                **self.metadata,
                'split_type': 'train',
                'train_ratio': train_ratio,
                'parent_dataset': self.name
            }
        )

        # Create test dataset
        test_dataset = Dataset(
            name=f"{self.name}_test",
            sessions=self.sessions[split_index:],
            metadata={
                **self.metadata,
                'split_type': 'test',
                'train_ratio': train_ratio,
                'parent_dataset': self.name
            }
        )

        return train_dataset, test_dataset

    def save_to_parquet(self, filepath: Path) -> None:
        """Save the dataset to a parquet file.

        Args:
            filepath: Path where the parquet file will be saved.
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow is required for parquet functionality. Install with: pip install pyarrow")

        # Flatten all calls from all sessions into a list of dictionaries
        records = []
        for session in self.sessions:
            for call in session.calls:
                record = call.to_dict()
                # Add session-level information
                record['session_user_type'] = session.user_type
                record['session_start'] = session.start_timestamp.isoformat()
                record['session_end'] = session.end_timestamp.isoformat() if session.end_timestamp else None
                records.append(record)

        # Convert to PyArrow table
        table = pa.Table.from_pylist(records)

        # Save metadata as custom metadata
        metadata_json = json.dumps({
            'dataset_name': self.name,
            'metadata': self.metadata,
            'num_sessions': len(self.sessions),
            'total_calls': self.total_calls
        })

        # Add custom metadata to the table
        existing_metadata = table.schema.metadata or {}
        combined_metadata = {
            **existing_metadata,
            b'dataset_metadata': metadata_json.encode('utf-8')
        }
        table = table.replace_schema_metadata(combined_metadata)

        # Write to parquet file
        pq.write_table(table, str(filepath))

    @classmethod
    def load_from_parquet(cls, filepath: Path) -> 'Dataset':
        """Load a dataset from a parquet file.

        Args:
            filepath: Path to the parquet file to load.

        Returns:
            Dataset object loaded from the parquet file.
        """
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow is required for parquet functionality. Install with: pip install pyarrow")

        # Read the parquet file
        table = pq.read_table(str(filepath))

        # Extract custom metadata
        schema_metadata = table.schema.metadata or {}
        dataset_metadata_json = schema_metadata.get(b'dataset_metadata', b'{}').decode('utf-8')
        dataset_metadata = json.loads(dataset_metadata_json)

        # Convert to list of dictionaries
        records = table.to_pylist()

        # Group calls by session
        sessions_dict = {}
        for record in records:
            session_id = record['session_id']

            if session_id not in sessions_dict:
                # Create new session
                sessions_dict[session_id] = {
                    'session_id': session_id,
                    'user_id': record['user_id'],
                    'user_type': record['session_user_type'],
                    'start_timestamp': datetime.fromisoformat(record['session_start']),
                    'end_timestamp': datetime.fromisoformat(record['session_end']) if record['session_end'] else None,
                    'calls': []
                }

            # Create APICall object
            call = APICall(
                call_id=record['call_id'],
                endpoint=record['endpoint'],
                method=record['method'],
                params=record['params'],
                user_id=record['user_id'],
                session_id=record['session_id'],
                timestamp=datetime.fromisoformat(record['timestamp']),
                response_time_ms=record['response_time_ms'],
                status_code=record['status_code'],
                response_size_bytes=record['response_size_bytes'],
                user_type=record['user_type']
            )

            sessions_dict[session_id]['calls'].append(call)

        # Create Session objects
        sessions = [Session(**session_data) for session_data in sessions_dict.values()]

        # Create and return Dataset
        return cls(
            name=dataset_metadata.get('dataset_name', 'unnamed'),
            sessions=sessions,
            metadata=dataset_metadata.get('metadata', {})
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"Dataset(name={self.name!r}, num_sessions={len(self.sessions)}, "
                f"total_calls={self.total_calls}, num_users={self.num_unique_users}, "
                f"num_endpoints={len(self.unique_endpoints)})")


"""Session extraction module for grouping API calls into user sessions.

This module provides functionality to group raw API call logs into meaningful
user sessions based on temporal patterns and inactivity timeouts.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
from collections import defaultdict
import json

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from preprocessing.models import APICall, Session


@dataclass
class SessionExtractionConfig:
    """Configuration for session extraction."""

    inactivity_timeout_minutes: float = 30.0
    min_session_length: int = 1
    max_session_length: Optional[int] = None
    show_progress: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.inactivity_timeout_minutes <= 0:
            raise ValueError(f"inactivity_timeout_minutes must be positive, got {self.inactivity_timeout_minutes}")

        if self.min_session_length < 1:
            raise ValueError(f"min_session_length must be at least 1, got {self.min_session_length}")

        if self.max_session_length is not None and self.max_session_length < self.min_session_length:
            raise ValueError(f"max_session_length ({self.max_session_length}) must be >= min_session_length ({self.min_session_length})")


@dataclass
class SessionStatistics:
    """Statistics about session extraction."""

    total_calls_processed: int = 0
    total_sessions_created: int = 0
    sessions_kept: int = 0
    sessions_filtered_too_short: int = 0
    sessions_filtered_too_long: int = 0
    unique_users: int = 0
    average_session_length: float = 0.0
    average_session_duration_seconds: float = 0.0
    min_session_length: int = 0
    max_session_length: int = 0
    min_session_duration: float = 0.0
    max_session_duration: float = 0.0
    calls_with_missing_timestamps: int = 0
    calls_with_malformed_data: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            'total_calls_processed': self.total_calls_processed,
            'total_sessions_created': self.total_sessions_created,
            'sessions_kept': self.sessions_kept,
            'sessions_filtered_too_short': self.sessions_filtered_too_short,
            'sessions_filtered_too_long': self.sessions_filtered_too_long,
            'unique_users': self.unique_users,
            'average_session_length': self.average_session_length,
            'average_session_duration_seconds': self.average_session_duration_seconds,
            'min_session_length': self.min_session_length,
            'max_session_length': self.max_session_length,
            'min_session_duration_seconds': self.min_session_duration,
            'max_session_duration_seconds': self.max_session_duration,
            'calls_with_missing_timestamps': self.calls_with_missing_timestamps,
            'calls_with_malformed_data': self.calls_with_malformed_data,
        }


class SessionExtractor:
    """Extracts user sessions from raw API call logs.

    Groups individual API calls into sessions based on temporal patterns,
    user activity, and configurable thresholds.

    Example:
        >>> extractor = SessionExtractor(inactivity_timeout_minutes=30, min_session_length=2)
        >>> sessions = extractor.extract_sessions(calls)
        >>> print(extractor.get_statistics())
    """

    def __init__(
        self,
        inactivity_timeout_minutes: float = 30.0,
        min_session_length: int = 1,
        max_session_length: Optional[int] = None,
        show_progress: bool = True
    ):
        """Initialize the SessionExtractor.

        Args:
            inactivity_timeout_minutes: Time in minutes after which a new session starts.
            min_session_length: Minimum number of calls for a valid session.
            max_session_length: Maximum number of calls for a valid session (None = no limit).
            show_progress: Whether to show progress bar during extraction.
        """
        self.config = SessionExtractionConfig(
            inactivity_timeout_minutes=inactivity_timeout_minutes,
            min_session_length=min_session_length,
            max_session_length=max_session_length,
            show_progress=show_progress
        )

        self.statistics = SessionStatistics()
        self._reset_statistics()

    def _reset_statistics(self) -> None:
        """Reset statistics to initial state."""
        self.statistics = SessionStatistics()

    def extract_sessions(self, calls: List[APICall]) -> List[Session]:
        """Extract sessions from a list of APICall objects.

        Args:
            calls: List of APICall objects to process.

        Returns:
            List of Session objects, filtered according to configuration.
        """
        self._reset_statistics()

        # Handle empty input
        if not calls:
            return []

        # Filter out calls with missing timestamps
        valid_calls = []
        for call in calls:
            if call.timestamp is None:
                self.statistics.calls_with_missing_timestamps += 1
            else:
                valid_calls.append(call)

        self.statistics.total_calls_processed = len(valid_calls)

        if not valid_calls:
            return []

        # Group calls by user
        calls_by_user = self._group_calls_by_user(valid_calls)
        self.statistics.unique_users = len(calls_by_user)

        # Extract sessions for each user
        all_sessions = []

        # Set up progress bar if available and enabled
        iterator = calls_by_user.items()
        if TQDM_AVAILABLE and self.config.show_progress:
            iterator = tqdm(
                iterator,
                desc="Extracting sessions",
                unit="users",
                total=len(calls_by_user)
            )

        for user_id, user_calls in iterator:
            user_sessions = self._extract_user_sessions(user_id, user_calls)
            all_sessions.extend(user_sessions)

        # Calculate statistics
        self._calculate_statistics(all_sessions)

        return all_sessions

    def _group_calls_by_user(self, calls: List[APICall]) -> Dict[str, List[APICall]]:
        """Group API calls by user ID.

        Args:
            calls: List of APICall objects.

        Returns:
            Dictionary mapping user_id to list of their calls.
        """
        calls_by_user = defaultdict(list)

        for call in calls:
            calls_by_user[call.user_id].append(call)

        # Sort each user's calls by timestamp
        for user_id in calls_by_user:
            calls_by_user[user_id].sort(key=lambda c: c.timestamp)

        return dict(calls_by_user)

    def _extract_user_sessions(self, user_id: str, calls: List[APICall]) -> List[Session]:
        """Extract sessions for a single user.

        Args:
            user_id: User identifier.
            calls: List of APICall objects for this user, sorted by timestamp.

        Returns:
            List of Session objects for this user.
        """
        if not calls:
            return []

        sessions = []
        current_session_calls = [calls[0]]
        session_counter = 0

        timeout_delta = timedelta(minutes=self.config.inactivity_timeout_minutes)

        for i in range(1, len(calls)):
            prev_call = calls[i - 1]
            curr_call = calls[i]

            # Calculate time gap between calls
            time_gap = curr_call.timestamp - prev_call.timestamp

            # Check if we should start a new session
            if time_gap > timeout_delta:
                # Finish current session
                session = self._create_session(
                    user_id,
                    current_session_calls,
                    session_counter
                )
                if session:
                    sessions.append(session)
                    session_counter += 1

                # Start new session
                current_session_calls = [curr_call]
            else:
                # Continue current session
                current_session_calls.append(curr_call)

        # Don't forget the last session
        if current_session_calls:
            session = self._create_session(
                user_id,
                current_session_calls,
                session_counter
            )
            if session:
                sessions.append(session)

        return sessions

    def _create_session(
        self,
        user_id: str,
        calls: List[APICall],
        session_number: int
    ) -> Optional[Session]:
        """Create a Session object from a list of calls.

        Args:
            user_id: User identifier.
            calls: List of APICall objects for this session.
            session_number: Session number for this user (0-indexed).

        Returns:
            Session object if valid, None if filtered out.
        """
        if not calls:
            return None

        self.statistics.total_sessions_created += 1

        # Check session length constraints
        num_calls = len(calls)

        if num_calls < self.config.min_session_length:
            self.statistics.sessions_filtered_too_short += 1
            return None

        if self.config.max_session_length is not None and num_calls > self.config.max_session_length:
            self.statistics.sessions_filtered_too_long += 1
            return None

        # Sort calls by timestamp to ensure correct order
        sorted_calls = sorted(calls, key=lambda c: c.timestamp)

        # Generate unique session ID
        session_id = f"{user_id}_session_{session_number}"

        # Update session_id for all calls
        for call in sorted_calls:
            call.session_id = session_id

        # Create session
        session = Session(
            session_id=session_id,
            user_id=user_id,
            user_type=sorted_calls[0].user_type,
            start_timestamp=sorted_calls[0].timestamp,
            end_timestamp=sorted_calls[-1].timestamp,
            calls=sorted_calls
        )

        self.statistics.sessions_kept += 1

        return session

    def _calculate_statistics(self, sessions: List[Session]) -> None:
        """Calculate statistics for extracted sessions.

        Args:
            sessions: List of Session objects.
        """
        if not sessions:
            return

        session_lengths = [session.num_calls for session in sessions]
        session_durations = [session.duration_seconds for session in sessions]

        self.statistics.average_session_length = sum(session_lengths) / len(sessions)
        self.statistics.average_session_duration_seconds = sum(session_durations) / len(sessions)

        self.statistics.min_session_length = min(session_lengths)
        self.statistics.max_session_length = max(session_lengths)

        self.statistics.min_session_duration = min(session_durations)
        self.statistics.max_session_duration = max(session_durations)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from the last extraction.

        Returns:
            Dictionary containing extraction statistics.
        """
        return self.statistics.to_dict()

    def extract_from_dataframe(
        self,
        df: 'pd.DataFrame',
        column_mapping: Optional[Dict[str, str]] = None
    ) -> List[Session]:
        """Extract sessions from a pandas DataFrame.

        Args:
            df: DataFrame containing API call logs.
            column_mapping: Optional mapping of DataFrame columns to APICall fields.
                           Default mapping: {
                               'call_id': 'call_id',
                               'endpoint': 'endpoint',
                               'method': 'method',
                               'params': 'params',
                               'user_id': 'user_id',
                               'session_id': 'session_id',
                               'timestamp': 'timestamp',
                               'response_time_ms': 'response_time_ms',
                               'status_code': 'status_code',
                               'response_size_bytes': 'response_size_bytes',
                               'user_type': 'user_type'
                           }

        Returns:
            List of Session objects.

        Raises:
            ImportError: If pandas is not installed.
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for DataFrame extraction. Install with: pip install pandas")

        # Default column mapping
        default_mapping = {
            'call_id': 'call_id',
            'endpoint': 'endpoint',
            'method': 'method',
            'params': 'params',
            'user_id': 'user_id',
            'session_id': 'session_id',
            'timestamp': 'timestamp',
            'response_time_ms': 'response_time_ms',
            'status_code': 'status_code',
            'response_size_bytes': 'response_size_bytes',
            'user_type': 'user_type'
        }

        # Use provided mapping or default
        mapping = column_mapping if column_mapping else default_mapping

        # Convert DataFrame to APICall objects
        calls = []

        for idx, row in df.iterrows():
            try:
                # Handle params field (might be string or dict)
                params = row.get(mapping.get('params', 'params'), {})
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except json.JSONDecodeError:
                        params = {}

                # Handle timestamp (might be string or datetime)
                timestamp = row.get(mapping.get('timestamp', 'timestamp'))
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                elif isinstance(timestamp, pd.Timestamp):
                    timestamp = timestamp.to_pydatetime()

                call = APICall(
                    call_id=str(row.get(mapping.get('call_id', 'call_id'), idx)),
                    endpoint=str(row.get(mapping.get('endpoint', 'endpoint'), '')),
                    method=str(row.get(mapping.get('method', 'method'), 'GET')),
                    params=params,
                    user_id=str(row.get(mapping.get('user_id', 'user_id'), '')),
                    session_id=str(row.get(mapping.get('session_id', 'session_id'), '')),
                    timestamp=timestamp,
                    response_time_ms=float(row.get(mapping.get('response_time_ms', 'response_time_ms'), 0)),
                    status_code=int(row.get(mapping.get('status_code', 'status_code'), 200)),
                    response_size_bytes=int(row.get(mapping.get('response_size_bytes', 'response_size_bytes'), 0)),
                    user_type=str(row.get(mapping.get('user_type', 'user_type'), 'free'))
                )
                calls.append(call)
            except Exception as e:
                self.statistics.calls_with_malformed_data += 1
                # Skip malformed rows
                continue

        return self.extract_sessions(calls)

    def extract_from_file(
        self,
        filepath: Path,
        column_mapping: Optional[Dict[str, str]] = None
    ) -> List[Session]:
        """Extract sessions from a file (CSV, JSON, or Parquet).

        Args:
            filepath: Path to the file.
            column_mapping: Optional column mapping for DataFrame-based formats.

        Returns:
            List of Session objects.

        Raises:
            ValueError: If file format is not supported.
            FileNotFoundError: If file does not exist.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        suffix = filepath.suffix.lower()

        # CSV format
        if suffix == '.csv':
            if not PANDAS_AVAILABLE:
                raise ImportError("pandas is required for CSV files. Install with: pip install pandas")
            df = pd.read_csv(filepath)
            return self.extract_from_dataframe(df, column_mapping)

        # JSON format
        elif suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, list):
                calls = [APICall.from_dict(item) for item in data]
            elif isinstance(data, dict) and 'calls' in data:
                calls = [APICall.from_dict(item) for item in data['calls']]
            else:
                raise ValueError(f"Unsupported JSON structure in {filepath}")

            return self.extract_sessions(calls)

        # Parquet format
        elif suffix == '.parquet':
            if not PANDAS_AVAILABLE:
                raise ImportError("pandas is required for Parquet files. Install with: pip install pandas pyarrow")
            df = pd.read_parquet(filepath)
            return self.extract_from_dataframe(df, column_mapping)

        else:
            raise ValueError(f"Unsupported file format: {suffix}. Supported formats: .csv, .json, .parquet")

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"SessionExtractor(inactivity_timeout={self.config.inactivity_timeout_minutes}min, "
                f"min_length={self.config.min_session_length}, "
                f"max_length={self.config.max_session_length})")


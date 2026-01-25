"""Data validation utilities for ensuring data quality.

This module provides comprehensive validation and quality checks for API trace data,
helping identify issues before training ML models.
"""

from typing import List, Dict, Any
from datetime import datetime
import numpy as np
from collections import Counter

from preprocessing.models import APICall, Session, Dataset


class DataValidator:
    """Validates data quality and detects anomalies in API trace datasets.

    Provides methods to check individual calls, sessions, and entire datasets
    for common data quality issues.
    """

    # Valid HTTP status code ranges
    MIN_STATUS_CODE = 100
    MAX_STATUS_CODE = 599

    # Thresholds for anomaly detection
    OUTLIER_RESPONSE_TIME_MULTIPLIER = 3.0  # IQR multiplier for outliers
    MIN_SESSION_LENGTH = 1  # Minimum calls per session
    MAX_SESSION_LENGTH = 100  # Maximum calls per session (suspiciously long)
    HIGH_ERROR_RATE_THRESHOLD = 0.5  # 50% errors is suspicious

    def __init__(
        self,
        min_session_length: int = 1,
        max_session_length: int = 100,
        high_error_rate: float = 0.5
    ):
        """Initialize the data validator.

        Args:
            min_session_length: Minimum required calls per session
            max_session_length: Maximum reasonable calls per session
            high_error_rate: Threshold for flagging high-error sessions
        """
        self.min_session_length = min_session_length
        self.max_session_length = max_session_length
        self.high_error_rate = high_error_rate

    def validate_api_call(self, call: APICall) -> List[str]:
        """Check a single APICall for issues.

        Validates:
        - Required fields are present and non-empty
        - Timestamp is valid
        - Status code is reasonable (100-599)
        - Response time is non-negative

        Args:
            call: APICall to validate

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check required fields are non-empty
        if not call.call_id:
            errors.append(f"Call has empty call_id")

        if not call.endpoint:
            errors.append(f"Call {call.call_id}: endpoint is empty")

        if not call.method:
            errors.append(f"Call {call.call_id}: method is empty")

        if not call.user_id:
            errors.append(f"Call {call.call_id}: user_id is empty")

        if not call.session_id:
            errors.append(f"Call {call.call_id}: session_id is empty")

        if not call.user_type:
            errors.append(f"Call {call.call_id}: user_type is empty")

        # Check timestamp is valid
        if call.timestamp is None:
            errors.append(f"Call {call.call_id}: timestamp is None")
        elif not isinstance(call.timestamp, datetime):
            errors.append(f"Call {call.call_id}: timestamp is not a datetime object")

        # Check status code is in valid range
        if not (self.MIN_STATUS_CODE <= call.status_code <= self.MAX_STATUS_CODE):
            errors.append(
                f"Call {call.call_id}: invalid status code {call.status_code} "
                f"(must be {self.MIN_STATUS_CODE}-{self.MAX_STATUS_CODE})"
            )

        # Check response time is non-negative
        if call.response_time_ms < 0:
            errors.append(
                f"Call {call.call_id}: negative response time {call.response_time_ms}ms"
            )

        # Check response size is non-negative
        if call.response_size_bytes < 0:
            errors.append(
                f"Call {call.call_id}: negative response size {call.response_size_bytes} bytes"
            )

        # Check user type is valid
        if call.user_type not in ['premium', 'free', 'guest']:
            errors.append(
                f"Call {call.call_id}: invalid user_type '{call.user_type}' "
                f"(must be 'premium', 'free', or 'guest')"
            )

        # Check endpoint starts with /
        if not call.endpoint.startswith('/'):
            errors.append(
                f"Call {call.call_id}: endpoint '{call.endpoint}' should start with '/'"
            )

        return errors

    def validate_session(self, session: Session) -> List[str]:
        """Check a Session for issues.

        Validates:
        - All calls share the same session ID
        - Calls are in chronological order
        - Session has at least minimum required length
        - Timestamps are consistent

        Args:
            session: Session to validate

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check session has minimum length
        if len(session.calls) < self.min_session_length:
            errors.append(
                f"Session {session.session_id}: has {len(session.calls)} calls "
                f"(minimum {self.min_session_length} required)"
            )

        # Check all calls belong to this session
        for idx, call in enumerate(session.calls):
            if call.session_id != session.session_id:
                errors.append(
                    f"Session {session.session_id}: call #{idx+1} has mismatched "
                    f"session_id '{call.session_id}'"
                )

        # Check calls are in chronological order
        for idx in range(1, len(session.calls)):
            prev_call = session.calls[idx - 1]
            curr_call = session.calls[idx]

            if curr_call.timestamp < prev_call.timestamp:
                errors.append(
                    f"Session {session.session_id}: calls are not in chronological order, "
                    f"call #{idx+1} timestamp is before call #{idx}"
                )

        # Check session start/end timestamps are consistent
        if session.calls:
            first_call_time = session.calls[0].timestamp
            last_call_time = session.calls[-1].timestamp

            if session.start_timestamp > first_call_time:
                errors.append(
                    f"Session {session.session_id}: start_timestamp is after first call timestamp"
                )

            if session.end_timestamp and session.end_timestamp < last_call_time:
                errors.append(
                    f"Session {session.session_id}: end_timestamp is before last call timestamp"
                )

        # Check all calls have same user_id
        if session.calls:
            user_ids = {call.user_id for call in session.calls}
            if len(user_ids) > 1:
                errors.append(
                    f"Session {session.session_id}: calls have inconsistent user_ids {user_ids}"
                )
            elif session.user_id not in user_ids:
                errors.append(
                    f"Session {session.session_id}: session user_id '{session.user_id}' "
                    f"doesn't match call user_ids {user_ids}"
                )

        # Check all calls have same user_type
        if session.calls:
            user_types = {call.user_type for call in session.calls}
            if len(user_types) > 1:
                errors.append(
                    f"Session {session.session_id}: calls have inconsistent user_types {user_types}"
                )
            elif session.user_type not in user_types:
                errors.append(
                    f"Session {session.session_id}: session user_type '{session.user_type}' "
                    f"doesn't match call user_types {user_types}"
                )

        return errors

    def validate_dataset(self, dataset: Dataset) -> Dict[str, Any]:
        """Comprehensive dataset validation.

        Runs validation on all sessions and calls, collecting all errors.

        Args:
            dataset: Dataset to validate

        Returns:
            Dictionary with validation results:
            {
                'valid': bool,
                'total_sessions': int,
                'total_calls': int,
                'invalid_sessions': int,
                'invalid_calls': int,
                'errors': List[str],
                'warnings': List[str]
            }
        """
        all_errors = []
        all_warnings = []

        invalid_sessions = 0
        invalid_calls = 0

        # Track for duplicate detection
        seen_session_ids = set()
        seen_call_ids = set()

        # Validate each session
        for session in dataset.sessions:
            # Check for duplicate session IDs
            if session.session_id in seen_session_ids:
                all_errors.append(f"Duplicate session_id: {session.session_id}")
            seen_session_ids.add(session.session_id)

            # Validate session
            session_errors = self.validate_session(session)
            if session_errors:
                invalid_sessions += 1
                all_errors.extend(session_errors)

            # Validate each call in session
            for call in session.calls:
                # Check for duplicate call IDs
                if call.call_id in seen_call_ids:
                    all_warnings.append(f"Duplicate call_id: {call.call_id}")
                seen_call_ids.add(call.call_id)

                # Validate call
                call_errors = self.validate_api_call(call)
                if call_errors:
                    invalid_calls += 1
                    all_errors.extend(call_errors)

        # Check for empty dataset
        if not dataset.sessions:
            all_errors.append("Dataset has no sessions")

        return {
            'valid': len(all_errors) == 0,
            'total_sessions': len(dataset.sessions),
            'total_calls': dataset.total_calls,
            'invalid_sessions': invalid_sessions,
            'invalid_calls': invalid_calls,
            'errors': all_errors,
            'warnings': all_warnings
        }

    def check_data_quality(self, dataset: Dataset) -> Dict[str, float]:
        """Compute quality metrics for a dataset.

        Calculates:
        - Fraction of fields with missing values
        - Fraction of duplicate calls
        - Fraction of response time outliers

        Args:
            dataset: Dataset to analyze

        Returns:
            Dictionary of metric names to values (0.0 to 1.0)
        """
        if not dataset.sessions or dataset.total_calls == 0:
            return {
                'missing_values_fraction': 0.0,
                'duplicate_calls_fraction': 0.0,
                'response_time_outliers_fraction': 0.0,
                'error_rate': 0.0,
                'empty_endpoint_fraction': 0.0
            }

        total_calls = dataset.total_calls

        # Count missing values
        missing_count = 0
        for session in dataset.sessions:
            for call in session.calls:
                # Check for None or empty string fields
                if not call.call_id:
                    missing_count += 1
                if not call.endpoint:
                    missing_count += 1
                if not call.method:
                    missing_count += 1
                if not call.user_id:
                    missing_count += 1
                if not call.session_id:
                    missing_count += 1
                if call.timestamp is None:
                    missing_count += 1

        # Count duplicate call IDs
        call_ids = [call.call_id for session in dataset.sessions for call in session.calls]
        unique_call_ids = len(set(call_ids))
        duplicate_calls = total_calls - unique_call_ids

        # Detect response time outliers using IQR method
        response_times = [call.response_time_ms for session in dataset.sessions for call in session.calls]

        if len(response_times) > 0:
            q1 = np.percentile(response_times, 25)
            q3 = np.percentile(response_times, 75)
            iqr = q3 - q1

            lower_bound = q1 - self.OUTLIER_RESPONSE_TIME_MULTIPLIER * iqr
            upper_bound = q3 + self.OUTLIER_RESPONSE_TIME_MULTIPLIER * iqr

            outlier_count = sum(
                1 for rt in response_times
                if rt < lower_bound or rt > upper_bound
            )
        else:
            outlier_count = 0

        # Count errors (non-2xx status codes)
        error_count = sum(
            1 for session in dataset.sessions
            for call in session.calls
            if not (200 <= call.status_code < 300)
        )

        # Count empty endpoints
        empty_endpoint_count = sum(
            1 for session in dataset.sessions
            for call in session.calls
            if not call.endpoint or call.endpoint.strip() == ''
        )

        return {
            'missing_values_fraction': missing_count / (total_calls * 6),  # 6 key fields
            'duplicate_calls_fraction': duplicate_calls / total_calls if total_calls > 0 else 0.0,
            'response_time_outliers_fraction': outlier_count / total_calls if total_calls > 0 else 0.0,
            'error_rate': error_count / total_calls if total_calls > 0 else 0.0,
            'empty_endpoint_fraction': empty_endpoint_count / total_calls if total_calls > 0 else 0.0
        }

    def detect_anomalies(self, dataset: Dataset) -> List[Dict[str, Any]]:
        """Find suspicious sessions that may indicate data quality issues.

        Detects:
        - Sessions with unusual length (very short or very long)
        - Sessions with many errors
        - Sessions with abnormal patterns

        Args:
            dataset: Dataset to analyze

        Returns:
            List of flagged sessions with reasons for manual review
        """
        anomalies = []

        for session in dataset.sessions:
            flags = []

            # Check session length
            if len(session.calls) < self.min_session_length:
                flags.append(f"Too short: {len(session.calls)} calls (min {self.min_session_length})")

            if len(session.calls) > self.max_session_length:
                flags.append(f"Too long: {len(session.calls)} calls (max {self.max_session_length})")

            # Check error rate
            if session.calls:
                error_count = sum(1 for call in session.calls if not call.is_successful())
                error_rate = error_count / len(session.calls)

                if error_rate >= self.high_error_rate:
                    flags.append(
                        f"High error rate: {error_rate:.1%} "
                        f"({error_count}/{len(session.calls)} calls failed)"
                    )

            # Check for repeated identical endpoints (potential bot behavior)
            if session.calls and len(session.calls) > 5:
                endpoints = [call.endpoint for call in session.calls]
                most_common = Counter(endpoints).most_common(1)[0]
                most_common_endpoint, count = most_common

                if count / len(session.calls) > 0.8:  # 80% same endpoint
                    flags.append(
                        f"Repetitive behavior: {count}/{len(session.calls)} calls "
                        f"to same endpoint '{most_common_endpoint}'"
                    )

            # Check for suspiciously fast calls (< 1ms)
            if session.calls:
                very_fast_calls = sum(
                    1 for call in session.calls
                    if call.response_time_ms < 1
                )

                if very_fast_calls / len(session.calls) > 0.5:
                    flags.append(
                        f"Suspiciously fast: {very_fast_calls}/{len(session.calls)} "
                        f"calls with < 1ms response time"
                    )

            # Check for missing timestamps
            missing_timestamps = sum(
                1 for call in session.calls
                if call.timestamp is None
            )

            if missing_timestamps > 0:
                flags.append(f"Missing timestamps: {missing_timestamps}/{len(session.calls)} calls")

            # Check session duration is reasonable
            if session.duration_seconds > 7200:  # > 2 hours
                flags.append(
                    f"Very long session: {session.duration_seconds/3600:.1f} hours"
                )

            if session.calls and session.duration_seconds < 1 and len(session.calls) > 10:
                flags.append(
                    f"Unrealistic pace: {len(session.calls)} calls in {session.duration_seconds:.2f} seconds"
                )

            # If any flags were raised, add to anomalies
            if flags:
                anomalies.append({
                    'session_id': session.session_id,
                    'user_id': session.user_id,
                    'user_type': session.user_type,
                    'num_calls': len(session.calls),
                    'duration_seconds': session.duration_seconds,
                    'flags': flags
                })

        return anomalies

    def get_validation_summary(self, dataset: Dataset) -> str:
        """Get a human-readable validation summary.

        Args:
            dataset: Dataset to validate

        Returns:
            Formatted string with validation results
        """
        result = self.validate_dataset(dataset)
        quality = self.check_data_quality(dataset)
        anomalies = self.detect_anomalies(dataset)

        lines = [
            "="*70,
            "DATA VALIDATION SUMMARY",
            "="*70,
            "",
            f"Dataset: {dataset.name}",
            f"Total Sessions: {result['total_sessions']}",
            f"Total Calls: {result['total_calls']}",
            "",
            "VALIDATION RESULTS:",
            f"  Valid: {'[OK] YES' if result['valid'] else '[FAIL] NO'}",
            f"  Invalid Sessions: {result['invalid_sessions']}",
            f"  Invalid Calls: {result['invalid_calls']}",
            f"  Errors: {len(result['errors'])}",
            f"  Warnings: {len(result['warnings'])}",
            "",
            "QUALITY METRICS:",
            f"  Missing Values: {quality['missing_values_fraction']:.1%}",
            f"  Duplicate Calls: {quality['duplicate_calls_fraction']:.1%}",
            f"  Response Time Outliers: {quality['response_time_outliers_fraction']:.1%}",
            f"  Error Rate: {quality['error_rate']:.1%}",
            f"  Empty Endpoints: {quality['empty_endpoint_fraction']:.1%}",
            "",
            f"ANOMALIES DETECTED: {len(anomalies)}",
        ]

        if result['errors']:
            lines.append("")
            lines.append("FIRST FEW ERRORS:")
            for error in result['errors'][:5]:
                lines.append(f"  • {error}")
            if len(result['errors']) > 5:
                lines.append(f"  ... and {len(result['errors']) - 5} more")

        if anomalies:
            lines.append("")
            lines.append("FIRST FEW ANOMALIES:")
            for anomaly in anomalies[:3]:
                lines.append(f"  • Session {anomaly['session_id']}:")
                for flag in anomaly['flags']:
                    lines.append(f"      - {flag}")
            if len(anomalies) > 3:
                lines.append(f"  ... and {len(anomalies) - 3} more")

        lines.append("")
        lines.append("="*70)

        return "\n".join(lines)


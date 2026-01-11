"""Sequence building module for converting sessions into API call sequences suitable for Markov chain training.

This module transforms Session objects into various sequence formats needed for learning
transition probabilities between API endpoints.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from datetime import datetime
import re

from preprocessing.models import Session


@dataclass
class ContextualSequence:
    """A sequence with contextual metadata for context-aware modeling."""
    sequence: List[str]
    user_type: str
    time_of_day: str  # morning/afternoon/evening/night
    day_type: str  # weekday/weekend
    session_length_category: str  # short/medium/long


class SequenceBuilder:
    """Converts sessions into API call sequences suitable for Markov chain training.

    The Markov chain learns transition probabilities between API endpoints.
    This class provides various methods to extract and transform sequences for training.

    Attributes:
        normalize_endpoints: Whether to normalize endpoints (lowercase, remove IDs, etc.)
        min_sequence_length: Minimum length for sequences to be included in extraction
    """

    def __init__(self, normalize_endpoints: bool = True, min_sequence_length: int = 1):
        """Initialize the SequenceBuilder.

        Args:
            normalize_endpoints: If True, normalizes endpoints to improve pattern recognition
            min_sequence_length: Minimum number of endpoints in a sequence to include it
        """
        self.normalize_endpoints = normalize_endpoints
        self.min_sequence_length = min_sequence_length

    def normalize_endpoint(self, endpoint: str) -> str:
        """Normalize an endpoint to improve pattern recognition.

        Normalization steps:
        - Convert to lowercase
        - Remove trailing slashes
        - Strip query parameters
        - Replace numeric IDs with {id} placeholder

        This ensures that requests to the same logical endpoint are treated identically.
        For example: /users/1/profile and /users/999/profile both become /users/{id}/profile

        Args:
            endpoint: The endpoint path to normalize

        Returns:
            Normalized endpoint string
        """
        if not endpoint:
            return endpoint

        # Convert to lowercase
        normalized = endpoint.lower()

        # Strip query parameters (everything after ?)
        normalized = normalized.split('?')[0]

        # Remove trailing slashes
        normalized = normalized.rstrip('/')

        # Replace numeric IDs with {id} placeholder
        # Matches patterns like /users/123/ or /products/456
        # This regex finds path segments that are purely numeric
        normalized = re.sub(r'/\d+(?=/|$)', '/{id}', normalized)

        # Also handle UUIDs (8-4-4-4-12 hex pattern)
        normalized = re.sub(
            r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?=/|$)',
            '/{id}',
            normalized
        )

        return normalized

    def _process_sequence(self, sequence: List[str]) -> List[str]:
        """Process a sequence by applying normalization if enabled.

        Args:
            sequence: List of endpoint strings

        Returns:
            Processed sequence (normalized if enabled)
        """
        if self.normalize_endpoints:
            return [self.normalize_endpoint(endpoint) for endpoint in sequence]
        return sequence

    def build_sequences(self, sessions: List[Session], include_metadata: bool = False) -> List[List[str]]:
        """Extract endpoint sequences from a list of sessions.

        Args:
            sessions: List of Session objects to extract sequences from
            include_metadata: If True, returns ContextualSequence objects instead of plain lists

        Returns:
            List of endpoint sequences. Each sequence is a list of endpoint paths in chronological order.
            Sequences shorter than min_sequence_length are filtered out.
        """
        sequences = []

        for session in sessions:
            sequence = session.endpoint_sequence

            # Apply minimum length filter
            if len(sequence) < self.min_sequence_length:
                continue

            # Process the sequence
            processed_sequence = self._process_sequence(sequence)

            sequences.append(processed_sequence)

        return sequences

    def build_labeled_sequences(
        self,
        sessions: List[Session]
    ) -> List[Tuple[List[str], str]]:
        """Create (history, next_endpoint) pairs for evaluating prediction accuracy.

        For a sequence [A, B, C, D], this generates:
        - ([A], B)
        - ([A, B], C)
        - ([A, B, C], D)

        This allows testing: "given we've seen these endpoints, can we predict the next one?"

        Args:
            sessions: List of Session objects to extract labeled sequences from

        Returns:
            List of tuples where each tuple contains:
            - history: List of endpoints seen so far
            - next_endpoint: The next endpoint in the sequence
        """
        labeled_sequences = []

        for session in sessions:
            sequence = self._process_sequence(session.endpoint_sequence)

            # Need at least 2 endpoints to create a labeled pair
            if len(sequence) < 2:
                continue

            # Generate all prefixes with their next endpoint
            for i in range(1, len(sequence)):
                history = sequence[:i]
                next_endpoint = sequence[i]
                labeled_sequences.append((history, next_endpoint))

        return labeled_sequences

    def build_ngrams(
        self,
        sessions: List[Session],
        n: int = 2
    ) -> List[Tuple[str, ...]]:
        """Extract overlapping tuples of N consecutive endpoints.

        For bigrams (N=2) from [A, B, C, D]: [(A,B), (B,C), (C,D)]
        For trigrams (N=3): [(A,B,C), (B,C,D)]

        Args:
            sessions: List of Session objects to extract n-grams from
            n: Size of n-grams to extract (2 for bigrams, 3 for trigrams, etc.)

        Returns:
            List of tuples, where each tuple contains N consecutive endpoints
        """
        if n < 2:
            raise ValueError(f"n must be at least 2, got {n}")

        ngrams = []

        for session in sessions:
            sequence = self._process_sequence(session.endpoint_sequence)

            # Need at least n endpoints to create an n-gram
            if len(sequence) < n:
                continue

            # Extract overlapping n-grams
            for i in range(len(sequence) - n + 1):
                ngram = tuple(sequence[i:i + n])
                ngrams.append(ngram)

        return ngrams

    def build_contextual_sequences(
        self,
        sessions: List[Session]
    ) -> List[ContextualSequence]:
        """Include metadata alongside each sequence for context-aware modeling.

        For each sequence, also returns:
        - user_type: premium/free/guest
        - time_of_day: morning/afternoon/evening/night
        - day_type: weekday/weekend
        - session_length_category: short/medium/long

        Args:
            sessions: List of Session objects to extract contextual sequences from

        Returns:
            List of ContextualSequence objects containing sequences with metadata
        """
        contextual_sequences = []

        for session in sessions:
            sequence = self._process_sequence(session.endpoint_sequence)

            # Apply minimum length filter
            if len(sequence) < self.min_sequence_length:
                continue

            # Extract metadata
            user_type = session.user_type

            # Get time of day from session start
            time_of_day = self._get_time_of_day(session.start_timestamp)

            # Get day type (weekday/weekend)
            day_type = 'weekend' if session.start_timestamp.weekday() >= 5 else 'weekday'

            # Categorize session length
            duration = session.duration_seconds
            if duration < 60:  # Less than 1 minute
                session_length_category = 'short'
            elif duration < 600:  # Less than 10 minutes
                session_length_category = 'medium'
            else:
                session_length_category = 'long'

            contextual_seq = ContextualSequence(
                sequence=sequence,
                user_type=user_type,
                time_of_day=time_of_day,
                day_type=day_type,
                session_length_category=session_length_category
            )

            contextual_sequences.append(contextual_seq)

        return contextual_sequences

    @staticmethod
    def _get_time_of_day(timestamp: datetime) -> str:
        """Convert hour (0-23) to time of day category.

        Categories:
        - night: 0-5 (midnight to 6am)
        - morning: 6-11 (6am to noon)
        - afternoon: 12-17 (noon to 6pm)
        - evening: 18-23 (6pm to midnight)

        Args:
            timestamp: Datetime object to categorize

        Returns:
            Time of day category as a string
        """
        hour = timestamp.hour

        if 0 <= hour < 6:
            return 'night'
        elif 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        else:  # 18 <= hour < 24
            return 'evening'

    def get_transition_counts(
        self,
        sessions: List[Session]
    ) -> Dict[Tuple[str, str], int]:
        """Count the frequency of each endpoint transition.

        This is useful for building the transition probability matrix for the Markov chain.

        Args:
            sessions: List of Session objects to count transitions from

        Returns:
            Dictionary mapping (from_endpoint, to_endpoint) tuples to occurrence counts
        """
        transition_counts = {}

        for session in sessions:
            sequence = self._process_sequence(session.endpoint_sequence)

            # Extract transitions from this sequence
            for i in range(len(sequence) - 1):
                from_endpoint = sequence[i]
                to_endpoint = sequence[i + 1]
                transition = (from_endpoint, to_endpoint)

                transition_counts[transition] = transition_counts.get(transition, 0) + 1

        return transition_counts

    def get_transition_probabilities(
        self,
        sessions: List[Session]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate transition probabilities for each endpoint.

        Returns a nested dictionary where:
        result[from_endpoint][to_endpoint] = probability

        The probability represents: P(next = to_endpoint | current = from_endpoint)

        Args:
            sessions: List of Session objects to calculate probabilities from

        Returns:
            Nested dictionary of transition probabilities
        """
        # First, count all transitions
        transition_counts = self.get_transition_counts(sessions)

        # Count total transitions from each endpoint
        from_endpoint_totals = {}
        for (from_endpoint, to_endpoint), count in transition_counts.items():
            from_endpoint_totals[from_endpoint] = from_endpoint_totals.get(from_endpoint, 0) + count

        # Calculate probabilities
        probabilities = {}
        for (from_endpoint, to_endpoint), count in transition_counts.items():
            if from_endpoint not in probabilities:
                probabilities[from_endpoint] = {}

            probability = count / from_endpoint_totals[from_endpoint]
            probabilities[from_endpoint][to_endpoint] = probability

        return probabilities

    def split_sequences(
        self,
        sessions: List[Session],
        train_ratio: float = 0.8
    ) -> Tuple[List[Session], List[Session]]:
        """Split sessions into train and test sets for model evaluation.

        Args:
            sessions: List of Session objects to split
            train_ratio: Proportion of sessions to include in training set (0.0 to 1.0)

        Returns:
            Tuple of (train_sessions, test_sessions)
        """
        if not 0 < train_ratio < 1:
            raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

        split_index = int(len(sessions) * train_ratio)

        train_sessions = sessions[:split_index]
        test_sessions = sessions[split_index:]

        return train_sessions, test_sessions

    def get_unique_endpoints(self, sessions: List[Session]) -> List[str]:
        """Get all unique endpoints across all sessions.

        Args:
            sessions: List of Session objects

        Returns:
            Sorted list of unique endpoints (normalized if enabled)
        """
        unique_endpoints = set()

        for session in sessions:
            sequence = self._process_sequence(session.endpoint_sequence)
            unique_endpoints.update(sequence)

        return sorted(unique_endpoints)

    def get_sequence_statistics(self, sessions: List[Session]) -> Dict[str, Any]:
        """Get statistical information about the sequences.

        Args:
            sessions: List of Session objects to analyze

        Returns:
            Dictionary containing various statistics about the sequences
        """
        sequences = self.build_sequences(sessions)

        if not sequences:
            return {
                'total_sequences': 0,
                'total_calls': 0,
                'avg_sequence_length': 0.0,
                'min_sequence_length': 0,
                'max_sequence_length': 0,
                'unique_endpoints': 0,
                'total_transitions': 0
            }

        sequence_lengths = [len(seq) for seq in sequences]
        total_calls = sum(sequence_lengths)

        return {
            'total_sequences': len(sequences),
            'total_calls': total_calls,
            'avg_sequence_length': total_calls / len(sequences),
            'min_sequence_length': min(sequence_lengths),
            'max_sequence_length': max(sequence_lengths),
            'unique_endpoints': len(self.get_unique_endpoints(sessions)),
            'total_transitions': sum(len(seq) - 1 for seq in sequences if len(seq) > 1)
        }


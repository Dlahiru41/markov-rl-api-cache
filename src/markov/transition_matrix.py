"""
Efficient sparse transition matrix for Markov chain transition probabilities.

This module provides a TransitionMatrix class that efficiently stores and queries
transition counts and probabilities using a sparse dictionary-based representation.
"""

import json
import heapq
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path
from collections import defaultdict

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class TransitionMatrix:
    """
    Sparse transition matrix for Markov chain state transitions.

    Uses a dictionary-of-dictionaries structure for efficient storage of sparse
    transition data. Supports operations for incrementing counts, computing
    probabilities with optional Laplace smoothing, and serialization.

    Attributes:
        smoothing (float): Laplace smoothing parameter for probability calculation.
        transitions (Dict[str, Dict[str, int]]): Sparse matrix of transition counts.
        total_from (Dict[str, int]): Total outgoing transitions per state.

    Example:
        >>> tm = TransitionMatrix(smoothing=0.001)
        >>> tm.increment("login", "profile", 80)
        >>> tm.increment("login", "browse", 20)
        >>> prob = tm.get_probability("login", "profile")
        >>> print(f"{prob:.2f}")
        0.80
    """

    def __init__(self, smoothing: float = 0.0):
        """
        Initialize a new TransitionMatrix.

        Args:
            smoothing (float): Laplace smoothing parameter. Default is 0 (no smoothing).
                Higher values prevent zero probabilities for unseen transitions.

        Raises:
            ValueError: If smoothing is negative.
        """
        if smoothing < 0:
            raise ValueError(f"Smoothing parameter must be non-negative, got {smoothing}")

        self.smoothing = smoothing
        # Sparse matrix: {from_state: {to_state: count}}
        self.transitions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Total outgoing transitions per state
        self.total_from: Dict[str, int] = defaultdict(int)

    def increment(self, from_state: str, to_state: str, count: int = 1) -> None:
        """
        Add observations of a state transition.

        Time complexity: O(1) average case for dictionary operations.

        Args:
            from_state (str): The source state.
            to_state (str): The destination state.
            count (int): Number of times this transition was observed. Default is 1.

        Raises:
            ValueError: If count is negative.

        Example:
            >>> tm = TransitionMatrix()
            >>> tm.increment("A", "B", count=5)
            >>> tm.get_count("A", "B")
            5
        """
        if count < 0:
            raise ValueError(f"Count must be non-negative, got {count}")

        if count == 0:
            return

        self.transitions[from_state][to_state] += count
        self.total_from[from_state] += count

    def get_count(self, from_state: str, to_state: str) -> int:
        """
        Get the raw count for a specific transition.

        Time complexity: O(1) average case for dictionary lookup.

        Args:
            from_state (str): The source state.
            to_state (str): The destination state.

        Returns:
            int: Number of times this transition was observed. Returns 0 if unseen.

        Example:
            >>> tm = TransitionMatrix()
            >>> tm.increment("A", "B", count=10)
            >>> tm.get_count("A", "B")
            10
            >>> tm.get_count("A", "C")
            0
        """
        return self.transitions.get(from_state, {}).get(to_state, 0)

    def get_probability(self, from_state: str, to_state: str) -> float:
        """
        Calculate the transition probability P(to_state | from_state).

        Uses Laplace smoothing if configured:
        P(to|from) = (count + smoothing) / (total + smoothing * vocab_size)

        Time complexity: O(1) for no smoothing, O(n) for smoothing where n is vocab size.

        Args:
            from_state (str): The source state.
            to_state (str): The destination state.

        Returns:
            float: Probability of transitioning from from_state to to_state.
                Returns 0.0 if from_state has never been observed.

        Example:
            >>> tm = TransitionMatrix(smoothing=0.0)
            >>> tm.increment("A", "B", 8)
            >>> tm.increment("A", "C", 2)
            >>> tm.get_probability("A", "B")
            0.8
        """
        total = self.total_from.get(from_state, 0)

        if total == 0:
            return 0.0

        count = self.get_count(from_state, to_state)

        if self.smoothing == 0.0:
            return count / total

        # Apply Laplace smoothing
        vocab_size = self.num_states
        return (count + self.smoothing) / (total + self.smoothing * vocab_size)

    def get_row(self, from_state: str) -> Dict[str, float]:
        """
        Get all transitions from a state as probabilities.

        Time complexity: O(k) where k is the number of transitions from from_state.

        Args:
            from_state (str): The source state.

        Returns:
            Dict[str, float]: Dictionary mapping destination states to probabilities.
                Returns empty dict if from_state has never been observed.

        Example:
            >>> tm = TransitionMatrix()
            >>> tm.increment("A", "B", 8)
            >>> tm.increment("A", "C", 2)
            >>> row = tm.get_row("A")
            >>> row["B"]
            0.8
            >>> row["C"]
            0.2
        """
        if from_state not in self.transitions:
            return {}

        result = {}
        for to_state in self.transitions[from_state]:
            result[to_state] = self.get_probability(from_state, to_state)

        return result

    def get_top_k(self, from_state: str, k: int) -> List[Tuple[str, float]]:
        """
        Get the k most likely next states with their probabilities.

        Uses a min-heap for efficiency, avoiding full sorting of all transitions.

        Time complexity: O(n log k) where n is the number of transitions from from_state.

        Args:
            from_state (str): The source state.
            k (int): Number of top transitions to return.

        Returns:
            List[Tuple[str, float]]: List of (state, probability) tuples, sorted by
                probability in descending order. May return fewer than k items if
                from_state has fewer than k outgoing transitions.

        Example:
            >>> tm = TransitionMatrix()
            >>> tm.increment("A", "B", 10)
            >>> tm.increment("A", "C", 5)
            >>> tm.increment("A", "D", 3)
            >>> top = tm.get_top_k("A", k=2)
            >>> top[0][0]  # Most likely state
            'B'
            >>> top[0][1]  # Its probability
            0.555...
        """
        if from_state not in self.transitions or k <= 0:
            return []

        row = self.get_row(from_state)

        if len(row) <= k:
            # If we have k or fewer transitions, just sort them all
            return sorted(row.items(), key=lambda x: x[1], reverse=True)

        # Use heap to efficiently get top k
        # heapq is a min-heap, so we negate probabilities to get largest k
        top_k_heap = []
        for state, prob in row.items():
            if len(top_k_heap) < k:
                heapq.heappush(top_k_heap, (prob, state))
            elif prob > top_k_heap[0][0]:
                heapq.heapreplace(top_k_heap, (prob, state))

        # Convert back to (state, prob) tuples and sort in descending order
        result = [(state, prob) for prob, state in top_k_heap]
        result.sort(key=lambda x: x[1], reverse=True)

        return result

    def merge(self, other: 'TransitionMatrix') -> 'TransitionMatrix':
        """
        Combine counts from another TransitionMatrix.

        Creates a new TransitionMatrix with the sum of counts from both matrices.
        The new matrix uses the smoothing parameter from the current matrix.

        Time complexity: O(n + m) where n and m are the number of non-zero entries
            in the two matrices.

        Args:
            other (TransitionMatrix): Another transition matrix to merge with.

        Returns:
            TransitionMatrix: New matrix with combined counts.

        Example:
            >>> tm1 = TransitionMatrix()
            >>> tm1.increment("A", "B", 10)
            >>> tm2 = TransitionMatrix()
            >>> tm2.increment("A", "B", 5)
            >>> tm2.increment("A", "C", 3)
            >>> merged = tm1.merge(tm2)
            >>> merged.get_count("A", "B")
            15
            >>> merged.get_count("A", "C")
            3
        """
        result = TransitionMatrix(smoothing=self.smoothing)

        # Add all transitions from self
        for from_state, to_states in self.transitions.items():
            for to_state, count in to_states.items():
                result.increment(from_state, to_state, count)

        # Add all transitions from other
        for from_state, to_states in other.transitions.items():
            for to_state, count in to_states.items():
                result.increment(from_state, to_state, count)

        return result

    @property
    def num_states(self) -> int:
        """
        Get the number of unique states in the matrix.

        Time complexity: O(n) where n is the number of from_states.

        Returns:
            int: Number of unique states (both source and destination).

        Example:
            >>> tm = TransitionMatrix()
            >>> tm.increment("A", "B")
            >>> tm.increment("B", "C")
            >>> tm.num_states
            3
        """
        states = set(self.transitions.keys())
        for to_states in self.transitions.values():
            states.update(to_states.keys())
        return len(states)

    @property
    def num_transitions(self) -> int:
        """
        Get the number of non-zero transition entries.

        Time complexity: O(n) where n is the number of from_states.

        Returns:
            int: Number of non-zero transitions in the sparse matrix.

        Example:
            >>> tm = TransitionMatrix()
            >>> tm.increment("A", "B")
            >>> tm.increment("A", "C")
            >>> tm.num_transitions
            2
        """
        return sum(len(to_states) for to_states in self.transitions.values())

    @property
    def sparsity(self) -> float:
        """
        Calculate the fraction of zero entries in the complete matrix.

        Time complexity: O(n) where n is the number of from_states.

        Returns:
            float: Fraction of entries that are zero (between 0 and 1).
                Returns 0.0 if matrix is empty.

        Example:
            >>> tm = TransitionMatrix()
            >>> tm.increment("A", "B")
            >>> tm.increment("A", "C")
            >>> # Matrix is 3x3 = 9 cells, 2 non-zero = 7 zeros
            >>> f"{tm.sparsity:.2f}"
            '0.78'
        """
        n = self.num_states
        if n == 0:
            return 0.0

        total_possible = n * n
        non_zero = self.num_transitions
        zeros = total_possible - non_zero

        return zeros / total_possible

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics about the transition matrix.

        Time complexity: O(n log k) where n is total transitions, k=10 for top transitions.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - num_states: Number of unique states
                - num_transitions: Number of non-zero transitions
                - sparsity: Fraction of zero entries
                - avg_transitions_per_state: Average outgoing transitions per state
                - most_common_transitions: Top 10 transitions with counts

        Example:
            >>> tm = TransitionMatrix()
            >>> tm.increment("A", "B", 100)
            >>> tm.increment("A", "C", 50)
            >>> stats = tm.get_statistics()
            >>> stats["num_states"]
            3
            >>> stats["num_transitions"]
            2
        """
        # Collect all transitions with counts
        all_transitions = []
        for from_state, to_states in self.transitions.items():
            for to_state, count in to_states.items():
                all_transitions.append(((from_state, to_state), count))

        # Sort by count and get top 10
        all_transitions.sort(key=lambda x: x[1], reverse=True)
        top_10 = [
            {
                "from": from_state,
                "to": to_state,
                "count": count
            }
            for (from_state, to_state), count in all_transitions[:10]
        ]

        num_states = self.num_states
        num_from_states = len(self.transitions)
        avg_transitions = self.num_transitions / num_from_states if num_from_states > 0 else 0.0

        return {
            "num_states": num_states,
            "num_transitions": self.num_transitions,
            "sparsity": self.sparsity,
            "avg_transitions_per_state": avg_transitions,
            "most_common_transitions": top_10
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a JSON-serializable dictionary.

        Time complexity: O(n) where n is the number of non-zero transitions.

        Returns:
            Dict[str, Any]: Dictionary representation suitable for JSON serialization.

        Example:
            >>> tm = TransitionMatrix(smoothing=0.01)
            >>> tm.increment("A", "B", 5)
            >>> d = tm.to_dict()
            >>> d["smoothing"]
            0.01
            >>> d["transitions"]["A"]["B"]
            5
        """
        # Convert defaultdict to regular dict for JSON serialization
        transitions_dict = {
            from_state: dict(to_states)
            for from_state, to_states in self.transitions.items()
        }

        return {
            "smoothing": self.smoothing,
            "transitions": transitions_dict,
            "total_from": dict(self.total_from)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransitionMatrix':
        """
        Reconstruct a TransitionMatrix from a dictionary.

        Time complexity: O(n) where n is the number of non-zero transitions.

        Args:
            data (Dict[str, Any]): Dictionary created by to_dict().

        Returns:
            TransitionMatrix: Reconstructed transition matrix.

        Raises:
            KeyError: If required keys are missing from data.

        Example:
            >>> tm1 = TransitionMatrix(smoothing=0.01)
            >>> tm1.increment("A", "B", 5)
            >>> data = tm1.to_dict()
            >>> tm2 = TransitionMatrix.from_dict(data)
            >>> tm2.get_count("A", "B")
            5
        """
        matrix = cls(smoothing=data["smoothing"])

        # Restore transitions
        for from_state, to_states in data["transitions"].items():
            for to_state, count in to_states.items():
                matrix.transitions[from_state][to_state] = count

        # Restore totals
        for from_state, total in data["total_from"].items():
            matrix.total_from[from_state] = total

        return matrix

    def save(self, path: str) -> None:
        """
        Save transition matrix to a JSON file.

        Time complexity: O(n) where n is the number of non-zero transitions.

        Args:
            path (str): File path to save to. Parent directories will be created
                if they don't exist.

        Raises:
            IOError: If file cannot be written.

        Example:
            >>> tm = TransitionMatrix()
            >>> tm.increment("A", "B", 5)
            >>> tm.save("matrix.json")
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'TransitionMatrix':
        """
        Load transition matrix from a JSON file.

        Time complexity: O(n) where n is the number of non-zero transitions.

        Args:
            path (str): File path to load from.

        Returns:
            TransitionMatrix: Loaded transition matrix.

        Raises:
            FileNotFoundError: If file does not exist.
            json.JSONDecodeError: If file is not valid JSON.

        Example:
            >>> tm = TransitionMatrix.load("matrix.json")
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return cls.from_dict(data)

    def to_dataframe(self):
        """
        Convert transition matrix to a pandas DataFrame.

        Creates a DataFrame with columns: from_state, to_state, count, probability.
        Useful for visualization and analysis.

        Time complexity: O(n) where n is the number of non-zero transitions.

        Returns:
            pandas.DataFrame: DataFrame with transition data.

        Raises:
            ImportError: If pandas is not installed.

        Example:
            >>> tm = TransitionMatrix()
            >>> tm.increment("A", "B", 8)
            >>> tm.increment("A", "C", 2)
            >>> df = tm.to_dataframe()
            >>> len(df)
            2
            >>> df.columns.tolist()
            ['from_state', 'to_state', 'count', 'probability']
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for to_dataframe(). Install it with: pip install pandas")

        rows = []
        for from_state, to_states in self.transitions.items():
            for to_state, count in to_states.items():
                prob = self.get_probability(from_state, to_state)
                rows.append({
                    'from_state': from_state,
                    'to_state': to_state,
                    'count': count,
                    'probability': prob
                })

        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        """String representation of the TransitionMatrix."""
        return (f"TransitionMatrix(states={self.num_states}, "
                f"transitions={self.num_transitions}, "
                f"sparsity={self.sparsity:.2%})")

    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()


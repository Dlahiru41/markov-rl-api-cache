"""
First-order Markov chain for API call prediction.

A first-order Markov chain models P(next_API | current_API), using only the
current state to predict the next state. This serves as the baseline prediction
model for adaptive API caching.
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set, Any
from collections import defaultdict

from .transition_matrix import TransitionMatrix


class FirstOrderMarkovChain:
    """
    First-order Markov chain for predicting next API calls.

    Uses only the current state to predict the next state: P(next | current).
    Internally uses a TransitionMatrix for efficient storage and querying.

    Attributes:
        smoothing (float): Laplace smoothing parameter.
        transition_matrix (TransitionMatrix): Internal transition probability storage.

    Example:
        >>> mc = FirstOrderMarkovChain(smoothing=0.001)
        >>> sequences = [
        ...     ['login', 'profile', 'orders'],
        ...     ['login', 'browse', 'product']
        ... ]
        >>> mc.fit(sequences)
        >>> predictions = mc.predict('login', k=2)
        >>> print(predictions)
        [('profile', 0.5), ('browse', 0.5)]
    """

    def __init__(self, smoothing: float = 0.0):
        """
        Initialize a first-order Markov chain.

        Args:
            smoothing (float): Laplace smoothing parameter. Default is 0 (no smoothing).
                Values like 0.001-0.01 are recommended for robust predictions.

        Example:
            >>> mc = FirstOrderMarkovChain(smoothing=0.001)
        """
        self.smoothing = smoothing
        self.transition_matrix = TransitionMatrix(smoothing=smoothing)
        self._is_fitted = False

    def fit(self, sequences: List[List[str]]) -> 'FirstOrderMarkovChain':
        """
        Train the Markov chain on sequences of API calls.

        Extracts all consecutive pairs from the sequences and builds the
        transition matrix by counting occurrences.

        Args:
            sequences (List[List[str]]): List of sequences, where each sequence
                is a list of API endpoint strings.

        Returns:
            FirstOrderMarkovChain: Self, for method chaining.

        Example:
            >>> mc = FirstOrderMarkovChain()
            >>> sequences = [['A', 'B', 'C'], ['A', 'B', 'D']]
            >>> mc.fit(sequences)
            >>> mc.is_fitted
            True
        """
        # Reset the transition matrix
        self.transition_matrix = TransitionMatrix(smoothing=self.smoothing)

        # Extract and count all consecutive pairs
        for sequence in sequences:
            if len(sequence) < 2:
                continue  # Need at least 2 states for a transition

            for i in range(len(sequence) - 1):
                current = sequence[i]
                next_state = sequence[i + 1]
                self.transition_matrix.increment(current, next_state)

        self._is_fitted = True
        return self

    def partial_fit(self, sequences: List[List[str]]) -> 'FirstOrderMarkovChain':
        """
        Incrementally update the model with new sequences.

        Adds new observations without losing existing counts. Useful for
        online learning as new data arrives.

        Args:
            sequences (List[List[str]]): New sequences to learn from.

        Returns:
            FirstOrderMarkovChain: Self, for method chaining.

        Example:
            >>> mc = FirstOrderMarkovChain()
            >>> mc.fit([['A', 'B']])
            >>> mc.partial_fit([['A', 'C']])  # Add new data
            >>> # Model now knows both A->B and A->C transitions
        """
        # Extract and count all consecutive pairs
        for sequence in sequences:
            if len(sequence) < 2:
                continue

            for i in range(len(sequence) - 1):
                current = sequence[i]
                next_state = sequence[i + 1]
                self.transition_matrix.increment(current, next_state)

        self._is_fitted = True
        return self

    def update(self, current: str, next_state: str, count: int = 1) -> 'FirstOrderMarkovChain':
        """
        Add a single transition observation.

        Args:
            current (str): Current state.
            next_state (str): Next state.
            count (int): Number of times this transition was observed. Default is 1.

        Returns:
            FirstOrderMarkovChain: Self, for method chaining.

        Example:
            >>> mc = FirstOrderMarkovChain()
            >>> mc.update('login', 'profile', count=10)
            >>> mc.update('login', 'browse', count=5)
        """
        self.transition_matrix.increment(current, next_state, count)
        self._is_fitted = True
        return self

    def predict(self, current: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict the top-k most likely next API calls.

        Args:
            current (str): Current API endpoint.
            k (int): Number of predictions to return. Default is 5.

        Returns:
            List[Tuple[str, float]]: List of (api, probability) tuples, sorted
                by probability in descending order. Returns empty list if
                current state has never been seen.

        Example:
            >>> mc = FirstOrderMarkovChain()
            >>> mc.fit([['A', 'B', 'C'], ['A', 'B', 'D']])
            >>> predictions = mc.predict('A', k=2)
            >>> predictions[0][0]  # Most likely next state
            'B'
        """
        if not self._is_fitted:
            return []

        return self.transition_matrix.get_top_k(current, k=k)

    def predict_proba(self, current: str, target: str) -> float:
        """
        Get the probability of a specific transition.

        Args:
            current (str): Current state.
            target (str): Target next state.

        Returns:
            float: Probability P(target | current). Returns 0 if the transition
                was never observed (or smoothed value if smoothing is enabled).

        Example:
            >>> mc = FirstOrderMarkovChain()
            >>> mc.fit([['A', 'B'], ['A', 'B'], ['A', 'C']])
            >>> mc.predict_proba('A', 'B')
            0.666...
        """
        if not self._is_fitted:
            return 0.0

        return self.transition_matrix.get_probability(current, target)

    def generate_sequence(
        self,
        start: str,
        length: int = 10,
        stop_states: Optional[Set[str]] = None,
        seed: Optional[int] = None
    ) -> List[str]:
        """
        Generate a synthetic sequence using the learned transition probabilities.

        Args:
            start (str): Starting state.
            length (int): Maximum sequence length. Default is 10.
            stop_states (Optional[Set[str]]): States that terminate sequence generation.
                If None, only length limit applies.
            seed (Optional[int]): Random seed for reproducibility.

        Returns:
            List[str]: Generated sequence starting with start state.

        Example:
            >>> mc = FirstOrderMarkovChain()
            >>> mc.fit([['A', 'B', 'C'], ['A', 'B', 'D']])
            >>> seq = mc.generate_sequence('A', length=5, seed=42)
            >>> seq[0]
            'A'
        """
        if not self._is_fitted:
            return [start]

        if seed is not None:
            random.seed(seed)

        sequence = [start]
        current = start

        for _ in range(length - 1):
            # Check if we've reached a stop state
            if stop_states and current in stop_states:
                break

            # Get transition probabilities
            row = self.transition_matrix.get_row(current)

            if not row:
                # No transitions from current state, stop here
                break

            # Sample next state based on probabilities
            states = list(row.keys())
            probs = list(row.values())

            # Normalize probabilities (should already be normalized, but just in case)
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
            else:
                break

            # Sample next state
            next_state = random.choices(states, weights=probs, k=1)[0]
            sequence.append(next_state)
            current = next_state

        return sequence

    def score_sequence(self, sequence: List[str]) -> float:
        """
        Calculate the log-likelihood of an observed sequence.

        Computes the sum of log(P(next|current)) for all transitions in the
        sequence. Useful for anomaly detection - unusual sequences have low scores.

        Args:
            sequence (List[str]): Sequence to score.

        Returns:
            float: Log-likelihood of the sequence. Returns -inf if any transition
                has zero probability (without smoothing).

        Example:
            >>> mc = FirstOrderMarkovChain()
            >>> mc.fit([['A', 'B', 'C']])
            >>> score = mc.score_sequence(['A', 'B', 'C'])
            >>> score < 0  # Log probabilities are negative
            True
        """
        if not self._is_fitted or len(sequence) < 2:
            return 0.0

        log_likelihood = 0.0

        for i in range(len(sequence) - 1):
            current = sequence[i]
            next_state = sequence[i + 1]

            prob = self.transition_matrix.get_probability(current, next_state)

            if prob == 0:
                return float('-inf')

            log_likelihood += np.log(prob)

        return log_likelihood

    def evaluate(
        self,
        test_sequences: List[List[str]],
        k_values: List[int] = [1, 3, 5]
    ) -> Dict[str, float]:
        """
        Evaluate prediction accuracy on test sequences.

        For each transition in test data, checks if the correct next state
        was in the top-k predictions.

        Args:
            test_sequences (List[List[str]]): Test sequences to evaluate on.
            k_values (List[int]): List of k values for top-k accuracy.
                Default is [1, 3, 5].

        Returns:
            Dict[str, float]: Dictionary with metrics:
                - top_k_accuracy: Fraction of correct predictions in top-k (for each k)
                - mrr: Mean Reciprocal Rank
                - coverage: Fraction of test states we could predict for
                - perplexity: exp(avg negative log likelihood)

        Example:
            >>> mc = FirstOrderMarkovChain()
            >>> mc.fit([['A', 'B', 'C']])
            >>> metrics = mc.evaluate([['A', 'B', 'C']], k_values=[1, 3])
            >>> metrics['top_1_accuracy']
            1.0
        """
        if not self._is_fitted:
            return {
                f'top_{k}_accuracy': 0.0 for k in k_values
            } | {'mrr': 0.0, 'coverage': 0.0, 'perplexity': float('inf')}

        total_transitions = 0
        predictable_transitions = 0
        reciprocal_ranks = []
        log_likelihoods = []

        # Initialize counters for each k
        correct_in_top_k = {k: 0 for k in k_values}

        # Evaluate each transition in test sequences
        for sequence in test_sequences:
            if len(sequence) < 2:
                continue

            for i in range(len(sequence) - 1):
                current = sequence[i]
                actual_next = sequence[i + 1]

                total_transitions += 1

                # Get predictions
                predictions = self.predict(current, k=max(k_values))

                if not predictions:
                    # Cannot make predictions for this state
                    continue

                predictable_transitions += 1

                # Find rank of actual next state
                rank = None
                for idx, (pred_state, prob) in enumerate(predictions, 1):
                    if pred_state == actual_next:
                        rank = idx
                        break

                if rank is not None:
                    # Update reciprocal rank
                    reciprocal_ranks.append(1.0 / rank)

                    # Update top-k accuracy
                    for k in k_values:
                        if rank <= k:
                            correct_in_top_k[k] += 1
                else:
                    reciprocal_ranks.append(0.0)

                # Calculate log likelihood for perplexity
                prob = self.predict_proba(current, actual_next)
                if prob > 0:
                    log_likelihoods.append(np.log(prob))

        # Calculate metrics
        metrics = {}

        # Top-k accuracies
        for k in k_values:
            if predictable_transitions > 0:
                accuracy = correct_in_top_k[k] / predictable_transitions
            else:
                accuracy = 0.0
            metrics[f'top_{k}_accuracy'] = accuracy

        # Mean Reciprocal Rank
        if reciprocal_ranks:
            metrics['mrr'] = np.mean(reciprocal_ranks)
        else:
            metrics['mrr'] = 0.0

        # Coverage: fraction of states we could predict for
        if total_transitions > 0:
            metrics['coverage'] = predictable_transitions / total_transitions
        else:
            metrics['coverage'] = 0.0

        # Perplexity: exp(avg negative log likelihood)
        if log_likelihoods:
            avg_neg_log_likelihood = -np.mean(log_likelihoods)
            metrics['perplexity'] = np.exp(avg_neg_log_likelihood)
        else:
            metrics['perplexity'] = float('inf')

        return metrics

    def save(self, path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            path (str): File path to save to. Parent directories will be created
                if they don't exist.

        Example:
            >>> mc = FirstOrderMarkovChain()
            >>> mc.fit([['A', 'B']])
            >>> mc.save('model.json')
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'smoothing': self.smoothing,
            'is_fitted': self._is_fitted,
            'transition_matrix': self.transition_matrix.to_dict()
        }

        with open(path_obj, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'FirstOrderMarkovChain':
        """
        Load a trained model from disk.

        Args:
            path (str): File path to load from.

        Returns:
            FirstOrderMarkovChain: Loaded model.

        Example:
            >>> mc = FirstOrderMarkovChain.load('model.json')
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create instance
        mc = cls(smoothing=data['smoothing'])
        mc._is_fitted = data['is_fitted']

        # Load transition matrix
        mc.transition_matrix = TransitionMatrix.from_dict(data['transition_matrix'])

        return mc

    @property
    def is_fitted(self) -> bool:
        """
        Check if the model has been trained.

        Returns:
            bool: True if model has been fitted to data.

        Example:
            >>> mc = FirstOrderMarkovChain()
            >>> mc.is_fitted
            False
            >>> mc.fit([['A', 'B']])
            >>> mc.is_fitted
            True
        """
        return self._is_fitted

    @property
    def states(self) -> Set[str]:
        """
        Get all known states in the model.

        Returns:
            Set[str]: Set of all states that have been observed.

        Example:
            >>> mc = FirstOrderMarkovChain()
            >>> mc.fit([['A', 'B', 'C']])
            >>> 'A' in mc.states
            True
        """
        if not self._is_fitted:
            return set()

        # Get all states from transition matrix
        states = set(self.transition_matrix.transitions.keys())
        for to_states in self.transition_matrix.transitions.values():
            states.update(to_states.keys())

        return states

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the model.

        Returns:
            Dict[str, Any]: Dictionary with model statistics including
                transition matrix stats and model properties.

        Example:
            >>> mc = FirstOrderMarkovChain()
            >>> mc.fit([['A', 'B', 'C']])
            >>> stats = mc.get_statistics()
            >>> stats['num_states']
            3
        """
        if not self._is_fitted:
            return {
                'is_fitted': False,
                'num_states': 0,
                'num_transitions': 0
            }

        matrix_stats = self.transition_matrix.get_statistics()

        return {
            'is_fitted': True,
            'smoothing': self.smoothing,
            'num_states': matrix_stats['num_states'],
            'num_transitions': matrix_stats['num_transitions'],
            'sparsity': matrix_stats['sparsity'],
            'avg_transitions_per_state': matrix_stats['avg_transitions_per_state'],
            'most_common_transitions': matrix_stats['most_common_transitions'][:5]
        }

    def __repr__(self) -> str:
        """String representation of the Markov chain."""
        if self._is_fitted:
            num_states = len(self.states)
            num_transitions = self.transition_matrix.num_transitions
            return (f"FirstOrderMarkovChain(states={num_states}, "
                   f"transitions={num_transitions}, "
                   f"smoothing={self.smoothing})")
        else:
            return f"FirstOrderMarkovChain(fitted=False, smoothing={self.smoothing})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()


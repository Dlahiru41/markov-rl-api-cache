"""
Second-order Markov chain for API call prediction.

A second-order Markov chain models P(next | current, previous), using the last
two states to predict the next state. This captures more context than first-order
chains - for example, the behavior after "login → profile" might be different from
"browse → profile".

The tradeoff is that second-order models require more data to reliably estimate
these higher-order transitions, as the state space grows quadratically.

Usage Guidelines:
    - Use second-order when: API patterns depend on context, you have lots of data
    - Use first-order when: Limited data, patterns are mostly local, need robustness
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set, Any
from collections import defaultdict

from .transition_matrix import TransitionMatrix
from .first_order import FirstOrderMarkovChain


# Special token to represent the start of a sequence
START_TOKEN = "<START>"


class SecondOrderMarkovChain:
    """
    Second-order Markov chain for predicting next API calls.

    Uses the last two states to predict the next state: P(next | current, previous).
    Internally maintains a TransitionMatrix where states are (previous, current) pairs,
    and optionally falls back to first-order predictions for unseen pairs.

    Attributes:
        smoothing (float): Laplace smoothing parameter.
        fallback_to_first_order (bool): Whether to use first-order fallback for unseen pairs.
        transition_matrix (TransitionMatrix): Second-order transition storage.
        first_order_model (Optional[FirstOrderMarkovChain]): Fallback first-order model.

    Example:
        >>> mc2 = SecondOrderMarkovChain(smoothing=0.001, fallback_to_first_order=True)
        >>> sequences = [
        ...     ['login', 'profile', 'orders'],
        ...     ['login', 'browse', 'product']
        ... ]
        >>> mc2.fit(sequences)
        >>> predictions = mc2.predict('login', 'profile', k=2)
        >>> print(predictions)
        [('orders', 0.8), ...]
    """

    # Delimiter for creating composite state keys
    STATE_DELIMITER = "|"

    def __init__(self, smoothing: float = 0.0, fallback_to_first_order: bool = True):
        """
        Initialize a second-order Markov chain.

        Args:
            smoothing (float): Laplace smoothing parameter. Default is 0 (no smoothing).
                Values like 0.001-0.01 are recommended for robust predictions.
            fallback_to_first_order (bool): If True, train a first-order model alongside
                the second-order one and use it as fallback for unseen state pairs.
                Default is True.

        Example:
            >>> mc2 = SecondOrderMarkovChain(smoothing=0.001, fallback_to_first_order=True)
        """
        self.smoothing = smoothing
        self.fallback_to_first_order = fallback_to_first_order
        self.transition_matrix = TransitionMatrix(smoothing=smoothing)
        self.first_order_model: Optional[FirstOrderMarkovChain] = None
        self._is_fitted = False

        if fallback_to_first_order:
            self.first_order_model = FirstOrderMarkovChain(smoothing=smoothing)

    @staticmethod
    def _make_state_key(previous: str, current: str) -> str:
        """
        Create a composite state key from two API endpoints.

        Args:
            previous (str): Previous API endpoint.
            current (str): Current API endpoint.

        Returns:
            str: Composite key in format "previous|current".

        Example:
            >>> SecondOrderMarkovChain._make_state_key('login', 'profile')
            'login|profile'
        """
        return f"{previous}{SecondOrderMarkovChain.STATE_DELIMITER}{current}"

    @staticmethod
    def _parse_state_key(key: str) -> Tuple[str, str]:
        """
        Parse a composite state key back into two API endpoints.

        Args:
            key (str): Composite key in format "previous|current".

        Returns:
            Tuple[str, str]: (previous, current) API endpoints.

        Example:
            >>> SecondOrderMarkovChain._parse_state_key('login|profile')
            ('login', 'profile')
        """
        parts = key.split(SecondOrderMarkovChain.STATE_DELIMITER, 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid state key format: {key}")
        return parts[0], parts[1]

    def fit(self, sequences: List[List[str]]) -> 'SecondOrderMarkovChain':
        """
        Train the second-order Markov chain on sequences of API calls.

        Extracts all (previous, current, next) triples from the sequences and builds
        the transition matrix. If fallback is enabled, also trains the first-order model.

        Args:
            sequences (List[List[str]]): List of sequences, where each sequence
                is a list of API endpoint strings.

        Returns:
            SecondOrderMarkovChain: Self, for method chaining.

        Example:
            >>> mc2 = SecondOrderMarkovChain()
            >>> sequences = [['A', 'B', 'C'], ['A', 'B', 'D']]
            >>> mc2.fit(sequences)
            >>> mc2.is_fitted
            True
        """
        # Reset the transition matrix
        self.transition_matrix = TransitionMatrix(smoothing=self.smoothing)

        # Train first-order model if fallback is enabled
        if self.fallback_to_first_order and self.first_order_model is not None:
            self.first_order_model.fit(sequences)

        # Extract and count all (previous, current, next) triples
        for sequence in sequences:
            if len(sequence) < 2:
                continue  # Need at least 2 states

            # Handle the first transition: (START, first, second)
            if len(sequence) >= 2:
                state_key = self._make_state_key(START_TOKEN, sequence[0])
                self.transition_matrix.increment(state_key, sequence[1])

            # Handle subsequent transitions
            for i in range(len(sequence) - 2):
                previous = sequence[i]
                current = sequence[i + 1]
                next_state = sequence[i + 2]

                state_key = self._make_state_key(previous, current)
                self.transition_matrix.increment(state_key, next_state)

        self._is_fitted = True
        return self

    def partial_fit(self, sequences: List[List[str]]) -> 'SecondOrderMarkovChain':
        """
        Incrementally update the model with new sequences.

        Adds new observations without losing existing counts. Useful for
        online learning as new data arrives.

        Args:
            sequences (List[List[str]]): New sequences to learn from.

        Returns:
            SecondOrderMarkovChain: Self, for method chaining.

        Example:
            >>> mc2 = SecondOrderMarkovChain()
            >>> mc2.fit([['A', 'B', 'C']])
            >>> mc2.partial_fit([['A', 'C', 'D']])  # Add new data
            >>> # Model now knows both patterns
        """
        # Update first-order model if fallback is enabled
        if self.fallback_to_first_order and self.first_order_model is not None:
            self.first_order_model.partial_fit(sequences)

        # Extract and count all (previous, current, next) triples
        for sequence in sequences:
            if len(sequence) < 2:
                continue

            # Handle the first transition
            if len(sequence) >= 2:
                state_key = self._make_state_key(START_TOKEN, sequence[0])
                self.transition_matrix.increment(state_key, sequence[1])

            # Handle subsequent transitions
            for i in range(len(sequence) - 2):
                previous = sequence[i]
                current = sequence[i + 1]
                next_state = sequence[i + 2]

                state_key = self._make_state_key(previous, current)
                self.transition_matrix.increment(state_key, next_state)

        self._is_fitted = True
        return self

    def update(
        self,
        previous: str,
        current: str,
        next_state: str,
        count: int = 1
    ) -> 'SecondOrderMarkovChain':
        """
        Add a single second-order transition observation.

        Args:
            previous (str): Previous state.
            current (str): Current state.
            next_state (str): Next state.
            count (int): Number of times this transition was observed. Default is 1.

        Returns:
            SecondOrderMarkovChain: Self, for method chaining.

        Example:
            >>> mc2 = SecondOrderMarkovChain()
            >>> mc2.update('login', 'profile', 'orders', count=10)
            >>> mc2.update('browse', 'profile', 'settings', count=5)
        """
        state_key = self._make_state_key(previous, current)
        self.transition_matrix.increment(state_key, next_state, count)

        # Also update first-order model if fallback is enabled
        if self.fallback_to_first_order and self.first_order_model is not None:
            self.first_order_model.update(current, next_state, count)

        self._is_fitted = True
        return self

    def predict(
        self,
        previous: str,
        current: str,
        k: int = 5,
        use_fallback: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Predict the top-k most likely next API calls given the last two APIs.

        Args:
            previous (str): Previous API endpoint.
            current (str): Current API endpoint.
            k (int): Number of predictions to return. Default is 5.
            use_fallback (bool): If True and the (previous, current) pair is unseen,
                fall back to first-order predictions using just current. Default is True.

        Returns:
            List[Tuple[str, float]]: List of (api, probability) tuples, sorted
                by probability in descending order. Returns empty list if the
                state pair has never been seen and fallback is disabled or unavailable.

        Example:
            >>> mc2 = SecondOrderMarkovChain(fallback_to_first_order=True)
            >>> mc2.fit([['A', 'B', 'C'], ['A', 'B', 'D']])
            >>> predictions = mc2.predict('A', 'B', k=2)
            >>> len(predictions) <= 2
            True
        """
        if not self._is_fitted:
            return []

        state_key = self._make_state_key(previous, current)
        predictions = self.transition_matrix.get_top_k(state_key, k=k)

        # If no predictions and fallback is enabled, use first-order model
        if not predictions and use_fallback and self.first_order_model is not None:
            predictions = self.first_order_model.predict(current, k=k)

        return predictions

    def predict_proba(
        self,
        previous: str,
        current: str,
        target: str,
        use_fallback: bool = True
    ) -> float:
        """
        Get the probability of a specific transition given the last two states.

        Args:
            previous (str): Previous state.
            current (str): Current state.
            target (str): Target next state.
            use_fallback (bool): If True and the (previous, current) pair is unseen,
                fall back to first-order probability. Default is True.

        Returns:
            float: Probability P(target | current, previous). Returns 0 if the
                transition was never observed and fallback is disabled (or smoothed
                value if smoothing is enabled).

        Example:
            >>> mc2 = SecondOrderMarkovChain()
            >>> mc2.fit([['A', 'B', 'C'], ['A', 'B', 'C'], ['A', 'B', 'D']])
            >>> mc2.predict_proba('A', 'B', 'C')
            0.666...
        """
        if not self._is_fitted:
            return 0.0

        state_key = self._make_state_key(previous, current)
        prob = self.transition_matrix.get_probability(state_key, target)

        # If probability is 0 (unseen pair) and fallback is enabled
        if prob == 0.0 and use_fallback and self.first_order_model is not None:
            # Check if we have any transitions from this state pair
            row = self.transition_matrix.get_row(state_key)
            if not row:  # No transitions from this pair, use fallback
                prob = self.first_order_model.predict_proba(current, target)

        return prob

    def generate_sequence(
        self,
        start: str,
        length: int = 10,
        stop_states: Optional[Set[str]] = None,
        seed: Optional[int] = None,
        use_fallback: bool = True
    ) -> List[str]:
        """
        Generate a synthetic sequence using the learned transition probabilities.

        Args:
            start (str): Starting state (will be treated as first state after START).
            length (int): Maximum sequence length. Default is 10.
            stop_states (Optional[Set[str]]): States that terminate sequence generation.
                If None, only length limit applies.
            seed (Optional[int]): Random seed for reproducibility.
            use_fallback (bool): Whether to use first-order fallback for unseen pairs.

        Returns:
            List[str]: Generated sequence starting with start state.

        Example:
            >>> mc2 = SecondOrderMarkovChain()
            >>> mc2.fit([['A', 'B', 'C'], ['A', 'B', 'D']])
            >>> seq = mc2.generate_sequence('A', length=5, seed=42)
            >>> seq[0]
            'A'
        """
        if not self._is_fitted:
            return [start]

        if seed is not None:
            random.seed(seed)

        # Start with START_TOKEN -> start
        previous = START_TOKEN
        current = start
        sequence = [start]

        for _ in range(length - 1):
            # Check if we've reached a stop state
            if stop_states and current in stop_states:
                break

            # Get predictions
            predictions = self.predict(previous, current, k=100, use_fallback=use_fallback)

            if not predictions:
                # No transitions available, stop here
                break

            # Sample next state based on probabilities
            states = [pred[0] for pred in predictions]
            probs = [pred[1] for pred in predictions]

            # Normalize probabilities (should already be normalized, but just in case)
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
            else:
                break

            # Sample next state
            next_state = random.choices(states, weights=probs, k=1)[0]
            sequence.append(next_state)

            # Update state history
            previous = current
            current = next_state

        return sequence

    def score_sequence(self, sequence: List[str], use_fallback: bool = True) -> float:
        """
        Calculate the log-likelihood of an observed sequence.

        Computes the sum of log(P(next|current, previous)) for all transitions.
        Useful for anomaly detection - unusual sequences have low scores.

        Args:
            sequence (List[str]): Sequence to score.
            use_fallback (bool): Whether to use first-order fallback for unseen pairs.

        Returns:
            float: Log-likelihood of the sequence. Returns -inf if any transition
                has zero probability (without smoothing or fallback).

        Example:
            >>> mc2 = SecondOrderMarkovChain()
            >>> mc2.fit([['A', 'B', 'C']])
            >>> score = mc2.score_sequence(['A', 'B', 'C'])
            >>> score < 0  # Log probabilities are negative
            True
        """
        if not self._is_fitted or len(sequence) < 2:
            return 0.0

        log_likelihood = 0.0

        # Score first transition: START -> sequence[0] -> sequence[1]
        if len(sequence) >= 2:
            prob = self.predict_proba(START_TOKEN, sequence[0], sequence[1], use_fallback)
            if prob == 0:
                return float('-inf')
            log_likelihood += np.log(prob)

        # Score subsequent transitions
        for i in range(len(sequence) - 2):
            previous = sequence[i]
            current = sequence[i + 1]
            next_state = sequence[i + 2]

            prob = self.predict_proba(previous, current, next_state, use_fallback)

            if prob == 0:
                return float('-inf')

            log_likelihood += np.log(prob)

        return log_likelihood

    def evaluate(
        self,
        test_sequences: List[List[str]],
        k_values: List[int] = [1, 3, 5],
        track_fallback: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate prediction accuracy on test sequences.

        For each transition in test data, checks if the correct next state
        was in the top-k predictions. Also tracks how often fallback was used.

        Args:
            test_sequences (List[List[str]]): Test sequences to evaluate on.
            k_values (List[int]): List of k values for top-k accuracy.
                Default is [1, 3, 5].
            track_fallback (bool): If True, track and report fallback usage rate.

        Returns:
            Dict[str, float]: Dictionary with metrics:
                - top_k_accuracy: Fraction of correct predictions in top-k (for each k)
                - mrr: Mean Reciprocal Rank
                - coverage: Fraction of test states we could predict for
                - perplexity: exp(avg negative log likelihood)
                - fallback_rate: Fraction of predictions using first-order fallback
                    (only if track_fallback=True and fallback is enabled)

        Example:
            >>> mc2 = SecondOrderMarkovChain()
            >>> mc2.fit([['A', 'B', 'C']])
            >>> metrics = mc2.evaluate([['A', 'B', 'C']], k_values=[1, 3])
            >>> metrics['top_1_accuracy']
            1.0
        """
        if not self._is_fitted:
            return {
                f'top_{k}_accuracy': 0.0 for k in k_values
            } | {'mrr': 0.0, 'coverage': 0.0, 'perplexity': float('inf'), 'fallback_rate': 0.0}

        total_transitions = 0
        predictable_transitions = 0
        fallback_used = 0
        reciprocal_ranks = []
        log_likelihoods = []

        # Initialize counters for each k
        correct_in_top_k = {k: 0 for k in k_values}

        # Evaluate each transition in test sequences
        for sequence in test_sequences:
            if len(sequence) < 2:
                continue

            # Evaluate first transition: START -> sequence[0] -> sequence[1]
            if len(sequence) >= 2:
                previous = START_TOKEN
                current = sequence[0]
                actual_next = sequence[1]

                total_transitions += 1

                # Check if we need fallback
                state_key = self._make_state_key(previous, current)
                second_order_predictions = self.transition_matrix.get_top_k(state_key, k=max(k_values))
                used_fallback = False

                if not second_order_predictions and self.fallback_to_first_order and self.first_order_model:
                    predictions = self.first_order_model.predict(current, k=max(k_values))
                    used_fallback = True
                else:
                    predictions = second_order_predictions

                if track_fallback and used_fallback:
                    fallback_used += 1

                if predictions:
                    predictable_transitions += 1

                    # Find rank of actual next state
                    rank = None
                    for idx, (pred_state, prob) in enumerate(predictions, 1):
                        if pred_state == actual_next:
                            rank = idx
                            break

                    if rank is not None:
                        reciprocal_ranks.append(1.0 / rank)
                        for k in k_values:
                            if rank <= k:
                                correct_in_top_k[k] += 1
                    else:
                        reciprocal_ranks.append(0.0)

                    # Calculate log likelihood
                    prob = self.predict_proba(previous, current, actual_next, use_fallback=True)
                    if prob > 0:
                        log_likelihoods.append(np.log(prob))

            # Evaluate subsequent transitions
            for i in range(len(sequence) - 2):
                previous = sequence[i]
                current = sequence[i + 1]
                actual_next = sequence[i + 2]

                total_transitions += 1

                # Check if we need fallback
                state_key = self._make_state_key(previous, current)
                second_order_predictions = self.transition_matrix.get_top_k(state_key, k=max(k_values))
                used_fallback = False

                if not second_order_predictions and self.fallback_to_first_order and self.first_order_model:
                    predictions = self.first_order_model.predict(current, k=max(k_values))
                    used_fallback = True
                else:
                    predictions = second_order_predictions

                if track_fallback and used_fallback:
                    fallback_used += 1

                if not predictions:
                    continue

                predictable_transitions += 1

                # Find rank of actual next state
                rank = None
                for idx, (pred_state, prob) in enumerate(predictions, 1):
                    if pred_state == actual_next:
                        rank = idx
                        break

                if rank is not None:
                    reciprocal_ranks.append(1.0 / rank)
                    for k in k_values:
                        if rank <= k:
                            correct_in_top_k[k] += 1
                else:
                    reciprocal_ranks.append(0.0)

                # Calculate log likelihood
                prob = self.predict_proba(previous, current, actual_next, use_fallback=True)
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

        # Coverage
        if total_transitions > 0:
            metrics['coverage'] = predictable_transitions / total_transitions
        else:
            metrics['coverage'] = 0.0

        # Perplexity
        if log_likelihoods:
            avg_neg_log_likelihood = -np.mean(log_likelihoods)
            metrics['perplexity'] = np.exp(avg_neg_log_likelihood)
        else:
            metrics['perplexity'] = float('inf')

        # Fallback rate
        if track_fallback and predictable_transitions > 0:
            metrics['fallback_rate'] = fallback_used / total_transitions
        else:
            metrics['fallback_rate'] = 0.0

        return metrics

    def compare_with_first_order(
        self,
        test_sequences: List[List[str]],
        k_values: List[int] = [1, 3, 5]
    ) -> Dict[str, Any]:
        """
        Compare second-order accuracy directly with first-order on the same test data.

        Args:
            test_sequences (List[List[str]]): Test sequences to evaluate on.
            k_values (List[int]): List of k values for comparison. Default is [1, 3, 5].

        Returns:
            Dict[str, Any]: Dictionary with comparison metrics:
                - second_order_metrics: Full metrics dict from second-order evaluation
                - first_order_metrics: Full metrics dict from first-order evaluation
                - improvement: Dict of {metric: improvement_percentage} for each metric
                - fallback_rate: How often second-order fell back to first-order

        Example:
            >>> mc2 = SecondOrderMarkovChain(fallback_to_first_order=True)
            >>> mc2.fit(train_sequences)
            >>> comparison = mc2.compare_with_first_order(test_sequences)
            >>> print(f"Improvement: {comparison['improvement']['top_1_accuracy']:.1f}%")
        """
        if not self._is_fitted:
            return {
                'second_order_metrics': {},
                'first_order_metrics': {},
                'improvement': {},
                'fallback_rate': 0.0
            }

        # Evaluate second-order model
        second_order_metrics = self.evaluate(test_sequences, k_values=k_values, track_fallback=True)

        # Evaluate first-order model (or train a temporary one if fallback is disabled)
        if self.first_order_model is not None:
            first_order_metrics = self.first_order_model.evaluate(test_sequences, k_values=k_values)
        else:
            # Train a temporary first-order model for comparison
            temp_first_order = FirstOrderMarkovChain(smoothing=self.smoothing)
            temp_first_order.fit(test_sequences)
            first_order_metrics = temp_first_order.evaluate(test_sequences, k_values=k_values)

        # Calculate improvements
        improvement = {}
        for metric in ['mrr', 'coverage'] + [f'top_{k}_accuracy' for k in k_values]:
            if metric in second_order_metrics and metric in first_order_metrics:
                first_val = first_order_metrics[metric]
                second_val = second_order_metrics[metric]

                if first_val > 0:
                    improvement[metric] = ((second_val - first_val) / first_val) * 100
                else:
                    improvement[metric] = 0.0

        # Perplexity improvement (lower is better, so reverse the calculation)
        if 'perplexity' in second_order_metrics and 'perplexity' in first_order_metrics:
            first_perp = first_order_metrics['perplexity']
            second_perp = second_order_metrics['perplexity']
            if first_perp != float('inf') and second_perp != float('inf') and first_perp > 0:
                improvement['perplexity'] = ((first_perp - second_perp) / first_perp) * 100
            else:
                improvement['perplexity'] = 0.0

        return {
            'second_order_metrics': second_order_metrics,
            'first_order_metrics': first_order_metrics,
            'improvement': improvement,
            'fallback_rate': second_order_metrics.get('fallback_rate', 0.0)
        }

    def save(self, path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            path (str): File path to save to. Parent directories will be created
                if they don't exist.

        Example:
            >>> mc2 = SecondOrderMarkovChain()
            >>> mc2.fit([['A', 'B', 'C']])
            >>> mc2.save('model2.json')
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'smoothing': self.smoothing,
            'fallback_to_first_order': self.fallback_to_first_order,
            'is_fitted': self._is_fitted,
            'transition_matrix': self.transition_matrix.to_dict()
        }

        # Save first-order model if it exists
        if self.first_order_model is not None:
            data['first_order_model'] = {
                'smoothing': self.first_order_model.smoothing,
                'is_fitted': self.first_order_model.is_fitted,
                'transition_matrix': self.first_order_model.transition_matrix.to_dict()
            }

        with open(path_obj, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'SecondOrderMarkovChain':
        """
        Load a trained model from disk.

        Args:
            path (str): File path to load from.

        Returns:
            SecondOrderMarkovChain: Loaded model.

        Example:
            >>> mc2 = SecondOrderMarkovChain.load('model2.json')
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create instance
        mc2 = cls(
            smoothing=data['smoothing'],
            fallback_to_first_order=data['fallback_to_first_order']
        )
        mc2._is_fitted = data['is_fitted']

        # Load transition matrix
        mc2.transition_matrix = TransitionMatrix.from_dict(data['transition_matrix'])

        # Load first-order model if it exists
        if 'first_order_model' in data and mc2.first_order_model is not None:
            fo_data = data['first_order_model']
            mc2.first_order_model = FirstOrderMarkovChain(smoothing=fo_data['smoothing'])
            mc2.first_order_model._is_fitted = fo_data['is_fitted']
            mc2.first_order_model.transition_matrix = TransitionMatrix.from_dict(
                fo_data['transition_matrix']
            )

        return mc2

    @property
    def is_fitted(self) -> bool:
        """
        Check if the model has been trained.

        Returns:
            bool: True if model has been fitted to data.

        Example:
            >>> mc2 = SecondOrderMarkovChain()
            >>> mc2.is_fitted
            False
            >>> mc2.fit([['A', 'B', 'C']])
            >>> mc2.is_fitted
            True
        """
        return self._is_fitted

    @property
    def states(self) -> Set[str]:
        """
        Get all known individual states (API endpoints) in the model.

        Note: This returns individual states, not state pairs. The internal
        representation uses pairs, but this property extracts unique states.

        Returns:
            Set[str]: Set of all individual states that have been observed.

        Example:
            >>> mc2 = SecondOrderMarkovChain()
            >>> mc2.fit([['A', 'B', 'C']])
            >>> 'A' in mc2.states
            True
        """
        if not self._is_fitted:
            return set()

        states = set()

        # Extract states from state keys
        for state_key in self.transition_matrix.transitions.keys():
            if state_key.startswith(f"{START_TOKEN}{self.STATE_DELIMITER}"):
                # Special case: START|state
                _, state = self._parse_state_key(state_key)
                states.add(state)
            else:
                try:
                    prev, curr = self._parse_state_key(state_key)
                    if prev != START_TOKEN:
                        states.add(prev)
                    states.add(curr)
                except ValueError:
                    continue

        # Also get states from destinations
        for to_states in self.transition_matrix.transitions.values():
            states.update(to_states.keys())

        # Remove START_TOKEN if it got included
        states.discard(START_TOKEN)

        return states

    @property
    def state_pairs(self) -> Set[Tuple[str, str]]:
        """
        Get all known state pairs (previous, current) in the model.

        Returns:
            Set[Tuple[str, str]]: Set of all (previous, current) state pairs observed.

        Example:
            >>> mc2 = SecondOrderMarkovChain()
            >>> mc2.fit([['A', 'B', 'C']])
            >>> ('A', 'B') in mc2.state_pairs
            True
        """
        if not self._is_fitted:
            return set()

        pairs = set()
        for state_key in self.transition_matrix.transitions.keys():
            try:
                prev, curr = self._parse_state_key(state_key)
                pairs.add((prev, curr))
            except ValueError:
                continue

        return pairs

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the model.

        Returns:
            Dict[str, Any]: Dictionary with model statistics including
                transition matrix stats, state counts, and fallback info.

        Example:
            >>> mc2 = SecondOrderMarkovChain()
            >>> mc2.fit([['A', 'B', 'C']])
            >>> stats = mc2.get_statistics()
            >>> stats['num_individual_states']
            3
        """
        if not self._is_fitted:
            return {
                'is_fitted': False,
                'num_individual_states': 0,
                'num_state_pairs': 0,
                'num_transitions': 0
            }

        matrix_stats = self.transition_matrix.get_statistics()

        stats = {
            'is_fitted': True,
            'smoothing': self.smoothing,
            'fallback_to_first_order': self.fallback_to_first_order,
            'num_individual_states': len(self.states),
            'num_state_pairs': len(self.state_pairs),
            'num_transitions': matrix_stats['num_transitions'],
            'sparsity': matrix_stats['sparsity'],
            'avg_transitions_per_state_pair': matrix_stats['avg_transitions_per_state'],
            'most_common_transitions': matrix_stats['most_common_transitions'][:5]
        }

        # Add first-order model stats if available
        if self.first_order_model is not None and self.first_order_model.is_fitted:
            fo_stats = self.first_order_model.get_statistics()
            stats['first_order_stats'] = {
                'num_states': fo_stats['num_states'],
                'num_transitions': fo_stats['num_transitions']
            }

        return stats

    def __repr__(self) -> str:
        """String representation of the Markov chain."""
        if self._is_fitted:
            num_states = len(self.states)
            num_pairs = len(self.state_pairs)
            num_transitions = self.transition_matrix.num_transitions
            return (f"SecondOrderMarkovChain(states={num_states}, "
                   f"pairs={num_pairs}, transitions={num_transitions}, "
                   f"smoothing={self.smoothing}, fallback={self.fallback_to_first_order})")
        else:
            return (f"SecondOrderMarkovChain(fitted=False, smoothing={self.smoothing}, "
                   f"fallback={self.fallback_to_first_order})")

    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()


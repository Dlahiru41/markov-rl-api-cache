"""
Unified Markov chain predictor with RL integration.

Provides a consistent interface for all Markov chain variants (first-order,
second-order, context-aware) and integrates cleanly with the RL system.
The RL agent doesn't need to know which type of Markov chain is being used.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Union
from collections import deque

from .first_order import FirstOrderMarkovChain
from .second_order import SecondOrderMarkovChain
from .context_aware import ContextAwareMarkovChain


class MarkovPredictor:
    """
    Unified predictor interface for all Markov chain variants.

    Provides a clean abstraction layer for RL integration. Manages call history
    automatically and provides fixed-size state vectors for the RL agent.

    Attributes:
        order (int): Markov chain order (1 or 2).
        context_aware (bool): Whether to use context-aware predictions.
        context_features (List[str]): List of context feature names (if context_aware).
        smoothing (float): Laplace smoothing parameter.
        history_size (int): Maximum size of call history window.
        chain (Union[...]): Internal Markov chain instance.
        history (deque): Sliding window of recent API calls.
        prediction_count (int): Number of predictions made.
        correct_predictions (Dict[int, int]): Correct predictions at each k.

    Example:
        >>> predictor = MarkovPredictor(
        ...     order=1,
        ...     context_aware=True,
        ...     context_features=['user_type', 'time_of_day']
        ... )
        >>> predictor.fit(sequences, contexts)
        >>> predictor.observe('login')
        >>> predictions = predictor.predict(k=5, context={'user_type': 'premium', 'hour': 10})
        >>> state = predictor.get_state_vector(k=5)
    """

    def __init__(
        self,
        order: int = 1,
        context_aware: bool = False,
        context_features: Optional[List[str]] = None,
        smoothing: float = 0.001,
        history_size: int = 10,
        fallback_strategy: str = 'global'
    ):
        """
        Initialize the unified predictor.

        Args:
            order (int): Markov chain order (1 or 2). Default is 1.
            context_aware (bool): Whether to use context-aware predictions. Default is False.
            context_features (Optional[List[str]]): Context feature names. Required if context_aware.
            smoothing (float): Laplace smoothing parameter. Default is 0.001.
            history_size (int): Maximum size of call history window. Default is 10.
            fallback_strategy (str): Fallback strategy for context-aware chains. Default is 'global'.

        Raises:
            ValueError: If order is invalid or context_features missing when context_aware is True.

        Example:
            >>> predictor = MarkovPredictor(order=1, context_aware=False)
            >>> predictor = MarkovPredictor(order=2, context_aware=True,
            ...                            context_features=['user_type'])
        """
        if order not in [1, 2]:
            raise ValueError(f"order must be 1 or 2, got {order}")

        if context_aware and not context_features:
            raise ValueError("context_features required when context_aware=True")

        self.order = order
        self.context_aware = context_aware
        self.context_features = context_features or []
        self.smoothing = smoothing
        self.history_size = history_size
        self.fallback_strategy = fallback_strategy

        # Create appropriate chain type
        if context_aware:
            self.chain = ContextAwareMarkovChain(
                context_features=context_features,
                order=order,
                fallback_strategy=fallback_strategy,
                smoothing=smoothing
            )
        elif order == 1:
            self.chain = FirstOrderMarkovChain(smoothing=smoothing)
        else:  # order == 2
            self.chain = SecondOrderMarkovChain(
                smoothing=smoothing,
                fallback_to_first_order=True
            )

        # Call history (sliding window)
        self.history: deque = deque(maxlen=history_size)

        # Context history (for context-aware predictions)
        self.context_history: deque = deque(maxlen=history_size)

        # Tracking metrics
        self.prediction_count = 0
        self.correct_predictions: Dict[int, int] = {}  # k -> count
        self.total_confidence = 0.0
        self.all_predictions: List[List[Tuple[str, float]]] = []
        self.all_actuals: List[str] = []

        # API vocabulary (for encoding)
        self.api_vocab: Dict[str, int] = {}
        self.vocab_size = 0

        self._is_fitted = False

    def fit(
        self,
        sequences: List[List[str]],
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> 'MarkovPredictor':
        """
        Train the predictor on sequences.

        Args:
            sequences (List[List[str]]): List of API call sequences.
            contexts (Optional[List[Dict[str, Any]]]): Context for each sequence
                (required if context_aware=True).

        Returns:
            MarkovPredictor: Self, for method chaining.

        Raises:
            ValueError: If contexts required but not provided.

        Example:
            >>> predictor.fit(sequences, contexts)
        """
        if self.context_aware:
            if contexts is None:
                raise ValueError("contexts required when context_aware=True")
            if len(sequences) != len(contexts):
                raise ValueError("sequences and contexts must have same length")
            self.chain.fit(sequences, contexts)
        else:
            self.chain.fit(sequences)

        # Build API vocabulary
        self._build_vocabulary(sequences)

        self._is_fitted = True
        return self

    def _build_vocabulary(self, sequences: List[List[str]]) -> None:
        """
        Build API vocabulary for encoding.

        Args:
            sequences (List[List[str]]): Training sequences.
        """
        unique_apis = set()
        for sequence in sequences:
            unique_apis.update(sequence)

        # Sort for consistent ordering
        sorted_apis = sorted(unique_apis)
        self.api_vocab = {api: idx for idx, api in enumerate(sorted_apis)}
        self.vocab_size = len(self.api_vocab)

    def observe(
        self,
        api: str,
        context: Optional[Dict[str, Any]] = None,
        update: bool = False
    ) -> None:
        """
        Record a new API call observation.

        Adds to history and optionally updates the model (online learning).

        Args:
            api (str): API endpoint observed.
            context (Optional[Dict[str, Any]]): Context for this call (if context_aware).
            update (bool): If True, update the underlying model. Default is False.

        Example:
            >>> predictor.observe('login')
            >>> predictor.observe('profile', context={'user_type': 'premium', 'hour': 10})
            >>> predictor.observe('orders', update=True)  # Online learning
        """
        # Add to history
        self.history.append(api)

        if self.context_aware and context is not None:
            self.context_history.append(context)

        # Add to vocabulary if new
        if api not in self.api_vocab:
            self.api_vocab[api] = self.vocab_size
            self.vocab_size += 1

        # Online learning
        if update and self._is_fitted:
            if len(self.history) >= 2:
                # Create a mini-sequence from recent history
                recent_seq = list(self.history)[-min(5, len(self.history)):]

                if self.context_aware and context is not None:
                    # Use most recent context
                    recent_contexts = [context] * len([recent_seq])
                    self.chain.partial_fit([recent_seq], recent_contexts)
                else:
                    self.chain.partial_fit([recent_seq])

    def reset_history(self) -> None:
        """
        Clear call history (e.g., when starting a new session).

        Example:
            >>> predictor.reset_history()
        """
        self.history.clear()
        self.context_history.clear()

    def predict(
        self,
        k: int = 5,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """
        Get top-k predictions based on current history.

        Args:
            k (int): Number of predictions to return. Default is 5.
            context (Optional[Dict[str, Any]]): Context for prediction (if context_aware).

        Returns:
            List[Tuple[str, float]]: List of (api, probability) tuples.

        Example:
            >>> predictions = predictor.predict(k=5, context={'user_type': 'premium', 'hour': 10})
            >>> for api, prob in predictions:
            ...     print(f"{api}: {prob:.3f}")
        """
        if not self._is_fitted:
            return []

        if not self.history:
            return []

        current = self.history[-1]

        # Make prediction based on chain type
        if self.context_aware:
            if context is None and self.context_history:
                context = self.context_history[-1]

            if context is None:
                return []

            if self.order == 1 or len(self.history) < 2:
                predictions = self.chain.predict(current, context, k=k)
            else:
                prev = self.history[-2]
                predictions = self.chain.predict(current, context, k=k, prev=prev)
        else:
            if self.order == 1 or len(self.history) < 2:
                predictions = self.chain.predict(current, k=k)
            else:
                prev = self.history[-2]
                predictions = self.chain.predict(prev, current, k=k)

        return predictions

    def predict_sequence(
        self,
        length: int = 5,
        context: Optional[Dict[str, Any]] = None
    ) -> List[List[Tuple[str, float]]]:
        """
        Look-ahead predictions for next N positions.

        Useful for prefetch planning - predict multiple steps ahead.

        Args:
            length (int): Number of future positions to predict. Default is 5.
            context (Optional[Dict[str, Any]]): Context for predictions (if context_aware).

        Returns:
            List[List[Tuple[str, float]]]: Predictions for each position.
                predictions[0] = predictions for immediate next API
                predictions[1] = predictions for API after that, etc.

        Example:
            >>> seq_predictions = predictor.predict_sequence(length=5)
            >>> for i, preds in enumerate(seq_predictions, 1):
            ...     print(f"Position {i}: {preds[0][0]}")  # Top prediction
        """
        if not self._is_fitted or not self.history:
            return [[] for _ in range(length)]

        all_predictions = []

        # Use a temporary history for lookahead
        temp_history = list(self.history)

        for _ in range(length):
            # Get predictions based on temp history
            current = temp_history[-1]

            if self.context_aware:
                if context is None and self.context_history:
                    context = self.context_history[-1]

                if context is None:
                    all_predictions.append([])
                    break

                if self.order == 1 or len(temp_history) < 2:
                    preds = self.chain.predict(current, context, k=5)
                else:
                    prev = temp_history[-2]
                    preds = self.chain.predict(current, context, k=5, prev=prev)
            else:
                if self.order == 1 or len(temp_history) < 2:
                    preds = self.chain.predict(current, k=5)
                else:
                    prev = temp_history[-2]
                    preds = self.chain.predict(prev, current, k=5)

            all_predictions.append(preds)

            # Add most likely prediction to temp history for next iteration
            if preds:
                temp_history.append(preds[0][0])
            else:
                break

        return all_predictions

    def get_state_vector(
        self,
        k: int = 5,
        context: Optional[Dict[str, Any]] = None,
        include_history: bool = True
    ) -> np.ndarray:
        """
        Get fixed-size state vector for RL agent.

        This is crucial for RL integration - returns a fixed-size numpy array
        that encodes the current prediction state.

        Args:
            k (int): Number of top predictions to include. Default is 5.
            context (Optional[Dict[str, Any]]): Context for prediction (if context_aware).
            include_history (bool): Whether to include encoded history. Default is True.

        Returns:
            np.ndarray: Fixed-size state vector with:
                - Top-k predicted API indices (normalized to 0-1)
                - Top-k prediction probabilities
                - Confidence score (max probability)
                - Optionally: encoded recent history
                - Optionally: encoded context features

        Example:
            >>> state = predictor.get_state_vector(k=5)
            >>> state.shape
            (21,)  # 5 indices + 5 probs + 1 confidence + 10 history
        """
        # Get predictions
        predictions = self.predict(k=k, context=context)

        # Initialize state vector components
        pred_indices = np.zeros(k)
        pred_probs = np.zeros(k)

        # Fill in predictions (normalized indices)
        for i, (api, prob) in enumerate(predictions[:k]):
            if api in self.api_vocab:
                # Normalize index to [0, 1]
                pred_indices[i] = self.api_vocab[api] / max(self.vocab_size, 1)
            pred_probs[i] = prob

        # Confidence score (max probability or entropy-based)
        if predictions:
            confidence = predictions[0][1]  # Max probability
        else:
            confidence = 0.0

        # Build state vector
        state_components = [pred_indices, pred_probs, np.array([confidence])]

        # Add history encoding
        if include_history:
            history_encoding = np.zeros(self.history_size)
            for i, api in enumerate(list(self.history)[-self.history_size:]):
                if api in self.api_vocab:
                    # Normalize index
                    history_encoding[i] = self.api_vocab[api] / max(self.vocab_size, 1)
            state_components.append(history_encoding)

        # Add context encoding (if context-aware)
        if self.context_aware and context is not None:
            context_encoding = self._encode_context(context)
            state_components.append(context_encoding)

        # Concatenate all components
        state_vector = np.concatenate(state_components)

        return state_vector

    def _encode_context(self, context: Dict[str, Any]) -> np.ndarray:
        """
        Encode context features as numeric vector.

        Args:
            context (Dict[str, Any]): Context dictionary.

        Returns:
            np.ndarray: Encoded context vector.
        """
        encoding = []

        for feature in self.context_features:
            if feature == 'time_of_day':
                # One-hot encode time of day
                time_categories = ['morning', 'afternoon', 'evening', 'night']
                time_value = context.get('time_of_day', 'unknown')

                # If hour provided instead, discretize
                if 'hour' in context and 'time_of_day' not in context:
                    hour = context['hour']
                    if 6 <= hour < 12:
                        time_value = 'morning'
                    elif 12 <= hour < 18:
                        time_value = 'afternoon'
                    elif 18 <= hour < 22:
                        time_value = 'evening'
                    else:
                        time_value = 'night'

                one_hot = [1.0 if cat == time_value else 0.0 for cat in time_categories]
                encoding.extend(one_hot)

            elif feature == 'day_type':
                # Binary encode day type
                day_value = context.get('day_type', 'weekday')

                # If day provided instead, discretize
                if 'day' in context and 'day_type' not in context:
                    day = context['day']
                    day_value = 'weekday' if day < 5 else 'weekend'

                encoding.append(1.0 if day_value == 'weekday' else 0.0)

            elif feature == 'user_type':
                # One-hot encode user type (assuming common types)
                user_types = ['free', 'premium', 'enterprise']
                user_value = context.get('user_type', 'free')
                one_hot = [1.0 if ut == user_value else 0.0 for ut in user_types]
                encoding.extend(one_hot)

            else:
                # Generic numeric encoding for other features
                value = context.get(feature, 0)
                if isinstance(value, (int, float)):
                    encoding.append(float(value))
                else:
                    # Hash string to number
                    encoding.append(float(hash(str(value)) % 1000) / 1000.0)

        return np.array(encoding, dtype=np.float32)

    def record_outcome(self, actual_next: str) -> None:
        """
        Record the actual next API call after prediction.

        Used for tracking accuracy metrics in real-time.

        Args:
            actual_next (str): The actual API that was called next.

        Example:
            >>> predictions = predictor.predict(k=5)
            >>> # ... actual API call happens ...
            >>> predictor.record_outcome('profile')
        """
        if not self.all_predictions or len(self.all_predictions) == 0:
            # No prediction to evaluate
            return

        # Get the last predictions made
        last_predictions = self.all_predictions[-1] if self.all_predictions else []

        if not last_predictions:
            return

        self.prediction_count += 1
        self.all_actuals.append(actual_next)

        # Check if actual_next was in top-k for various k
        predicted_apis = [api for api, prob in last_predictions]

        for k in range(1, len(predicted_apis) + 1):
            if actual_next in predicted_apis[:k]:
                if k not in self.correct_predictions:
                    self.correct_predictions[k] = 0
                self.correct_predictions[k] += 1

        # Track confidence (probability of top prediction)
        if last_predictions:
            self.total_confidence += last_predictions[0][1]

    def get_metrics(self) -> Dict[str, float]:
        """
        Get current accuracy and performance metrics.

        Returns:
            Dict[str, float]: Dictionary with metrics:
                - top_k_accuracy: Accuracy at various k values
                - avg_confidence: Average confidence of predictions
                - prediction_count: Total predictions made
                - coverage: Fraction of predictions where we could make a prediction

        Example:
            >>> metrics = predictor.get_metrics()
            >>> print(f"Top-1 accuracy: {metrics['top_1_accuracy']:.2%}")
            >>> print(f"Top-5 accuracy: {metrics['top_5_accuracy']:.2%}")
        """
        metrics = {
            'prediction_count': self.prediction_count,
            'avg_confidence': (
                self.total_confidence / self.prediction_count
                if self.prediction_count > 0 else 0.0
            ),
        }

        # Calculate top-k accuracies
        for k, correct_count in self.correct_predictions.items():
            accuracy = correct_count / self.prediction_count if self.prediction_count > 0 else 0.0
            metrics[f'top_{k}_accuracy'] = accuracy

        # Coverage (fraction where we could make predictions)
        metrics['coverage'] = 1.0 if self.prediction_count > 0 else 0.0

        return metrics

    def save(self, path: str) -> None:
        """
        Save predictor state to disk.

        Args:
            path (str): File path to save to.

        Example:
            >>> predictor.save('models/predictor.json')
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Save configuration and state
        data = {
            'order': self.order,
            'context_aware': self.context_aware,
            'context_features': self.context_features,
            'smoothing': self.smoothing,
            'history_size': self.history_size,
            'fallback_strategy': self.fallback_strategy,
            'is_fitted': self._is_fitted,
            'api_vocab': self.api_vocab,
            'vocab_size': self.vocab_size,
            'history': list(self.history),
            'context_history': list(self.context_history) if self.context_history else [],
            'prediction_count': self.prediction_count,
            'correct_predictions': self.correct_predictions,
            'total_confidence': self.total_confidence,
        }

        with open(path_obj, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        # Save underlying chain
        chain_path = path_obj.with_suffix('.chain.json')
        self.chain.save(str(chain_path))

    @classmethod
    def load(cls, path: str) -> 'MarkovPredictor':
        """
        Load predictor from disk.

        Args:
            path (str): File path to load from.

        Returns:
            MarkovPredictor: Loaded predictor instance.

        Example:
            >>> predictor = MarkovPredictor.load('models/predictor.json')
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create instance
        predictor = cls(
            order=data['order'],
            context_aware=data['context_aware'],
            context_features=data.get('context_features'),
            smoothing=data['smoothing'],
            history_size=data['history_size'],
            fallback_strategy=data.get('fallback_strategy', 'global')
        )

        # Restore state
        predictor._is_fitted = data['is_fitted']
        predictor.api_vocab = data['api_vocab']
        predictor.vocab_size = data['vocab_size']
        predictor.history = deque(data['history'], maxlen=predictor.history_size)
        predictor.context_history = deque(
            data.get('context_history', []),
            maxlen=predictor.history_size
        )
        predictor.prediction_count = data['prediction_count']
        predictor.correct_predictions = {
            int(k): v for k, v in data['correct_predictions'].items()
        }
        predictor.total_confidence = data['total_confidence']

        # Load underlying chain
        path_obj = Path(path)
        chain_path = path_obj.with_suffix('.chain.json')

        if predictor.context_aware:
            predictor.chain = ContextAwareMarkovChain.load(str(chain_path))
        elif predictor.order == 1:
            predictor.chain = FirstOrderMarkovChain.load(str(chain_path))
        else:
            predictor.chain = SecondOrderMarkovChain.load(str(chain_path))

        return predictor

    @property
    def is_fitted(self) -> bool:
        """
        Check if predictor has been trained.

        Returns:
            bool: True if fitted.
        """
        return self._is_fitted

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return (
                f"MarkovPredictor(order={self.order}, "
                f"context_aware={self.context_aware}, "
                f"vocab_size={self.vocab_size}, "
                f"history={len(self.history)}/{self.history_size})"
            )
        else:
            return (
                f"MarkovPredictor(fitted=False, order={self.order}, "
                f"context_aware={self.context_aware})"
            )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()


def create_predictor(config: Any) -> MarkovPredictor:
    """
    Factory function to create predictor from project configuration.

    Args:
        config: Configuration object with markov prediction settings.
            Expected attributes:
                - markov_order (int): Order of Markov chain (1 or 2)
                - context_aware (bool): Whether to use context-aware predictions
                - context_features (List[str]): Context feature names
                - smoothing (float): Laplace smoothing parameter
                - history_size (int): Call history window size
                - fallback_strategy (str): Fallback strategy for context-aware

    Returns:
        MarkovPredictor: Configured predictor instance.

    Example:
        >>> from src.utils.config import get_config
        >>> config = get_config()
        >>> predictor = create_predictor(config)
    """
    # Extract configuration with defaults
    order = getattr(config, 'markov_order', 1)
    context_aware = getattr(config, 'context_aware', False)
    context_features = getattr(config, 'context_features', None)
    smoothing = getattr(config, 'smoothing', 0.001)
    history_size = getattr(config, 'history_size', 10)
    fallback_strategy = getattr(config, 'fallback_strategy', 'global')

    predictor = MarkovPredictor(
        order=order,
        context_aware=context_aware,
        context_features=context_features,
        smoothing=smoothing,
        history_size=history_size,
        fallback_strategy=fallback_strategy
    )

    return predictor


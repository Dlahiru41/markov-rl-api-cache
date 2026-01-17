"""
Context-aware Markov chain for API call prediction.

A context-aware Markov chain maintains separate transition models for different
contexts (e.g., user types, times of day). This captures the fact that different
users behave differently - premium users might have different browsing patterns
than free users, and morning traffic might differ from evening traffic.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Set, Union
from collections import defaultdict

from .first_order import FirstOrderMarkovChain
from .second_order import SecondOrderMarkovChain


class ContextAwareMarkovChain:
    """
    Context-aware Markov chain that conditions predictions on user context.

    Maintains separate Markov chains for different contexts (e.g., user types,
    times of day) plus a global fallback chain. This allows capturing behavioral
    differences across user segments.

    Attributes:
        context_features (List[str]): List of context feature names to use.
        order (int): Markov chain order (1 or 2).
        fallback_strategy (str): Strategy for unknown contexts ("global", "similar", "none").
        smoothing (float): Laplace smoothing parameter.
        chains (Dict[str, Union[FirstOrderMarkovChain, SecondOrderMarkovChain]]):
            Dictionary mapping context keys to their Markov chains.
        global_chain (Union[FirstOrderMarkovChain, SecondOrderMarkovChain]):
            Chain trained on all data regardless of context.

    Example:
        >>> mc_ctx = ContextAwareMarkovChain(
        ...     context_features=['user_type', 'time_of_day'],
        ...     order=1,
        ...     fallback_strategy='global'
        ... )
        >>> mc_ctx.fit(sequences, contexts)
        >>> predictions = mc_ctx.predict('login', {'user_type': 'premium', 'hour': 10})
    """

    # Context delimiter for creating composite keys
    CONTEXT_DELIMITER = "|"

    # Time of day discretization (hour -> category)
    TIME_CATEGORIES = {
        'morning': range(6, 12),    # 6-11
        'afternoon': range(12, 18), # 12-17
        'evening': range(18, 22),   # 18-21
        'night': list(range(22, 24)) + list(range(0, 6))  # 22-5
    }

    # Day of week discretization (0=Monday, 6=Sunday)
    DAY_CATEGORIES = {
        'weekday': range(0, 5),  # Monday-Friday
        'weekend': range(5, 7)   # Saturday-Sunday
    }

    def __init__(
        self,
        context_features: List[str],
        order: int = 1,
        fallback_strategy: str = 'global',
        smoothing: float = 0.001
    ):
        """
        Initialize a context-aware Markov chain.

        Args:
            context_features (List[str]): List of context feature names to use.
                Examples: ['user_type'], ['user_type', 'time_of_day'], ['time_of_day', 'day_type']
            order (int): Markov chain order (1 or 2). Default is 1.
            fallback_strategy (str): Strategy for unknown contexts. Options:
                - 'global': Use global chain trained on all data (default)
                - 'similar': Try to find a similar context
                - 'none': Return empty predictions
            smoothing (float): Laplace smoothing parameter. Default is 0.001.

        Raises:
            ValueError: If order is not 1 or 2, or fallback_strategy is invalid.

        Example:
            >>> mc_ctx = ContextAwareMarkovChain(
            ...     context_features=['user_type', 'time_of_day'],
            ...     order=1,
            ...     fallback_strategy='global',
            ...     smoothing=0.001
            ... )
        """
        if order not in [1, 2]:
            raise ValueError(f"Order must be 1 or 2, got {order}")

        if fallback_strategy not in ['global', 'similar', 'none']:
            raise ValueError(
                f"fallback_strategy must be 'global', 'similar', or 'none', got {fallback_strategy}"
            )

        self.context_features = context_features
        self.order = order
        self.fallback_strategy = fallback_strategy
        self.smoothing = smoothing

        # Dictionary of context-specific chains
        self.chains: Dict[str, Union[FirstOrderMarkovChain, SecondOrderMarkovChain]] = {}

        # Global chain (trained on all data)
        self.global_chain = self._create_chain()

        # Track how many samples each context has seen
        self.context_sample_counts: Dict[str, int] = defaultdict(int)

        self._is_fitted = False

    def _create_chain(self) -> Union[FirstOrderMarkovChain, SecondOrderMarkovChain]:
        """
        Create a new Markov chain of the appropriate order.

        Returns:
            Union[FirstOrderMarkovChain, SecondOrderMarkovChain]: New chain instance.
        """
        if self.order == 1:
            return FirstOrderMarkovChain(smoothing=self.smoothing)
        else:
            return SecondOrderMarkovChain(
                smoothing=self.smoothing,
                fallback_to_first_order=True
            )

    def _discretize_hour(self, hour: int) -> str:
        """
        Discretize hour (0-23) to time of day category.

        Args:
            hour (int): Hour in 24-hour format (0-23).

        Returns:
            str: Time category ('morning', 'afternoon', 'evening', 'night').

        Example:
            >>> mc._discretize_hour(10)
            'morning'
            >>> mc._discretize_hour(14)
            'afternoon'
        """
        for category, hours in self.TIME_CATEGORIES.items():
            if hour in hours:
                return category
        return 'night'  # Default

    def _discretize_day(self, day: int) -> str:
        """
        Discretize day of week (0-6) to weekday/weekend.

        Args:
            day (int): Day of week (0=Monday, 6=Sunday).

        Returns:
            str: Day category ('weekday' or 'weekend').

        Example:
            >>> mc._discretize_day(0)  # Monday
            'weekday'
            >>> mc._discretize_day(6)  # Sunday
            'weekend'
        """
        for category, days in self.DAY_CATEGORIES.items():
            if day in days:
                return category
        return 'weekday'  # Default

    def _discretize_context(self, context: Dict[str, Any]) -> Dict[str, str]:
        """
        Discretize context features (convert continuous to categorical).

        Handles special features:
        - 'hour' (0-23) → 'time_of_day' (morning/afternoon/evening/night)
        - 'day' (0-6) → 'day_type' (weekday/weekend)

        Args:
            context (Dict[str, Any]): Raw context dictionary.

        Returns:
            Dict[str, str]: Discretized context with categorical values.

        Example:
            >>> mc._discretize_context({'user_type': 'premium', 'hour': 10})
            {'user_type': 'premium', 'time_of_day': 'morning'}
        """
        discretized = {}

        for feature in self.context_features:
            if feature == 'time_of_day':
                # Convert hour to time category
                if 'hour' in context:
                    discretized['time_of_day'] = self._discretize_hour(context['hour'])
                elif 'time_of_day' in context:
                    discretized['time_of_day'] = context['time_of_day']
            elif feature == 'day_type':
                # Convert day to weekday/weekend
                if 'day' in context:
                    discretized['day_type'] = self._discretize_day(context['day'])
                elif 'day_type' in context:
                    discretized['day_type'] = context['day_type']
            else:
                # Pass through other features as-is
                if feature in context:
                    discretized[feature] = str(context[feature])

        return discretized

    def _make_context_key(self, context: Dict[str, Any]) -> str:
        """
        Create a context key from discretized context features.

        Args:
            context (Dict[str, Any]): Context dictionary.

        Returns:
            str: Context key (e.g., "premium|morning").

        Example:
            >>> mc._make_context_key({'user_type': 'premium', 'time_of_day': 'morning'})
            'premium|morning'
        """
        discretized = self._discretize_context(context)

        # Build key from features in order
        parts = []
        for feature in self.context_features:
            value = discretized.get(feature, 'unknown')
            parts.append(value)

        return self.CONTEXT_DELIMITER.join(parts)

    def _parse_context_key(self, key: str) -> Dict[str, str]:
        """
        Parse a context key back to a dictionary.

        Args:
            key (str): Context key (e.g., "premium|morning").

        Returns:
            Dict[str, str]: Dictionary mapping feature names to values.

        Example:
            >>> mc._parse_context_key('premium|morning')
            {'user_type': 'premium', 'time_of_day': 'morning'}
        """
        parts = key.split(self.CONTEXT_DELIMITER)
        return dict(zip(self.context_features, parts))

    def _find_similar_context(self, context_key: str) -> Optional[str]:
        """
        Find a similar context key that exists in the chains.

        Tries to find a context that matches on some (but not all) features.
        Priority: match more features > match fewer features.

        Args:
            context_key (str): Target context key.

        Returns:
            Optional[str]: Similar context key if found, None otherwise.

        Example:
            >>> # If 'premium|morning' not found, might return 'premium|afternoon'
            >>> mc._find_similar_context('premium|morning')
            'premium|afternoon'
        """
        if not self.chains:
            return None

        target_ctx = self._parse_context_key(context_key)

        # Try to find contexts that match on some features
        best_match = None
        best_match_count = 0

        for existing_key in self.chains.keys():
            existing_ctx = self._parse_context_key(existing_key)

            # Count matching features
            match_count = sum(
                1 for feature in self.context_features
                if target_ctx.get(feature) == existing_ctx.get(feature)
            )

            if match_count > best_match_count:
                best_match = existing_key
                best_match_count = match_count

        # Only return if we matched at least one feature
        return best_match if best_match_count > 0 else None

    def fit(
        self,
        sequences: List[List[str]],
        contexts: List[Dict[str, Any]]
    ) -> 'ContextAwareMarkovChain':
        """
        Train context-aware Markov chains on sequences with contexts.

        Args:
            sequences (List[List[str]]): List of API call sequences.
            contexts (List[Dict[str, Any]]): Context for each sequence.
                Each dict should contain values for the configured context_features.

        Returns:
            ContextAwareMarkovChain: Self, for method chaining.

        Raises:
            ValueError: If sequences and contexts have different lengths.

        Example:
            >>> sequences = [['A', 'B', 'C'], ['A', 'B', 'D']]
            >>> contexts = [{'user_type': 'premium', 'hour': 10},
            ...             {'user_type': 'free', 'hour': 14}]
            >>> mc_ctx.fit(sequences, contexts)
        """
        if len(sequences) != len(contexts):
            raise ValueError(
                f"sequences and contexts must have same length, "
                f"got {len(sequences)} and {len(contexts)}"
            )

        # Reset all chains
        self.chains = {}
        self.global_chain = self._create_chain()
        self.context_sample_counts = defaultdict(int)

        # Group sequences by context
        context_to_sequences: Dict[str, List[List[str]]] = defaultdict(list)

        for sequence, context in zip(sequences, contexts):
            context_key = self._make_context_key(context)
            context_to_sequences[context_key].append(sequence)
            self.context_sample_counts[context_key] += 1

        # Train context-specific chains
        for context_key, ctx_sequences in context_to_sequences.items():
            chain = self._create_chain()
            chain.fit(ctx_sequences)
            self.chains[context_key] = chain

        # Train global chain on all data
        self.global_chain.fit(sequences)

        self._is_fitted = True
        return self

    def partial_fit(
        self,
        sequences: List[List[str]],
        contexts: List[Dict[str, Any]]
    ) -> 'ContextAwareMarkovChain':
        """
        Incrementally update the model with new sequences and contexts.

        Args:
            sequences (List[List[str]]): New sequences to learn from.
            contexts (List[Dict[str, Any]]): Context for each sequence.

        Returns:
            ContextAwareMarkovChain: Self, for method chaining.

        Raises:
            ValueError: If sequences and contexts have different lengths.

        Example:
            >>> mc_ctx.partial_fit(new_sequences, new_contexts)
        """
        if len(sequences) != len(contexts):
            raise ValueError(
                f"sequences and contexts must have same length, "
                f"got {len(sequences)} and {len(contexts)}"
            )

        # Group sequences by context
        context_to_sequences: Dict[str, List[List[str]]] = defaultdict(list)

        for sequence, context in zip(sequences, contexts):
            context_key = self._make_context_key(context)
            context_to_sequences[context_key].append(sequence)
            self.context_sample_counts[context_key] += 1

        # Update context-specific chains
        for context_key, ctx_sequences in context_to_sequences.items():
            if context_key in self.chains:
                self.chains[context_key].partial_fit(ctx_sequences)
            else:
                # Create new chain for this context
                chain = self._create_chain()
                chain.fit(ctx_sequences)
                self.chains[context_key] = chain

        # Update global chain
        self.global_chain.partial_fit(sequences)

        self._is_fitted = True
        return self

    def predict(
        self,
        current: str,
        context: Dict[str, Any],
        k: int = 5,
        prev: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Make context-aware predictions.

        Args:
            current (str): Current API endpoint.
            context (Dict[str, Any]): Context dictionary (e.g., {'user_type': 'premium', 'hour': 10}).
            k (int): Number of predictions to return. Default is 5.
            prev (Optional[str]): Previous API endpoint (only used if order=2).

        Returns:
            List[Tuple[str, float]]: List of (api, probability) tuples.

        Example:
            >>> predictions = mc_ctx.predict('login', {'user_type': 'premium', 'hour': 10}, k=3)
            >>> predictions
            [('premium_features', 0.8), ('profile', 0.15), ...]
        """
        if not self._is_fitted:
            return []

        context_key = self._make_context_key(context)

        # Try to get context-specific chain
        chain = self.chains.get(context_key)

        # Apply fallback strategy if needed
        if chain is None:
            if self.fallback_strategy == 'global':
                chain = self.global_chain
            elif self.fallback_strategy == 'similar':
                similar_key = self._find_similar_context(context_key)
                if similar_key:
                    chain = self.chains[similar_key]
                else:
                    chain = self.global_chain
            else:  # 'none'
                return []

        # Make prediction based on order
        if self.order == 1 or prev is None:
            return chain.predict(current, k=k)
        else:
            # Second-order prediction
            return chain.predict(prev, current, k=k)

    def predict_with_confidence(
        self,
        current: str,
        context: Dict[str, Any],
        k: int = 5,
        prev: Optional[str] = None
    ) -> List[Tuple[str, float, float]]:
        """
        Make predictions with confidence scores.

        Confidence is based on how much training data we have for this context.

        Args:
            current (str): Current API endpoint.
            context (Dict[str, Any]): Context dictionary.
            k (int): Number of predictions to return. Default is 5.
            prev (Optional[str]): Previous API endpoint (only used if order=2).

        Returns:
            List[Tuple[str, float, float]]: List of (api, probability, confidence) tuples.
                Confidence ranges from 0 to 1, higher means more training data.

        Example:
            >>> predictions = mc_ctx.predict_with_confidence('login', {'user_type': 'premium', 'hour': 10})
            >>> for api, prob, conf in predictions:
            ...     print(f"{api}: {prob:.2f} (confidence: {conf:.2f})")
        """
        if not self._is_fitted:
            return []

        context_key = self._make_context_key(context)

        # Get predictions
        predictions = self.predict(current, context, k=k, prev=prev)

        # Calculate confidence based on sample count for this context
        sample_count = self.context_sample_counts.get(context_key, 0)

        # Confidence formula: tanh(samples / 100)
        # Reaches ~0.76 at 100 samples, ~0.96 at 200 samples
        import math
        confidence = math.tanh(sample_count / 100.0)

        # If we fell back to global or similar, reduce confidence
        if context_key not in self.chains:
            confidence *= 0.5  # Reduce by half for fallback

        # Return predictions with confidence
        return [(api, prob, confidence) for api, prob in predictions]

    def get_context_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about contexts in the model.

        Returns:
            Dict[str, Any]: Statistics including:
                - num_contexts: Number of unique contexts
                - contexts: List of context keys
                - samples_per_context: Dict mapping context to sample count
                - low_data_contexts: Contexts with < 10 samples
                - context_features: List of feature names used

        Example:
            >>> stats = mc_ctx.get_context_statistics()
            >>> print(f"Unique contexts: {stats['num_contexts']}")
            >>> print(f"Low data contexts: {stats['low_data_contexts']}")
        """
        if not self._is_fitted:
            return {
                'num_contexts': 0,
                'contexts': [],
                'samples_per_context': {},
                'low_data_contexts': [],
                'context_features': self.context_features
            }

        # Find contexts with low data
        low_data_threshold = 10
        low_data_contexts = [
            ctx for ctx, count in self.context_sample_counts.items()
            if count < low_data_threshold
        ]

        return {
            'num_contexts': len(self.chains),
            'contexts': list(self.chains.keys()),
            'samples_per_context': dict(self.context_sample_counts),
            'low_data_contexts': low_data_contexts,
            'context_features': self.context_features,
            'total_samples': sum(self.context_sample_counts.values()),
            'avg_samples_per_context': (
                sum(self.context_sample_counts.values()) / len(self.chains)
                if self.chains else 0
            )
        }

    def get_context_importance(
        self,
        test_sequences: List[List[str]],
        test_contexts: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Measure how much each context feature improves predictions.

        Compares accuracy with and without each feature to determine importance.

        Args:
            test_sequences (List[List[str]]): Test sequences.
            test_contexts (List[Dict[str, Any]]): Context for each test sequence.

        Returns:
            Dict[str, float]: Mapping of feature name to importance score.
                Higher scores mean the feature is more important for predictions.
                Score is the percentage improvement in accuracy when using the feature.

        Example:
            >>> importance = mc_ctx.get_context_importance(test_sequences, test_contexts)
            >>> print(f"user_type importance: {importance['user_type']:.1f}%")
        """
        if len(test_sequences) != len(test_contexts):
            raise ValueError("test_sequences and test_contexts must have same length")

        if not self._is_fitted:
            return {feature: 0.0 for feature in self.context_features}

        # Evaluate with full context (baseline)
        full_context_correct = 0
        total_predictions = 0

        for sequence, context in zip(test_sequences, test_contexts):
            if len(sequence) < 2:
                continue

            for i in range(len(sequence) - 1):
                current = sequence[i]
                actual_next = sequence[i + 1]
                prev = sequence[i - 1] if i > 0 and self.order == 2 else None

                predictions = self.predict(current, context, k=1, prev=prev)
                if predictions and predictions[0][0] == actual_next:
                    full_context_correct += 1
                total_predictions += 1

        full_context_accuracy = (
            full_context_correct / total_predictions if total_predictions > 0 else 0
        )

        # Measure importance of each feature
        importance = {}

        for feature in self.context_features:
            # Create a version without this feature
            reduced_features = [f for f in self.context_features if f != feature]

            if not reduced_features:
                # Can't remove the only feature, use global chain accuracy
                global_correct = 0
                for sequence, context in zip(test_sequences, test_contexts):
                    if len(sequence) < 2:
                        continue

                    for i in range(len(sequence) - 1):
                        current = sequence[i]
                        actual_next = sequence[i + 1]
                        prev = sequence[i - 1] if i > 0 and self.order == 2 else None

                        if self.order == 1 or prev is None:
                            predictions = self.global_chain.predict(current, k=1)
                        else:
                            predictions = self.global_chain.predict(prev, current, k=1)

                        if predictions and predictions[0][0] == actual_next:
                            global_correct += 1

                reduced_accuracy = (
                    global_correct / total_predictions if total_predictions > 0 else 0
                )
            else:
                # Evaluate with reduced context
                reduced_correct = 0
                for sequence, context in zip(test_sequences, test_contexts):
                    if len(sequence) < 2:
                        continue

                    # Create reduced context (only features except this one)
                    reduced_context = {
                        f: context.get(f) for f in reduced_features
                        if f in context or f.replace('_type', '') in context or f.replace('_of_day', '') in context
                    }

                    for i in range(len(sequence) - 1):
                        current = sequence[i]
                        actual_next = sequence[i + 1]
                        prev = sequence[i - 1] if i > 0 and self.order == 2 else None

                        # Use a temporary model with reduced features to simulate
                        # For simplicity, use the closest matching context
                        reduced_key = self._make_context_key(reduced_context) if reduced_context else None

                        if reduced_key and reduced_key in self.chains:
                            chain = self.chains[reduced_key]
                        else:
                            chain = self.global_chain

                        if self.order == 1 or prev is None:
                            predictions = chain.predict(current, k=1)
                        else:
                            predictions = chain.predict(prev, current, k=1)

                        if predictions and predictions[0][0] == actual_next:
                            reduced_correct += 1

                reduced_accuracy = (
                    reduced_correct / total_predictions if total_predictions > 0 else 0
                )

            # Importance is the improvement from adding this feature
            improvement = full_context_accuracy - reduced_accuracy
            importance_score = (improvement / reduced_accuracy * 100) if reduced_accuracy > 0 else 0
            importance[feature] = importance_score

        return importance

    def save(self, path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            path (str): File path to save to.

        Example:
            >>> mc_ctx.save('models/context_aware.json')
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data
        data = {
            'context_features': self.context_features,
            'order': self.order,
            'fallback_strategy': self.fallback_strategy,
            'smoothing': self.smoothing,
            'is_fitted': self._is_fitted,
            'context_sample_counts': dict(self.context_sample_counts)
        }

        # Save context-specific chains
        chains_data = {}
        for context_key, chain in self.chains.items():
            chain_data = {
                'transition_matrix': chain.transition_matrix.to_dict()
            }
            if self.order == 2 and hasattr(chain, 'first_order_model') and chain.first_order_model:
                chain_data['first_order_model'] = {
                    'transition_matrix': chain.first_order_model.transition_matrix.to_dict()
                }
            chains_data[context_key] = chain_data

        data['chains'] = chains_data

        # Save global chain
        data['global_chain'] = {
            'transition_matrix': self.global_chain.transition_matrix.to_dict()
        }
        if self.order == 2 and hasattr(self.global_chain, 'first_order_model') and self.global_chain.first_order_model:
            data['global_chain']['first_order_model'] = {
                'transition_matrix': self.global_chain.first_order_model.transition_matrix.to_dict()
            }

        with open(path_obj, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ContextAwareMarkovChain':
        """
        Load a trained model from disk.

        Args:
            path (str): File path to load from.

        Returns:
            ContextAwareMarkovChain: Loaded model.

        Example:
            >>> mc_ctx = ContextAwareMarkovChain.load('models/context_aware.json')
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create instance
        from .transition_matrix import TransitionMatrix

        mc_ctx = cls(
            context_features=data['context_features'],
            order=data['order'],
            fallback_strategy=data['fallback_strategy'],
            smoothing=data['smoothing']
        )
        mc_ctx._is_fitted = data['is_fitted']
        mc_ctx.context_sample_counts = defaultdict(int, data['context_sample_counts'])

        # Load context-specific chains
        for context_key, chain_data in data['chains'].items():
            chain = mc_ctx._create_chain()
            chain.transition_matrix = TransitionMatrix.from_dict(chain_data['transition_matrix'])
            chain._is_fitted = True

            if mc_ctx.order == 2 and 'first_order_model' in chain_data:
                chain.first_order_model = FirstOrderMarkovChain(smoothing=mc_ctx.smoothing)
                chain.first_order_model.transition_matrix = TransitionMatrix.from_dict(
                    chain_data['first_order_model']['transition_matrix']
                )
                chain.first_order_model._is_fitted = True

            mc_ctx.chains[context_key] = chain

        # Load global chain
        mc_ctx.global_chain = mc_ctx._create_chain()
        mc_ctx.global_chain.transition_matrix = TransitionMatrix.from_dict(
            data['global_chain']['transition_matrix']
        )
        mc_ctx.global_chain._is_fitted = True

        if mc_ctx.order == 2 and 'first_order_model' in data['global_chain']:
            mc_ctx.global_chain.first_order_model = FirstOrderMarkovChain(smoothing=mc_ctx.smoothing)
            mc_ctx.global_chain.first_order_model.transition_matrix = TransitionMatrix.from_dict(
                data['global_chain']['first_order_model']['transition_matrix']
            )
            mc_ctx.global_chain.first_order_model._is_fitted = True

        return mc_ctx

    @property
    def is_fitted(self) -> bool:
        """
        Check if the model has been trained.

        Returns:
            bool: True if model has been fitted.
        """
        return self._is_fitted

    @property
    def contexts(self) -> Set[str]:
        """
        Get all known context keys.

        Returns:
            Set[str]: Set of context keys.
        """
        return set(self.chains.keys())

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return (
                f"ContextAwareMarkovChain(contexts={len(self.chains)}, "
                f"features={self.context_features}, order={self.order}, "
                f"fallback={self.fallback_strategy})"
            )
        else:
            return (
                f"ContextAwareMarkovChain(fitted=False, features={self.context_features}, "
                f"order={self.order})"
            )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()


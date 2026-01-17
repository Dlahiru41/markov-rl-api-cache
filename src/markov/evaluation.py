"""
Comprehensive evaluation module for Markov chain prediction performance.

Provides rigorous evaluation metrics, breakdowns, and visualizations for
understanding prediction quality and comparing different models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any, Union
from collections import defaultdict
from sklearn.model_selection import KFold

from .predictor import MarkovPredictor


class MarkovEvaluator:
    """
    Evaluator for computing metrics and analyzing Markov chain performance.

    Provides comprehensive evaluation including accuracy metrics, per-endpoint
    breakdowns, calibration analysis, and cross-validation.

    Attributes:
        predictor (MarkovPredictor): The predictor to evaluate.

    Example:
        >>> evaluator = MarkovEvaluator(predictor)
        >>> results = evaluator.evaluate_accuracy(test_sequences, test_contexts)
        >>> print(f"Top-1 Accuracy: {results['top_1_accuracy']:.3f}")
    """

    def __init__(self, predictor: MarkovPredictor):
        """
        Initialize the evaluator.

        Args:
            predictor (MarkovPredictor): The predictor to evaluate.

        Example:
            >>> evaluator = MarkovEvaluator(predictor)
        """
        self.predictor = predictor

    def evaluate_accuracy(
        self,
        test_sequences: List[List[str]],
        contexts: Optional[List[Dict[str, Any]]] = None,
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Compute core accuracy metrics on test data.

        For each transition in test sequences, checks if the correct next API
        was in the top-k predictions.

        Args:
            test_sequences (List[List[str]]): Test sequences to evaluate on.
            contexts (Optional[List[Dict[str, Any]]]): Context for each sequence
                (required if predictor is context-aware).
            k_values (List[int]): List of k values for top-k accuracy. Default [1,3,5,10].

        Returns:
            Dict[str, float]: Dictionary with metrics:
                - top_k_accuracy: Accuracy at each k value
                - mrr: Mean Reciprocal Rank
                - coverage: Fraction of states we could predict for
                - perplexity: Information-theoretic uncertainty measure

        Example:
            >>> results = evaluator.evaluate_accuracy(test_sequences, k_values=[1,3,5])
            >>> print(f"Top-1: {results['top_1_accuracy']:.3f}")
            >>> print(f"MRR: {results['mrr']:.3f}")
        """
        if contexts is not None and len(test_sequences) != len(contexts):
            raise ValueError("test_sequences and contexts must have same length")

        total_transitions = 0
        predictable_transitions = 0
        correct_in_top_k = {k: 0 for k in k_values}
        reciprocal_ranks = []
        log_likelihoods = []

        # Evaluate each sequence
        for seq_idx, sequence in enumerate(test_sequences):
            if len(sequence) < 2:
                continue

            context = contexts[seq_idx] if contexts is not None else None

            # Reset predictor history for each sequence
            self.predictor.reset_history()

            # Process each transition
            for i in range(len(sequence) - 1):
                current = sequence[i]
                actual_next = sequence[i + 1]

                # Observe current (builds history for next prediction)
                if i > 0:  # Skip first observation (no history yet)
                    self.predictor.observe(sequence[i - 1], context=context)

                self.predictor.observe(current, context=context)

                total_transitions += 1

                # Get predictions
                predictions = self.predictor.predict(k=max(k_values), context=context)

                if not predictions:
                    # Cannot make prediction for this state
                    continue

                predictable_transitions += 1

                # Find rank of actual next state
                rank = None
                for idx, (pred_api, prob) in enumerate(predictions, 1):
                    if pred_api == actual_next:
                        rank = idx
                        # Get probability for perplexity
                        if prob > 0:
                            log_likelihoods.append(np.log(prob))
                        break

                # Update metrics
                if rank is not None:
                    reciprocal_ranks.append(1.0 / rank)
                    for k in k_values:
                        if rank <= k:
                            correct_in_top_k[k] += 1
                else:
                    reciprocal_ranks.append(0.0)

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

        # Additional useful metrics
        metrics['total_transitions'] = total_transitions
        metrics['predictable_transitions'] = predictable_transitions

        return metrics

    def evaluate_per_endpoint(
        self,
        test_sequences: List[List[str]],
        contexts: Optional[List[Dict[str, Any]]] = None,
        k_values: List[int] = [1, 3]
    ) -> pd.DataFrame:
        """
        Accuracy broken down by which endpoint we're predicting FROM.

        Helps identify which APIs are easy or hard to predict after.

        Args:
            test_sequences (List[List[str]]): Test sequences.
            contexts (Optional[List[Dict[str, Any]]]): Contexts (if applicable).
            k_values (List[int]): K values for accuracy. Default [1, 3].

        Returns:
            pd.DataFrame: DataFrame with columns:
                - endpoint: The API we're predicting from
                - sample_count: Number of transitions from this endpoint
                - top_1_accuracy, top_3_accuracy: Accuracy metrics
                - mrr: Mean Reciprocal Rank for this endpoint

        Example:
            >>> df = evaluator.evaluate_per_endpoint(test_sequences)
            >>> print(df.sort_values('top_1_accuracy', ascending=False).head())
        """
        endpoint_stats = defaultdict(lambda: {
            'total': 0,
            'correct_in_top_k': {k: 0 for k in k_values},
            'reciprocal_ranks': []
        })

        # Evaluate each sequence
        for seq_idx, sequence in enumerate(test_sequences):
            if len(sequence) < 2:
                continue

            context = contexts[seq_idx] if contexts is not None else None
            self.predictor.reset_history()

            for i in range(len(sequence) - 1):
                current = sequence[i]
                actual_next = sequence[i + 1]

                if i > 0:
                    self.predictor.observe(sequence[i - 1], context=context)
                self.predictor.observe(current, context=context)

                endpoint_stats[current]['total'] += 1

                predictions = self.predictor.predict(k=max(k_values), context=context)

                if not predictions:
                    continue

                # Find rank
                rank = None
                for idx, (pred_api, prob) in enumerate(predictions, 1):
                    if pred_api == actual_next:
                        rank = idx
                        break

                if rank is not None:
                    endpoint_stats[current]['reciprocal_ranks'].append(1.0 / rank)
                    for k in k_values:
                        if rank <= k:
                            endpoint_stats[current]['correct_in_top_k'][k] += 1
                else:
                    endpoint_stats[current]['reciprocal_ranks'].append(0.0)

        # Build DataFrame
        rows = []
        for endpoint, stats in endpoint_stats.items():
            row = {
                'endpoint': endpoint,
                'sample_count': stats['total']
            }

            # Accuracies
            for k in k_values:
                if stats['total'] > 0:
                    accuracy = stats['correct_in_top_k'][k] / stats['total']
                else:
                    accuracy = 0.0
                row[f'top_{k}_accuracy'] = accuracy

            # MRR
            if stats['reciprocal_ranks']:
                row['mrr'] = np.mean(stats['reciprocal_ranks'])
            else:
                row['mrr'] = 0.0

            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by sample count descending
        if not df.empty:
            df = df.sort_values('sample_count', ascending=False)

        return df

    def evaluate_per_context(
        self,
        test_sequences: List[List[str]],
        contexts: List[Dict[str, Any]],
        k_values: List[int] = [1, 3]
    ) -> pd.DataFrame:
        """
        Accuracy broken down by context values.

        Helps identify if predictions work better for certain user types,
        times of day, etc.

        Args:
            test_sequences (List[List[str]]): Test sequences.
            contexts (List[Dict[str, Any]]): Context for each sequence.
            k_values (List[int]): K values for accuracy. Default [1, 3].

        Returns:
            pd.DataFrame: DataFrame with columns:
                - Context feature columns (e.g., user_type, time_of_day)
                - sample_count: Number of transitions for this context
                - top_1_accuracy, top_3_accuracy: Accuracy metrics
                - mrr: Mean Reciprocal Rank

        Example:
            >>> df = evaluator.evaluate_per_context(test_sequences, contexts)
            >>> print(df[df['user_type'] == 'premium'])
        """
        if not self.predictor.context_aware:
            raise ValueError("Predictor must be context-aware for per-context evaluation")

        if len(test_sequences) != len(contexts):
            raise ValueError("test_sequences and contexts must have same length")

        # Discretize contexts
        discretized_contexts = []
        for context in contexts:
            discretized = self.predictor.chain._discretize_context(context)
            # Convert to tuple for hashing
            context_tuple = tuple(sorted(discretized.items()))
            discretized_contexts.append(context_tuple)

        context_stats = defaultdict(lambda: {
            'total': 0,
            'correct_in_top_k': {k: 0 for k in k_values},
            'reciprocal_ranks': [],
            'context_dict': {}
        })

        # Evaluate
        for seq_idx, sequence in enumerate(test_sequences):
            if len(sequence) < 2:
                continue

            context = contexts[seq_idx]
            context_key = discretized_contexts[seq_idx]

            # Store context dict for later
            if not context_stats[context_key]['context_dict']:
                context_stats[context_key]['context_dict'] = dict(context_key)

            self.predictor.reset_history()

            for i in range(len(sequence) - 1):
                current = sequence[i]
                actual_next = sequence[i + 1]

                if i > 0:
                    self.predictor.observe(sequence[i - 1], context=context)
                self.predictor.observe(current, context=context)

                context_stats[context_key]['total'] += 1

                predictions = self.predictor.predict(k=max(k_values), context=context)

                if not predictions:
                    continue

                rank = None
                for idx, (pred_api, prob) in enumerate(predictions, 1):
                    if pred_api == actual_next:
                        rank = idx
                        break

                if rank is not None:
                    context_stats[context_key]['reciprocal_ranks'].append(1.0 / rank)
                    for k in k_values:
                        if rank <= k:
                            context_stats[context_key]['correct_in_top_k'][k] += 1
                else:
                    context_stats[context_key]['reciprocal_ranks'].append(0.0)

        # Build DataFrame
        rows = []
        for context_key, stats in context_stats.items():
            row = stats['context_dict'].copy()
            row['sample_count'] = stats['total']

            for k in k_values:
                if stats['total'] > 0:
                    accuracy = stats['correct_in_top_k'][k] / stats['total']
                else:
                    accuracy = 0.0
                row[f'top_{k}_accuracy'] = accuracy

            if stats['reciprocal_ranks']:
                row['mrr'] = np.mean(stats['reciprocal_ranks'])
            else:
                row['mrr'] = 0.0

            rows.append(row)

        df = pd.DataFrame(rows)

        if not df.empty:
            df = df.sort_values('sample_count', ascending=False)

        return df

    def evaluate_calibration(
        self,
        test_sequences: List[List[str]],
        contexts: Optional[List[Dict[str, Any]]] = None,
        num_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate prediction calibration.

        When we predict 80% probability, are we actually right 80% of the time?
        Bins predictions by confidence and compares predicted vs actual accuracy.

        Args:
            test_sequences (List[List[str]]): Test sequences.
            contexts (Optional[List[Dict[str, Any]]]): Contexts (if applicable).
            num_bins (int): Number of bins for calibration curve. Default 10.

        Returns:
            Dict[str, Any]: Calibration data with:
                - bin_centers: Center of each probability bin
                - predicted_probs: Average predicted probability per bin
                - actual_accuracy: Actual accuracy per bin
                - sample_counts: Number of samples per bin

        Example:
            >>> calibration = evaluator.evaluate_calibration(test_sequences)
            >>> MarkovVisualizer.plot_calibration_curve(calibration)
        """
        # Collect predictions and outcomes
        predicted_probs = []
        was_correct = []

        for seq_idx, sequence in enumerate(test_sequences):
            if len(sequence) < 2:
                continue

            context = contexts[seq_idx] if contexts is not None else None
            self.predictor.reset_history()

            for i in range(len(sequence) - 1):
                current = sequence[i]
                actual_next = sequence[i + 1]

                if i > 0:
                    self.predictor.observe(sequence[i - 1], context=context)
                self.predictor.observe(current, context=context)

                predictions = self.predictor.predict(k=1, context=context)

                if not predictions:
                    continue

                pred_api, prob = predictions[0]
                predicted_probs.append(prob)
                was_correct.append(1.0 if pred_api == actual_next else 0.0)

        if not predicted_probs:
            return {
                'bin_centers': [],
                'predicted_probs': [],
                'actual_accuracy': [],
                'sample_counts': []
            }

        # Bin by predicted probability
        predicted_probs = np.array(predicted_probs)
        was_correct = np.array(was_correct)

        bins = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(predicted_probs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)

        bin_centers = []
        bin_predicted_probs = []
        bin_actual_accuracy = []
        bin_sample_counts = []

        for i in range(num_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                bin_predicted_probs.append(np.mean(predicted_probs[mask]))
                bin_actual_accuracy.append(np.mean(was_correct[mask]))
                bin_sample_counts.append(np.sum(mask))

        return {
            'bin_centers': bin_centers,
            'predicted_probs': bin_predicted_probs,
            'actual_accuracy': bin_actual_accuracy,
            'sample_counts': bin_sample_counts
        }

    def cross_validate(
        self,
        sequences: List[List[str]],
        contexts: Optional[List[Dict[str, Any]]] = None,
        k_folds: int = 5,
        k_values: List[int] = [1, 3, 5]
    ) -> Dict[str, Tuple[float, float]]:
        """
        Run k-fold cross-validation.

        Provides confidence intervals for accuracy estimates.

        Args:
            sequences (List[List[str]]): All sequences for cross-validation.
            contexts (Optional[List[Dict[str, Any]]]): Contexts (if applicable).
            k_folds (int): Number of folds. Default 5.
            k_values (List[int]): K values for accuracy. Default [1, 3, 5].

        Returns:
            Dict[str, Tuple[float, float]]: Dictionary mapping metric name to
                (mean, std) tuple. Provides mean and standard deviation across folds.

        Example:
            >>> cv_results = evaluator.cross_validate(sequences, contexts, k_folds=5)
            >>> mean, std = cv_results['top_1_accuracy']
            >>> print(f"Top-1: {mean:.3f} ± {std:.3f}")
        """
        if contexts is not None and len(sequences) != len(contexts):
            raise ValueError("sequences and contexts must have same length")

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(sequences)):
            print(f"Fold {fold_idx + 1}/{k_folds}...", end=' ')

            # Split data
            train_sequences = [sequences[i] for i in train_idx]
            test_sequences = [sequences[i] for i in test_idx]

            if contexts is not None:
                train_contexts = [contexts[i] for i in train_idx]
                test_contexts = [contexts[i] for i in test_idx]
            else:
                train_contexts = None
                test_contexts = None

            # Train predictor on train fold
            if self.predictor.context_aware:
                self.predictor.fit(train_sequences, train_contexts)
            else:
                self.predictor.fit(train_sequences)

            # Evaluate on test fold
            results = self.evaluate_accuracy(test_sequences, test_contexts, k_values)
            fold_results.append(results)
            print(f"Top-1: {results['top_1_accuracy']:.3f}")

        # Aggregate results
        cv_results = {}
        metric_names = fold_results[0].keys()

        for metric in metric_names:
            values = [fold[metric] for fold in fold_results]
            # Filter out inf values for perplexity
            if metric == 'perplexity':
                values = [v for v in values if v != float('inf')]
            if values:
                cv_results[metric] = (np.mean(values), np.std(values))
            else:
                cv_results[metric] = (0.0, 0.0)

        return cv_results

    def compare_models(
        self,
        models: Dict[str, MarkovPredictor],
        test_sequences: List[List[str]],
        contexts: Optional[List[Dict[str, Any]]] = None,
        k_values: List[int] = [1, 3, 5]
    ) -> pd.DataFrame:
        """
        Evaluate and compare multiple models on the same test data.

        Args:
            models (Dict[str, MarkovPredictor]): Dictionary mapping model name
                to predictor instance.
            test_sequences (List[List[str]]): Test sequences.
            contexts (Optional[List[Dict[str, Any]]]): Contexts (if applicable).
            k_values (List[int]): K values for accuracy. Default [1, 3, 5].

        Returns:
            pd.DataFrame: DataFrame comparing all models on all metrics.
                Rows = models, Columns = metrics.

        Example:
            >>> models = {
            ...     'first_order': predictor1,
            ...     'second_order': predictor2,
            ...     'context_aware': predictor3
            ... }
            >>> df = evaluator.compare_models(models, test_sequences, contexts)
            >>> print(df.sort_values('top_1_accuracy', ascending=False))
        """
        results = []

        for model_name, predictor in models.items():
            print(f"Evaluating {model_name}...", end=' ')

            # Create temporary evaluator for this model
            temp_evaluator = MarkovEvaluator(predictor)

            # Evaluate
            metrics = temp_evaluator.evaluate_accuracy(test_sequences, contexts, k_values)

            # Add model name
            metrics['model'] = model_name
            results.append(metrics)

            print(f"Top-1: {metrics['top_1_accuracy']:.3f}")

        df = pd.DataFrame(results)

        # Move model column to front
        if not df.empty:
            cols = ['model'] + [col for col in df.columns if col != 'model']
            df = df[cols]

        return df


class MarkovVisualizer:
    """
    Visualizer for creating plots of Markov chain performance.

    Provides static methods for generating various analysis plots including
    transition heatmaps, accuracy curves, calibration plots, and more.
    """

    @staticmethod
    def plot_transition_heatmap(
        predictor: MarkovPredictor,
        top_k: int = 20,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10)
    ) -> None:
        """
        Show most common transitions as a heatmap.

        Rows = from states, columns = to states, color = probability.

        Args:
            predictor (MarkovPredictor): The predictor to visualize.
            top_k (int): Number of top APIs to include. Default 20.
            output_path (Optional[str]): Path to save figure. If None, displays.
            figsize (Tuple[int, int]): Figure size. Default (12, 10).

        Example:
            >>> MarkovVisualizer.plot_transition_heatmap(predictor, top_k=15,
            ...                                          output_path='heatmap.png')
        """
        # Get transition matrix from underlying chain
        if predictor.context_aware:
            # Use global chain for context-aware
            chain = predictor.chain.global_chain
        else:
            chain = predictor.chain

        # Get all states and their frequencies
        state_counts = defaultdict(int)
        for from_state, to_states in chain.transition_matrix.transitions.items():
            for to_state, count in to_states.items():
                state_counts[from_state] += count
                state_counts[to_state] += count

        # Get top-k states
        top_states = sorted(state_counts.items(), key=lambda x: -x[1])[:top_k]
        top_state_names = [state for state, count in top_states]

        # Build transition probability matrix
        matrix = np.zeros((top_k, top_k))

        for i, from_state in enumerate(top_state_names):
            row = chain.transition_matrix.get_row(from_state)
            for j, to_state in enumerate(top_state_names):
                if to_state in row:
                    matrix[i, j] = row[to_state]

        # Create heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(
            matrix,
            xticklabels=top_state_names,
            yticklabels=top_state_names,
            cmap='YlOrRd',
            cbar_kws={'label': 'Transition Probability'},
            fmt='.2f',
            linewidths=0.5
        )
        plt.title(f'Top-{top_k} API Transition Probabilities', fontsize=14, fontweight='bold')
        plt.xlabel('To API', fontsize=12)
        plt.ylabel('From API', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved heatmap to {output_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_accuracy_by_position(
        sequences: List[List[str]],
        predictor: MarkovPredictor,
        contexts: Optional[List[Dict[str, Any]]] = None,
        max_position: int = 20,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Plot how accuracy changes by position in sequence.

        Shows if early predictions are better than late ones.

        Args:
            sequences (List[List[str]]): Test sequences.
            predictor (MarkovPredictor): The predictor to evaluate.
            contexts (Optional[List[Dict[str, Any]]]): Contexts (if applicable).
            max_position (int): Maximum position to plot. Default 20.
            output_path (Optional[str]): Path to save figure. If None, displays.
            figsize (Tuple[int, int]): Figure size. Default (10, 6).

        Example:
            >>> MarkovVisualizer.plot_accuracy_by_position(test_sequences, predictor)
        """
        position_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

        # Evaluate
        for seq_idx, sequence in enumerate(sequences):
            if len(sequence) < 2:
                continue

            context = contexts[seq_idx] if contexts is not None else None
            predictor.reset_history()

            for i in range(len(sequence) - 1):
                if i >= max_position:
                    break

                current = sequence[i]
                actual_next = sequence[i + 1]

                if i > 0:
                    predictor.observe(sequence[i - 1], context=context)
                predictor.observe(current, context=context)

                predictions = predictor.predict(k=1, context=context)

                if predictions and predictions[0][0] == actual_next:
                    position_stats[i]['correct'] += 1
                position_stats[i]['total'] += 1

        # Calculate accuracies
        positions = sorted(position_stats.keys())
        accuracies = []

        for pos in positions:
            if position_stats[pos]['total'] > 0:
                acc = position_stats[pos]['correct'] / position_stats[pos]['total']
            else:
                acc = 0.0
            accuracies.append(acc)

        # Plot
        plt.figure(figsize=figsize)
        plt.plot(positions, accuracies, marker='o', linewidth=2, markersize=6)
        plt.xlabel('Position in Sequence', fontsize=12)
        plt.ylabel('Top-1 Accuracy', fontsize=12)
        plt.title('Prediction Accuracy by Sequence Position', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved accuracy by position plot to {output_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_calibration_curve(
        calibration_data: Dict[str, Any],
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 8)
    ) -> None:
        """
        Plot calibration curve.

        X axis = predicted probability, Y axis = actual accuracy.
        Perfect calibration is the diagonal line.

        Args:
            calibration_data (Dict[str, Any]): Output from evaluate_calibration().
            output_path (Optional[str]): Path to save figure. If None, displays.
            figsize (Tuple[int, int]): Figure size. Default (8, 8).

        Example:
            >>> calibration = evaluator.evaluate_calibration(test_sequences)
            >>> MarkovVisualizer.plot_calibration_curve(calibration)
        """
        plt.figure(figsize=figsize)

        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')

        # Plot actual calibration
        if calibration_data['bin_centers']:
            plt.plot(
                calibration_data['predicted_probs'],
                calibration_data['actual_accuracy'],
                marker='o',
                linewidth=2,
                markersize=8,
                label='Model Calibration'
            )

            # Add sample counts as labels
            for i, (x, y, count) in enumerate(zip(
                calibration_data['predicted_probs'],
                calibration_data['actual_accuracy'],
                calibration_data['sample_counts']
            )):
                plt.annotate(
                    f'n={int(count)}',
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=8,
                    alpha=0.7
                )

        plt.xlabel('Predicted Probability', fontsize=12)
        plt.ylabel('Actual Accuracy', fontsize=12)
        plt.title('Calibration Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved calibration curve to {output_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_prediction_confidence_distribution(
        sequences: List[List[str]],
        predictor: MarkovPredictor,
        contexts: Optional[List[Dict[str, Any]]] = None,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Plot histogram of prediction confidence values.

        Shows if model is generally confident or uncertain.

        Args:
            sequences (List[List[str]]): Test sequences.
            predictor (MarkovPredictor): The predictor to evaluate.
            contexts (Optional[List[Dict[str, Any]]]): Contexts (if applicable).
            output_path (Optional[str]): Path to save figure. If None, displays.
            figsize (Tuple[int, int]): Figure size. Default (10, 6).

        Example:
            >>> MarkovVisualizer.plot_prediction_confidence_distribution(
            ...     test_sequences, predictor)
        """
        confidences = []

        # Collect confidences
        for seq_idx, sequence in enumerate(sequences):
            if len(sequence) < 2:
                continue

            context = contexts[seq_idx] if contexts is not None else None
            predictor.reset_history()

            for i in range(len(sequence) - 1):
                if i > 0:
                    predictor.observe(sequence[i - 1], context=context)
                predictor.observe(sequence[i], context=context)

                predictions = predictor.predict(k=1, context=context)

                if predictions:
                    confidences.append(predictions[0][1])

        if not confidences:
            print("No predictions to plot")
            return

        # Plot
        plt.figure(figsize=figsize)
        plt.hist(confidences, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Prediction Confidence (Max Probability)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Prediction Confidence', fontsize=14, fontweight='bold')
        plt.axvline(
            np.mean(confidences),
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Mean: {np.mean(confidences):.3f}'
        )
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved confidence distribution to {output_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_model_comparison(
        comparison_df: pd.DataFrame,
        metrics: List[str] = ['top_1_accuracy', 'top_3_accuracy', 'mrr'],
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Plot comparison of multiple models.

        Args:
            comparison_df (pd.DataFrame): Output from compare_models().
            metrics (List[str]): Metrics to plot. Default ['top_1_accuracy', 'top_3_accuracy', 'mrr'].
            output_path (Optional[str]): Path to save figure. If None, displays.
            figsize (Tuple[int, int]): Figure size. Default (12, 6).

        Example:
            >>> comparison = evaluator.compare_models(models, test_sequences)
            >>> MarkovVisualizer.plot_model_comparison(comparison)
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)

        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            if metric in comparison_df.columns:
                comparison_df.plot(
                    x='model',
                    y=metric,
                    kind='bar',
                    ax=ax,
                    legend=False,
                    color='steelblue'
                )
                ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
                ax.set_xlabel('')
                ax.set_ylabel('Score', fontsize=10)
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved model comparison to {output_path}")
        else:
            plt.show()

        plt.close()


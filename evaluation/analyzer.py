"""
Comprehensive results analysis and visualization module for generating thesis-quality
figures and statistical analysis.

This module provides tools for loading experiment results, performing statistical tests,
and creating publication-quality visualizations for academic thesis work.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict
import warnings

# Statistical tests
from scipy import stats
from scipy.stats import (
    ttest_ind, mannwhitneyu, shapiro, kstest, anderson,
    f_oneway, kruskal, chi2_contingency
)

# For multiple comparisons
try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Some statistical tests will be limited.")

from .experiment_runner import ExperimentResult, ExperimentConfig


class ResultsAnalyzer:
    """
    Statistical analyzer for experiment results.

    Provides comprehensive statistical analysis including hypothesis testing,
    confidence intervals, effect sizes, and comparisons across multiple methods.

    Features:
    - Load and parse experiment results
    - Statistical comparisons (t-test, Mann-Whitney U, ANOVA, etc.)
    - Effect size calculations (Cohen's d)
    - Confidence intervals (parametric and bootstrap)
    - Normality testing
    - Learning curve analysis
    - Best configuration identification

    Example:
        >>> analyzer = ResultsAnalyzer('results/experiments')
        >>> dqn_results = analyzer.load_experiment('dqn_baseline')
        >>> lru_results = analyzer.load_experiment('lru_baseline')
        >>> comparison = analyzer.compare_two_methods(
        ...     dqn_results.final_eval_metrics['rewards'],
        ...     lru_results.final_eval_metrics['rewards']
        ... )
        >>> print(f"p-value: {comparison['t_test_pvalue']:.4f}")
    """

    def __init__(self, results_dir: str):
        """
        Initialize the analyzer.

        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            raise ValueError(f"Results directory does not exist: {results_dir}")

        self.experiments: Dict[str, List[ExperimentResult]] = {}
        self.load_all_experiments()

    def load_all_experiments(self) -> Dict[str, List[ExperimentResult]]:
        """
        Load all experiment results from directory.

        Returns:
            Dictionary mapping experiment IDs to lists of results (one per seed)
        """
        # Try to load from consolidated file first
        all_results_file = self.results_dir / 'all_results.json'
        if all_results_file.exists():
            try:
                with open(all_results_file, 'r') as f:
                    data = json.load(f)
                    for exp_id, result_list in data.items():
                        self.experiments[exp_id] = [
                            ExperimentResult.from_dict(r) for r in result_list
                        ]
                print(f"✓ Loaded {len(self.experiments)} experiments from {all_results_file}")
                return self.experiments
            except Exception as e:
                print(f"Warning: Failed to load {all_results_file}: {e}")

        # Fall back to loading from individual result files
        results_subdir = self.results_dir / 'results'
        if results_subdir.exists():
            for result_file in results_subdir.glob('*.json'):
                try:
                    with open(result_file, 'r') as f:
                        result_data = json.load(f)
                        result = ExperimentResult.from_dict(result_data)
                        exp_id = result_file.stem
                        if exp_id not in self.experiments:
                            self.experiments[exp_id] = []
                        self.experiments[exp_id].append(result)
                except Exception as e:
                    print(f"Warning: Failed to load {result_file}: {e}")

        print(f"✓ Loaded {len(self.experiments)} experiments from individual files")
        return self.experiments

    def load_experiment(self, experiment_id: str) -> Optional[List[ExperimentResult]]:
        """
        Load results for a specific experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            List of ExperimentResult objects (one per seed), or None if not found
        """
        if experiment_id in self.experiments:
            return self.experiments[experiment_id]

        # Try loading from file
        result_file = self.results_dir / 'results' / f'{experiment_id}.json'
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                    result = ExperimentResult.from_dict(result_data)
                    self.experiments[experiment_id] = [result]
                    return [result]
            except Exception as e:
                print(f"Error loading {experiment_id}: {e}")

        return None

    def filter_by_tags(self, tags: List[str]) -> Dict[str, List[ExperimentResult]]:
        """
        Filter experiments by tags.

        Args:
            tags: List of tags to filter by (any match)

        Returns:
            Dictionary of filtered experiments
        """
        filtered = {}
        for exp_id, results in self.experiments.items():
            if results and any(tag in results[0].config.tags for tag in tags):
                filtered[exp_id] = results
        return filtered

    def compare_two_methods(
        self,
        method_a_results: List[float],
        method_b_results: List[float],
        metric: str = 'reward',
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform statistical comparison between two methods.

        Runs multiple statistical tests:
        - Independent t-test (parametric)
        - Mann-Whitney U test (non-parametric)
        - Cohen's d effect size
        - Confidence intervals

        Args:
            method_a_results: List of metric values for method A
            method_b_results: List of metric values for method B
            metric: Name of metric being compared (for reporting)
            alpha: Significance level (default 0.05)

        Returns:
            Dictionary with test results:
                - t_test_pvalue: p-value from t-test
                - mannwhitneyu_pvalue: p-value from Mann-Whitney U
                - cohens_d: Effect size (Cohen's d)
                - mean_a, mean_b: Mean values
                - std_a, std_b: Standard deviations
                - ci_a, ci_b: 95% confidence intervals
                - significant: Whether difference is significant at alpha
                - interpretation: Text interpretation
        """
        a = np.array(method_a_results)
        b = np.array(method_b_results)

        # Basic statistics
        mean_a, mean_b = np.mean(a), np.mean(b)
        std_a, std_b = np.std(a, ddof=1), np.std(b, ddof=1)

        # T-test
        t_stat, t_pvalue = ttest_ind(a, b)

        # Mann-Whitney U (non-parametric alternative)
        try:
            u_stat, u_pvalue = mannwhitneyu(a, b, alternative='two-sided')
        except Exception:
            u_pvalue = np.nan

        # Cohen's d effect size
        pooled_std = np.sqrt(((len(a) - 1) * std_a**2 + (len(b) - 1) * std_b**2) / (len(a) + len(b) - 2))
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

        # Confidence intervals
        ci_a = self.compute_confidence_intervals(a, confidence=0.95)
        ci_b = self.compute_confidence_intervals(b, confidence=0.95)

        # Determine significance
        significant = t_pvalue < alpha

        # Interpretation
        if significant:
            if mean_a > mean_b:
                interpretation = f"Method A significantly outperforms Method B (p={t_pvalue:.4f}, d={cohens_d:.2f})"
            else:
                interpretation = f"Method B significantly outperforms Method A (p={t_pvalue:.4f}, d={cohens_d:.2f})"
        else:
            interpretation = f"No significant difference (p={t_pvalue:.4f}, d={cohens_d:.2f})"

        return {
            'metric': metric,
            'mean_a': float(mean_a),
            'mean_b': float(mean_b),
            'std_a': float(std_a),
            'std_b': float(std_b),
            'ci_a': ci_a,
            'ci_b': ci_b,
            't_test_pvalue': float(t_pvalue),
            'mannwhitneyu_pvalue': float(u_pvalue),
            'cohens_d': float(cohens_d),
            'significant': significant,
            'alpha': alpha,
            'interpretation': interpretation,
            'sample_size_a': len(a),
            'sample_size_b': len(b)
        }

    def compare_multiple_methods(
        self,
        results_dict: Dict[str, List[float]],
        metric: str = 'reward',
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Compare multiple methods simultaneously.

        Performs:
        - ANOVA (parametric)
        - Kruskal-Wallis (non-parametric)
        - Post-hoc pairwise comparisons (Tukey HSD if available)
        - Rankings

        Args:
            results_dict: Dictionary mapping method names to lists of metric values
            metric: Name of metric being compared
            alpha: Significance level

        Returns:
            Dictionary with:
                - anova_pvalue: Overall ANOVA p-value
                - kruskal_pvalue: Overall Kruskal-Wallis p-value
                - significant: Whether overall difference is significant
                - rankings: Methods ranked by performance
                - means: Mean values for each method
                - pairwise_comparisons: Pairwise test results (if statsmodels available)
        """
        method_names = list(results_dict.keys())
        method_values = [np.array(results_dict[name]) for name in method_names]

        # ANOVA
        f_stat, anova_pvalue = f_oneway(*method_values)

        # Kruskal-Wallis (non-parametric alternative)
        h_stat, kruskal_pvalue = kruskal(*method_values)

        # Compute means and rankings
        means = {name: np.mean(results_dict[name]) for name in method_names}
        rankings = sorted(means.items(), key=lambda x: x[1], reverse=True)

        # Determine overall significance
        significant = anova_pvalue < alpha

        result = {
            'metric': metric,
            'anova_pvalue': float(anova_pvalue),
            'kruskal_pvalue': float(kruskal_pvalue),
            'significant': significant,
            'alpha': alpha,
            'rankings': [(name, float(mean)) for name, mean in rankings],
            'means': {name: float(mean) for name, mean in means.items()},
            'num_methods': len(method_names)
        }

        # Post-hoc pairwise comparisons using Tukey HSD
        if STATSMODELS_AVAILABLE and significant:
            # Prepare data for Tukey HSD
            all_values = []
            all_labels = []
            for name in method_names:
                all_values.extend(results_dict[name])
                all_labels.extend([name] * len(results_dict[name]))

            try:
                tukey_result = pairwise_tukeyhsd(all_values, all_labels, alpha=alpha)

                # Parse Tukey results
                pairwise = []
                for i in range(len(tukey_result.summary().data) - 1):  # Skip header
                    row = tukey_result.summary().data[i + 1]
                    pairwise.append({
                        'group1': row[0],
                        'group2': row[1],
                        'meandiff': float(row[2]),
                        'pvalue': float(row[3]),
                        'lower': float(row[4]),
                        'upper': float(row[5]),
                        'reject': bool(row[6])
                    })

                result['pairwise_comparisons'] = pairwise
            except Exception as e:
                print(f"Warning: Tukey HSD failed: {e}")

        return result

    def compute_confidence_intervals(
        self,
        values: Union[List[float], np.ndarray],
        confidence: float = 0.95,
        method: str = 'parametric'
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for a list of values.

        Args:
            values: List or array of values
            confidence: Confidence level (default 0.95)
            method: 'parametric' or 'bootstrap'

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        values = np.array(values)

        if method == 'parametric':
            mean = np.mean(values)
            sem = stats.sem(values)
            interval = sem * stats.t.ppf((1 + confidence) / 2, len(values) - 1)
            return (mean - interval, mean + interval)

        elif method == 'bootstrap':
            # Bootstrap CI
            n_bootstrap = 10000
            bootstrap_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(values, size=len(values), replace=True)
                bootstrap_means.append(np.mean(sample))

            lower = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
            upper = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)
            return (lower, upper)

        else:
            raise ValueError(f"Unknown method: {method}")

    def test_normality(self, values: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """
        Test if values follow a normal distribution.

        Uses Shapiro-Wilk test (good for small samples).

        Args:
            values: List or array of values

        Returns:
            Dictionary with:
                - shapiro_pvalue: p-value from Shapiro-Wilk test
                - is_normal: Whether data appears normal (p > 0.05)
                - recommendation: Which test to use
        """
        values = np.array(values)

        # Shapiro-Wilk test
        stat, pvalue = shapiro(values)

        is_normal = pvalue > 0.05

        recommendation = (
            "Use parametric tests (t-test, ANOVA)"
            if is_normal
            else "Use non-parametric tests (Mann-Whitney U, Kruskal-Wallis)"
        )

        return {
            'shapiro_statistic': float(stat),
            'shapiro_pvalue': float(pvalue),
            'is_normal': is_normal,
            'sample_size': len(values),
            'recommendation': recommendation
        }

    def summarize_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Compute summary statistics for one experiment across all seeds.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Dictionary with summary statistics for all metrics
        """
        results = self.load_experiment(experiment_id)
        if not results:
            raise ValueError(f"Experiment not found: {experiment_id}")

        # Extract all metrics
        all_metrics = defaultdict(list)
        for result in results:
            if result.status == 'completed':
                for metric, value in result.final_eval_metrics.items():
                    all_metrics[metric].append(value)

        # Compute statistics for each metric
        summary = {
            'experiment_id': experiment_id,
            'name': results[0].config.name,
            'num_seeds': len(results),
            'num_completed': sum(1 for r in results if r.status == 'completed'),
            'metrics': {}
        }

        for metric, values in all_metrics.items():
            values = np.array(values)
            summary['metrics'][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values, ddof=1)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'ci_95': self.compute_confidence_intervals(values, confidence=0.95)
            }

        return summary

    def create_summary_table(
        self,
        experiment_ids: List[str],
        metrics: List[str]
    ) -> pd.DataFrame:
        """
        Create a summary table comparing experiments.

        Args:
            experiment_ids: List of experiment IDs to include
            metrics: List of metrics to include in table

        Returns:
            DataFrame with experiments as rows and metrics as columns
        """
        rows = []

        for exp_id in experiment_ids:
            summary = self.summarize_experiment(exp_id)

            row = {
                'Experiment': summary['name'],
                'Seeds': summary['num_completed']
            }

            for metric in metrics:
                if metric in summary['metrics']:
                    stats = summary['metrics'][metric]
                    row[f'{metric}_mean'] = stats['mean']
                    row[f'{metric}_std'] = stats['std']
                    row[f'{metric}_ci_lower'] = stats['ci_95'][0]
                    row[f'{metric}_ci_upper'] = stats['ci_95'][1]
                else:
                    row[f'{metric}_mean'] = np.nan
                    row[f'{metric}_std'] = np.nan

            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def identify_best_configuration(
        self,
        experiment_ids: List[str],
        metric: str = 'mean_reward',
        maximize: bool = True
    ) -> Dict[str, Any]:
        """
        Identify the best performing configuration.

        Args:
            experiment_ids: List of experiment IDs to compare
            metric: Metric to optimize
            maximize: Whether higher is better (default True)

        Returns:
            Dictionary with best configuration and statistical significance
        """
        # Collect results for all experiments
        exp_results = {}
        for exp_id in experiment_ids:
            results = self.load_experiment(exp_id)
            if results:
                completed = [r for r in results if r.status == 'completed']
                if completed:
                    values = [r.final_eval_metrics.get(metric, np.nan) for r in completed]
                    values = [v for v in values if not np.isnan(v)]
                    if values:
                        exp_results[exp_id] = {
                            'values': values,
                            'mean': np.mean(values),
                            'name': completed[0].config.name
                        }

        if not exp_results:
            raise ValueError("No valid results found")

        # Find best
        sorted_exps = sorted(
            exp_results.items(),
            key=lambda x: x[1]['mean'],
            reverse=maximize
        )

        best_id, best_data = sorted_exps[0]
        second_id, second_data = sorted_exps[1] if len(sorted_exps) > 1 else (None, None)

        result = {
            'best_experiment_id': best_id,
            'best_name': best_data['name'],
            'best_mean': best_data['mean'],
            'best_std': np.std(best_data['values'], ddof=1),
            'metric': metric,
            'all_rankings': [(exp_id, data['name'], data['mean']) for exp_id, data in sorted_exps]
        }

        # Compare with second best if available
        if second_id:
            comparison = self.compare_two_methods(
                best_data['values'],
                second_data['values'],
                metric=metric
            )
            result['vs_second_best'] = {
                'second_experiment_id': second_id,
                'second_name': second_data['name'],
                'second_mean': second_data['mean'],
                'improvement': (best_data['mean'] - second_data['mean']) / abs(second_data['mean']) * 100,
                'significant': comparison['significant'],
                'pvalue': comparison['t_test_pvalue'],
                'effect_size': comparison['cohens_d']
            }

        return result

    def analyze_learning_curve(self, experiment_id: str) -> Dict[str, Any]:
        """
        Analyze training progress and learning dynamics.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Dictionary with learning curve analysis:
                - convergence_episode: When training converged (if detected)
                - final_performance: Average performance in last 10% of training
                - stability: Coefficient of variation in final phase
                - trend: Overall trend (improving/degrading/stable)
        """
        results = self.load_experiment(experiment_id)
        if not results or not results[0].training_history:
            raise ValueError(f"No training history found for {experiment_id}")

        # Average across seeds
        all_episodes = []
        all_rewards = []

        for result in results:
            if result.training_history:
                episodes, rewards = [], []
                for ep, metrics in result.training_history:
                    episodes.append(ep)
                    rewards.append(metrics.get('reward', np.nan))
                all_episodes.append(episodes)
                all_rewards.append(rewards)

        # Use first seed's episodes (assuming all have same)
        episodes = all_episodes[0]
        mean_rewards = np.nanmean(all_rewards, axis=0)

        # Analyze final phase (last 10% of training)
        final_phase_start = int(len(mean_rewards) * 0.9)
        final_rewards = mean_rewards[final_phase_start:]

        final_performance = np.mean(final_rewards)
        final_std = np.std(final_rewards)
        stability = final_std / abs(final_performance) if final_performance != 0 else np.inf

        # Detect convergence (when moving average stabilizes)
        window_size = max(10, len(mean_rewards) // 10)
        moving_avg = np.convolve(mean_rewards, np.ones(window_size)/window_size, mode='valid')
        moving_std = np.array([np.std(mean_rewards[i:i+window_size]) for i in range(len(mean_rewards) - window_size + 1)])

        # Convergence when std drops below threshold
        converged_idx = np.where(moving_std < 0.05 * np.abs(moving_avg))[0]
        convergence_episode = episodes[converged_idx[0]] if len(converged_idx) > 0 else None

        # Overall trend (linear regression on rewards)
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(range(len(mean_rewards)), mean_rewards)

        if slope > 0.1:
            trend = 'improving'
        elif slope < -0.1:
            trend = 'degrading'
        else:
            trend = 'stable'

        return {
            'experiment_id': experiment_id,
            'total_episodes': len(episodes),
            'convergence_episode': convergence_episode,
            'final_performance': float(final_performance),
            'final_std': float(final_std),
            'stability_cv': float(stability),
            'trend': trend,
            'trend_slope': float(slope),
            'trend_r_squared': float(r_value**2)
        }

    def compare_learning_speeds(
        self,
        experiment_ids: List[str],
        target_performance: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Compare how quickly different methods learn.

        Args:
            experiment_ids: List of experiments to compare
            target_performance: Performance threshold to reach (if None, use 90% of best final)

        Returns:
            Dictionary with learning speed metrics for each experiment
        """
        learning_speeds = {}

        # First pass: determine target if not specified
        if target_performance is None:
            final_performances = []
            for exp_id in experiment_ids:
                analysis = self.analyze_learning_curve(exp_id)
                final_performances.append(analysis['final_performance'])
            target_performance = 0.9 * max(final_performances)

        # Second pass: find when each experiment reaches target
        for exp_id in experiment_ids:
            results = self.load_experiment(exp_id)
            if not results or not results[0].training_history:
                continue

            # Average rewards across seeds
            all_rewards = []
            for result in results:
                if result.training_history:
                    rewards = [metrics.get('reward', np.nan) for _, metrics in result.training_history]
                    all_rewards.append(rewards)

            mean_rewards = np.nanmean(all_rewards, axis=0)

            # Find first episode where target is reached
            reached_idx = np.where(mean_rewards >= target_performance)[0]
            episodes_to_target = reached_idx[0] if len(reached_idx) > 0 else None

            # Compute AUC (area under learning curve - sample efficiency)
            auc = np.trapz(mean_rewards) / len(mean_rewards)

            learning_speeds[exp_id] = {
                'name': results[0].config.name,
                'episodes_to_target': episodes_to_target,
                'auc': float(auc),
                'final_performance': float(mean_rewards[-1]) if len(mean_rewards) > 0 else np.nan
            }

        # Rank by learning speed
        valid_speeds = {k: v for k, v in learning_speeds.items() if v['episodes_to_target'] is not None}
        rankings = sorted(valid_speeds.items(), key=lambda x: x[1]['episodes_to_target'])

        return {
            'target_performance': target_performance,
            'learning_speeds': learning_speeds,
            'rankings': [(exp_id, data['name'], data['episodes_to_target']) for exp_id, data in rankings]
        }


class ResultsVisualizer:
    """
    Visualizer for generating publication-quality plots.

    Creates thesis-ready figures with consistent styling, proper fonts,
    and support for both color and grayscale output.

    Features:
    - Learning curves with confidence intervals
    - Bar charts with significance stars
    - Box plots for distribution comparison
    - Radar charts for multi-metric comparison
    - Heatmaps for hyperparameter analysis
    - Action distribution analysis
    - System performance plots

    Example:
        >>> visualizer = ResultsVisualizer()
        >>> visualizer.plot_learning_curves(
        ...     ['dqn_baseline', 'dqn_tuned'],
        ...     output_path='figures/learning.pdf'
        ... )
    """

    def __init__(self, style: str = 'seaborn-v0_8-paper', dpi: int = 300):
        """
        Initialize the visualizer with publication settings.

        Args:
            style: Matplotlib style
            dpi: DPI for saved figures (300+ recommended for publications)
        """
        self.dpi = dpi
        self.setup_publication_style(style)

    def setup_publication_style(self, style: str = 'seaborn-v0_8-paper'):
        """Configure matplotlib for publication-quality plots."""
        # Try to set the style, fall back to default if not available
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')

        # Publication settings
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 11,
            'figure.titlesize': 18,
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'pdf.fonttype': 42,  # TrueType fonts for PDF
            'ps.fonttype': 42,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'axes.linewidth': 1.0,
            'grid.linewidth': 0.5,
            'lines.linewidth': 2.0,
        })

        # Color palette
        self.colors = sns.color_palette('Set2', 10)
        self.grayscale = sns.color_palette('Greys', 10)

    def plot_learning_curves(
        self,
        analyzer: ResultsAnalyzer,
        experiment_ids: List[str],
        metric: str = 'reward',
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        smooth_window: int = 10,
        show_confidence: bool = True
    ):
        """
        Plot training curves for multiple experiments.

        Args:
            analyzer: ResultsAnalyzer instance
            experiment_ids: List of experiment IDs to plot
            metric: Metric to plot (default 'reward')
            output_path: Path to save figure (if None, displays instead)
            title: Plot title (if None, auto-generated)
            smooth_window: Window size for smoothing (0 for no smoothing)
            show_confidence: Whether to show confidence intervals
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        for idx, exp_id in enumerate(experiment_ids):
            results = analyzer.load_experiment(exp_id)
            if not results:
                continue

            # Collect training histories
            all_episodes = []
            all_values = []

            for result in results:
                if result.training_history:
                    episodes = []
                    values = []
                    for ep, metrics in result.training_history:
                        episodes.append(ep)
                        values.append(metrics.get(metric, np.nan))
                    all_episodes.append(episodes)
                    all_values.append(values)

            if not all_values:
                continue

            # Average across seeds
            episodes = all_episodes[0]
            mean_values = np.nanmean(all_values, axis=0)
            std_values = np.nanstd(all_values, axis=0)

            # Smooth if requested
            if smooth_window > 1:
                mean_values = np.convolve(mean_values, np.ones(smooth_window)/smooth_window, mode='valid')
                std_values = np.convolve(std_values, np.ones(smooth_window)/smooth_window, mode='valid')
                episodes = episodes[:len(mean_values)]

            # Plot
            label = results[0].config.name
            color = self.colors[idx % len(self.colors)]

            ax.plot(episodes, mean_values, label=label, color=color, linewidth=2)

            if show_confidence and len(all_values) > 1:
                ax.fill_between(
                    episodes,
                    mean_values - std_values,
                    mean_values + std_values,
                    alpha=0.2,
                    color=color
                )

        ax.set_xlabel('Training Episode')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title or f'Learning Curves: {metric.replace("_", " ").title()}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved learning curves to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_training_comparison(
        self,
        analyzer: ResultsAnalyzer,
        experiment_ids: List[str],
        metrics: List[str] = ['reward', 'loss', 'cache_hit_rate'],
        output_path: Optional[str] = None
    ):
        """
        Side-by-side comparison of training progress across multiple metrics.

        Args:
            analyzer: ResultsAnalyzer instance
            experiment_ids: List of experiment IDs
            metrics: List of metrics to plot
            output_path: Path to save figure
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx]

            for exp_idx, exp_id in enumerate(experiment_ids):
                results = analyzer.load_experiment(exp_id)
                if not results:
                    continue

                # Collect data
                all_values = []
                for result in results:
                    if result.training_history:
                        episodes = []
                        values = []
                        for ep, metrics_dict in result.training_history:
                            episodes.append(ep)
                            values.append(metrics_dict.get(metric, np.nan))
                        all_values.append(values)

                if not all_values:
                    continue

                mean_values = np.nanmean(all_values, axis=0)
                color = self.colors[exp_idx % len(self.colors)]
                label = results[0].config.name

                ax.plot(episodes, mean_values, label=label, color=color, linewidth=2)

            ax.set_xlabel('Episode')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(metric.replace('_', ' ').title())
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved training comparison to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_bar_comparison(
        self,
        analyzer: ResultsAnalyzer,
        experiment_ids: List[str],
        metric: str,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        show_significance: bool = True,
        sort_by_performance: bool = True
    ):
        """
        Bar chart comparing methods on one metric.

        Args:
            analyzer: ResultsAnalyzer instance
            experiment_ids: List of experiment IDs
            metric: Metric to compare
            output_path: Path to save figure
            title: Plot title
            show_significance: Whether to add significance stars
            sort_by_performance: Whether to sort bars by performance
        """
        # Collect data
        data = []
        for exp_id in experiment_ids:
            results = analyzer.load_experiment(exp_id)
            if results:
                completed = [r for r in results if r.status == 'completed']
                if completed:
                    values = [r.final_eval_metrics.get(metric, np.nan) for r in completed]
                    values = [v for v in values if not np.isnan(v)]
                    if values:
                        data.append({
                            'name': completed[0].config.name,
                            'mean': np.mean(values),
                            'std': np.std(values, ddof=1) if len(values) > 1 else 0,
                            'values': values,
                            'exp_id': exp_id
                        })

        if not data:
            print("No data to plot")
            return

        # Sort if requested
        if sort_by_performance:
            data = sorted(data, key=lambda x: x['mean'], reverse=True)

        # Create plot
        fig, ax = plt.subplots(figsize=(max(8, len(data) * 1.2), 6))

        names = [d['name'] for d in data]
        means = [d['mean'] for d in data]
        stds = [d['std'] for d in data]

        x_pos = np.arange(len(names))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                      color=self.colors[:len(names)], alpha=0.8, edgecolor='black')

        # Add significance stars if requested
        if show_significance and len(data) > 1:
            # Compare best with others
            best_values = data[0]['values']
            y_max = max(means) + max(stds) * 1.1

            for i in range(1, len(data)):
                comparison = analyzer.compare_two_methods(best_values, data[i]['values'], metric)
                if comparison['significant']:
                    # Add star
                    pval = comparison['t_test_pvalue']
                    if pval < 0.001:
                        stars = '***'
                    elif pval < 0.01:
                        stars = '**'
                    else:
                        stars = '*'

                    ax.text(i, means[i] + stds[i] + y_max * 0.02, stars,
                           ha='center', va='bottom', fontsize=14, fontweight='bold')

        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title or f'{metric.replace("_", " ").title()} Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (bar, mean) in enumerate(zip(bars, means)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + stds[i],
                   f'{mean:.2f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved bar comparison to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_box_comparison(
        self,
        analyzer: ResultsAnalyzer,
        experiment_ids: List[str],
        metric: str,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ):
        """
        Box plot showing distribution of results across methods.

        Args:
            analyzer: ResultsAnalyzer instance
            experiment_ids: List of experiment IDs
            metric: Metric to compare
            output_path: Path to save figure
            title: Plot title
        """
        # Collect data
        data_dict = {}
        names = []

        for exp_id in experiment_ids:
            results = analyzer.load_experiment(exp_id)
            if results:
                completed = [r for r in results if r.status == 'completed']
                if completed:
                    values = [r.final_eval_metrics.get(metric, np.nan) for r in completed]
                    values = [v for v in values if not np.isnan(v)]
                    if values:
                        name = completed[0].config.name
                        data_dict[name] = values
                        names.append(name)

        if not data_dict:
            print("No data to plot")
            return

        # Create plot
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 6))

        bp = ax.boxplot(data_dict.values(), labels=names, patch_artist=True,
                        showmeans=True, meanline=True)

        # Color boxes
        for patch, color in zip(bp['boxes'], self.colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title or f'{metric.replace("_", " ").title()} Distribution')
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved box plot to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_radar_chart(
        self,
        analyzer: ResultsAnalyzer,
        experiment_ids: List[str],
        metrics: List[str],
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        normalize: bool = True
    ):
        """
        Radar/spider chart comparing methods across multiple metrics.

        Args:
            analyzer: ResultsAnalyzer instance
            experiment_ids: List of experiment IDs
            metrics: List of metrics to include
            output_path: Path to save figure
            title: Plot title
            normalize: Whether to normalize metrics to [0, 1]
        """
        from math import pi

        # Collect data
        data = []
        for exp_id in experiment_ids:
            summary = analyzer.summarize_experiment(exp_id)
            method_data = {'name': summary['name']}

            for metric in metrics:
                if metric in summary['metrics']:
                    method_data[metric] = summary['metrics'][metric]['mean']
                else:
                    method_data[metric] = 0

            data.append(method_data)

        if not data:
            print("No data to plot")
            return

        # Normalize if requested
        if normalize:
            for metric in metrics:
                values = [d[metric] for d in data]
                min_val, max_val = min(values), max(values)
                if max_val > min_val:
                    for d in data:
                        d[metric] = (d[metric] - min_val) / (max_val - min_val)

        # Setup radar chart
        num_vars = len(metrics)
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Plot each method
        for idx, method_data in enumerate(data):
            values = [method_data[m] for m in metrics]
            values += values[:1]

            color = self.colors[idx % len(self.colors)]
            ax.plot(angles, values, 'o-', linewidth=2, label=method_data['name'], color=color)
            ax.fill(angles, values, alpha=0.15, color=color)

        # Fix axis
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_ylim(0, 1 if normalize else None)
        ax.set_title(title or 'Multi-Metric Comparison', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved radar chart to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_heatmap(
        self,
        results_df: pd.DataFrame,
        x_param: str,
        y_param: str,
        metric: str,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ):
        """
        Heatmap for hyperparameter sensitivity analysis.

        Args:
            results_df: DataFrame with experiment results
            x_param: Column name for x-axis parameter
            y_param: Column name for y-axis parameter
            metric: Column name for metric to visualize
            output_path: Path to save figure
            title: Plot title
        """
        # Pivot data for heatmap
        pivot_table = results_df.pivot_table(
            values=metric,
            index=y_param,
            columns=x_param,
            aggfunc='mean'
        )

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd',
                   ax=ax, cbar_kws={'label': metric.replace('_', ' ').title()})

        ax.set_xlabel(x_param.replace('_', ' ').title())
        ax.set_ylabel(y_param.replace('_', ' ').title())
        ax.set_title(title or f'{metric.replace("_", " ").title()} Heatmap')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved heatmap to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_action_distribution(
        self,
        analyzer: ResultsAnalyzer,
        experiment_id: str,
        output_path: Optional[str] = None,
        plot_type: str = 'bar'
    ):
        """
        Visualize action distribution (if available in results).

        Args:
            analyzer: ResultsAnalyzer instance
            experiment_id: Experiment identifier
            output_path: Path to save figure
            plot_type: 'bar' or 'pie'
        """
        results = analyzer.load_experiment(experiment_id)
        if not results:
            print(f"Experiment {experiment_id} not found")
            return

        # Try to extract action distribution from metrics
        action_counts = {}
        for result in results:
            if 'action_distribution' in result.final_eval_metrics:
                dist = result.final_eval_metrics['action_distribution']
                for action, count in dist.items():
                    action_counts[action] = action_counts.get(action, 0) + count

        if not action_counts:
            print("No action distribution data found")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        actions = list(action_counts.keys())
        counts = list(action_counts.values())

        if plot_type == 'pie':
            ax.pie(counts, labels=actions, autopct='%1.1f%%', colors=self.colors)
            ax.set_title(f'Action Distribution: {results[0].config.name}')
        else:  # bar
            x_pos = np.arange(len(actions))
            ax.bar(x_pos, counts, color=self.colors[:len(actions)])
            ax.set_xticks(x_pos)
            ax.set_xticklabels(actions, rotation=45, ha='right')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Action Distribution: {results[0].config.name}')
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved action distribution to {output_path}")
        else:
            plt.show()

        plt.close()

    def generate_all_thesis_figures(
        self,
        analyzer: ResultsAnalyzer,
        experiment_ids: List[str],
        output_dir: str,
        metrics: List[str] = ['mean_reward', 'cache_hit_rate', 'latency']
    ) -> Dict[str, str]:
        """
        Generate all figures needed for thesis.

        Args:
            analyzer: ResultsAnalyzer instance
            experiment_ids: List of experiment IDs
            output_dir: Directory to save figures
            metrics: Metrics to visualize

        Returns:
            Dictionary mapping figure names to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        figures = {}

        # 1. Learning curves
        for metric in metrics:
            fig_path = output_path / f'learning_curves_{metric}.pdf'
            self.plot_learning_curves(analyzer, experiment_ids, metric=metric, output_path=str(fig_path))
            figures[f'learning_curves_{metric}'] = str(fig_path)

        # 2. Training comparison
        fig_path = output_path / 'training_comparison.pdf'
        self.plot_training_comparison(analyzer, experiment_ids, metrics=metrics[:3], output_path=str(fig_path))
        figures['training_comparison'] = str(fig_path)

        # 3. Bar comparisons
        for metric in metrics:
            fig_path = output_path / f'bar_comparison_{metric}.pdf'
            self.plot_bar_comparison(analyzer, experiment_ids, metric=metric, output_path=str(fig_path))
            figures[f'bar_comparison_{metric}'] = str(fig_path)

        # 4. Box plots
        for metric in metrics:
            fig_path = output_path / f'box_plot_{metric}.pdf'
            self.plot_box_comparison(analyzer, experiment_ids, metric=metric, output_path=str(fig_path))
            figures[f'box_plot_{metric}'] = str(fig_path)

        # 5. Radar chart
        fig_path = output_path / 'radar_comparison.pdf'
        self.plot_radar_chart(analyzer, experiment_ids, metrics=metrics, output_path=str(fig_path))
        figures['radar_comparison'] = str(fig_path)

        print(f"\n✓ Generated {len(figures)} thesis figures in {output_dir}")

        # Save manifest
        manifest_path = output_path / 'figures_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(figures, f, indent=2)
        print(f"✓ Saved manifest to {manifest_path}")

        return figures


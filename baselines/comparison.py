"""
Baseline comparison framework for evaluating caching policies.

This module provides tools to compare multiple baseline policies against each other
and against trained RL agents in a fair and comprehensive manner.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import json
from scipy import stats
import time

from .base_policy import CachingPolicy, PolicyWrapper

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ComparisonConfig:
    """Configuration for baseline comparison."""
    num_episodes: int = 100
    seed: int = 42
    parallel: bool = False
    track_detailed_metrics: bool = True
    save_episode_data: bool = True
    confidence_level: float = 0.95


@dataclass
class PolicyResults:
    """Results for a single policy."""
    policy_name: str

    # Episode-level metrics
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    cache_hit_rates: List[float] = field(default_factory=list)
    prediction_accuracies: List[float] = field(default_factory=list)
    cascade_occurred: List[bool] = field(default_factory=list)
    prefetch_efficiencies: List[float] = field(default_factory=list)
    avg_latencies: List[float] = field(default_factory=list)
    bandwidth_used: List[float] = field(default_factory=list)

    # Aggregated statistics
    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_hit_rate: float = 0.0
    cascade_rate: float = 0.0
    mean_prefetch_efficiency: float = 0.0
    mean_latency_improvement: float = 0.0
    total_episodes: int = 0

    def compute_statistics(self):
        """Compute aggregated statistics from episode data."""
        if not self.episode_rewards:
            return

        self.total_episodes = len(self.episode_rewards)
        self.mean_reward = float(np.mean(self.episode_rewards))
        self.std_reward = float(np.std(self.episode_rewards))
        self.mean_hit_rate = float(np.mean(self.cache_hit_rates))
        self.cascade_rate = float(sum(self.cascade_occurred) / len(self.cascade_occurred))
        self.mean_prefetch_efficiency = float(np.mean(self.prefetch_efficiencies))
        self.mean_latency_improvement = float(np.mean(self.avg_latencies))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'policy_name': self.policy_name,
            'mean_reward': self.mean_reward,
            'std_reward': self.std_reward,
            'mean_hit_rate': self.mean_hit_rate,
            'cascade_rate': self.cascade_rate,
            'mean_prefetch_efficiency': self.mean_prefetch_efficiency,
            'mean_latency_improvement': self.mean_latency_improvement,
            'total_episodes': self.total_episodes
        }


class BaselineComparator:
    """
    Unified comparison framework for baseline policies and RL agents.

    This class provides tools to:
    - Register multiple policies for comparison
    - Run fair evaluations on the same environment
    - Track comprehensive metrics
    - Generate comparison reports and visualizations
    - Perform statistical significance tests

    Example:
        >>> comparator = BaselineComparator()
        >>> comparator.add_policy('LRU', LRUPolicy())
        >>> comparator.add_policy('LFU', LFUPolicy())
        >>> comparator.add_trained_agent('DQN', trained_agent)
        >>> results = comparator.run_comparison(env, num_episodes=100)
        >>> report_path = comparator.generate_report(results, 'results/')
    """

    def __init__(self, config: Optional[ComparisonConfig] = None):
        """
        Initialize comparator.

        Args:
            config: Comparison configuration
        """
        self.config = config or ComparisonConfig()
        self.policies: Dict[str, CachingPolicy] = {}
        self.results: Dict[str, PolicyResults] = {}

        logger.info("BaselineComparator initialized")

    def add_policy(self, name: str, policy: CachingPolicy):
        """
        Register a policy for comparison.

        Args:
            name: Unique name for the policy
            policy: CachingPolicy instance
        """
        if name in self.policies:
            logger.warning(f"Policy '{name}' already registered, overwriting")

        self.policies[name] = policy
        logger.info(f"Registered policy: {name}")

    def add_trained_agent(self, name: str, agent: Any):
        """
        Register a trained RL agent for comparison.

        The agent must have a predict(observation) method that returns (action, _).

        Args:
            name: Unique name for the agent
            agent: Trained agent (e.g., DQN from stable-baselines3)
        """
        # Wrap agent in adapter that implements CachingPolicy interface
        from .agent_adapter import RLAgentAdapter
        adapter = RLAgentAdapter(agent, name)
        self.add_policy(name, adapter)

    def run_comparison(
        self,
        env,
        num_episodes: Optional[int] = None,
        seed: Optional[int] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Run comparison of all registered policies.

        Args:
            env: Gym environment to evaluate on
            num_episodes: Number of episodes per policy (uses config default if None)
            seed: Random seed (uses config default if None)
            verbose: Print progress

        Returns:
            DataFrame with comparison results
        """
        num_episodes = num_episodes or self.config.num_episodes
        seed = seed or self.config.seed

        if not self.policies:
            raise ValueError("No policies registered. Use add_policy() first.")

        logger.info(f"Starting comparison of {len(self.policies)} policies over {num_episodes} episodes")

        # Evaluate each policy
        for policy_name, policy in self.policies.items():
            if verbose:
                print(f"\nEvaluating {policy_name}...")

            start_time = time.time()
            results = self._evaluate_policy(env, policy, num_episodes, seed)
            elapsed = time.time() - start_time

            self.results[policy_name] = results

            if verbose:
                print(f"  Mean reward: {results.mean_reward:.2f} ± {results.std_reward:.2f}")
                print(f"  Cache hit rate: {results.mean_hit_rate:.2%}")
                print(f"  Cascade rate: {results.cascade_rate:.2%}")
                print(f"  Time: {elapsed:.1f}s")

        # Convert to DataFrame
        df = self._results_to_dataframe()

        logger.info("Comparison complete")
        return df

    def _evaluate_policy(
        self,
        env,
        policy: CachingPolicy,
        num_episodes: int,
        seed: int
    ) -> PolicyResults:
        """Evaluate a single policy."""
        results = PolicyResults(policy_name=policy.get_name())

        for episode_idx in range(num_episodes):
            episode_seed = seed + episode_idx
            obs, _ = env.reset(seed=episode_seed)
            policy.reset()

            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done:
                # Get Markov predictions from environment
                # This assumes env exposes predictions somehow
                predictions = self._get_predictions_from_env(env)

                # Select action
                action = policy.select_action(obs, predictions)

                # Take step
                obs, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward
                episode_length += 1
                done = terminated or truncated

            # Record episode results
            metrics = env.get_episode_metrics()
            results.episode_rewards.append(episode_reward)
            results.episode_lengths.append(episode_length)
            results.cache_hit_rates.append(metrics.get('cache_hit_rate', 0.0))
            results.prediction_accuracies.append(metrics.get('prediction_accuracy', 0.0))
            results.cascade_occurred.append(metrics.get('cascade_occurred', False))
            results.prefetch_efficiencies.append(metrics.get('prefetch_efficiency', 0.0))
            results.avg_latencies.append(metrics.get('average_latency', 0.0))
            results.bandwidth_used.append(metrics.get('bandwidth_used', 0.0))

        # Compute statistics
        results.compute_statistics()

        return results

    def _get_predictions_from_env(self, env) -> List:
        """Extract predictions from environment."""
        # Try to get predictions from environment
        if hasattr(env, 'get_current_predictions'):
            return env.get_current_predictions()
        elif hasattr(env, 'predictor'):
            return env.predictor.predict(k=5)
        else:
            # Return empty if predictions not available
            return []

    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        data = []
        for policy_name, results in self.results.items():
            data.append(results.to_dict())

        df = pd.DataFrame(data)
        df = df.sort_values('mean_reward', ascending=False)
        return df

    def generate_report(
        self,
        results: pd.DataFrame,
        output_path: Union[str, Path],
        include_plots: bool = True
    ) -> str:
        """
        Generate comprehensive comparison report.

        Args:
            results: DataFrame from run_comparison()
            output_path: Directory to save report
            include_plots: Whether to generate and include plots

        Returns:
            Path to generated report file
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        report_file = output_path / 'baseline_comparison_report.md'

        with open(report_file, 'w') as f:
            # Header
            f.write("# Baseline Caching Policy Comparison Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Policies compared: {len(results)}\n")
            f.write(f"Episodes per policy: {self.config.num_episodes}\n\n")

            # Summary table
            f.write("## Summary Statistics\n\n")
            f.write(results.to_markdown(index=False))
            f.write("\n\n")

            # Best policy
            best_policy = results.iloc[0]
            f.write("## Best Policy\n\n")
            f.write(f"**{best_policy['policy_name']}**\n\n")
            f.write(f"- Mean Reward: {best_policy['mean_reward']:.2f} ± {best_policy['std_reward']:.2f}\n")
            f.write(f"- Cache Hit Rate: {best_policy['mean_hit_rate']:.2%}\n")
            f.write(f"- Cascade Rate: {best_policy['cascade_rate']:.2%}\n")
            f.write(f"- Prefetch Efficiency: {best_policy['mean_prefetch_efficiency']:.2%}\n\n")

            # Statistical significance
            f.write("## Statistical Significance Tests\n\n")
            sig_results = self._compute_significance_tests()
            f.write(sig_results)
            f.write("\n\n")

            # Detailed analysis
            f.write("## Detailed Analysis\n\n")
            for policy_name, policy_results in self.results.items():
                f.write(f"### {policy_name}\n\n")
                f.write(f"- Episodes: {policy_results.total_episodes}\n")
                f.write(f"- Reward: {policy_results.mean_reward:.2f} ± {policy_results.std_reward:.2f}\n")
                f.write(f"- Hit Rate: {policy_results.mean_hit_rate:.2%}\n")
                f.write(f"- Cascade Rate: {policy_results.cascade_rate:.2%}\n")
                f.write(f"- Prefetch Efficiency: {policy_results.mean_prefetch_efficiency:.2%}\n\n")

            # Include plots
            if include_plots:
                f.write("## Visualizations\n\n")
                self._generate_plots(output_path)
                f.write(f"![Reward Comparison](reward_comparison.png)\n\n")
                f.write(f"![Hit Rate Comparison](hitrate_comparison.png)\n\n")
                f.write(f"![Cascade Rate Comparison](cascade_comparison.png)\n\n")

        logger.info(f"Report generated: {report_file}")
        return str(report_file)

    def _compute_significance_tests(self) -> str:
        """Compute pairwise statistical significance tests."""
        if len(self.results) < 2:
            return "Need at least 2 policies for significance testing.\n"

        output = []
        policy_names = list(self.results.keys())

        # Compare each pair
        for i in range(len(policy_names)):
            for j in range(i + 1, len(policy_names)):
                policy1 = policy_names[i]
                policy2 = policy_names[j]

                rewards1 = self.results[policy1].episode_rewards
                rewards2 = self.results[policy2].episode_rewards

                # Perform t-test
                t_stat, p_value = stats.ttest_ind(rewards1, rewards2)

                # Determine significance
                if p_value < 0.001:
                    sig_level = "***"
                elif p_value < 0.01:
                    sig_level = "**"
                elif p_value < 0.05:
                    sig_level = "*"
                else:
                    sig_level = "ns"

                output.append(
                    f"- **{policy1}** vs **{policy2}**: "
                    f"t={t_stat:.2f}, p={p_value:.4f} {sig_level}\n"
                )

        return "".join(output)

    def _generate_plots(self, output_path: Path):
        """Generate comparison plots."""
        # Set style
        sns.set_style("whitegrid")

        # Plot 1: Reward comparison (box plot)
        self.plot_comparison(
            metric='reward',
            output_path=str(output_path / 'reward_comparison.png')
        )

        # Plot 2: Hit rate comparison
        self.plot_comparison(
            metric='cache_hit_rate',
            output_path=str(output_path / 'hitrate_comparison.png')
        )

        # Plot 3: Cascade rate comparison
        self.plot_comparison(
            metric='cascade_rate',
            output_path=str(output_path / 'cascade_comparison.png')
        )

    def plot_comparison(
        self,
        metric: str = 'reward',
        output_path: Optional[str] = None,
        figsize: tuple = (12, 6)
    ):
        """
        Create comparison plot for a specific metric.

        Args:
            metric: Metric to plot ('reward', 'cache_hit_rate', 'cascade_rate', etc.)
            output_path: Path to save plot (if None, displays instead)
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Prepare data
        if metric == 'reward':
            data = [results.episode_rewards for results in self.results.values()]
            labels = list(self.results.keys())
            ylabel = 'Episode Reward'
            title = 'Policy Comparison: Episode Rewards'

        elif metric == 'cache_hit_rate':
            data = [results.cache_hit_rates for results in self.results.values()]
            labels = list(self.results.keys())
            ylabel = 'Cache Hit Rate'
            title = 'Policy Comparison: Cache Hit Rate'

        elif metric == 'cascade_rate':
            # Bar plot for cascade rates
            cascade_rates = [results.cascade_rate for results in self.results.values()]
            labels = list(self.results.keys())

            ax.bar(range(len(labels)), cascade_rates)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Cascade Rate')
            ax.set_title('Policy Comparison: Cascade Rate')
            ax.set_ylim(0, 1)

            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved: {output_path}")
            else:
                plt.show()

            plt.close()
            return

        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Create box plot
        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        # Color boxes
        colors = sns.color_palette("husl", len(data))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved: {output_path}")
        else:
            plt.show()

        plt.close()

    def save_results(self, output_path: Union[str, Path]):
        """
        Save detailed results to JSON.

        Args:
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert results to JSON-serializable format
        results_dict = {}
        for policy_name, results in self.results.items():
            results_dict[policy_name] = {
                'summary': results.to_dict(),
                'episodes': {
                    'rewards': results.episode_rewards,
                    'lengths': results.episode_lengths,
                    'hit_rates': results.cache_hit_rates,
                    'cascade_occurred': results.cascade_occurred
                }
            }

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"Results saved: {output_path}")

    def load_results(self, input_path: Union[str, Path]):
        """
        Load previously saved results.

        Args:
            input_path: Path to load results from
        """
        with open(input_path, 'r') as f:
            results_dict = json.load(f)

        # Reconstruct PolicyResults objects
        for policy_name, data in results_dict.items():
            results = PolicyResults(policy_name=policy_name)
            results.episode_rewards = data['episodes']['rewards']
            results.episode_lengths = data['episodes']['lengths']
            results.cache_hit_rates = data['episodes']['hit_rates']
            results.cascade_occurred = data['episodes']['cascade_occurred']
            results.compute_statistics()

            self.results[policy_name] = results

        logger.info(f"Results loaded: {input_path}")


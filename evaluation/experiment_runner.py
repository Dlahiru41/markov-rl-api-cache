"""
Comprehensive experiment runner for systematic evaluation of configurations.

This module automates experiment execution, tracks results, and ensures reproducibility
for thesis evaluation. Supports hyperparameter sweeps, ablation studies, and baseline
comparisons with statistical significance testing.
"""

import os
import sys
import json
import yaml
import hashlib
import logging
import time
import pickle
import shutil
import psutil
import platform
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime
from collections import defaultdict
import itertools
import copy

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """
    Configuration for a single experiment.

    Defines all parameters needed to run an experiment, including training settings,
    evaluation parameters, and metadata for organization.
    """
    name: str
    description: str
    hypothesis: str
    controller_config: Dict[str, Any]
    num_training_episodes: int
    num_eval_episodes: int
    baselines_to_compare: List[str] = field(default_factory=list)
    seeds: List[int] = field(default_factory=lambda: [42])
    tags: List[str] = field(default_factory=list)
    priority: int = 0

    # Optional overrides
    early_stopping_patience: Optional[int] = None
    checkpoint_interval: int = 100
    log_interval: int = 10

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ExperimentResult:
    """
    Results from a single experiment run.

    Captures all metrics, artifacts, and metadata from an experiment execution.
    """
    config: ExperimentConfig
    seed: int
    training_history: List[Tuple[int, Dict[str, float]]] = field(default_factory=list)
    final_eval_metrics: Dict[str, float] = field(default_factory=dict)
    baseline_comparisons: Dict[str, Dict[str, float]] = field(default_factory=dict)
    training_time_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    artifacts: Dict[str, str] = field(default_factory=dict)
    status: str = 'pending'  # 'pending', 'running', 'completed', 'failed', 'interrupted'
    error_message: Optional[str] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['config'] = self.config.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentResult':
        """Create from dictionary."""
        config = ExperimentConfig.from_dict(data.pop('config'))
        return cls(config=config, **data)


class ExperimentRunner:
    """
    Runner for executing and managing experiments systematically.

    Features:
    - Queue-based experiment management
    - Hyperparameter sweep generation
    - Ablation study automation
    - Baseline comparison integration
    - Reproducibility guarantees
    - Resource management
    - Progress tracking and resumption
    - Results aggregation and export

    Example:
        >>> runner = ExperimentRunner(output_dir='results/experiments')
        >>> config = ExperimentConfig(name='test', ...)
        >>> exp_id = runner.add_experiment(config)
        >>> runner.run_all()
        >>> results = runner.get_results()
    """

    def __init__(self, output_dir: str, max_parallel: int = 1):
        """
        Initialize experiment runner.

        Args:
            output_dir: Directory to save all experiment results
            max_parallel: Maximum number of experiments to run simultaneously
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_parallel = max_parallel

        # Experiment queue
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.results: Dict[str, List[ExperimentResult]] = defaultdict(list)

        # Tracking
        self.experiment_counter = 0
        self.completed_experiments: set = set()

        # Setup directories
        self._setup_directories()

        # Load existing results if any
        self._load_existing_results()

        logger.info(f"ExperimentRunner initialized: {self.output_dir}")

    def _setup_directories(self):
        """Create directory structure for experiments."""
        (self.output_dir / 'configs').mkdir(exist_ok=True)
        (self.output_dir / 'results').mkdir(exist_ok=True)
        (self.output_dir / 'artifacts').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)

    def _load_existing_results(self):
        """Load previously saved results."""
        results_file = self.output_dir / 'all_results.json'
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    for exp_id, result_list in data.items():
                        self.results[exp_id] = [
                            ExperimentResult.from_dict(r) for r in result_list
                        ]
                        if result_list:
                            self.completed_experiments.add(exp_id)
                logger.info(f"Loaded {len(self.results)} existing experiments")
            except Exception as e:
                logger.warning(f"Failed to load existing results: {e}")

    def add_experiment(self, config: ExperimentConfig) -> str:
        """
        Add a single experiment to the queue.

        Args:
            config: Experiment configuration

        Returns:
            Experiment ID
        """
        # Generate unique ID
        config_hash = self._hash_config(config)
        exp_id = f"{config.name}_{config_hash[:8]}"

        # Check if already exists
        if exp_id in self.experiments:
            logger.warning(f"Experiment {exp_id} already exists, skipping")
            return exp_id

        # Save config
        self.experiments[exp_id] = config
        config_path = self.output_dir / 'configs' / f'{exp_id}.json'
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

        logger.info(f"Added experiment: {exp_id} (priority={config.priority})")
        return exp_id

    def add_hyperparameter_sweep(
        self,
        base_config: ExperimentConfig,
        param_grid: Dict[str, List[Any]]
    ) -> List[str]:
        """
        Generate experiments for hyperparameter sweep.

        Args:
            base_config: Base configuration to modify
            param_grid: Dictionary mapping parameter paths to lists of values
                       e.g., {'learning_rate': [0.001, 0.0001], 'gamma': [0.95, 0.99]}

        Returns:
            List of experiment IDs
        """
        exp_ids = []

        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        for combination in itertools.product(*param_values):
            # Create modified config
            config = copy.deepcopy(base_config)

            # Build experiment name
            name_parts = [base_config.name]

            # Apply parameter values
            for param_name, param_value in zip(param_names, combination):
                # Set value in config
                self._set_nested_value(config.controller_config, param_name, param_value)

                # Add to name
                short_name = param_name.split('.')[-1]
                name_parts.append(f"{short_name}={param_value}")

            # Update config
            config.name = "_".join(name_parts)
            config.description = f"Hyperparameter sweep: {dict(zip(param_names, combination))}"
            config.tags = base_config.tags + ['hyperparameter_sweep']

            # Add experiment
            exp_id = self.add_experiment(config)
            exp_ids.append(exp_id)

        logger.info(f"Generated {len(exp_ids)} experiments for hyperparameter sweep")
        return exp_ids

    def add_ablation_study(
        self,
        base_config: ExperimentConfig,
        components: Dict[str, bool]
    ) -> List[str]:
        """
        Generate experiments for ablation study.

        Tests contribution of each component by disabling them one at a time.

        Args:
            base_config: Base configuration with all components enabled
            components: Dictionary of component names to their default values
                       e.g., {'markov_context': True, 'dueling_dqn': True}

        Returns:
            List of experiment IDs
        """
        exp_ids = []

        # Add full configuration
        full_config = copy.deepcopy(base_config)
        full_config.name = f"{base_config.name}_full"
        full_config.description = "Full configuration with all components"
        full_config.tags = base_config.tags + ['ablation', 'full']
        exp_ids.append(self.add_experiment(full_config))

        # Add configuration with each component disabled
        for component_name, default_value in components.items():
            config = copy.deepcopy(base_config)

            # Disable this component
            disabled_value = not default_value if isinstance(default_value, bool) else None
            self._set_nested_value(config.controller_config, component_name, disabled_value)

            # Update metadata
            config.name = f"{base_config.name}_no_{component_name}"
            config.description = f"Ablation: without {component_name}"
            config.tags = base_config.tags + ['ablation', f'no_{component_name}']

            exp_id = self.add_experiment(config)
            exp_ids.append(exp_id)

        logger.info(f"Generated {len(exp_ids)} experiments for ablation study")
        return exp_ids

    def run_all(self, progress_callback: Optional[Callable[[str, str, float], None]] = None):
        """
        Run all queued experiments.

        Args:
            progress_callback: Optional callback(exp_id, status, progress)
        """
        # Sort experiments by priority
        sorted_experiments = sorted(
            self.experiments.items(),
            key=lambda x: x[1].priority,
            reverse=True
        )

        total_experiments = sum(len(config.seeds) for _, config in sorted_experiments)
        completed = 0

        logger.info(f"Starting {total_experiments} experiment runs")

        for exp_id, config in sorted_experiments:
            # Skip if already completed
            if exp_id in self.completed_experiments:
                logger.info(f"Skipping completed experiment: {exp_id}")
                completed += len(config.seeds)
                continue

            # Run for each seed
            for seed_idx, seed in enumerate(config.seeds):
                try:
                    if progress_callback:
                        progress = completed / total_experiments
                        progress_callback(exp_id, 'running', progress)

                    logger.info(f"Running {exp_id} with seed {seed} ({seed_idx + 1}/{len(config.seeds)})")

                    # Run experiment
                    result = self.run_experiment(config, seed)

                    # Save result
                    self.results[exp_id].append(result)
                    self._save_results()

                    completed += 1

                    if progress_callback:
                        progress = completed / total_experiments
                        progress_callback(exp_id, result.status, progress)

                except KeyboardInterrupt:
                    logger.warning("Interrupted by user")
                    if progress_callback:
                        progress_callback(exp_id, 'interrupted', completed / total_experiments)
                    raise

                except Exception as e:
                    logger.error(f"Experiment {exp_id} (seed {seed}) failed: {e}")

                    # Create failed result
                    result = ExperimentResult(
                        config=config,
                        seed=seed,
                        status='failed',
                        error_message=str(e),
                        timestamp=datetime.now().isoformat()
                    )
                    self.results[exp_id].append(result)
                    self._save_results()

                    completed += 1

                    if progress_callback:
                        progress_callback(exp_id, 'failed', completed / total_experiments)

            # Mark experiment as completed
            self.completed_experiments.add(exp_id)

        logger.info(f"Completed {completed} experiment runs")

    def run_experiment(self, config: ExperimentConfig, seed: int) -> ExperimentResult:
        """
        Run a single experiment with given seed.

        Args:
            config: Experiment configuration
            seed: Random seed

        Returns:
            Experiment result
        """
        start_time = time.time()

        # Create result object
        result = ExperimentResult(
            config=config,
            seed=seed,
            status='running',
            timestamp=datetime.now().isoformat()
        )

        # Setup logging for this experiment
        exp_id = self._hash_config(config)[:8]
        log_file = self.output_dir / 'logs' / f'{config.name}_{seed}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        try:
            # 1. Set all random seeds
            self._set_all_seeds(seed)

            # 2. Save environment info
            env_info_path = self.output_dir / 'artifacts' / f'{config.name}_{seed}_env.json'
            self._save_environment_info(env_info_path)

            # 3. Check resources
            self._check_resources()

            # 4. Create controller (this would integrate with your actual controller)
            logger.info("Setting up controller...")
            # controller = self._create_controller(config.controller_config, seed)

            # 5. Run training
            logger.info(f"Training for {config.num_training_episodes} episodes...")
            training_history = self._run_training(config, seed)
            result.training_history = training_history

            # 6. Run evaluation
            logger.info(f"Evaluating for {config.num_eval_episodes} episodes...")
            eval_metrics = self._run_evaluation(config, seed)
            result.final_eval_metrics = eval_metrics

            # 7. Run baseline comparisons
            if config.baselines_to_compare:
                logger.info(f"Comparing against {len(config.baselines_to_compare)} baselines...")
                baseline_results = self._run_baseline_comparisons(config, seed)
                result.baseline_comparisons = baseline_results

            # 8. Save artifacts
            artifacts = self._save_artifacts(config, seed, result)
            result.artifacts = artifacts

            # 9. Update result
            result.status = 'completed'
            result.training_time_seconds = time.time() - start_time
            result.peak_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024

            logger.info(f"Experiment completed in {result.training_time_seconds:.1f}s")

        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            result.status = 'failed'
            result.error_message = str(e)
            result.training_time_seconds = time.time() - start_time

        finally:
            logger.removeHandler(file_handler)
            file_handler.close()

        return result

    def _run_training(self, config: ExperimentConfig, seed: int) -> List[Tuple[int, Dict[str, float]]]:
        """Run training phase."""
        # This is a placeholder - integrate with your actual training code
        history = []

        # Simulate training
        for episode in range(config.num_training_episodes):
            if episode % config.log_interval == 0:
                metrics = {
                    'reward': np.random.randn() * 10 + 50,  # Placeholder
                    'loss': np.random.rand() * 0.1,
                    'epsilon': max(0.01, 1.0 - episode / config.num_training_episodes)
                }
                history.append((episode, metrics))

                if episode % 100 == 0:
                    logger.info(f"Episode {episode}: reward={metrics['reward']:.2f}")

        return history

    def _run_evaluation(self, config: ExperimentConfig, seed: int) -> Dict[str, float]:
        """Run evaluation phase."""
        # Placeholder - integrate with your actual evaluation code
        return {
            'mean_reward': np.random.randn() * 10 + 100,
            'std_reward': np.random.rand() * 5,
            'cache_hit_rate': np.random.rand() * 0.3 + 0.6,
            'cascade_rate': np.random.rand() * 0.1
        }

    def _run_baseline_comparisons(
        self,
        config: ExperimentConfig,
        seed: int
    ) -> Dict[str, Dict[str, float]]:
        """Run baseline comparisons."""
        # Placeholder - integrate with baselines module
        results = {}

        for baseline_name in config.baselines_to_compare:
            results[baseline_name] = {
                'mean_reward': np.random.randn() * 10 + 80,
                'cache_hit_rate': np.random.rand() * 0.3 + 0.5
            }

        return results

    def _save_artifacts(
        self,
        config: ExperimentConfig,
        seed: int,
        result: ExperimentResult
    ) -> Dict[str, str]:
        """Save experiment artifacts."""
        artifacts = {}
        artifact_dir = self.output_dir / 'artifacts' / f'{config.name}_{seed}'
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Save model (placeholder)
        model_path = artifact_dir / 'model.pt'
        artifacts['model'] = str(model_path)

        # Save training history plot (placeholder)
        plot_path = artifact_dir / 'training_history.png'
        artifacts['training_plot'] = str(plot_path)

        return artifacts

    def resume_from(self, checkpoint_dir: str):
        """Resume interrupted experiments from checkpoint."""
        checkpoint_path = Path(checkpoint_dir)

        # Load experiment queue
        queue_file = checkpoint_path / 'experiment_queue.json'
        if queue_file.exists():
            with open(queue_file, 'r') as f:
                queue_data = json.load(f)
                for exp_id, config_dict in queue_data.items():
                    if exp_id not in self.completed_experiments:
                        config = ExperimentConfig.from_dict(config_dict)
                        self.experiments[exp_id] = config

        logger.info(f"Resumed {len(self.experiments)} experiments from checkpoint")

    def get_results(self, experiment_id: Optional[str] = None) -> List[ExperimentResult]:
        """
        Get experiment results.

        Args:
            experiment_id: Optional specific experiment ID

        Returns:
            List of experiment results
        """
        if experiment_id:
            return self.results.get(experiment_id, [])
        else:
            all_results = []
            for results_list in self.results.values():
                all_results.extend(results_list)
            return all_results

    def aggregate_results(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """
        Aggregate results across seeds for statistical analysis.

        Args:
            experiment_ids: List of experiment IDs to aggregate

        Returns:
            Dictionary with aggregated statistics
        """
        all_rewards = []
        all_configs = []

        for exp_id in experiment_ids:
            results = self.results.get(exp_id, [])
            if not results:
                continue

            # Extract rewards
            rewards = [r.final_eval_metrics.get('mean_reward', 0) for r in results]
            all_rewards.append(np.mean(rewards))
            all_configs.append(results[0].config)

        if not all_rewards:
            return {}

        # Find best configuration
        best_idx = np.argmax(all_rewards)
        best_config = all_configs[best_idx]
        best_mean = all_rewards[best_idx]

        # Compute statistics for best config
        best_results = self.results.get(experiment_ids[best_idx], [])
        best_rewards = [r.final_eval_metrics.get('mean_reward', 0) for r in best_results]

        return {
            'best_config': best_config.name,
            'best_config_full': best_config,
            'best_mean_reward': np.mean(best_rewards),
            'best_std_reward': np.std(best_rewards),
            'best_exp_id': experiment_ids[best_idx],
            'all_mean_rewards': all_rewards,
            'num_configs': len(all_rewards)
        }

    def export_results(self, format: str = 'csv') -> str:
        """
        Export results to file.

        Args:
            format: Export format ('csv', 'json', 'latex')

        Returns:
            Path to exported file
        """
        # Collect all results
        data = []
        for exp_id, results_list in self.results.items():
            for result in results_list:
                row = {
                    'experiment_id': exp_id,
                    'name': result.config.name,
                    'seed': result.seed,
                    'status': result.status,
                    'training_time': result.training_time_seconds,
                    **result.final_eval_metrics
                }
                data.append(row)

        df = pd.DataFrame(data)

        # Export
        if format == 'csv':
            output_path = self.output_dir / 'results_export.csv'
            df.to_csv(output_path, index=False)

        elif format == 'json':
            output_path = self.output_dir / 'results_export.json'
            df.to_json(output_path, orient='records', indent=2)

        elif format == 'latex':
            output_path = self.output_dir / 'results_export.tex'
            with open(output_path, 'w') as f:
                f.write(df.to_latex(index=False))

        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Exported results to: {output_path}")
        return str(output_path)

    # === Reproducibility Methods ===

    def _set_all_seeds(self, seed: int):
        """Set all random seeds for reproducibility."""
        import random
        random.seed(seed)

        np.random.seed(seed)

        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

        # Set gymnasium seed (will be set when env is created)
        os.environ['PYTHONHASHSEED'] = str(seed)

        logger.debug(f"Set all seeds to {seed}")

    def _save_environment_info(self, path: Path):
        """Save environment information for reproducibility."""
        import subprocess

        env_info = {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'processor': platform.processor(),
            'timestamp': datetime.now().isoformat()
        }

        # Get package versions
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'freeze'],
                capture_output=True,
                text=True
            )
            env_info['packages'] = result.stdout.split('\n')
        except Exception:
            env_info['packages'] = []

        # GPU info
        try:
            import torch
            if torch.cuda.is_available():
                env_info['cuda_version'] = torch.version.cuda
                env_info['gpu_name'] = torch.cuda.get_device_name(0)
                env_info['gpu_count'] = torch.cuda.device_count()
        except ImportError:
            pass

        with open(path, 'w') as f:
            json.dump(env_info, f, indent=2)

    def _hash_config(self, config: ExperimentConfig) -> str:
        """Create hash of configuration for deduplication."""
        config_str = json.dumps(config.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    # === Resource Management ===

    def _estimate_time(self, config: ExperimentConfig) -> float:
        """Estimate experiment duration based on similar past runs."""
        # Find similar experiments
        similar_times = []

        for results_list in self.results.values():
            for result in results_list:
                if result.config.num_training_episodes == config.num_training_episodes:
                    similar_times.append(result.training_time_seconds)

        if similar_times:
            return np.median(similar_times)
        else:
            # Rough estimate: 0.1 seconds per episode
            return config.num_training_episodes * 0.1

    def _check_resources(self):
        """Check if sufficient resources are available."""
        # Check disk space
        disk_usage = psutil.disk_usage(str(self.output_dir))
        if disk_usage.percent > 90:
            logger.warning(f"Low disk space: {disk_usage.percent:.1f}% used")

        # Check memory
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            logger.warning(f"Low memory: {memory.percent:.1f}% used")

        # Check GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                    logger.debug(f"GPU {i}: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
        except ImportError:
            pass

    def _cleanup_old_experiments(self, keep_last: int = 10):
        """Remove old checkpoint files to save disk space."""
        checkpoint_dir = self.output_dir / 'checkpoints'
        if not checkpoint_dir.exists():
            return

        # Get all checkpoint directories
        checkpoints = sorted(checkpoint_dir.iterdir(), key=lambda p: p.stat().st_mtime)

        # Remove oldest ones
        for checkpoint in checkpoints[:-keep_last]:
            if checkpoint.is_dir():
                shutil.rmtree(checkpoint)
                logger.info(f"Removed old checkpoint: {checkpoint.name}")

    # === Helper Methods ===

    def _set_nested_value(self, d: Dict, key_path: str, value: Any):
        """Set value in nested dictionary using dot notation."""
        keys = key_path.split('.')
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    def _save_results(self):
        """Save all results to disk."""
        results_file = self.output_dir / 'all_results.json'

        # Convert to serializable format
        data = {}
        for exp_id, results_list in self.results.items():
            data[exp_id] = [r.to_dict() for r in results_list]

        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)

    def generate_summary_report(self) -> str:
        """Generate a summary report of all experiments."""
        report = []
        report.append("="*80)
        report.append("EXPERIMENT SUMMARY REPORT")
        report.append("="*80)
        report.append(f"Total experiments: {len(self.experiments)}")
        report.append(f"Completed experiments: {len(self.completed_experiments)}")
        report.append(f"Total runs: {sum(len(r) for r in self.results.values())}")
        report.append("")

        # Group by tags
        tag_groups = defaultdict(list)
        for exp_id, config in self.experiments.items():
            for tag in config.tags:
                tag_groups[tag].append(exp_id)

        report.append("Experiments by tag:")
        for tag, exp_ids in sorted(tag_groups.items()):
            report.append(f"  {tag}: {len(exp_ids)} experiments")

        report.append("")
        report.append("="*80)

        return "\n".join(report)


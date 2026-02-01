"""
Utility for loading experiment configurations from YAML files.

Provides functions to parse YAML experiment definitions and create
ExperimentConfig objects for the experiment runner.
"""

import yaml
from pathlib import Path
from typing import List, Dict, Any
import itertools
import copy

from .experiment_runner import ExperimentConfig


def load_yaml_experiments(yaml_path: str) -> List[ExperimentConfig]:
    """
    Load experiments from YAML file.

    Args:
        yaml_path: Path to YAML experiment definition

    Returns:
        List of ExperimentConfig objects
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Determine experiment type
    if 'hyperparameter_search' in str(yaml_path) or 'param_grid' in data:
        return _load_hyperparameter_search(data)
    elif 'ablation' in str(yaml_path) or 'components' in data:
        return _load_ablation_study(data)
    elif 'baseline_comparison' in str(yaml_path):
        return _load_baseline_comparison(data)
    elif 'scalability' in str(yaml_path):
        return _load_scalability_test(data)
    else:
        # Generic loading
        return _load_generic_experiments(data)


def _load_hyperparameter_search(data: Dict[str, Any]) -> List[ExperimentConfig]:
    """Load hyperparameter search experiments from YAML."""
    experiments = []

    base_config = data.get('experiment_set', {}).get('base_config', {})

    # Process each parameter search
    for key, value in data.items():
        if key == 'experiment_set':
            continue

        if isinstance(value, dict) and 'parameter' in value:
            # Single parameter search
            param_name = value['parameter']
            param_values = value['values']

            for param_value in param_values:
                config = _create_experiment_config(
                    name=f"{key}_{param_value}",
                    description=value.get('hypothesis', ''),
                    base_config=base_config,
                    param_overrides={param_name: param_value},
                    priority=value.get('priority', 0)
                )
                experiments.append(config)

        elif isinstance(value, dict) and 'parameters' in value:
            # Multi-parameter search
            param_names = []
            param_values_list = []

            for param_name, param_spec in value['parameters'].items():
                param_names.append(param_spec['path'])
                param_values_list.append(param_spec['values'])

            # Generate all combinations
            for combination in itertools.product(*param_values_list):
                overrides = dict(zip(param_names, combination))
                name_suffix = "_".join([f"{k.split('.')[-1]}={v}" for k, v in overrides.items()])

                config = _create_experiment_config(
                    name=f"{key}_{name_suffix}",
                    description=value.get('description', ''),
                    base_config=base_config,
                    param_overrides=overrides,
                    priority=value.get('priority', 0)
                )
                experiments.append(config)

    return experiments


def _load_ablation_study(data: Dict[str, Any]) -> List[ExperimentConfig]:
    """Load ablation study experiments from YAML."""
    experiments = []

    base_config = data.get('experiment_set', {}).get('base_config', {})
    components = data.get('components', {})
    ablation_tests = data.get('ablation_tests', {})

    for test_name, test_spec in ablation_tests.items():
        if test_name == 'full':
            # Full configuration with all components
            config = _create_experiment_config(
                name=test_spec['name'],
                description=test_spec['description'],
                base_config=base_config,
                param_overrides={},
                priority=test_spec.get('priority', 0),
                tags=test_spec.get('tags', [])
            )
            experiments.append(config)

        else:
            # Ablation with components disabled
            disabled_components = test_spec.get('disable', [])
            param_overrides = {}

            for component_name in disabled_components:
                if component_name in components:
                    component = components[component_name]
                    param_path = component['path']
                    disabled_value = component.get('disabled_value', False)
                    param_overrides[param_path] = disabled_value

            config = _create_experiment_config(
                name=test_spec['name'],
                description=test_spec['description'],
                base_config=base_config,
                param_overrides=param_overrides,
                priority=test_spec.get('priority', 0),
                tags=test_spec.get('tags', [])
            )
            experiments.append(config)

    return experiments


def _load_baseline_comparison(data: Dict[str, Any]) -> List[ExperimentConfig]:
    """Load baseline comparison experiments from YAML."""
    experiments = []

    base_config = data.get('experiment_set', {}).get('base_config', {})

    # RL agent configuration
    if 'rl_agent' in data:
        rl_spec = data['rl_agent']
        config = _create_experiment_config(
            name=rl_spec['name'],
            description=rl_spec['description'],
            base_config=base_config,
            controller_config=rl_spec.get('controller_config', {}),
            priority=rl_spec.get('priority', 10)
        )
        experiments.append(config)

    # Test on different environments
    if 'environments' in data:
        for env_name, env_spec in data['environments'].items():
            env_config = copy.deepcopy(base_config)
            env_config.update(env_spec.get('config', {}))

            config = _create_experiment_config(
                name=f"rl_agent_{env_name}",
                description=env_spec['description'],
                base_config=env_config,
                priority=env_spec.get('priority', 5)
            )
            experiments.append(config)

    return experiments


def _load_scalability_test(data: Dict[str, Any]) -> List[ExperimentConfig]:
    """Load scalability test experiments from YAML."""
    experiments = []

    base_config = data.get('experiment_set', {}).get('base_config', {})

    # Process each scalability dimension
    for dimension_name, dimension_spec in data.items():
        if dimension_name in ['experiment_set', 'metrics', 'analysis', 'deployment_scenarios', 'thesis_claims']:
            continue

        if isinstance(dimension_spec, dict) and 'configurations' in dimension_spec:
            for config_name, config_spec in dimension_spec['configurations'].items():
                # Extract configuration parameters
                param_overrides = {}
                for key, value in config_spec.items():
                    if key not in ['description', 'expected']:
                        param_overrides[key] = value

                config = _create_experiment_config(
                    name=f"{dimension_name}_{config_name}",
                    description=config_spec.get('description', ''),
                    base_config=base_config,
                    param_overrides=param_overrides,
                    priority=5
                )
                experiments.append(config)

    return experiments


def _load_generic_experiments(data: Dict[str, Any]) -> List[ExperimentConfig]:
    """Load generic experiment definitions."""
    experiments = []

    # Try to extract experiments from various formats
    if 'experiments' in data:
        for exp_data in data['experiments']:
            config = ExperimentConfig(
                name=exp_data['name'],
                description=exp_data.get('description', ''),
                hypothesis=exp_data.get('hypothesis', ''),
                controller_config=exp_data.get('controller_config', {}),
                num_training_episodes=exp_data.get('num_training_episodes', 1000),
                num_eval_episodes=exp_data.get('num_eval_episodes', 100),
                baselines_to_compare=exp_data.get('baselines_to_compare', []),
                seeds=exp_data.get('seeds', [42]),
                tags=exp_data.get('tags', []),
                priority=exp_data.get('priority', 0)
            )
            experiments.append(config)

    return experiments


def _create_experiment_config(
    name: str,
    description: str,
    base_config: Dict[str, Any],
    param_overrides: Dict[str, Any] = None,
    controller_config: Dict[str, Any] = None,
    priority: int = 0,
    tags: List[str] = None
) -> ExperimentConfig:
    """Helper to create ExperimentConfig from YAML data."""

    # Start with base config
    final_controller_config = copy.deepcopy(base_config.get('controller_config', {}))

    # Apply parameter overrides
    if param_overrides:
        for param_path, value in param_overrides.items():
            _set_nested_dict(final_controller_config, param_path, value)

    # Apply full controller config if provided
    if controller_config:
        final_controller_config.update(controller_config)

    return ExperimentConfig(
        name=name,
        description=description,
        hypothesis=base_config.get('hypothesis', ''),
        controller_config=final_controller_config,
        num_training_episodes=base_config.get('num_training_episodes', 1000),
        num_eval_episodes=base_config.get('num_eval_episodes', 100),
        baselines_to_compare=base_config.get('baselines_to_compare', []),
        seeds=base_config.get('seeds', [42]),
        tags=tags or base_config.get('tags', []),
        priority=priority
    )


def _set_nested_dict(d: Dict, key_path: str, value: Any):
    """Set value in nested dictionary using dot notation."""
    keys = key_path.split('.')
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def load_all_experiments(experiments_dir: str = 'evaluation/experiments') -> Dict[str, List[ExperimentConfig]]:
    """
    Load all experiment YAML files from directory.

    Args:
        experiments_dir: Directory containing YAML files

    Returns:
        Dictionary mapping experiment set name to list of configs
    """
    experiments_path = Path(experiments_dir)
    all_experiments = {}

    for yaml_file in experiments_path.glob('*.yaml'):
        exp_name = yaml_file.stem
        try:
            experiments = load_yaml_experiments(str(yaml_file))
            all_experiments[exp_name] = experiments
            print(f"Loaded {len(experiments)} experiments from {yaml_file.name}")
        except Exception as e:
            print(f"Failed to load {yaml_file.name}: {e}")

    return all_experiments


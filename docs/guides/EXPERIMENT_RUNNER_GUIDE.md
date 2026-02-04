# Experiment Runner - Complete Guide

## Overview

The Experiment Runner provides systematic evaluation of different configurations, hyperparameters, and approaches. Essential for thesis work requiring many experiments with statistical rigor.

## Features

✅ **Automated Experiment Execution**
- Queue-based experiment management
- Progress tracking and resumption
- Graceful failure handling
- Incremental result saving

✅ **Experiment Generation**
- Hyperparameter grid search
- Ablation studies
- Baseline comparisons
- Scalability tests

✅ **Reproducibility**
- Automatic seed management
- Environment info capture
- Configuration hashing
- Deduplication

✅ **Resource Management**
- GPU/CPU usage tracking
- Memory monitoring
- Disk space checks
- Time estimation

✅ **Results Analysis**
- Statistical aggregation
- Cross-seed analysis
- Export to CSV/JSON/LaTeX
- Visualization generation

## Quick Start

### 1. Basic Usage

```python
from evaluation import ExperimentRunner, ExperimentConfig

# Create runner
runner = ExperimentRunner(output_dir='results/experiments')

# Define experiment
config = ExperimentConfig(
    name='baseline_dqn',
    description='Baseline DQN with default hyperparameters',
    hypothesis='DQN should outperform LRU baseline',
    controller_config={
        'agent_config': {
            'learning_rate': 0.0001,
            'gamma': 0.99
        }
    },
    num_training_episodes=1000,
    num_eval_episodes=100,
    baselines_to_compare=['lru', 'lfu', 'static_markov'],
    seeds=[42, 123, 456],
    tags=['baseline']
)

# Add to queue
exp_id = runner.add_experiment(config)

# Run all experiments
runner.run_all()

# Get results
results = runner.get_results()
print(f"Completed {len(results)} runs")
```

### 2. Hyperparameter Sweep

```python
# Base configuration
base_config = ExperimentConfig(
    name='hyperparam_sweep',
    description='Find optimal hyperparameters',
    hypothesis='Testing learning rate and gamma',
    controller_config={'agent_config': {}},
    num_training_episodes=1000,
    num_eval_episodes=100,
    seeds=[42, 123, 456]
)

# Define parameter grid
param_grid = {
    'agent_config.learning_rate': [0.001, 0.0001, 0.00001],
    'agent_config.gamma': [0.95, 0.99, 0.995]
}

# Generate experiments (3 x 3 = 9 configurations)
sweep_ids = runner.add_hyperparameter_sweep(base_config, param_grid)
print(f"Generated {len(sweep_ids)} experiments")

# Run sweep
runner.run_all()

# Find best configuration
aggregated = runner.aggregate_results(sweep_ids)
print(f"Best config: {aggregated['best_config']}")
print(f"Best reward: {aggregated['best_mean_reward']:.2f} ± {aggregated['best_std_reward']:.2f}")
```

### 3. Ablation Study

```python
# Full configuration
full_config = ExperimentConfig(
    name='ablation',
    description='Test component contributions',
    hypothesis='Each component contributes to performance',
    controller_config={
        'markov_config': {
            'context_aware': True,
            'order': 2
        },
        'agent_config': {
            'dueling': True,
            'double_dqn': True,
            'prioritized_replay': True
        }
    },
    num_training_episodes=1000,
    num_eval_episodes=100,
    seeds=[42, 123, 456]
)

# Define components to test
components = {
    'markov_config.context_aware': True,
    'markov_config.order': 2,
    'agent_config.dueling': True,
    'agent_config.double_dqn': True,
    'agent_config.prioritized_replay': True
}

# Generate ablation experiments
ablation_ids = runner.add_ablation_study(full_config, components)

# This creates:
# 1. Full configuration (all components)
# 2. Without context_aware
# 3. Without second-order (uses first-order)
# 4. Without dueling
# 5. Without double DQN
# 6. Without prioritized replay
```

### 4. Load from YAML

```python
from evaluation.yaml_loader import load_yaml_experiments, load_all_experiments

# Load single experiment file
experiments = load_yaml_experiments('evaluation/experiments/hyperparameter_search.yaml')
for exp in experiments:
    runner.add_experiment(exp)

# Load all YAML files
all_experiments = load_all_experiments('evaluation/experiments')
for exp_set_name, exp_list in all_experiments.items():
    print(f"Loaded {len(exp_list)} experiments from {exp_set_name}")
    for exp in exp_list:
        runner.add_experiment(exp)
```

## YAML Experiment Definitions

### Hyperparameter Search

File: `evaluation/experiments/hyperparameter_search.yaml`

```yaml
experiment_set:
  name: "hyperparameter_search"
  base_config:
    num_training_episodes: 1000
    num_eval_episodes: 100
    seeds: [42, 123, 456]

learning_rate_search:
  parameter: "agent_config.learning_rate"
  values: [0.001, 0.0001, 0.00001]
  hypothesis: "Optimal learning rate around 0.0001"
  priority: 10
```

### Ablation Study

File: `evaluation/experiments/ablation_study.yaml`

```yaml
components:
  markov_context_aware:
    path: "markov_config.context_aware"
    default: true
    description: "Context-aware predictions"
    priority: 10
  
  dueling_architecture:
    path: "agent_config.dueling"
    default: true
    description: "Dueling DQN architecture"
    priority: 8
```

### Baseline Comparison

File: `evaluation/experiments/baseline_comparison.yaml`

```yaml
baselines:
  lru:
    name: "baseline_lru"
    expected_improvement: "20-40%"
  
  static_markov:
    name: "baseline_static_markov"
    expected_improvement: "10-30%"
```

### Scalability Test

File: `evaluation/experiments/scalability_test.yaml`

```yaml
api_scale:
  configurations:
    small:
      num_apis: 10
    large:
      num_apis: 50
    massive:
      num_apis: 200
```

## Progress Tracking

```python
def progress_callback(exp_id, status, progress):
    """Called during experiment execution."""
    print(f"{exp_id}: {status} - {progress*100:.0f}% complete")

runner.run_all(progress_callback=progress_callback)
```

## Results Analysis

### Get Results

```python
# All results
all_results = runner.get_results()

# Specific experiment
exp_results = runner.get_results(experiment_id='baseline_dqn_abc12345')

# Iterate through results
for result in all_results:
    print(f"{result.config.name} (seed {result.seed}): {result.final_eval_metrics['mean_reward']:.2f}")
```

### Aggregate Across Seeds

```python
# Aggregate specific experiments
aggregated = runner.aggregate_results(['exp1', 'exp2', 'exp3'])

print(f"Best configuration: {aggregated['best_config']}")
print(f"Mean reward: {aggregated['best_mean_reward']:.2f}")
print(f"Std reward: {aggregated['best_std_reward']:.2f}")
```

### Export Results

```python
# Export to CSV
csv_path = runner.export_results(format='csv')

# Export to JSON
json_path = runner.export_results(format='json')

# Export to LaTeX table
latex_path = runner.export_results(format='latex')
```

## Resume from Checkpoint

```python
# Resume interrupted experiments
runner.resume_from('results/experiments/checkpoints/2026-02-01_12-00')

# Run remaining experiments
runner.run_all()
```

## Resource Management

### Check Available Resources

```python
# Automatic before each experiment
# Checks: GPU memory, disk space, RAM
# Logs warnings if resources low
```

### Cleanup Old Checkpoints

```python
# Keep only last 10 checkpoints
runner._cleanup_old_experiments(keep_last=10)
```

### Time Estimation

```python
# Estimate time for new experiment
estimated_time = runner._estimate_time(config)
print(f"Estimated time: {estimated_time/60:.1f} minutes")
```

## Reproducibility

### Automatic Seed Management

- Sets Python random, NumPy, PyTorch, Gymnasium seeds
- Saves environment information
- Hashes configurations for deduplication

### Environment Info Saved

- Python version
- Package versions (pip freeze)
- CUDA version
- GPU information
- System platform
- Timestamp

## Directory Structure

```
results/experiments/
├── configs/
│   ├── baseline_dqn_abc12345.json
│   └── hyperparam_sweep_def67890.json
├── results/
│   └── all_results.json
├── artifacts/
│   ├── baseline_dqn_42/
│   │   ├── model.pt
│   │   ├── training_history.png
│   │   └── env_info.json
│   └── hyperparam_sweep_123/
│       └── ...
├── logs/
│   ├── baseline_dqn_42.log
│   └── hyperparam_sweep_123.log
├── checkpoints/
│   └── 2026-02-01_12-00/
└── results_export.csv
```

## Advanced Usage

### Custom Progress Tracking

```python
class ExperimentTracker:
    def __init__(self):
        self.completed = 0
        self.total = 0
    
    def callback(self, exp_id, status, progress):
        if status == 'completed':
            self.completed += 1
        
        print(f"Progress: {self.completed}/{self.total} experiments")

tracker = ExperimentTracker()
runner.run_all(progress_callback=tracker.callback)
```

### Parallel Execution

```python
# Run up to 4 experiments in parallel (multi-GPU)
runner = ExperimentRunner(
    output_dir='results/experiments',
    max_parallel=4
)
```

### Priority Scheduling

```python
# Higher priority experiments run first
config_high = ExperimentConfig(
    name='critical_experiment',
    priority=10,  # High priority
    ...
)

config_low = ExperimentConfig(
    name='optional_experiment',
    priority=1,  # Low priority
    ...
)
```

## Thesis Integration

### Required Experiments

1. **Hyperparameter Search**
   - Find optimal configuration
   - Report best hyperparameters

2. **Ablation Study**
   - Demonstrate component contributions
   - Show each component's value

3. **Baseline Comparison**
   - Compare against all baselines
   - Show statistical significance

4. **Scalability Test**
   - Test across different scales
   - Demonstrate practical viability

### Generating Thesis Results

```python
# 1. Run all experiments
runner.run_all()

# 2. Export results
csv_path = runner.export_results('csv')
latex_path = runner.export_results('latex')

# 3. Generate summary report
report = runner.generate_summary_report()
with open('thesis/evaluation/summary.txt', 'w') as f:
    f.write(report)

# 4. Copy plots to thesis directory
import shutil
shutil.copy('results/experiments/artifacts/*/training_history.png', 
            'thesis/figures/')
```

## Troubleshooting

### Experiments Fail

- Check logs in `results/experiments/logs/`
- Results still saved (status='failed')
- Can resume with `runner.resume_from()`

### Out of Memory

- Reduce `max_parallel`
- Decrease `batch_size` in config
- Monitor with `runner._check_resources()`

### Disk Space

- Use `runner._cleanup_old_experiments()`
- Remove old artifacts manually
- Export results and delete raw data

## Best Practices

1. **Start Small**: Test with few episodes first
2. **Use Multiple Seeds**: At least 3 for significance
3. **Save Incrementally**: Results saved after each run
4. **Monitor Progress**: Use progress callbacks
5. **Document Hypotheses**: Write clear experiment descriptions
6. **Tag Experiments**: Use tags for organization
7. **Version Control**: Commit YAML files
8. **Backup Results**: Copy important results elsewhere

## Example: Complete Thesis Evaluation

```python
from evaluation import ExperimentRunner
from evaluation.yaml_loader import load_all_experiments

# 1. Create runner
runner = ExperimentRunner(output_dir='results/thesis_evaluation')

# 2. Load all experiment definitions
all_experiments = load_all_experiments('evaluation/experiments')

# 3. Add to queue
for exp_set, experiments in all_experiments.items():
    print(f"Loading {exp_set}...")
    for exp in experiments:
        runner.add_experiment(exp)

print(f"Total experiments queued: {len(runner.experiments)}")

# 4. Run all (this may take hours/days)
def progress(exp_id, status, prog):
    print(f"[{prog*100:.0f}%] {exp_id}: {status}")

runner.run_all(progress_callback=progress)

# 5. Export results
runner.export_results('csv')
runner.export_results('latex')

# 6. Generate summary
report = runner.generate_summary_report()
print(report)

print("✓ Thesis evaluation complete!")
```

---

## API Reference

See `evaluation/experiment_runner.py` for complete API documentation.

**Key Classes:**
- `ExperimentConfig`: Experiment definition
- `ExperimentResult`: Experiment outcome
- `ExperimentRunner`: Main runner class

**Key Methods:**
- `add_experiment()`: Add single experiment
- `add_hyperparameter_sweep()`: Generate sweep
- `add_ablation_study()`: Generate ablation
- `run_all()`: Execute all experiments
- `get_results()`: Retrieve results
- `aggregate_results()`: Statistical aggregation
- `export_results()`: Export to file

---

**Ready to run systematic experiments for your thesis!** 🎓


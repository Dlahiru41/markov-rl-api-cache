# Experiment Runner - Quick Reference Card

## 🚀 Quick Start (30 seconds)

```python
from evaluation import ExperimentRunner, ExperimentConfig

runner = ExperimentRunner('results/experiments')
config = ExperimentConfig(
    name='test', description='Test run', hypothesis='Testing',
    controller_config={}, num_training_episodes=100, num_eval_episodes=20
)
exp_id = runner.add_experiment(config)
runner.run_all()
```

## 📋 Common Operations

### Add Single Experiment
```python
config = ExperimentConfig(name='exp1', ...)
runner.add_experiment(config)
```

### Hyperparameter Sweep
```python
param_grid = {
    'agent_config.learning_rate': [0.001, 0.0001],
    'agent_config.gamma': [0.95, 0.99]
}
runner.add_hyperparameter_sweep(base_config, param_grid)
```

### Ablation Study
```python
components = {
    'markov_config.context_aware': True,
    'agent_config.dueling': True
}
runner.add_ablation_study(base_config, components)
```

### Load from YAML
```python
from evaluation.yaml_loader import load_all_experiments
experiments = load_all_experiments('evaluation/experiments')
```

### Run All
```python
runner.run_all(progress_callback=lambda id, s, p: print(f"{id}: {s}"))
```

### Get Results
```python
results = runner.get_results()
results = runner.get_results(experiment_id='exp123')
aggregated = runner.aggregate_results(['exp1', 'exp2'])
```

### Export
```python
csv_path = runner.export_results('csv')
latex_path = runner.export_results('latex')
json_path = runner.export_results('json')
```

## 📁 YAML Files

### hyperparameter_search.yaml
- Learning rate: 5 values
- Gamma: 4 values
- Network size: 4 configs
- Batch size: 4 values
- Combined searches

### ablation_study.yaml
- 10+ components
- Full vs minimal
- Component-by-component tests

### baseline_comparison.yaml
- RL vs 6 baselines
- 4 environment configs
- Statistical tests

### scalability_test.yaml
- 6 scaling dimensions
- 30+ configurations
- Deployment scenarios

## 🔧 Configuration

### ExperimentConfig Fields
- `name`: Unique ID
- `description`: What you're testing
- `hypothesis`: Expected result
- `controller_config`: Full config dict
- `num_training_episodes`: Training length
- `num_eval_episodes`: Evaluation length
- `baselines_to_compare`: List of baselines
- `seeds`: List of random seeds
- `tags`: Organization tags
- `priority`: Execution order (higher first)

### ExperimentResult Fields
- `config`: Original config
- `seed`: Random seed used
- `training_history`: [(episode, metrics)]
- `final_eval_metrics`: Dict of metrics
- `baseline_comparisons`: Dict per baseline
- `training_time_seconds`: Duration
- `peak_memory_mb`: Max memory
- `artifacts`: Dict of file paths
- `status`: completed/failed/interrupted
- `error_message`: If failed
- `timestamp`: When finished

## 📊 Analysis Functions

```python
# Aggregate across seeds
agg = runner.aggregate_results(['exp1', 'exp2', 'exp3'])
print(f"Best: {agg['best_config']}")
print(f"Reward: {agg['best_mean_reward']:.2f} ± {agg['best_std_reward']:.2f}")

# Generate summary
report = runner.generate_summary_report()
print(report)

# Export for thesis
csv = runner.export_results('csv')
latex = runner.export_results('latex')
```

## 🎯 Typical Workflow

1. **Define Experiments** (YAML or Python)
2. **Add to Queue** (`add_experiment()`)
3. **Run All** (`run_all()`)
4. **Analyze Results** (`get_results()`, `aggregate_results()`)
5. **Export for Thesis** (`export_results()`)

## ⚡ One-Liners

```bash
# Run demo
python demo_experiment_runner.py

# Load and run all YAML experiments
python -c "from evaluation import *; from evaluation.yaml_loader import *; r=ExperimentRunner('results/thesis'); [r.add_experiment(e) for es in load_all_experiments().values() for e in es]; r.run_all()"
```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Import error | Check `sys.path`, run from project root |
| Out of memory | Reduce `batch_size`, `max_parallel` |
| Disk full | Run `_cleanup_old_experiments()` |
| Failed experiment | Check logs in `results/experiments/logs/` |
| Resume needed | Use `runner.resume_from(checkpoint_dir)` |

## 📈 Expected Performance

| Experiment Type | # Configs | Time Estimate |
|-----------------|-----------|---------------|
| Single | 1 | 30-60 min |
| Small sweep | 3-10 | 2-5 hours |
| Large sweep | 20-50 | 10-24 hours |
| Ablation | 5-15 | 3-10 hours |
| Full baseline | 6+ | 6-12 hours |
| Scalability | 30+ | 12-48 hours |

## 🎓 Thesis Checklist

- [ ] Define all experiments in YAML
- [ ] Run hyperparameter search
- [ ] Run ablation study
- [ ] Run baseline comparison
- [ ] Run scalability tests
- [ ] Export results to CSV/LaTeX
- [ ] Generate summary report
- [ ] Copy plots to thesis/figures/
- [ ] Write methodology section
- [ ] Write results section
- [ ] Include statistical tests
- [ ] Document reproducibility

## 💡 Tips

✓ Start with small `num_training_episodes` to test  
✓ Use at least 3 seeds for statistical significance  
✓ Tag experiments for organization  
✓ Use progress callbacks to monitor  
✓ Save important results elsewhere (backup)  
✓ Review logs if experiments fail  
✓ Clean up old checkpoints periodically  
✓ Version control your YAML files  

## 📞 Help

- Full guide: `EXPERIMENT_RUNNER_GUIDE.md`
- Implementation: `evaluation/experiment_runner.py`
- YAML examples: `evaluation/experiments/*.yaml`
- Demo: `demo_experiment_runner.py`

---

**Ready to run systematic experiments!** 🚀


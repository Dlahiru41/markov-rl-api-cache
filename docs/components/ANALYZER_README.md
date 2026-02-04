# Results Analysis and Visualization Module

Comprehensive results analysis and visualization module for generating thesis-quality figures and statistical analysis.

## 📋 Overview

This module provides powerful tools for analyzing experiment results, performing rigorous statistical tests, and creating publication-quality visualizations for academic thesis work.

## 🎯 Features

### ResultsAnalyzer
- **Data Loading**: Load and parse experiment results from directory
- **Statistical Tests**: t-test, Mann-Whitney U, ANOVA, Kruskal-Wallis, Tukey HSD
- **Effect Sizes**: Cohen's d calculation
- **Confidence Intervals**: Parametric and bootstrap methods
- **Normality Testing**: Shapiro-Wilk test
- **Learning Curve Analysis**: Convergence detection, stability metrics
- **Best Configuration Identification**: Automatic ranking with significance testing

### ResultsVisualizer
- **Learning Curves**: With confidence intervals and smoothing
- **Bar Charts**: With significance stars and error bars
- **Box Plots**: Distribution comparison
- **Radar Charts**: Multi-metric comparison
- **Heatmaps**: Hyperparameter sensitivity analysis
- **Action Distribution**: Policy behavior analysis
- **Publication-Quality Output**: High DPI, proper fonts, LaTeX-compatible

### ReportGenerator
- **Single Experiment Reports**: Markdown, HTML, LaTeX
- **Comparison Reports**: Multi-experiment statistical analysis
- **Thesis Chapter Generation**: Complete LaTeX chapter with figures
- **Automated Figure Integration**: All plots with consistent styling

### CLI Tool (scripts/analyze.py)
- `summarize`: Overview of all experiments
- `compare`: Statistical comparison between methods
- `plot`: Generate specific visualizations
- `report`: Create comprehensive reports
- `thesis`: Generate all thesis materials

## 📦 Installation

### Required Packages

You need to install the following additional packages:

```bash
pip install jinja2 statsmodels
```

These packages are **required** for full functionality:
- **jinja2**: For HTML report generation (template engine)
- **statsmodels**: For advanced statistical tests (Tukey HSD, post-hoc comparisons)

The following packages are already in your `requirements.txt`:
- numpy, pandas, scipy (statistical analysis)
- matplotlib, seaborn (visualization)

## 🚀 Quick Start

### 1. Basic Usage

```python
from evaluation.analyzer import ResultsAnalyzer, ResultsVisualizer
from evaluation.report_generator import ReportGenerator

# Load results
analyzer = ResultsAnalyzer('results/experiments')

# Get experiment results
dqn_results = analyzer.load_experiment('dqn_baseline')
lru_results = analyzer.load_experiment('lru_baseline')

# Statistical comparison
dqn_rewards = [r.final_eval_metrics['mean_reward'] for r in dqn_results]
lru_rewards = [r.final_eval_metrics['mean_reward'] for r in lru_results]

comparison = analyzer.compare_two_methods(dqn_rewards, lru_rewards)
print(f"p-value: {comparison['t_test_pvalue']:.4f}")
print(f"Effect size: {comparison['cohens_d']:.2f}")
print(f"Significant: {comparison['significant']}")
```

### 2. Generate Visualizations

```python
visualizer = ResultsVisualizer()

# Learning curves
visualizer.plot_learning_curves(
    analyzer,
    ['dqn_baseline', 'dqn_tuned'],
    metric='reward',
    output_path='figures/learning_curves.pdf'
)

# Bar comparison with significance stars
visualizer.plot_bar_comparison(
    analyzer,
    ['dqn', 'lru', 'lfu', 'random'],
    metric='cache_hit_rate',
    output_path='figures/comparison.pdf'
)
```

### 3. Generate Reports

```python
generator = ReportGenerator(analyzer, visualizer)

# Comparison report
generator.generate_comparison_report(
    ['dqn_baseline', 'lru_baseline', 'lfu_baseline'],
    output_path='reports/comparison.html',
    format='html',
    metrics=['mean_reward', 'cache_hit_rate']
)

# Thesis chapter
generator.generate_thesis_chapter(
    'results/experiments',
    output_path='thesis/chapters/results.tex'
)
```

### 4. Command-Line Interface

```bash
# Summarize all experiments
python scripts/analyze.py summarize --results-dir results/experiments

# Statistical comparison
python scripts/analyze.py compare \
  --experiments dqn_baseline,lru_baseline \
  --metric mean_reward

# Generate learning curves
python scripts/analyze.py plot learning_curves \
  --experiments dqn_baseline,dqn_tuned \
  --output figures/ \
  --format pdf

# Generate all thesis materials
python scripts/analyze.py thesis \
  --results-dir results/experiments \
  --output thesis/generated/
```

## 📊 Examples

### Example 1: Statistical Comparison

```python
# Load experiments
analyzer = ResultsAnalyzer('results/experiments')

# Compare DQN vs LRU
dqn_results = analyzer.load_experiment('dqn_baseline')
lru_results = analyzer.load_experiment('lru_baseline')

dqn_rewards = [r.final_eval_metrics['mean_reward'] for r in dqn_results]
lru_rewards = [r.final_eval_metrics['mean_reward'] for r in lru_results]

comparison = analyzer.compare_two_methods(dqn_rewards, lru_rewards)

print(f"DQN: {comparison['mean_a']:.2f} ± {comparison['std_a']:.2f}")
print(f"LRU: {comparison['mean_b']:.2f} ± {comparison['std_b']:.2f}")
print(f"Improvement: {(comparison['mean_a'] - comparison['mean_b']) / comparison['mean_b'] * 100:.1f}%")
print(f"Significant: {comparison['significant']} (p={comparison['t_test_pvalue']:.4f})")
```

### Example 2: Multi-Method Comparison

```python
# Compare multiple methods
all_results = {
    'DQN': [r.final_eval_metrics['mean_reward'] for r in analyzer.load_experiment('dqn')],
    'LRU': [r.final_eval_metrics['mean_reward'] for r in analyzer.load_experiment('lru')],
    'LFU': [r.final_eval_metrics['mean_reward'] for r in analyzer.load_experiment('lfu')],
    'Random': [r.final_eval_metrics['mean_reward'] for r in analyzer.load_experiment('random')]
}

comparison = analyzer.compare_multiple_methods(all_results)
print(f"ANOVA p-value: {comparison['anova_pvalue']:.4f}")
print("\nRankings:")
for rank, (name, mean) in enumerate(comparison['rankings'], 1):
    print(f"  {rank}. {name}: {mean:.2f}")
```

### Example 3: Learning Curve Analysis

```python
# Analyze training dynamics
analysis = analyzer.analyze_learning_curve('dqn_baseline')

print(f"Total Episodes: {analysis['total_episodes']}")
print(f"Convergence Episode: {analysis['convergence_episode']}")
print(f"Final Performance: {analysis['final_performance']:.2f} ± {analysis['final_std']:.2f}")
print(f"Trend: {analysis['trend']}")
```

### Example 4: Generate All Thesis Figures

```python
visualizer = ResultsVisualizer()

# Generate all figures needed for thesis
figures = visualizer.generate_all_thesis_figures(
    analyzer,
    ['dqn_baseline', 'lru_baseline', 'lfu_baseline'],
    output_dir='thesis/figures',
    metrics=['mean_reward', 'cache_hit_rate', 'latency']
)

print(f"Generated {len(figures)} figures:")
for name, path in figures.items():
    print(f"  {name}: {path}")
```

## 📁 File Structure

```
evaluation/
├── __init__.py              # Module exports
├── analyzer.py              # ResultsAnalyzer and ResultsVisualizer classes
├── report_generator.py      # ReportGenerator class
└── experiment_runner.py     # (Existing) ExperimentRunner

scripts/
└── analyze.py              # Command-line interface

validate_analyzer.py        # Comprehensive validation tests
demo_analyzer.py           # Usage demonstrations
```

## 🧪 Testing

### Run Validation Tests

```bash
python validate_analyzer.py
```

This will:
1. Create mock experiment data
2. Test all analyzer functionality
3. Test all visualizer functionality
4. Test report generation
5. Run integration tests

### Run Demonstrations

```bash
python demo_analyzer.py
```

This will demonstrate:
1. Basic analysis workflow
2. Statistical comparisons
3. Multi-method comparison
4. Learning curve analysis
5. Visualization generation
6. Report generation
7. Complete thesis workflow

## 📖 API Reference

### ResultsAnalyzer

#### Initialization
```python
analyzer = ResultsAnalyzer(results_dir: str)
```

#### Methods
- `load_experiment(experiment_id) -> List[ExperimentResult]`
- `load_all_experiments() -> Dict[str, List[ExperimentResult]]`
- `filter_by_tags(tags) -> Dict[str, List[ExperimentResult]]`
- `compare_two_methods(method_a, method_b, metric) -> Dict`
- `compare_multiple_methods(results_dict, metric) -> Dict`
- `compute_confidence_intervals(values, confidence=0.95) -> Tuple[float, float]`
- `test_normality(values) -> Dict`
- `summarize_experiment(experiment_id) -> Dict`
- `create_summary_table(experiment_ids, metrics) -> pd.DataFrame`
- `identify_best_configuration(experiment_ids, metric) -> Dict`
- `analyze_learning_curve(experiment_id) -> Dict`
- `compare_learning_speeds(experiment_ids) -> Dict`

### ResultsVisualizer

#### Initialization
```python
visualizer = ResultsVisualizer(style='seaborn-v0_8-paper', dpi=300)
```

#### Methods
- `plot_learning_curves(analyzer, experiment_ids, metric, output_path)`
- `plot_training_comparison(analyzer, experiment_ids, metrics, output_path)`
- `plot_bar_comparison(analyzer, experiment_ids, metric, output_path)`
- `plot_box_comparison(analyzer, experiment_ids, metric, output_path)`
- `plot_radar_chart(analyzer, experiment_ids, metrics, output_path)`
- `plot_heatmap(results_df, x_param, y_param, metric, output_path)`
- `plot_action_distribution(analyzer, experiment_id, output_path)`
- `generate_all_thesis_figures(analyzer, experiment_ids, output_dir) -> Dict[str, str]`

### ReportGenerator

#### Initialization
```python
generator = ReportGenerator(analyzer, visualizer)
```

#### Methods
- `generate_experiment_report(experiment_id, output_path, format='markdown')`
- `generate_comparison_report(experiment_ids, output_path, format='html')`
- `generate_thesis_chapter(results_dir, output_path)`

## 🎨 Plot Styling

All plots are configured for publication quality:
- **DPI**: 300 (suitable for printing)
- **Fonts**: Times New Roman, proper sizing for readability
- **Colors**: Colorblind-friendly palette
- **Format**: PDF/PNG/SVG/EPS support
- **LaTeX Compatible**: Font types suitable for LaTeX inclusion

## 📊 Statistical Tests

### Two-Sample Comparison
- **t-test**: Parametric comparison of means
- **Mann-Whitney U**: Non-parametric alternative
- **Cohen's d**: Effect size measure
- **95% CI**: Confidence intervals

### Multi-Sample Comparison
- **ANOVA**: Parametric comparison across groups
- **Kruskal-Wallis**: Non-parametric alternative
- **Tukey HSD**: Post-hoc pairwise comparisons

### Normality Testing
- **Shapiro-Wilk**: Test for normal distribution
- Helps choose appropriate statistical test

## 🎓 Thesis Integration

The module is designed specifically for thesis work:

1. **Experimental Setup Section**: Automatically generated descriptions
2. **Results Tables**: LaTeX-formatted tables with proper formatting
3. **Figures**: Consistent high-quality plots with proper captions
4. **Statistical Analysis**: Rigorous hypothesis testing
5. **Comparison Section**: Rankings with significance tests
6. **Discussion Points**: Automated interpretation strings

## 🔧 Troubleshooting

### Missing Packages

If you see import errors:
```bash
pip install jinja2 statsmodels
```

### No Experiment Data

The analyzer expects results in the format created by `ExperimentRunner`. If you have custom result format, you may need to adapt the loading logic.

### Plot Not Showing

Set `output_path=None` to display plots instead of saving them.

### StatModels Not Available

The module will work without statsmodels, but Tukey HSD post-hoc tests will not be available. Basic ANOVA will still work.

## 📝 Citation

If you use this module in your thesis, you can cite the statistical methods:

- **t-test**: Student, 1908
- **Mann-Whitney U**: Mann & Whitney, 1947
- **Cohen's d**: Cohen, 1988
- **ANOVA**: Fisher, 1925
- **Tukey HSD**: Tukey, 1949

## 🤝 Contributing

This module is part of the Markov-RL API Cache project. For issues or improvements, please update the evaluation module.

## 📄 License

Part of the Markov-RL API Cache thesis project.


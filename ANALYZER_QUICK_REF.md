# Results Analysis Module - Quick Reference Card

## 🚀 Installation
```bash
pip install jinja2 statsmodels
```

## 📊 Basic Usage

### 1. Load and Analyze Results
```python
from evaluation.analyzer import ResultsAnalyzer, ResultsVisualizer
from evaluation.report_generator import ReportGenerator

# Initialize
analyzer = ResultsAnalyzer('results/experiments')
visualizer = ResultsVisualizer()
generator = ReportGenerator(analyzer, visualizer)
```

### 2. Statistical Comparison
```python
# Load experiments
dqn = analyzer.load_experiment('dqn_baseline')
lru = analyzer.load_experiment('lru_baseline')

# Extract metric
dqn_rewards = [r.final_eval_metrics['mean_reward'] for r in dqn]
lru_rewards = [r.final_eval_metrics['mean_reward'] for r in lru]

# Compare
comparison = analyzer.compare_two_methods(dqn_rewards, lru_rewards)
print(f"p-value: {comparison['t_test_pvalue']:.4f}")
print(f"Significant: {comparison['significant']}")
```

### 3. Generate Thesis Figures
```python
# Generate all figures at once
figures = visualizer.generate_all_thesis_figures(
    analyzer,
    ['dqn', 'lru', 'lfu'],
    output_dir='thesis/figures',
    metrics=['mean_reward', 'cache_hit_rate']
)
```

### 4. Create Reports
```python
# HTML comparison report
generator.generate_comparison_report(
    ['dqn', 'lru', 'lfu'],
    'reports/comparison.html',
    format='html'
)

# LaTeX thesis chapter
generator.generate_thesis_chapter(
    'results/experiments',
    'thesis/chapters/results.tex'
)
```

## 💻 CLI Commands

### Summarize Experiments
```bash
python scripts/analyze.py summarize --results-dir results/experiments
```

### Statistical Comparison
```bash
python scripts/analyze.py compare \
  --experiments dqn_baseline,lru_baseline \
  --metric mean_reward
```

### Generate Plots
```bash
# Learning curves
python scripts/analyze.py plot learning_curves \
  --experiments dqn_baseline,dqn_tuned \
  --output figures/ \
  --format pdf

# Bar comparison
python scripts/analyze.py plot bar_comparison \
  --experiments dqn,lru,lfu \
  --metric cache_hit_rate \
  --output figures/
```

### Generate Reports
```bash
python scripts/analyze.py report \
  --experiments dqn,lru,lfu \
  --output reports/ \
  --format html
```

### Generate All Thesis Materials
```bash
python scripts/analyze.py thesis \
  --results-dir results/experiments \
  --output thesis/generated/
```

## 📈 Key Methods

### ResultsAnalyzer
| Method | Purpose |
|--------|---------|
| `load_experiment(id)` | Load one experiment |
| `compare_two_methods(a, b)` | Statistical comparison |
| `compare_multiple_methods(dict)` | ANOVA comparison |
| `summarize_experiment(id)` | Get summary stats |
| `identify_best_configuration(ids)` | Find best method |
| `analyze_learning_curve(id)` | Learning dynamics |

### ResultsVisualizer
| Method | Purpose |
|--------|---------|
| `plot_learning_curves()` | Training progress |
| `plot_bar_comparison()` | Performance bars |
| `plot_box_comparison()` | Distribution boxes |
| `plot_radar_chart()` | Multi-metric spider |
| `plot_heatmap()` | Hyperparameter heat |
| `generate_all_thesis_figures()` | All at once |

### ReportGenerator
| Method | Purpose |
|--------|---------|
| `generate_experiment_report()` | Single experiment |
| `generate_comparison_report()` | Multi-experiment |
| `generate_thesis_chapter()` | LaTeX chapter |

## 📊 Statistical Tests Available

| Test | Type | Use Case |
|------|------|----------|
| t-test | Parametric | Compare 2 means |
| Mann-Whitney U | Non-parametric | Compare 2 distributions |
| ANOVA | Parametric | Compare 3+ means |
| Kruskal-Wallis | Non-parametric | Compare 3+ distributions |
| Tukey HSD | Post-hoc | Pairwise after ANOVA |
| Cohen's d | Effect size | Magnitude of difference |
| Shapiro-Wilk | Normality | Check distribution |

## 🎨 Plot Formats

All plots support:
- **Formats**: PDF, PNG, SVG, EPS
- **DPI**: 300 (default, adjustable)
- **Fonts**: Times New Roman (thesis standard)
- **Size**: Optimized for printing

## 📁 File Structure

```
evaluation/
├── analyzer.py              # Analysis & visualization
├── report_generator.py      # Report creation
└── experiment_runner.py     # Experiment management

scripts/
└── analyze.py              # CLI tool

ANALYZER_README.md          # Full documentation
VALIDATION_RESULTS.md       # Test results
```

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | `pip install jinja2 statsmodels` |
| No data found | Check `results_dir` path |
| Plot not showing | Add `output_path` parameter |
| Font warnings | Normal, plots still work |

## 📞 Getting Help

1. Check `ANALYZER_README.md` for detailed docs
2. Run `python validate_analyzer.py` to test
3. Run `python demo_analyzer.py` for examples
4. Check docstrings: `help(ResultsAnalyzer)`

## ✅ Validation Status

**ALL TESTS PASSED** ✓
- ResultsAnalyzer: ✅
- ResultsVisualizer: ✅
- ReportGenerator: ✅
- Integration: ✅

See `VALIDATION_RESULTS.md` for details.

---

**Ready to use for thesis work!** 🎓


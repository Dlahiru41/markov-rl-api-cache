# Markov Chain Evaluation Module - Quick Reference

## Overview

The evaluation module provides rigorous metrics and visualizations for analyzing and comparing Markov chain prediction performance. Essential for thesis evaluation sections.

## Components

- **MarkovEvaluator**: Compute comprehensive metrics
- **MarkovVisualizer**: Create professional plots

## Quick Start

```python
from src.markov import MarkovPredictor, MarkovEvaluator, MarkovVisualizer

# Train predictor
predictor = MarkovPredictor(order=1)
predictor.fit(train_sequences)

# Create evaluator
evaluator = MarkovEvaluator(predictor)

# Evaluate
results = evaluator.evaluate_accuracy(test_sequences, k_values=[1,3,5,10])
print(f"Top-1 Accuracy: {results['top_1_accuracy']:.3f}")
print(f"MRR: {results['mrr']:.3f}")
```

## MarkovEvaluator API

### Constructor

```python
evaluator = MarkovEvaluator(predictor)
```

**Parameters:**
- `predictor`: MarkovPredictor instance to evaluate

### Core Metrics

#### `evaluate_accuracy(test_sequences, contexts=None, k_values=[1,3,5,10])`

Compute core accuracy metrics.

```python
results = evaluator.evaluate_accuracy(test_sequences, k_values=[1,3,5,10])
```

**Returns:**
```python
{
    'top_1_accuracy': 0.72,   # Accuracy at k=1
    'top_3_accuracy': 0.89,   # Accuracy at k=3
    'top_5_accuracy': 0.94,   # Accuracy at k=5
    'top_10_accuracy': 0.97,  # Accuracy at k=10
    'mrr': 0.85,              # Mean Reciprocal Rank
    'coverage': 0.98,         # Fraction of predictable states
    'perplexity': 2.5,        # Information-theoretic uncertainty
    'total_transitions': 1000,
    'predictable_transitions': 980
}
```

**Metrics Explained:**

- **top_k_accuracy**: Fraction of times correct answer was in top-k predictions
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank of correct answer (higher is better)
- **Coverage**: What fraction of states could we make predictions for
- **Perplexity**: exp(average negative log likelihood) - lower is better

### Breakdown Analyses

#### `evaluate_per_endpoint(test_sequences, contexts=None, k_values=[1,3])`

Accuracy broken down by which endpoint we're predicting FROM.

```python
df = evaluator.evaluate_per_endpoint(test_sequences)
print(df.head())
```

**Returns DataFrame:**
```
endpoint     sample_count  top_1_accuracy  top_3_accuracy  mrr
login        150          0.850           0.950           0.900
profile      120          0.800           0.917           0.875
browse       100          0.700           0.850           0.775
...
```

**Use case:** Identify which APIs are easy or hard to predict after.

#### `evaluate_per_context(test_sequences, contexts, k_values=[1,3])`

Accuracy broken down by context values.

```python
df = evaluator.evaluate_per_context(test_sequences, contexts)
print(df)
```

**Returns DataFrame:**
```
user_type  time_of_day  sample_count  top_1_accuracy  top_3_accuracy  mrr
premium    morning      80           0.900           0.975           0.925
free       morning      70           0.714           0.857           0.785
premium    evening      60           0.850           0.950           0.900
...
```

**Use case:** Do predictions work better for premium users? During morning?

#### `evaluate_calibration(test_sequences, contexts=None, num_bins=10)`

Calibration analysis: When we predict 80% probability, are we right 80% of the time?

```python
calibration = evaluator.evaluate_calibration(test_sequences, num_bins=10)
```

**Returns:**
```python
{
    'bin_centers': [0.1, 0.2, 0.3, ...],        # Probability bin centers
    'predicted_probs': [0.12, 0.21, 0.31, ...], # Avg predicted probability per bin
    'actual_accuracy': [0.15, 0.19, 0.29, ...], # Actual accuracy per bin
    'sample_counts': [50, 80, 120, ...]          # Samples per bin
}
```

**Use case:** Assess if probability estimates are reliable. Perfect calibration means predicted_probs ≈ actual_accuracy.

### Cross-Validation

#### `cross_validate(sequences, contexts=None, k_folds=5, k_values=[1,3,5])`

K-fold cross-validation for confidence intervals.

```python
cv_results = evaluator.cross_validate(sequences, contexts, k_folds=5)

for metric, (mean, std) in cv_results.items():
    print(f"{metric}: {mean:.3f} ± {std:.3f}")
```

**Returns:**
```python
{
    'top_1_accuracy': (0.720, 0.035),  # (mean, std)
    'top_3_accuracy': (0.890, 0.025),
    'mrr': (0.850, 0.030),
    ...
}
```

**Use case:** Provides confidence in accuracy estimates. Report as "72% ± 3.5%".

### Model Comparison

#### `compare_models(models, test_sequences, contexts=None, k_values=[1,3,5])`

Compare multiple models on same test data.

```python
models = {
    'first_order': predictor1,
    'second_order': predictor2,
    'context_aware': predictor3
}

comparison = evaluator.compare_models(models, test_sequences, contexts)
print(comparison)
```

**Returns DataFrame:**
```
model          top_1_accuracy  top_3_accuracy  top_5_accuracy  mrr    perplexity
first_order    0.720           0.890           0.940           0.850  2.50
second_order   0.785           0.925           0.965           0.895  2.15
context_aware  0.820           0.950           0.980           0.920  1.85
```

**Use case:** Systematic model selection. Which approach works best?

## MarkovVisualizer API

All visualization methods are static methods. They either display (if `output_path=None`) or save to file.

### Transition Heatmap

```python
MarkovVisualizer.plot_transition_heatmap(
    predictor,
    top_k=20,
    output_path='heatmap.png'
)
```

Shows most common transitions as a heatmap. Rows = from states, columns = to states, color = probability.

**Parameters:**
- `predictor`: MarkovPredictor to visualize
- `top_k`: Number of top APIs to include (default 20)
- `output_path`: Path to save figure (if None, displays)
- `figsize`: Figure size tuple (default (12, 10))

**Use case:** Understand common API access patterns.

### Accuracy by Position

```python
MarkovVisualizer.plot_accuracy_by_position(
    test_sequences,
    predictor,
    max_position=20,
    output_path='accuracy_by_position.png'
)
```

Shows how accuracy changes by position in sequence. Are early predictions better than late ones?

**Use case:** Understand if predictions degrade over time in a session.

### Calibration Curve

```python
calibration = evaluator.evaluate_calibration(test_sequences)
MarkovVisualizer.plot_calibration_curve(
    calibration,
    output_path='calibration.png'
)
```

X axis = predicted probability, Y axis = actual accuracy. Perfect calibration is diagonal line.

**Use case:** Visualize calibration quality for thesis.

### Confidence Distribution

```python
MarkovVisualizer.plot_prediction_confidence_distribution(
    test_sequences,
    predictor,
    output_path='confidence_dist.png'
)
```

Histogram of prediction confidence values. Are we generally confident or uncertain?

**Use case:** Understand model uncertainty characteristics.

### Model Comparison Plot

```python
comparison = evaluator.compare_models(models, test_sequences)
MarkovVisualizer.plot_model_comparison(
    comparison,
    metrics=['top_1_accuracy', 'top_3_accuracy', 'mrr'],
    output_path='model_comparison.png'
)
```

Bar chart comparing multiple models on selected metrics.

**Use case:** Visual comparison for presentations and thesis.

## Complete Example

```python
from src.markov import MarkovPredictor, MarkovEvaluator, MarkovVisualizer

# 1. Train models
predictor1 = MarkovPredictor(order=1)
predictor1.fit(train_sequences)

predictor2 = MarkovPredictor(order=2)
predictor2.fit(train_sequences)

# 2. Create evaluator
evaluator = MarkovEvaluator(predictor1)

# 3. Basic evaluation
results = evaluator.evaluate_accuracy(test_sequences, k_values=[1,3,5,10])
print(f"Top-1 Accuracy: {results['top_1_accuracy']:.1%}")
print(f"Top-5 Accuracy: {results['top_5_accuracy']:.1%}")
print(f"MRR: {results['mrr']:.3f}")
print(f"Perplexity: {results['perplexity']:.2f}")

# 4. Per-endpoint breakdown
per_endpoint = evaluator.evaluate_per_endpoint(test_sequences)
print("\nTop 5 endpoints by accuracy:")
print(per_endpoint.sort_values('top_1_accuracy', ascending=False).head(5))

# 5. Calibration
calibration = evaluator.evaluate_calibration(test_sequences, num_bins=10)
MarkovVisualizer.plot_calibration_curve(calibration, 'calibration.png')

# 6. Cross-validation
cv_results = evaluator.cross_validate(sequences, k_folds=5)
mean, std = cv_results['top_1_accuracy']
print(f"\nCV Top-1 Accuracy: {mean:.1%} ± {std:.1%}")

# 7. Model comparison
models = {'first_order': predictor1, 'second_order': predictor2}
comparison = evaluator.compare_models(models, test_sequences)
print("\nModel Comparison:")
print(comparison[['model', 'top_1_accuracy', 'mrr']])

# 8. Visualizations
MarkovVisualizer.plot_transition_heatmap(predictor1, top_k=15, output_path='heatmap.png')
MarkovVisualizer.plot_accuracy_by_position(test_sequences, predictor1, output_path='acc_pos.png')
MarkovVisualizer.plot_model_comparison(comparison, output_path='comparison.png')
```

## For Thesis Evaluation Section

### Recommended Metrics to Report

1. **Accuracy Metrics:**
   - Top-1, Top-3, Top-5 accuracy
   - Mean Reciprocal Rank (MRR)
   - Coverage

2. **With Confidence Intervals:**
   - Use cross-validation: "72.0% ± 3.5%"

3. **Breakdowns:**
   - Per-endpoint: Which APIs are hardest to predict?
   - Per-context (if applicable): How does context affect accuracy?

4. **Calibration:**
   - Are probability estimates reliable?
   - Include calibration curve

5. **Comparisons:**
   - Compare different approaches (first-order vs second-order vs context-aware)
   - Show improvement percentages

### Recommended Visualizations

1. **Transition Heatmap** - Shows learned patterns
2. **Calibration Curve** - Shows reliability of probabilities
3. **Model Comparison** - Bar chart of metrics
4. **Accuracy by Position** - Shows degradation over time (if any)

### Sample Results Table

```
Model          Top-1   Top-3   Top-5   MRR    Perplexity
First-Order    72.0%   89.0%   94.0%   0.850  2.50
Second-Order   78.5%   92.5%   96.5%   0.895  2.15
Context-Aware  82.0%   95.0%   98.0%   0.920  1.85
```

## Common Issues

### Issue: Low coverage

**Cause:** Many test states never seen in training

**Solution:**
```python
results = evaluator.evaluate_accuracy(test_sequences)
if results['coverage'] < 0.9:
    print("Warning: Low coverage - need more training data")
```

### Issue: Poor calibration

**Cause:** Model overconfident or underconfident

**Solution:** Check calibration curve. May need to adjust smoothing or use temperature scaling.

### Issue: Perplexity is inf

**Cause:** Model assigned zero probability to actual transitions

**Solution:** Increase smoothing parameter.

## Files

- **Implementation:** `src/markov/evaluation.py`
- **Demo:** `demo_evaluation.py`
- **Validation:** `validate_evaluation.py`
- **Quick Ref:** `EVALUATION_QUICK_REF.md` (this file)

---

**Status:** ✅ Complete and Ready for Thesis Evaluation  
**Date:** January 17, 2026


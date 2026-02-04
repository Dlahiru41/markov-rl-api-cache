# Preprocessing Pipeline CLI Guide

This guide explains how to use the preprocessing pipeline command-line interface to prepare API trace data for Markov chain and reinforcement learning training.

## Overview

The preprocessing pipeline combines all data preparation components into a single workflow:

1. **Load or Generate Data** - Load from file or generate synthetic data
2. **Validate** - Check data quality and detect anomalies
3. **Extract Sessions** - Group API calls into user sessions
4. **Build Sequences** - Create sequences for Markov chain training
5. **Fit Feature Engineer** - Learn feature encodings from training data
6. **Split Data** - Create train/test splits (chronological)
7. **Save Artifacts** - Save all processed data and models
8. **Generate Report** - Create human-readable summary

## Installation

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## Basic Usage

### Generate Synthetic Data

The easiest way to get started is to generate synthetic data:

```bash
python scripts/preprocess.py --synthetic --num-users 1000 -o data/processed
```

This generates realistic API traces for 1000 users and processes them completely.

### Process Real Data

To process your own API logs:

```bash
python scripts/preprocess.py -i data/raw/api_logs.parquet -o data/processed
```

Supported file formats:
- **Parquet** (`.parquet`) - Recommended for performance
- **JSON** (`.json`) - Must contain a Dataset structure

### Validate Data Only

To check data quality without processing:

```bash
python scripts/preprocess.py -i data/raw/api_logs.parquet --validate-only
```

This runs validation checks and reports issues without creating train/test splits.

## Command-Line Options

### Required (One of)

- `--input` / `-i` - Path to input data file (CSV, JSON, or Parquet)
- `--synthetic` - Generate synthetic data instead of loading from file

### Optional

- `--output` / `-o` - Output directory (default: `data/processed`)
- `--num-users` - Number of users for synthetic generation (default: 1000)
- `--sessions-per-user` - Average sessions per user (default: 5)
- `--seed` - Random seed for reproducibility (default: 42)
- `--train-ratio` - Fraction of data for training (default: 0.8)
- `--validate-only` - Just validate without full processing
- `--config` - Path to custom config file (YAML)
- `--verbose` / `-v` - Enable verbose output

## Examples

### Example 1: Quick Test with Small Dataset

```bash
python scripts/preprocess.py --synthetic --num-users 100 --output data/test --seed 42
```

**Output:**
```
======================================================================
  Preprocessing Pipeline
======================================================================

Mode: Synthetic data generation
  • Users: 100
  • Sessions per user: 5
  • Output: data/test
  • Seed: 42
  • Train ratio: 80.0%

Step 1: Generating synthetic data...
  ✓ Generated 435 sessions

Step 2: Validating data quality...
  Valid: YES
  Errors: 0
  Warnings: 0
  Anomalies: 1
  Error rate: 5.1%

...

Quick Statistics:
  • Raw sessions: 435
  • Raw calls: 5,032
  • Train sessions: 348
  • Test sessions: 87
  • Sequences: 348
  • Feature dimension: 31
  • Duration: 0.5s
```

### Example 2: Large-Scale Synthetic Dataset

```bash
python scripts/preprocess.py \
  --synthetic \
  --num-users 10000 \
  --sessions-per-user 10 \
  --output data/large_dataset \
  --seed 12345
```

### Example 3: Process Real Data with Custom Split

```bash
python scripts/preprocess.py \
  -i data/raw/production_logs.parquet \
  -o data/production_processed \
  --train-ratio 0.9 \
  --seed 42
```

This uses 90% of data for training and 10% for testing.

### Example 4: Validation Only

```bash
python scripts/preprocess.py \
  -i data/raw/suspicious_logs.parquet \
  --validate-only \
  --verbose
```

This checks data quality without processing, with verbose output for debugging.

### Example 5: Custom Configuration

Create a custom config file `configs/custom.yaml`:

```yaml
markov:
  order: 2
  smoothing: 1e-5

preprocessing:
  min_session_length: 3
  max_session_length: 50
```

Then run:

```bash
python scripts/preprocess.py \
  --synthetic \
  --config configs/custom.yaml \
  --output data/custom_config
```

## Output Files

The pipeline creates the following files in the output directory:

### 1. `train.parquet`
Training dataset in Parquet format. Contains:
- All API calls from training sessions
- Organized by session_id
- Ready for model training

### 2. `test.parquet`
Test dataset in Parquet format. Contains:
- All API calls from test sessions
- Chronologically after training data
- For evaluation only

### 3. `sequences.json`
Extracted sequences for Markov chain training. Format:
```json
[
  ["/api/login", "/api/users/{id}/profile", "/api/products/browse"],
  ["/api/products/search", "/api/products/{id}/details", "/api/cart/add"],
  ...
]
```

### 4. `feature_engineer.pkl`
Fitted FeatureEngineer model. Use for inference:
```python
import pickle
with open('data/processed/feature_engineer.pkl', 'rb') as f:
    feature_engineer = pickle.load(f)

# Transform new data
features = feature_engineer.transform_session(session)
```

### 5. `statistics.json`
Detailed statistics in JSON format:
```json
{
  "start_time": "2026-01-11T21:47:29.871822",
  "end_time": "2026-01-11T21:47:29.984831",
  "duration_seconds": 0.113009,
  "raw_sessions": 250,
  "raw_calls": 2703,
  "train_sessions": 200,
  "test_sessions": 50,
  "sequences_extracted": 200,
  "feature_dimension": 31,
  "validation_errors": 0,
  "quality_metrics": { ... }
}
```

### 6. `report.md`
Human-readable preprocessing report. Open with any Markdown viewer or text editor.

## Data Validation

The pipeline performs comprehensive validation:

### Checks Performed

1. **Required Fields** - Ensures all necessary fields are present
2. **Data Types** - Validates timestamps, numeric values, etc.
3. **Value Ranges** - Checks status codes, response times, etc.
4. **Consistency** - Verifies session IDs, chronological order
5. **Anomaly Detection** - Identifies outliers and suspicious patterns

### Quality Metrics

- **Missing Values Fraction** - Proportion of missing data
- **Duplicate Calls Fraction** - Duplicate entries
- **Response Time Outliers** - Unusually slow requests
- **Error Rate** - Proportion of failed requests (4xx/5xx)
- **Empty Endpoint Fraction** - Missing endpoint paths

## Loading Processed Data

### Python

```python
import pandas as pd
from pathlib import Path

# Load train data
train_df = pd.read_parquet('data/processed/train.parquet')
print(f"Training data: {len(train_df)} calls")

# Load test data
test_df = pd.read_parquet('data/processed/test.parquet')
print(f"Test data: {len(test_df)} calls")

# Load sequences
import json
with open('data/processed/sequences.json', 'r') as f:
    sequences = json.load(f)
print(f"Sequences: {len(sequences)}")

# Load feature engineer
import pickle
with open('data/processed/feature_engineer.pkl', 'rb') as f:
    fe = pickle.load(f)
print(f"Feature dimension: {fe.get_feature_dim()}")
```

### Load as Dataset Objects

```python
from preprocessing.models import Dataset

# Load training dataset
train_dataset = Dataset.load_from_parquet('data/processed/train.parquet')
print(f"Sessions: {len(train_dataset.sessions)}")
print(f"Calls: {train_dataset.total_calls}")
print(f"Users: {train_dataset.num_unique_users}")
```

## Troubleshooting

### Error: "Must specify either --input or --synthetic"

You need to provide either an input file or use `--synthetic`:
```bash
# Wrong
python scripts/preprocess.py -o data/output

# Correct
python scripts/preprocess.py --synthetic -o data/output
# OR
python scripts/preprocess.py -i data/raw/logs.parquet -o data/output
```

### Error: "Data validation failed"

Your input data has quality issues. Run with `--validate-only` to see details:
```bash
python scripts/preprocess.py -i data/raw/logs.parquet --validate-only --verbose
```

### Unicode Errors on Windows

If you see encoding errors when viewing the report, use UTF-8 capable editors:
- VS Code
- Notepad++ (with UTF-8 encoding)
- PowerShell: `Get-Content report.md -Encoding UTF8`

### Out of Memory

For large datasets, consider:
1. Processing in smaller batches
2. Reducing `--num-users` for synthetic data
3. Using a machine with more RAM
4. Using Parquet format (more memory efficient)

## Integration with Downstream Tasks

### Training a Markov Chain

```python
import json
from src.markov import MarkovChain

# Load sequences
with open('data/processed/sequences.json', 'r') as f:
    sequences = json.load(f)

# Train Markov chain
markov = MarkovChain(order=1)
markov.train(sequences)
markov.save('models/markov_model.pkl')
```

### Training RL Agent

```python
from preprocessing.models import Dataset
import pickle

# Load datasets
train_dataset = Dataset.load_from_parquet('data/processed/train.parquet')
test_dataset = Dataset.load_from_parquet('data/processed/test.parquet')

# Load feature engineer
with open('data/processed/feature_engineer.pkl', 'rb') as f:
    feature_engineer = pickle.load(f)

# Train RL agent
from src.rl import RLAgent

agent = RLAgent(
    state_dim=feature_engineer.get_feature_dim(),
    action_dim=num_cache_actions
)

# Training loop
for session in train_dataset.sessions:
    features = feature_engineer.transform_session(session)
    agent.train(features, rewards)
```

## Best Practices

1. **Always use a seed** - For reproducibility across experiments
2. **Validate first** - Use `--validate-only` before full processing
3. **Chronological splits** - Default behavior prevents data leakage
4. **Save the report** - Keep track of preprocessing parameters
5. **Version your data** - Include seed and date in output directory names

Example versioned output:
```bash
python scripts/preprocess.py \
  --synthetic \
  --num-users 10000 \
  --seed 42 \
  --output "data/processed_v1_$(date +%Y%m%d)_seed42"
```

## FAQ

**Q: What's the difference between sessions and sequences?**

A: Sessions are groups of API calls by the same user. Sequences are the ordered list of endpoints extracted from sessions for Markov training.

**Q: Why chronological splitting?**

A: Time-based splitting prevents data leakage where the model learns from "future" data. The model is evaluated on behavior it hasn't seen temporally.

**Q: Can I process CSV files?**

A: Currently only Parquet and JSON are supported. Convert CSV to Parquet first:
```python
import pandas as pd
df = pd.read_csv('data.csv')
df.to_parquet('data.parquet')
```

**Q: How do I know if my data is good quality?**

A: Check the validation report. Aim for:
- Error rate < 10%
- Response time outliers < 5%
- No missing values
- No duplicate calls

**Q: Can I customize the workflow?**

A: Yes! Import and use the `PreprocessingPipeline` class directly in Python for full control. The CLI is a convenience wrapper.

## Additional Resources

- **Pipeline Implementation**: `preprocessing/pipeline.py`
- **Data Models**: `preprocessing/models.py`
- **Session Extraction**: `preprocessing/session_extractor.py`
- **Feature Engineering**: `preprocessing/feature_engineer.py`
- **Synthetic Generator**: `preprocessing/synthetic_generator.py`

For more details, see the module documentation and tests.


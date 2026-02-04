# Preprocessing Pipeline - Quick Reference

## One-Line Commands

```bash
# Quick synthetic test (100 users)
python scripts/preprocess.py --synthetic --num-users 100 --output data/test

# Production-scale synthetic (10k users)
python scripts/preprocess.py --synthetic --num-users 10000 --output data/large

# Process real data
python scripts/preprocess.py -i data/raw/logs.parquet -o data/processed

# Validate only
python scripts/preprocess.py -i data/raw/logs.parquet --validate-only

# Custom train/test split (90/10)
python scripts/preprocess.py --synthetic --train-ratio 0.9 -o data/90_10_split
```

## Options Quick Reference

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--input` | `-i` | Path | - | Input data file (Parquet/JSON) |
| `--output` | `-o` | Path | `data/processed` | Output directory |
| `--synthetic` | - | Flag | False | Generate synthetic data |
| `--num-users` | - | Int | 1000 | Users for synthetic generation |
| `--sessions-per-user` | - | Int | 5 | Avg sessions per user |
| `--seed` | - | Int | 42 | Random seed |
| `--train-ratio` | - | Float | 0.8 | Training data fraction (0-1) |
| `--validate-only` | - | Flag | False | Skip processing, just validate |
| `--config` | - | Path | - | Custom YAML config |
| `--verbose` | `-v` | Flag | False | Verbose output |

## Output Files

| File | Description |
|------|-------------|
| `train.parquet` | Training dataset (80% by default) |
| `test.parquet` | Test dataset (20% by default) |
| `sequences.json` | Sequences for Markov training |
| `feature_engineer.pkl` | Fitted feature encoder |
| `statistics.json` | Detailed statistics |
| `report.md` | Human-readable report |

## Common Workflows

### Development/Testing
```bash
# Quick test with small dataset
python scripts/preprocess.py --synthetic --num-users 50 -o data/dev

# Validate your data
python scripts/preprocess.py -i data/raw/my_data.parquet --validate-only -v
```

### Production
```bash
# Process production logs
python scripts/preprocess.py \
  -i data/raw/production_logs.parquet \
  -o data/production_processed \
  --seed 42 \
  --train-ratio 0.9

# Generate large synthetic dataset
python scripts/preprocess.py \
  --synthetic \
  --num-users 100000 \
  --sessions-per-user 8 \
  -o data/synthetic_large \
  --seed 12345
```

### Experimentation
```bash
# Different random seeds for cross-validation
for seed in 42 123 456 789 1337; do
  python scripts/preprocess.py \
    --synthetic \
    --num-users 5000 \
    --seed $seed \
    -o "data/cv_fold_$seed"
done
```

## Python Usage

```python
from preprocessing.pipeline import PreprocessingPipeline

# Initialize
pipeline = PreprocessingPipeline(
    output_dir='data/processed',
    train_ratio=0.8,
    seed=42
)

# Run with synthetic data
results = pipeline.run(
    generate_synthetic=True,
    num_users=1000,
    sessions_per_user=(5, 2)
)

# Run with real data
results = pipeline.run(
    input_path='data/raw/logs.parquet'
)

# Validate only
results = pipeline.run(
    input_path='data/raw/logs.parquet',
    validate_only=True
)

# Access results
print(results['train'])      # Path to train.parquet
print(results['test'])       # Path to test.parquet
print(results['sequences'])  # Path to sequences.json
print(results['report'])     # Path to report.md
```

## Load Processed Data

```python
# Load as pandas DataFrames
import pandas as pd
train_df = pd.read_parquet('data/processed/train.parquet')
test_df = pd.read_parquet('data/processed/test.parquet')

# Load as Dataset objects
from preprocessing.models import Dataset
train_ds = Dataset.load_from_parquet('data/processed/train.parquet')
test_ds = Dataset.load_from_parquet('data/processed/test.parquet')

# Load sequences
import json
with open('data/processed/sequences.json') as f:
    sequences = json.load(f)

# Load feature engineer
import pickle
with open('data/processed/feature_engineer.pkl', 'rb') as f:
    fe = pickle.load(f)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Must specify --input or --synthetic" | Add one of these flags |
| "Data validation failed" | Run with `--validate-only -v` to see issues |
| Out of memory | Reduce `--num-users` or process in batches |
| Unicode errors (Windows) | View report with UTF-8 editor (VS Code, etc.) |
| Slow processing | Use Parquet format, smaller dataset |

## Exit Codes

- `0` - Success
- `1` - Error (validation failed, file not found, etc.)

## Validation Metrics

| Metric | Good | Warning | Bad |
|--------|------|---------|-----|
| Error rate | < 5% | 5-10% | > 10% |
| Missing values | 0% | < 1% | > 1% |
| Duplicates | 0% | < 0.1% | > 0.1% |
| Outliers | < 2% | 2-5% | > 5% |

## Tips

✅ **Do:**
- Always use `--seed` for reproducibility
- Run `--validate-only` first on new data
- Keep preprocessing reports for documentation
- Use Parquet format for large datasets

❌ **Don't:**
- Mix `--input` and `--synthetic` flags
- Use train_ratio < 0.5 or > 0.95
- Process without validating first
- Forget to version your processed data

## Need Help?

```bash
python scripts/preprocess.py --help
```

For detailed documentation, see `PREPROCESSING_CLI_GUIDE.md`


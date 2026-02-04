# Preprocessing Report

**Generated:** 2026-01-11 21:47:30
**Pipeline Version:** 1.0

---

## Input Data

- **Source:** synthetic
- **Total Sessions:** 250
- **Total API Calls:** 2,703
- **Unique Users:** 50
- **Unique Endpoints:** 17

## Data Validation

- **Valid:** ✓ YES
- **Errors:** 0
- **Warnings:** 0
- **Anomalies Detected:** 1

### Quality Metrics

- **missing_values_fraction:** 0.0%
- **duplicate_calls_fraction:** 0.0%
- **response_time_outliers_fraction:** 1.6%
- **error_rate:** 5.3%
- **empty_endpoint_fraction:** 0.0%

## Train/Test Split

- **Strategy:** Chronological (prevents time leakage)
- **Train Ratio:** 80.0%

### Training Set

- **Sessions:** 200
- **API Calls:** 2,143

### Test Set

- **Sessions:** 50
- **API Calls:** 560

## Sequence Extraction

- **Sequences Extracted:** 200
- **Purpose:** Markov chain training

## Feature Engineering

- **Feature Dimension:** 31
- **Purpose:** RL state representation

## Output Artifacts

**Location:** `data\test2`

- `train.parquet` - Training dataset
- `test.parquet` - Test dataset
- `sequences.json` - Extracted sequences for Markov training
- `feature_engineer.pkl` - Fitted feature engineer for inference
- `statistics.json` - Detailed statistics in JSON format
- `report.md` - This report

## Processing Statistics

- **Duration:** 0.1 seconds
- **Start Time:** 2026-01-11 21:47:29
- **End Time:** 2026-01-11 21:47:29

---

**Pipeline Status:** ✓ Complete
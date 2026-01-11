# Preprocessing Report

**Generated:** 2026-01-11 21:45:51
**Pipeline Version:** 1.0

---

## Input Data

- **Source:** synthetic
- **Total Sessions:** 435
- **Total API Calls:** 5,032
- **Unique Users:** 100
- **Unique Endpoints:** 17

## Data Validation

- **Valid:** ✓ YES
- **Errors:** 0
- **Warnings:** 0
- **Anomalies Detected:** 1

### Quality Metrics

- **missing_values_fraction:** 0.0%
- **duplicate_calls_fraction:** 0.0%
- **response_time_outliers_fraction:** 1.5%
- **error_rate:** 5.1%
- **empty_endpoint_fraction:** 0.0%

## Train/Test Split

- **Strategy:** Chronological (prevents time leakage)
- **Train Ratio:** 80.0%

### Training Set

- **Sessions:** 348
- **API Calls:** 3,913

### Test Set

- **Sessions:** 87
- **API Calls:** 1,119

## Sequence Extraction

- **Sequences Extracted:** 348
- **Purpose:** Markov chain training

## Feature Engineering

- **Feature Dimension:** 31
- **Purpose:** RL state representation

## Output Artifacts

**Location:** `data\test`

- `train.parquet` - Training dataset
- `test.parquet` - Test dataset
- `sequences.json` - Extracted sequences for Markov training
- `feature_engineer.pkl` - Fitted feature engineer for inference
- `statistics.json` - Detailed statistics in JSON format
- `report.md` - This report

## Processing Statistics

- **Duration:** 0.7 seconds
- **Start Time:** 2026-01-11 21:45:50
- **End Time:** In Progress

---

**Pipeline Status:** ✓ Complete
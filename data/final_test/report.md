# Preprocessing Report

**Generated:** 2026-01-11 21:49:02
**Pipeline Version:** 1.0

---

## Input Data

- **Source:** synthetic
- **Total Sessions:** 1,328
- **Total API Calls:** 15,221
- **Unique Users:** 200
- **Unique Endpoints:** 17

## Data Validation

- **Valid:** ✓ YES
- **Errors:** 0
- **Warnings:** 0
- **Anomalies Detected:** 3

### Quality Metrics

- **missing_values_fraction:** 0.0%
- **duplicate_calls_fraction:** 0.0%
- **response_time_outliers_fraction:** 1.4%
- **error_rate:** 5.1%
- **empty_endpoint_fraction:** 0.0%

## Train/Test Split

- **Strategy:** Chronological (prevents time leakage)
- **Train Ratio:** 75.0%

### Training Set

- **Sessions:** 996
- **API Calls:** 11,357

### Test Set

- **Sessions:** 332
- **API Calls:** 3,864

## Sequence Extraction

- **Sequences Extracted:** 996
- **Purpose:** Markov chain training

## Feature Engineering

- **Feature Dimension:** 31
- **Purpose:** RL state representation

## Output Artifacts

**Location:** `data\final_test`

- `train.parquet` - Training dataset
- `test.parquet` - Test dataset
- `sequences.json` - Extracted sequences for Markov training
- `feature_engineer.pkl` - Fitted feature engineer for inference
- `statistics.json` - Detailed statistics in JSON format
- `report.md` - This report

## Processing Statistics

- **Duration:** 0.3 seconds
- **Start Time:** 2026-01-11 21:49:01
- **End Time:** 2026-01-11 21:49:01

---

**Pipeline Status:** ✓ Complete
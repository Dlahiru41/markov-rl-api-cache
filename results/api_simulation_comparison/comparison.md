# API Simulation Comparison

Generated: 2026-03-15T10:20:16.762057

## Scenario Definitions

- **without_solution**: passive `DO_NOTHING` policy (no intelligent action)
- **with_solution**: intelligent policy mode = `heuristic`

## Results

| Metric | Without Solution | With Solution |
|---|---:|---:|
| Mean reward | 183.500 | 199.000 |
| Cache hit rate | 60.00% | 65.00% |
| Success rate | 97.50% | 97.50% |
| Cascade rate | 0.00% | 0.00% |
| Mean p95 latency (ms) | 37.48 | 32.13 |
| Prefetch efficiency | 0.00% | 0.00% |

## Improvement (With Solution vs Without)

- Reward improvement: **8.45%**
- Cache hit-rate improvement: **8.33%**
- Success-rate improvement: **0.00%**
- Cascade-rate reduction: **0.00%**
- Latency reduction: **14.29%**

## Prometheus Snapshots

- `without_solution.prom`: `C:/Users/dlahi/OneDrive/Desktop/4th YEAR/FYP/code/markov-rl-api-cache/results/api_simulation_comparison/without_solution.prom`
- `with_solution.prom`: `C:/Users/dlahi/OneDrive/Desktop/4th YEAR/FYP/code/markov-rl-api-cache/results/api_simulation_comparison/with_solution.prom`
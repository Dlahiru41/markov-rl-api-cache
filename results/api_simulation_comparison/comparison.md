# API Simulation Comparison

Generated: 2026-04-02T06:05:52.738616

## Scenario Definitions

- **without_solution**: passive `DO_NOTHING` policy (no intelligent action)
- **with_solution**: intelligent policy mode = `heuristic`

## Results

| Metric | Without Solution | With Solution |
|---|---:|---:|
| Mean reward | 758.458 | 773.958 |
| Cache hit rate | 87.16% | 88.83% |
| Success rate | 98.06% | 98.06% |
| Cascade rate | 0.00% | 0.00% |
| Mean p95 latency (ms) | 0.61 | 0.52 |
| Prefetch efficiency | 0.00% | 0.00% |

## Improvement (With Solution vs Without)

- Reward improvement: **2.04%**
- Cache hit-rate improvement: **1.91%**
- Success-rate improvement: **0.00%**
- Cascade-rate reduction: **0.00%**
- Latency reduction: **14.29%**

## Prometheus Snapshots

- `without_solution.prom`: `/home/runner/work/markov-rl-api-cache/markov-rl-api-cache/results/api_simulation_comparison/without_solution.prom`
- `with_solution.prom`: `/home/runner/work/markov-rl-api-cache/markov-rl-api-cache/results/api_simulation_comparison/with_solution.prom`
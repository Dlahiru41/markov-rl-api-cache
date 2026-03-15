# API Simulation Comparison (With vs Without Solution)

This guide runs an A/B simulation with identical traffic settings:

- `without_solution`: passive `DO_NOTHING` action policy
- `with_solution`: intelligent policy (DQN model if provided, otherwise Markov-aware heuristic)

## Run

```powershell
python scripts/api_simulation_compare.py
```

Optional (use your trained model checkpoint):

```powershell
python scripts/api_simulation_compare.py --agent-model test_agent.pt
```

Optional tuning:

```powershell
python scripts/api_simulation_compare.py --episodes 20 --max-steps 80 --session-length 80 --seed 123
```

## Output Files

Generated under `results/api_simulation_comparison/`:

- `comparison.json` - structured raw metrics and computed improvements
- `comparison.md` - readable summary table
- `without_solution.prom` - Prometheus snapshot for baseline run
- `with_solution.prom` - Prometheus snapshot for solution run

## Prometheus Metrics Included

Snapshots include these core metrics (with `markov_rl_` prefix):

- `request_count_total`
- `request_latency_seconds` (histogram)
- `cache_hits_total`
- `cache_misses_total`
- `cache_hit_rate`
- `prefetch_efficiency`
- `cascade_events_total`

You can ingest the `.prom` files in your monitoring workflow or inspect the computed snapshot in `comparison.json`.


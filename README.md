# markov-rl-api-cache

Markov Chain-based Reinforcement Learning framework for adaptive API caching in microservices.

This repository contains components for:
- Markov chain modeling of API call patterns
- Reinforcement learning agents (DQN) to adapt cache policies
- Simulators for microservice traffic and failures
- Integration with API gateway and cache backends (Redis)
- Evaluation, baselines, and monitoring setups

See the `src/`, `simulator/`, and `evaluation/` directories for implementation and experiment code.

## API Simulation Comparison (With vs Without Solution)

Run the A/B simulation and generate comparison reports:

```powershell
python scripts/api_simulation_compare.py
```

Use a trained DQN checkpoint (optional):

```powershell
python scripts/api_simulation_compare.py --agent-model test_agent.pt
```

Outputs are written to `results/api_simulation_comparison/`:
- `comparison.json`
- `comparison.md`
- `without_solution.prom`
- `with_solution.prom`

See `docs/API_SIMULATION_COMPARISON.md` for details.

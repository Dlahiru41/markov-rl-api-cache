"""
A/B API simulation for comparing baseline behavior vs the intelligent solution.

Scenarios:
- without_solution: passive cache behavior (DO_NOTHING action)
- with_solution: intelligent policy using a trained DQN model when provided,
  otherwise a Markov-aware heuristic fallback

Outputs:
- JSON summary report
- Markdown comparison report
- Prometheus metrics snapshot for each scenario
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np

# Ensure local imports work when run as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prometheus_client import generate_latest

from src.integration.gym_environment import CachingEnv, CacheEnvConfig, SimulatorConfig
from src.monitoring.metrics import MetricsCollector
from src.rl.actions import CacheAction
from src.rl.agents.dqn_agent import DQNAgent, DQNConfig


@dataclass
class ScenarioResult:
    name: str
    episodes: int
    total_requests: int
    mean_reward: float
    mean_cache_hit_rate: float
    mean_prediction_accuracy: float
    mean_prefetch_efficiency: float
    cascade_rate: float
    mean_p95_latency_ms: float
    success_rate: float
    action_distribution: Dict[str, int]
    prometheus: Dict[str, float]


class NoSolutionPolicy:
    """Baseline policy: no intelligent intervention."""

    def select_action(self, obs: np.ndarray, env: CachingEnv) -> int:
        _ = obs, env
        return int(CacheAction.DO_NOTHING)


class SolutionPolicy:
    """
    Intelligent policy wrapper.

    Priority:
    1) Use trained DQN model if provided and loadable
    2) Fall back to Markov-aware heuristic policy
    """

    def __init__(self, model_path: Optional[str] = None):
        self.agent: Optional[DQNAgent] = None
        self.mode = "heuristic"

        if model_path:
            model_file = Path(model_path)
            if model_file.exists():
                try:
                    config = DQNConfig(state_dim=60, action_dim=7, epsilon_start=0.0, epsilon_end=0.0)
                    self.agent = DQNAgent(config)
                    self.agent.load(str(model_file))
                    self.agent.epsilon = 0.0
                    self.mode = "dqn"
                except Exception:
                    # Safe fallback when model shape/checkpoint differs from runtime config
                    self.agent = None
                    self.mode = "heuristic"

    def select_action(self, obs: np.ndarray, env: CachingEnv) -> int:
        if self.agent is not None:
            return int(self.agent.select_action(obs, evaluate=True))

        predictions = env.predictor.predict(k=5, context=env._get_current_context())
        top_prob = predictions[0][1] if predictions else 0.0
        cache_hit_rate = env.cache_manager.get_metrics().get("hit_rate", 0.0)
        cpu = env.system_metrics.get("cpu", 0.0)
        error_rate = env.system_metrics.get("error_rate", 0.0)

        if cpu > 0.85 or error_rate > 0.08:
            return int(CacheAction.PREFETCH_CONSERVATIVE)
        if top_prob >= 0.70:
            return int(CacheAction.PREFETCH_MODERATE)
        if cache_hit_rate < 0.45:
            return int(CacheAction.CACHE_CURRENT)
        return int(CacheAction.DO_NOTHING)


def _extract_prometheus_snapshot(metrics_text: str) -> Dict[str, float]:
    """Extract a compact metric snapshot by summing label variants."""

    keys_of_interest = {
        "markov_rl_request_count_total",
        "markov_rl_cache_hits_total",
        "markov_rl_cache_misses_total",
        "markov_rl_cache_hit_rate",
        "markov_rl_prefetch_efficiency",
        "markov_rl_cascade_events_total",
        "markov_rl_request_latency_seconds_sum",
        "markov_rl_request_latency_seconds_count",
    }

    aggregates: Dict[str, float] = {k: 0.0 for k in keys_of_interest}

    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric_name = line.split("{", 1)[0].split()[0]
        if metric_name not in aggregates:
            continue
        try:
            value = float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
        aggregates[metric_name] += value

    req_count = aggregates["markov_rl_request_latency_seconds_count"]
    avg_latency_ms = (aggregates["markov_rl_request_latency_seconds_sum"] / req_count * 1000.0) if req_count else 0.0

    return {
        "request_count_total": aggregates["markov_rl_request_count_total"],
        "cache_hits_total": aggregates["markov_rl_cache_hits_total"],
        "cache_misses_total": aggregates["markov_rl_cache_misses_total"],
        "cache_hit_rate": aggregates["markov_rl_cache_hit_rate"],
        "prefetch_efficiency": aggregates["markov_rl_prefetch_efficiency"],
        "cascade_events_total": aggregates["markov_rl_cascade_events_total"],
        "request_latency_avg_ms": avg_latency_ms,
    }


def _run_scenario(
    name: str,
    policy,
    episodes: int,
    seed: int,
    max_steps: int,
    session_length: int,
    num_apis: int,
) -> Tuple[ScenarioResult, str]:
    """Run one simulation scenario and return summary + Prometheus text."""

    env_config = CacheEnvConfig(
        simulator_config=SimulatorConfig(
            num_apis=num_apis,
            # gym_environment uses randint(low, high), so high must be > low.
            session_length_range=(session_length, session_length + 1),
            cascade_threshold=0.8,
            base_latency_ms=50.0,
            cache_hit_latency_ms=5.0,
        ),
        max_steps_per_episode=max_steps,
        episode_end_on_cascade=False,
        log_episode_metrics=False,
        seed=seed,
    )

    env = CachingEnv(env_config)
    metrics = MetricsCollector(service=f"api-sim-{name}")
    rng = random.Random(seed)

    rewards = []
    hit_rates = []
    pred_accuracies = []
    prefetch_efficiencies = []
    p95_latencies = []
    cascade_events = 0
    total_requests = 0
    total_success = 0
    action_distribution: Dict[str, int] = {}

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode, options={"reset_cache": True})
        done = False

        while not done:
            action = policy.select_action(obs, env)
            action_name = CacheAction.get_name(action)
            action_distribution[action_name] = action_distribution.get(action_name, 0) + 1
            metrics.record_env_step(action_name=action_name, steps=1)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_requests += 1

            api = info.get("api", "unknown")
            p95_ms = float(info.get("system_metrics", {}).get("p95_latency", 50.0))
            latency_seconds = max(0.0, p95_ms / 1000.0)

            # Simulate status based on environment error rate.
            error_rate = float(info.get("system_metrics", {}).get("error_rate", 0.0))
            status = "500" if rng.random() < error_rate else "200"
            if status == "200":
                total_success += 1

            metrics.record_request(endpoint=api, latency_seconds=latency_seconds, status=status)

            if bool(info.get("cache_hit", False)):
                metrics.record_cache_hit(endpoint=api)
            else:
                metrics.record_cache_miss(endpoint=api)

            metrics.update_cascade_risk(float(info.get("cascade_risk", 0.0)))
            metrics.update_system_metrics(
                cpu=float(info.get("system_metrics", {}).get("cpu", 0.0)),
                memory=float(info.get("system_metrics", {}).get("memory", 0.0)),
            )

        ep = env.get_episode_metrics()
        rewards.append(float(ep.get("total_reward", 0.0)))
        hit_rates.append(float(ep.get("cache_hit_rate", 0.0)))
        pred_accuracies.append(float(ep.get("prediction_accuracy", 0.0)))
        prefetch_efficiencies.append(float(ep.get("prefetch_efficiency", 0.0)))
        p95_latencies.append(float(ep.get("final_system_metrics", {}).get("p95_latency", 0.0)))

        if bool(ep.get("cascade_occurred", False)):
            cascade_events += 1

        metrics.record_episode(
            reward=float(ep.get("total_reward", 0.0)),
            length=int(ep.get("total_steps", 0)),
            hit_rate=float(ep.get("cache_hit_rate", 0.0)),
            cascade_occurred=bool(ep.get("cascade_occurred", False)),
        )

        prefetch_used = int(ep.get("total_prefetch_hits", 0))
        prefetch_wasted = int(ep.get("total_prefetch_wasted", 0))
        if prefetch_used + prefetch_wasted > 0:
            metrics.record_prefetch(
                strategy="simulation",
                used=prefetch_used,
                wasted=prefetch_wasted,
                bandwidth_bytes=(prefetch_used + prefetch_wasted) * 1024,
            )

    env.close()

    prometheus_text = generate_latest(metrics.registry).decode("utf-8")
    prom_snapshot = _extract_prometheus_snapshot(prometheus_text)

    scenario_result = ScenarioResult(
        name=name,
        episodes=episodes,
        total_requests=total_requests,
        mean_reward=float(np.mean(rewards) if rewards else 0.0),
        mean_cache_hit_rate=float(np.mean(hit_rates) if hit_rates else 0.0),
        mean_prediction_accuracy=float(np.mean(pred_accuracies) if pred_accuracies else 0.0),
        mean_prefetch_efficiency=float(np.mean(prefetch_efficiencies) if prefetch_efficiencies else 0.0),
        cascade_rate=float(cascade_events / episodes if episodes > 0 else 0.0),
        mean_p95_latency_ms=float(np.mean(p95_latencies) if p95_latencies else 0.0),
        success_rate=float(total_success / total_requests if total_requests > 0 else 0.0),
        action_distribution=action_distribution,
        prometheus=prom_snapshot,
    )

    return scenario_result, prometheus_text


def _build_comparison(baseline: ScenarioResult, solution: ScenarioResult) -> Dict[str, Any]:
    """Create directional improvement metrics (positive means better with solution)."""

    def pct_up(new: float, old: float) -> float:
        if old == 0:
            return 0.0
        return ((new - old) / abs(old)) * 100.0

    def pct_down(new: float, old: float) -> float:
        # Lower-is-better metric represented as positive improvement if reduced.
        if old == 0:
            return 0.0
        return ((old - new) / abs(old)) * 100.0

    return {
        "reward_improvement_pct": pct_up(solution.mean_reward, baseline.mean_reward),
        "cache_hit_rate_improvement_pct": pct_up(solution.mean_cache_hit_rate, baseline.mean_cache_hit_rate),
        "success_rate_improvement_pct": pct_up(solution.success_rate, baseline.success_rate),
        "cascade_rate_reduction_pct": pct_down(solution.cascade_rate, baseline.cascade_rate),
        "latency_reduction_pct": pct_down(solution.mean_p95_latency_ms, baseline.mean_p95_latency_ms),
    }


def _as_dict(result: ScenarioResult) -> Dict[str, Any]:
    return {
        "name": result.name,
        "episodes": result.episodes,
        "total_requests": result.total_requests,
        "mean_reward": result.mean_reward,
        "mean_cache_hit_rate": result.mean_cache_hit_rate,
        "mean_prediction_accuracy": result.mean_prediction_accuracy,
        "mean_prefetch_efficiency": result.mean_prefetch_efficiency,
        "cascade_rate": result.cascade_rate,
        "mean_p95_latency_ms": result.mean_p95_latency_ms,
        "success_rate": result.success_rate,
        "action_distribution": result.action_distribution,
        "prometheus_snapshot": result.prometheus,
    }


def _build_markdown_report(
    baseline: ScenarioResult,
    solution: ScenarioResult,
    comparison: Dict[str, Any],
    dqn_mode: str,
    output_dir: Path,
) -> str:
    """Build a concise Markdown report."""

    return "\n".join([
        "# API Simulation Comparison",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Scenario Definitions",
        "",
        "- **without_solution**: passive `DO_NOTHING` policy (no intelligent action)",
        f"- **with_solution**: intelligent policy mode = `{dqn_mode}`",
        "",
        "## Results",
        "",
        "| Metric | Without Solution | With Solution |",
        "|---|---:|---:|",
        f"| Mean reward | {baseline.mean_reward:.3f} | {solution.mean_reward:.3f} |",
        f"| Cache hit rate | {baseline.mean_cache_hit_rate:.2%} | {solution.mean_cache_hit_rate:.2%} |",
        f"| Success rate | {baseline.success_rate:.2%} | {solution.success_rate:.2%} |",
        f"| Cascade rate | {baseline.cascade_rate:.2%} | {solution.cascade_rate:.2%} |",
        f"| Mean p95 latency (ms) | {baseline.mean_p95_latency_ms:.2f} | {solution.mean_p95_latency_ms:.2f} |",
        f"| Prefetch efficiency | {baseline.mean_prefetch_efficiency:.2%} | {solution.mean_prefetch_efficiency:.2%} |",
        "",
        "## Improvement (With Solution vs Without)",
        "",
        f"- Reward improvement: **{comparison['reward_improvement_pct']:.2f}%**",
        f"- Cache hit-rate improvement: **{comparison['cache_hit_rate_improvement_pct']:.2f}%**",
        f"- Success-rate improvement: **{comparison['success_rate_improvement_pct']:.2f}%**",
        f"- Cascade-rate reduction: **{comparison['cascade_rate_reduction_pct']:.2f}%**",
        f"- Latency reduction: **{comparison['latency_reduction_pct']:.2f}%**",
        "",
        "## Prometheus Snapshots",
        "",
        f"- `without_solution.prom`: `{(output_dir / 'without_solution.prom').as_posix()}`",
        f"- `with_solution.prom`: `{(output_dir / 'with_solution.prom').as_posix()}`",
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A/B API simulation comparison.")
    parser.add_argument("--episodes", type=int, default=12, help="Number of episodes per scenario")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--max-steps", type=int, default=60, help="Max steps per episode")
    parser.add_argument("--session-length", type=int, default=60, help="Fixed API calls per episode")
    parser.add_argument("--num-apis", type=int, default=20, help="Number of simulated endpoints")
    parser.add_argument(
        "--agent-model",
        type=str,
        default=None,
        help="Optional DQN checkpoint path. If omitted/unloadable, heuristic mode is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/api_simulation_comparison",
        help="Output directory for reports and Prometheus snapshots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_policy = NoSolutionPolicy()
    solution_policy = SolutionPolicy(model_path=args.agent_model)

    without_result, without_prom = _run_scenario(
        name="without_solution",
        policy=baseline_policy,
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        session_length=args.session_length,
        num_apis=args.num_apis,
    )

    with_result, with_prom = _run_scenario(
        name="with_solution",
        policy=solution_policy,
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        session_length=args.session_length,
        num_apis=args.num_apis,
    )

    comparison = _build_comparison(without_result, with_result)

    payload = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "seed": args.seed,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "session_length": args.session_length,
            "num_apis": args.num_apis,
            "solution_policy_mode": solution_policy.mode,
            "agent_model": args.agent_model,
        },
        "without_solution": _as_dict(without_result),
        "with_solution": _as_dict(with_result),
        "comparison": comparison,
    }

    json_path = out_dir / "comparison.json"
    md_path = out_dir / "comparison.md"
    prom_without_path = out_dir / "without_solution.prom"
    prom_with_path = out_dir / "with_solution.prom"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    prom_without_path.write_text(without_prom, encoding="utf-8")
    prom_with_path.write_text(with_prom, encoding="utf-8")

    report = _build_markdown_report(
        baseline=without_result,
        solution=with_result,
        comparison=comparison,
        dqn_mode=solution_policy.mode,
        output_dir=out_dir,
    )
    md_path.write_text(report, encoding="utf-8")

    print("\nAPI simulation comparison complete.")
    print(f"- Without solution mean reward: {without_result.mean_reward:.3f}")
    print(f"- With solution mean reward:    {with_result.mean_reward:.3f}")
    print(f"- Cache hit-rate improvement:   {comparison['cache_hit_rate_improvement_pct']:.2f}%")
    print(f"- Latency reduction:            {comparison['latency_reduction_pct']:.2f}%")
    print(f"- Report:                       {md_path}")
    print(f"- JSON:                         {json_path}")


if __name__ == "__main__":
    main()


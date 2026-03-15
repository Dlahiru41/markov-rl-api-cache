import json
from pathlib import Path

from scripts.api_simulation_compare import (
    NoSolutionPolicy,
    SolutionPolicy,
    _build_comparison,
    _run_scenario,
)


def test_api_simulation_compare_smoke(tmp_path: Path):
    without_result, without_prom = _run_scenario(
        name="without_solution",
        policy=NoSolutionPolicy(),
        episodes=1,
        seed=7,
        max_steps=5,
        session_length=5,
        num_apis=10,
    )

    with_result, with_prom = _run_scenario(
        name="with_solution",
        policy=SolutionPolicy(model_path=None),
        episodes=1,
        seed=7,
        max_steps=5,
        session_length=5,
        num_apis=10,
    )

    comparison = _build_comparison(without_result, with_result)

    assert without_result.total_requests > 0
    assert with_result.total_requests > 0
    assert "markov_rl_request_count_total" in without_prom
    assert "markov_rl_request_count_total" in with_prom
    assert "reward_improvement_pct" in comparison

    out_file = tmp_path / "comparison_smoke.json"
    out_file.write_text(
        json.dumps(
            {
                "without": without_result.name,
                "with": with_result.name,
                "comparison": comparison,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert out_file.exists()


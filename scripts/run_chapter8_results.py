#!/usr/bin/env python3
"""
Run all Chapter 8 evidence commands in one place and save structured results.

Default behavior executes the same command set used for the report and writes:
- per-command logs
- per-command metadata (status, duration, return code, parsed test summaries)
- collected artifact paths and parsed key outputs when available
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RunSpec:
    key: str
    description: str
    command: List[str]


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_pytest_summary(text: str) -> Dict[str, int]:
    summary = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    tail = "\n".join(text.splitlines()[-40:])
    patterns = {
        "passed": r"(\d+)\s+passed",
        "failed": r"(\d+)\s+failed",
        "skipped": r"(\d+)\s+skipped",
        "errors": r"(\d+)\s+errors?",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, tail)
        if match:
            summary[key] = int(match.group(1))
    return summary


def _read_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_csv(path: Path) -> List[Dict[str, str]] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return None


def _execute(spec: RunSpec, out_dir: Path, dry_run: bool) -> Dict[str, Any]:
    log_path = out_dir / f"{spec.key}.log"
    result: Dict[str, Any] = {
        "key": spec.key,
        "description": spec.description,
        "command": " ".join(spec.command),
        "log_file": str(log_path),
        "status": "not_started",
        "return_code": None,
        "duration_seconds": 0.0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    if dry_run:
        result["status"] = "dry_run"
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        return result

    start = time.perf_counter()
    proc = subprocess.run(
        spec.command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    duration = time.perf_counter() - start
    output_text = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    log_path.write_text(output_text, encoding="utf-8")

    result["status"] = "success" if proc.returncode == 0 else "failed"
    result["return_code"] = proc.returncode
    result["duration_seconds"] = round(duration, 3)
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    if "pytest" in " ".join(spec.command):
        result["pytest_summary"] = _parse_pytest_summary(output_text)
    return result


def _collect_artifacts(repo_root: Path) -> Dict[str, Any]:
    artifacts = {
        "chapter8_report": str(repo_root / "docs" / "evaluation" / "chapter_8_report.md"),
        "api_simulation_comparison_json": str(repo_root / "results" / "api_simulation_comparison" / "comparison.json"),
        "api_simulation_comparison_md": str(repo_root / "results" / "api_simulation_comparison" / "comparison.md"),
        "baseline_results_csv": str(repo_root / "results" / "ch8_baselines" / "results.csv"),
        "baseline_detailed_json": str(repo_root / "results" / "ch8_baselines" / "detailed_results.json"),
        "train_metrics_json": str(repo_root / "results" / "ch8_train" / "metrics.json"),
    }

    parsed: Dict[str, Any] = {
        "api_simulation_comparison": _read_json(Path(artifacts["api_simulation_comparison_json"])),
        "baseline_results_rows": _read_csv(Path(artifacts["baseline_results_csv"])),
        "train_metrics": _read_json(Path(artifacts["train_metrics_json"])),
    }
    return {"paths": artifacts, "parsed": parsed}


def _dependency_setup(repo_root: Path, out_dir: Path, dry_run: bool, skip_install_deps: bool) -> Dict[str, Any]:
    log_path = out_dir / "dependency_setup.log"
    result: Dict[str, Any] = {
        "key": "dependency_setup",
        "description": "Ensure runtime dependencies for Chapter 8 command set are available",
        "command": "",
        "log_file": str(log_path),
        "status": "not_started",
        "return_code": None,
        "duration_seconds": 0.0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    pytest_missing = importlib.util.find_spec("pytest") is None
    needs_install = pytest_missing and not skip_install_deps

    if skip_install_deps:
        result["status"] = "skipped"
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        return result

    install_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(repo_root / "requirements.txt"),
        "-r",
        str(repo_root / "requirements_gym.txt"),
        "-r",
        str(repo_root / "requirements_integration_tests.txt"),
        "scikit-learn",
    ]
    result["command"] = " ".join(install_cmd)

    if dry_run:
        result["status"] = "dry_run" if needs_install else "already_satisfied"
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        return result

    if not needs_install:
        result["status"] = "already_satisfied"
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        return result

    start = time.perf_counter()
    proc = subprocess.run(
        install_cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    duration = time.perf_counter() - start
    output_text = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    log_path.write_text(output_text, encoding="utf-8")

    result["status"] = "success" if proc.returncode == 0 else "failed"
    result["return_code"] = proc.returncode
    result["duration_seconds"] = round(duration, 3)
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all Chapter 8 result commands and store structured outputs.")
    parser.add_argument(
        "--output-dir",
        default=f"results/ch8_run_{_timestamp()}",
        help="Directory for structured run outputs (default: results/ch8_run_<timestamp>)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=["functional", "nonfunctional", "model", "performance", "simulation", "baselines", "training"],
        help="Run only selected groups.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print/record what would run without executing commands.")
    parser.add_argument(
        "--skip-install-deps",
        action="store_true",
        help="Skip dependency bootstrap step (default is to auto-install if pytest is missing).",
    )
    args = parser.parse_args()

    out_dir = (REPO_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    groups: Dict[str, List[RunSpec]] = {
        "functional": [
            RunSpec(
                key="functional",
                description="Functional requirements suite (FR-01..FR-20)",
                command=[sys.executable, "-m", "pytest", "tests/functional/test_functional_requirements.py", "-ra"],
            )
        ],
        "nonfunctional": [
            RunSpec(
                key="nonfunctional",
                description="Non-functional requirements suite (NFR-01..NFR-08)",
                command=[sys.executable, "-m", "pytest", "tests/nonfunctional/test_nfr.py", "-s", "-ra"],
            )
        ],
        "model": [
            RunSpec(
                key="model",
                description="Model evaluation suite (Markov + DQN checks)",
                command=[sys.executable, "-m", "pytest", "tests/model/test_model_evaluation.py", "-s", "-ra"],
            )
        ],
        "performance": [
            RunSpec(
                key="performance",
                description="Performance latency/throughput benchmark tests",
                command=[
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/performance/test_latency.py",
                    "tests/performance/test_throughput.py",
                    "-s",
                    "-q",
                ],
            )
        ],
        "simulation": [
            RunSpec(
                key="simulation_compare",
                description="A/B API simulation comparison",
                command=[sys.executable, "scripts/api_simulation_compare.py"],
            )
        ],
        "baselines": [
            RunSpec(
                key="baseline_compare",
                description="Baseline policy comparison for benchmark table support",
                command=[
                    sys.executable,
                    "scripts/compare_baselines.py",
                    "--episodes",
                    "30",
                    "--output",
                    "results/ch8_baselines",
                    "--save-json",
                ],
            )
        ],
        "training": [
            RunSpec(
                key="training_attempt",
                description="DQN training run attempted for Chapter 8 artifact completeness",
                command=[sys.executable, "scripts/train.py", "--episodes", "60", "--output", "results/ch8_train"],
            )
        ],
    }

    selected = args.only if args.only else list(groups.keys())
    specs: List[RunSpec] = []
    for group in selected:
        specs.extend(groups[group])

    summary: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "output_dir": str(out_dir),
        "dry_run": bool(args.dry_run),
        "selected_groups": selected,
        "runs": [],
    }

    dep_result = _dependency_setup(REPO_ROOT, out_dir, args.dry_run, args.skip_install_deps)
    summary["runs"].append(dep_result)

    for spec in specs:
        run_result = _execute(spec, out_dir, args.dry_run)
        summary["runs"].append(run_result)

    summary["artifacts"] = _collect_artifacts(REPO_ROOT)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nStructured summary written to: {summary_path}")
    for run in summary["runs"]:
        print(f"- {run['key']}: {run['status']} (rc={run['return_code']}) log={run['log_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Single-command runner for scheduler + training + performance comparison integration suite.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    target = "tests/test_scheduler_training_performance_integration.py"

    cmd = [sys.executable, "-m", "pytest", "-v", target]
    result = subprocess.run(cmd, cwd=repo_root)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

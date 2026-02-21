"""
Benchmark reporter for performance test results.

Usage (CLI)::

    python -m tests.performance.benchmark_report \\
        --results benchmark.json \\
        --baseline baselines/benchmark_baseline.json \\
        --output reports/performance_report.html

Usage (API)::

    from tests.performance.benchmark_report import BenchmarkReporter

    reporter = BenchmarkReporter()
    reporter.collect_results("benchmark.json")
    reporter.compare_with_baseline("baselines/benchmark_baseline.json")
    reporter.generate_report("reports/performance_report.html")
    reporter.save_as_baseline("baselines/benchmark_baseline.json")
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """A single benchmark measurement."""
    name: str
    ops_per_sec: Optional[float] = None
    mean_ms: Optional[float] = None
    p50_ms: Optional[float] = None
    p90_ms: Optional[float] = None
    p95_ms: Optional[float] = None
    p99_ms: Optional[float] = None
    min_ms: Optional[float] = None
    max_ms: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Comparison of a benchmark against its baseline."""
    name: str
    current: BenchmarkResult
    baseline: BenchmarkResult
    ops_change_pct: Optional[float] = None   # positive = improvement
    p99_change_pct: Optional[float] = None   # negative = improvement (lower is better)
    status: str = "unknown"                  # "improved", "regressed", "stable", "unknown"

    def _compute(self):
        """Compute percentage changes and determine status."""
        THRESHOLD_PCT = 5.0  # ±5 % is "stable"

        # Throughput: higher is better
        if (self.current.ops_per_sec is not None
                and self.baseline.ops_per_sec is not None
                and self.baseline.ops_per_sec > 0):
            self.ops_change_pct = (
                (self.current.ops_per_sec - self.baseline.ops_per_sec)
                / self.baseline.ops_per_sec * 100
            )

        # Latency: lower is better (compare p99)
        if (self.current.p99_ms is not None
                and self.baseline.p99_ms is not None
                and self.baseline.p99_ms > 0):
            self.p99_change_pct = (
                (self.current.p99_ms - self.baseline.p99_ms)
                / self.baseline.p99_ms * 100
            )

        # Determine overall status
        changes = []
        if self.ops_change_pct is not None:
            changes.append(self.ops_change_pct)
        if self.p99_change_pct is not None:
            changes.append(-self.p99_change_pct)   # flip sign (lower p99 = better)

        if not changes:
            self.status = "unknown"
        elif all(c > THRESHOLD_PCT for c in changes):
            self.status = "improved"
        elif any(c < -THRESHOLD_PCT for c in changes):
            self.status = "regressed"
        else:
            self.status = "stable"


# ---------------------------------------------------------------------------
# BenchmarkReporter
# ---------------------------------------------------------------------------

class BenchmarkReporter:
    """
    Collects pytest-benchmark results, optionally compares against a
    baseline, and generates an HTML (or JSON) report.
    """

    def __init__(self):
        self.results:     List[BenchmarkResult]    = []
        self.comparisons: List[ComparisonResult]   = []
        self._baseline:   Dict[str, BenchmarkResult] = {}
        self._raw:        Dict[str, Any]           = {}
        self._timestamp   = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_results(self, results_path: str) -> "BenchmarkReporter":
        """
        Parse a pytest-benchmark JSON file (produced by --benchmark-json=).

        Args:
            results_path: Path to the benchmark JSON file.

        Returns:
            self  (for method chaining)
        """
        path = Path(results_path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark results not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            self._raw = json.load(fh)

        self.results = self._parse_pytest_benchmark(self._raw)
        return self

    def compare_with_baseline(self, baseline_path: str) -> "BenchmarkReporter":
        """
        Load a saved baseline and compute percentage changes.

        Args:
            baseline_path: Path to a previously saved baseline JSON file.

        Returns:
            self
        """
        path = Path(baseline_path)
        if not path.exists():
            print(f"[BenchmarkReporter] Baseline not found at {path}; skipping comparison.")
            return self

        with open(path, "r", encoding="utf-8") as fh:
            raw_baseline = json.load(fh)

        baseline_list = self._parse_pytest_benchmark(raw_baseline)
        self._baseline = {r.name: r for r in baseline_list}

        self.comparisons = []
        for result in self.results:
            if result.name in self._baseline:
                comp = ComparisonResult(
                    name=result.name,
                    current=result,
                    baseline=self._baseline[result.name],
                )
                comp._compute()
                self.comparisons.append(comp)

        return self

    def generate_report(self, output_path: str,
                        fmt: str = "html") -> "BenchmarkReporter":
        """
        Generate a benchmark report.

        Args:
            output_path: Where to write the report.
            fmt:         "html" or "json".

        Returns:
            self
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "json":
            self._write_json_report(out)
        else:
            self._write_html_report(out)

        print(f"[BenchmarkReporter] Report written to {out}")
        return self

    def save_as_baseline(self, path: str) -> "BenchmarkReporter":
        """
        Save the current raw benchmark data as a new baseline.

        Args:
            path: Destination path for the baseline JSON.

        Returns:
            self
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if self._raw:
            with open(dest, "w", encoding="utf-8") as fh:
                json.dump(self._raw, fh, indent=2)
        else:
            # Fall back to serialising collected BenchmarkResults
            payload = {
                "benchmarks": [asdict(r) for r in self.results],
                "saved_at": self._timestamp,
            }
            with open(dest, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)

        print(f"[BenchmarkReporter] Baseline saved to {dest}")
        return self

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_pytest_benchmark(
        self, raw: Dict[str, Any]
    ) -> List[BenchmarkResult]:
        """
        Convert a pytest-benchmark JSON structure into BenchmarkResult objects.

        Supports both the standard ``benchmarks`` list format and a simple
        ``{"benchmarks": [{"name": ..., ...}]}`` dict format.
        """
        results: List[BenchmarkResult] = []
        entries = raw.get("benchmarks", [])

        for entry in entries:
            name  = entry.get("name") or entry.get("fullname") or "unknown"
            stats = entry.get("stats", entry)  # fallback: entry itself is stats

            results.append(BenchmarkResult(
                name      = name,
                ops_per_sec = stats.get("ops_per_sec"),
                mean_ms   = _to_ms(stats.get("mean")),
                p50_ms    = _to_ms(stats.get("median")),
                p90_ms    = _to_ms(stats.get("q_90") or stats.get("p90_ms")),
                p95_ms    = _to_ms(stats.get("q_95") or stats.get("p95_ms")),
                p99_ms    = _to_ms(stats.get("q_99") or stats.get("p99_ms")),
                min_ms    = _to_ms(stats.get("min")),
                max_ms    = _to_ms(stats.get("max")),
                extra     = {k: v for k, v in stats.items()
                             if k not in {"mean", "median", "min", "max",
                                          "ops_per_sec", "q_90", "q_95", "q_99"}},
            ))

        return results

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _write_json_report(self, path: Path):
        payload = {
            "generated_at": self._timestamp,
            "results": [asdict(r) for r in self.results],
            "comparisons": [
                {
                    "name":            c.name,
                    "status":          c.status,
                    "ops_change_pct":  c.ops_change_pct,
                    "p99_change_pct":  c.p99_change_pct,
                    "current":         asdict(c.current),
                    "baseline":        asdict(c.baseline),
                }
                for c in self.comparisons
            ],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def _write_html_report(self, path: Path):
        rows_results   = self._html_results_table()
        rows_compare   = self._html_comparison_table()
        compare_section = (
            f"<h2>Comparison with Baseline</h2>{rows_compare}"
            if self.comparisons
            else "<p><em>No baseline available for comparison.</em></p>"
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Performance Benchmark Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
    h1   {{ color: #1a73e8; }}
    h2   {{ color: #444; border-bottom: 2px solid #1a73e8; padding-bottom: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 2em; }}
    th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: right; }}
    th   {{ background: #1a73e8; color: #fff; text-align: center; }}
    tr:nth-child(even) {{ background: #f5f5f5; }}
    td.name {{ text-align: left; font-family: monospace; font-size: 0.85em; }}
    .improved {{ color: green; font-weight: bold; }}
    .regressed {{ color: red; font-weight: bold; }}
    .stable   {{ color: #555; }}
    .unknown  {{ color: #aaa; }}
    .footer {{ font-size: 0.8em; color: #888; margin-top: 3em; }}
  </style>
</head>
<body>
  <h1>Performance Benchmark Report</h1>
  <p>Generated: {self._timestamp} UTC</p>

  <h2>All Benchmarks</h2>
  {rows_results}

  {compare_section}

  <div class="footer">
    Generated by <code>tests.performance.benchmark_report</code>
  </div>
</body>
</html>"""

        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)

    def _html_results_table(self) -> str:
        if not self.results:
            return "<p><em>No results collected.</em></p>"

        header = (
            "<tr>"
            "<th>Test</th><th>ops/sec</th>"
            "<th>mean (ms)</th><th>p50 (ms)</th>"
            "<th>p90 (ms)</th><th>p95 (ms)</th>"
            "<th>p99 (ms)</th><th>min (ms)</th><th>max (ms)</th>"
            "</tr>"
        )
        rows = "\n".join(
            f"<tr>"
            f"<td class='name'>{r.name}</td>"
            f"<td>{_fmt(r.ops_per_sec, '.0f')}</td>"
            f"<td>{_fmt(r.mean_ms)}</td>"
            f"<td>{_fmt(r.p50_ms)}</td>"
            f"<td>{_fmt(r.p90_ms)}</td>"
            f"<td>{_fmt(r.p95_ms)}</td>"
            f"<td>{_fmt(r.p99_ms)}</td>"
            f"<td>{_fmt(r.min_ms)}</td>"
            f"<td>{_fmt(r.max_ms)}</td>"
            f"</tr>"
            for r in self.results
        )
        return f"<table>{header}\n{rows}\n</table>"

    def _html_comparison_table(self) -> str:
        if not self.comparisons:
            return ""

        header = (
            "<tr>"
            "<th>Test</th><th>Status</th>"
            "<th>ops/sec Δ%</th><th>p99 Δ%</th>"
            "<th>Current ops/sec</th><th>Baseline ops/sec</th>"
            "<th>Current p99</th><th>Baseline p99</th>"
            "</tr>"
        )
        rows = "\n".join(
            f"<tr>"
            f"<td class='name'>{c.name}</td>"
            f"<td class='{c.status}'>{c.status}</td>"
            f"<td>{_fmt_pct(c.ops_change_pct)}</td>"
            f"<td>{_fmt_pct(c.p99_change_pct, invert=True)}</td>"
            f"<td>{_fmt(c.current.ops_per_sec, '.0f')}</td>"
            f"<td>{_fmt(c.baseline.ops_per_sec, '.0f')}</td>"
            f"<td>{_fmt(c.current.p99_ms)}</td>"
            f"<td>{_fmt(c.baseline.p99_ms)}</td>"
            f"</tr>"
            for c in self.comparisons
        )
        return f"<table>{header}\n{rows}\n</table>"


# ---------------------------------------------------------------------------
# Formatting utilities
# ---------------------------------------------------------------------------

def _to_ms(value: Optional[float]) -> Optional[float]:
    """Convert seconds → milliseconds, leaving None unchanged."""
    if value is None:
        return None
    # pytest-benchmark stores times in seconds; check magnitude
    if value > 100:
        return value  # already in ms (our custom format)
    return value * 1000


def _fmt(value: Optional[float], fmt: str = ".4f") -> str:
    if value is None:
        return "—"
    return f"{value:{fmt}}"


def _fmt_pct(value: Optional[float], invert: bool = False) -> str:
    """Format a percentage change.  invert=True means lower is better."""
    if value is None:
        return "—"
    effective = -value if invert else value
    colour = "green" if effective >= 0 else "red"
    sign   = "+" if effective >= 0 else ""
    return f"<span style='color:{colour}'>{sign}{value:.1f}%</span>"


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a performance benchmark report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results",  required=True,
                   help="Path to pytest-benchmark JSON output file.")
    p.add_argument("--baseline", default=None,
                   help="Path to a previously saved baseline JSON file.")
    p.add_argument("--output",   default="reports/performance_report.html",
                   help="Output path for the report.")
    p.add_argument("--format",   choices=["html", "json"], default="html",
                   help="Report format.")
    p.add_argument("--save-baseline", metavar="PATH", default=None,
                   help="If set, also save current results as a new baseline.")
    return p


def main(argv: Optional[list] = None):
    args = _build_parser().parse_args(argv)

    reporter = BenchmarkReporter()
    reporter.collect_results(args.results)

    if args.baseline:
        reporter.compare_with_baseline(args.baseline)

    reporter.generate_report(args.output, fmt=args.format)

    if args.save_baseline:
        reporter.save_as_baseline(args.save_baseline)


if __name__ == "__main__":
    main()


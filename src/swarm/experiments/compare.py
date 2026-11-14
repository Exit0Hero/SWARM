"""Command-line entry point: paired comparison of two stored experiment results.

Example::

    python -m swarm.experiments.compare --baseline runs/a.json --candidate runs/b.json \\
        --out runs/a_vs_b.json

Refuses (exit code 2) unless both experiments used the same config hash, the same seed list in
the same order, and the same designation (ADR-0009).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from swarm.evaluation.statistics import BootstrapSpec
from swarm.experiments.experiment import ExperimentResult, compare_experiments


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m swarm.experiments.compare")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args(argv)
    try:
        a = ExperimentResult.from_dict(json.loads(args.baseline.read_text(encoding="utf-8")))
        b = ExperimentResult.from_dict(json.loads(args.candidate.read_text(encoding="utf-8")))
        spec = BootstrapSpec(args.resamples, args.confidence, args.bootstrap_seed)
        comparison = compare_experiments(a, b, bootstrap=spec)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "excluded": comparison["excluded_failed_seeds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line entry point: run a multi-seed experiment and write a JSON summary.

Example::

    python -m swarm.experiments.run_experiment --config configs/m2_default.json \\
        --policy random_valid --seeds 0-29 --out runs/m2_random.json

Exit codes: 0 all episodes completed; 3 result written but some episodes failed (listed in the
result); 2 invalid input.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from swarm.evaluation.statistics import BootstrapSpec
from swarm.experiments.experiment import parse_seeds, run_experiment
from swarm.experiments.run_episode import POLICIES
from swarm.scenarios.config import ConfigError, ScenarioConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m swarm.experiments.run_experiment",
        description="Run one policy on one config over a list of seeds and summarize metrics.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy", choices=sorted(POLICIES), default="random_valid")
    parser.add_argument("--seeds", required=True, help='e.g. "0-29" or "0-9,20-29"')
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--designation", choices=["evaluation", "development"], default="evaluation"
    )
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = ScenarioConfig.from_json(args.config.read_text(encoding="utf-8"))
        seeds = parse_seeds(args.seeds)
        spec = BootstrapSpec(args.resamples, args.confidence, args.bootstrap_seed)
        result = run_experiment(
            config, POLICIES[args.policy], seeds, bootstrap=spec, designation=args.designation
        )
    except (OSError, ConfigError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(result.to_json(), encoding="utf-8")
    record = result.to_dict()
    brief = {
        "out": str(args.out),
        "config_hash": record["config_hash"],
        "policy": record["policy"],
        "n_seeds": len(result.seeds),
        "complete": result.complete,
        "failed_seeds": record["failed_seeds"],
        "end_reasons": record["end_reasons"],
        "summary": {
            name: {k: s[k] for k in ("n_defined", "mean", "ci_low", "ci_high")}
            for name, s in record["summary"].items()
        },
    }
    print(json.dumps(brief, indent=2, sort_keys=True))
    return 0 if result.complete else 3


if __name__ == "__main__":
    raise SystemExit(main())

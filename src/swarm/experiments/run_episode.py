"""Command-line entry point: run one episode and write a JSON result.

Example::

    python -m swarm.experiments.run_episode --config configs/m1_default.json --seed 0 \\
        --out runs/m1_seed0.json --events runs/m1_seed0.events.jsonl --render
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from swarm.experiments.runner import run_episode, write_result
from swarm.policies.registry import POLICIES
from swarm.scenarios.config import ConfigError, ScenarioConfig
from swarm.scenarios.generator import ScenarioError
from swarm.visualization.text import render_snapshot_text

__all__ = ["POLICIES", "build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m swarm.experiments.run_episode",
        description="Run one SWARM episode and write a reproducible JSON result.",
    )
    parser.add_argument("--config", type=Path, required=True, help="scenario config JSON")
    parser.add_argument("--seed", type=int, required=True, help="root seed (non-negative)")
    parser.add_argument("--policy", choices=sorted(POLICIES), default="random_valid")
    parser.add_argument("--out", type=Path, required=True, help="result JSON path")
    parser.add_argument("--events", type=Path, help="optional event stream path (JSON Lines)")
    parser.add_argument(
        "--designation", choices=["evaluation", "development"], default="evaluation"
    )
    parser.add_argument("--render", action="store_true", help="print the final map")
    parser.add_argument(
        "--show-survivors", action="store_true", help="include ground truth in the render"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns 0 on success and 2 on invalid input (message on stderr)."""
    args = build_parser().parse_args(argv)
    try:
        config = ScenarioConfig.from_json(args.config.read_text(encoding="utf-8"))
        result = run_episode(config, POLICIES[args.policy], args.seed, designation=args.designation)
    except (OSError, ConfigError, ScenarioError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    write_result(result, args.out, args.events)
    if args.render:
        print(render_snapshot_text(result.final_snapshot, show_survivors=args.show_survivors))
    summary = {
        "out": str(args.out),
        "event_stream_sha256": result.event_stream_sha256,
        "scenario_sha256": result.scenario_sha256,
        "end_reason": result.end_reason,
        "steps": result.steps,
        "metrics": result.metrics.to_dict(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

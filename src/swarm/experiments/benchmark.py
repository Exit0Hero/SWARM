"""Performance benchmark of the simulator and policies (exploratory; wall clock).

Each case is one config, one policy and a fixed list of development seeds. The case runs
``--repetitions`` times through the normal runner, and is timed with ``time.perf_counter``
around whole episodes. The report gives, per case:

- the median over repetitions of total time / total steps (``ms_per_step``) and episodes per
  second;
- the steps run (identical in every repetition, because episodes are deterministic).

The cases vary one factor at a time from the ``basic`` example:

- extension overhead: no channel, belief sharing, auction, + perception, + failures, learned actor;
- robots per type (1, 2, 4, 8) and map side (20, 30, 40), both with ``auction_comm``;
- duration: the time limit at a recall target of 1.0.

Runtimes depend on the machine, so the report records Python, NumPy, OS, CPU count and the git
commit. Results are exploratory and are not compared across machines.

Example::

    python -m swarm.experiments.benchmark --out benchmark.json --repetitions 3
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from swarm.examples import example_text
from swarm.experiments.runner import run_episode
from swarm.experiments.study import deep_merge
from swarm.policies.registry import POLICIES
from swarm.scenarios.config import ScenarioConfig


def _config(overrides: dict[str, Any]) -> ScenarioConfig:
    base = json.loads(example_text("basic"))
    return ScenarioConfig.from_dict(deep_merge(base, overrides))


def _team(per_type: int) -> dict[str, Any]:
    roster = [{"agent_type": "drone"}] * per_type + [{"agent_type": "ground"}] * per_type
    return {"roster": roster}


def cases(models: Path | None) -> list[tuple[str, str, ScenarioConfig, Any]]:
    """(group, label, config, policy factory) for every benchmark case."""
    out: list[tuple[str, str, ScenarioConfig, Any]] = []
    basic = _config({})
    out.append(("overhead", "frontier, no channel", _config({"comms": None}), POLICIES["frontier"]))
    out.append(("overhead", "frontier_shared (belief sharing)", basic, POLICIES["frontier_shared"]))
    out.append(("overhead", "auction_comm", basic, POLICIES["auction_comm"]))
    noisy = {
        "perception": {
            "model": "noisy",
            "range_falloff": 0.3,
            "terrain_error": 0.05,
            "hazard_miss": 0.2,
            "hazard_false": 0.05,
            "prior": 0.02,
        }
    }
    out.append(("overhead", "auction_comm + perception", _config(noisy), POLICIES["auction_comm"]))
    fail = {"failures": {"count": 2, "start": 5, "end": 25}}
    out.append(("overhead", "auction_comm + failures", _config(fail), POLICIES["auction_comm"]))
    if models is not None:
        from swarm.dashboard.service import learned_catalog

        learned = learned_catalog(models)
        if "learned:mappo_s0" in learned:
            out.append(("overhead", "learned:mappo_s0", basic, learned["learned:mappo_s0"]))
    for n in (1, 2, 4, 8):
        out.append(("robots", f"{n}+{n} robots", _config(_team(n)), POLICIES["auction_comm"]))
    for side in (20, 30, 40):
        scale = (side / 20) ** 2
        config = _config(
            {
                "width": side,
                "height": side,
                "survivor_count": round(8 * scale),
                "time_limit": 10 * side,
            }
        )
        out.append(("map", f"{side}x{side} map", config, POLICIES["auction_comm"]))
    for limit in (50, 100, 200):
        config = _config({"time_limit": limit, "recall_target": 1.0})
        out.append(("duration", f"time limit {limit}", config, POLICIES["auction_comm"]))
    return out


def environment() -> dict[str, Any]:
    import subprocess

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        dirty: bool | None = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    import swarm

    return {
        "swarm": swarm.__version__,
        "commit": commit,
        "dirty": dirty,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def run(seeds: Sequence[int], repetitions: int, models: Path | None) -> dict[str, Any]:
    results = []
    for group, label, config, factory in cases(models):
        times, steps = [], 0
        for _ in range(repetitions):
            steps = 0
            start = time.perf_counter()
            for seed in seeds:
                steps += run_episode(config, factory, seed, designation="development").steps
            times.append(time.perf_counter() - start)
        median = statistics.median(times)
        results.append(
            {
                "group": group,
                "case": label,
                "config_hash": config.config_hash()[:12],
                "robots": len(config.roster),
                "cells": config.width * config.height,
                "episodes": len(seeds),
                "steps": steps,
                "seconds_median": median,
                "seconds_all": times,
                "ms_per_step": 1000 * median / max(steps, 1),
                "episodes_per_second": len(seeds) / median if median else None,
            }
        )
        print(
            f"{group:<9}{label:<36}{results[-1]['ms_per_step']:8.2f} ms/step"
            f"{results[-1]['episodes_per_second']:8.1f} ep/s",
            file=sys.stderr,
        )
    return {
        "kind": "benchmark",
        "note": "exploratory wall-clock timings on one machine; not a validated result",
        "method": "median over repetitions of total time for all seeds; time.perf_counter",
        "seeds": list(seeds),
        "repetitions": repetitions,
        "environment": environment(),
        "results": results,
    }


def to_markdown(report: dict[str, Any]) -> str:
    env = report["environment"]
    lines = [
        f"Environment: Python {env['python']}, NumPy {env['numpy']}, {env['platform']}, "
        f"{env['cpu_count']} CPUs, swarm {env['swarm']} ({env['commit']}). Seeds "
        f"{report['seeds'][0]}-{report['seeds'][-1]}, {report['repetitions']} repetitions.",
        "",
        "| Group | Case | Robots | Cells | Steps | ms/step | Episodes/s |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in report["results"]:
        lines.append(
            f"| {r['group']} | {r['case']} | {r['robots']} | {r['cells']} | {r['steps']} "
            f"| {r['ms_per_step']:.2f} | {r['episodes_per_second']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="swarm benchmark", description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--seeds", default="1000-1004", help="development seeds (default %(default)s)"
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--out", type=Path, help="write the JSON report here")
    parser.add_argument("--markdown", type=Path, help="also write a Markdown table")
    parser.add_argument("--models", type=Path, default=Path("docs/results/phase4/models"))
    args = parser.parse_args(argv)
    from swarm.experiments.experiment import parse_seeds

    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    report = run(list(parse_seeds(args.seeds)), args.repetitions, args.models)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(to_markdown(report), encoding="utf-8")
    print(to_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

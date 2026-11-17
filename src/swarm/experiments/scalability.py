"""Scalability benchmark (Phase 7): runtime and traffic as the problem grows.

Every run varies one factor from a base config: team size (drones and ground robots in equal
number), map side, or survivor count. For each point and policy it runs ``episodes`` episodes on
development seeds and records:

- wall-clock seconds per episode and per simulated step (descriptive; machine dependent);
- message units per step (communication traffic, deterministic);
- outcome metrics (deterministic).

With ``--profile``, one extra episode per point is run under ``cProfile`` and the five functions
with the most cumulative time inside ``swarm`` are recorded: the bottlenecks.

Example::

    python -m swarm.experiments.scalability --config configs/phase3/nominal.json \\
        --out docs/results/phase7/scalability.json --policies frontier_shared auction_comm
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from swarm.experiments.runner import environment_metadata, run_episode
from swarm.policies.registry import POLICIES
from swarm.scenarios.config import AgentSpec, ScenarioConfig

TEAM_SIZES = (1, 2, 4, 8)  # per agent type
MAP_SIDES = (20, 30, 40, 60)
SURVIVORS = (4, 8, 16, 32)
DEV_SEEDS = tuple(range(1000, 1020))


def variant(base: ScenarioConfig, factor: str, value: int) -> ScenarioConfig:
    """``base`` with one factor changed. Survivor density is kept when the map grows."""
    if factor == "team":
        roster = tuple(AgentSpec("drone") for _ in range(value)) + tuple(
            AgentSpec("ground") for _ in range(value)
        )
        return replace(base, roster=roster)
    if factor == "map":
        density = base.survivor_count / (base.height * base.width)
        return replace(
            base,
            height=value,
            width=value,
            base_position=None,
            survivor_count=max(1, round(density * value * value)),
            time_limit=max(base.time_limit, base.time_limit * value // base.height),
        )
    if factor == "survivors":
        return replace(base, survivor_count=value)
    raise ValueError(f"unknown factor {factor!r}")


def _hotspots(profile: cProfile.Profile, top: int = 5) -> list[dict[str, Any]]:
    stats = pstats.Stats(profile)
    rows = []
    for (path, line, name), (_, _, _, cumulative, _) in stats.stats.items():  # type: ignore[attr-defined]
        if "/swarm/" in path and name not in ("run_episode", "main"):
            rows.append(
                {"function": f"{Path(path).name}:{line}:{name}", "cumulative_s": cumulative}
            )
    rows.sort(key=lambda r: -float(r["cumulative_s"]))
    return rows[:top]


def benchmark(
    base: ScenarioConfig,
    policies: Sequence[str],
    *,
    episodes: int,
    profile: bool,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for factor, values in (("team", TEAM_SIZES), ("map", MAP_SIDES), ("survivors", SURVIVORS)):
        for value in values:
            config = variant(base, factor, value)
            for policy in policies:
                times, steps, units, recall, success = [], [], [], [], []
                for seed in DEV_SEEDS[:episodes]:
                    start = time.perf_counter()
                    r = run_episode(config, POLICIES[policy], seed, designation="development")
                    times.append(time.perf_counter() - start)
                    steps.append(r.steps)
                    m = r.metrics.to_dict()
                    recall.append(float(m["recall"] or 0.0))
                    success.append(float(bool(m["mission_success"])))
                    cm = r.comm_metrics or {}
                    units.append(float(cm.get("message_units_per_step") or 0.0))
                point: dict[str, Any] = {
                    "factor": factor,
                    "value": value,
                    "policy": policy,
                    "agents": len(config.roster),
                    "cells": config.height * config.width,
                    "survivors": config.survivor_count,
                    "episodes": len(times),
                    "seconds_per_episode": float(np.mean(times)),
                    "ms_per_step": 1000.0 * float(np.sum(times)) / max(1, int(np.sum(steps))),
                    "steps_mean": float(np.mean(steps)),
                    "message_units_per_step": float(np.mean(units)),
                    "recall_mean": float(np.mean(recall)),
                    "mission_success_rate": float(np.mean(success)),
                }
                if profile:
                    prof = cProfile.Profile()
                    prof.enable()
                    run_episode(config, POLICIES[policy], DEV_SEEDS[0], designation="development")
                    prof.disable()
                    point["hotspots"] = _hotspots(prof)
                points.append(point)
                print(
                    f"{factor}={value} {policy}: {point['ms_per_step']:.2f} ms/step",
                    file=sys.stderr,
                )
    return points


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m swarm.experiments.scalability")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", default=["frontier_shared", "auction_comm"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args(argv)
    unknown = sorted(set(args.policies) - set(POLICIES))
    if unknown or not 1 <= args.episodes <= len(DEV_SEEDS):
        print(f"error: unknown policies {unknown} or episodes outside 1..20", file=sys.stderr)
        return 2
    base = ScenarioConfig.from_json(args.config.read_text("utf-8"))
    points = benchmark(base, args.policies, episodes=args.episodes, profile=args.profile)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "base_config_hash": base.config_hash(),
        "seeds": list(DEV_SEEDS[: args.episodes]),
        "designation": "development",
        "note": "runtimes are wall clock on one machine; traffic and outcomes are deterministic",
        "points": points,
        **environment_metadata(),
    }
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

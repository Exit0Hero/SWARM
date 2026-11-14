"""Run every registered baseline on identical seeds and the pre-registered paired comparisons.

Example::

    python -m swarm.experiments.run_baselines --config configs/m2_default.json \\
        --seeds 0-49 --out-dir runs/phase2/m2

Writes one experiment JSON per policy, ``comparisons.json`` with the pairs pre-registered in
``docs/16_PHASE2_BASELINE_STUDY.md``, and ``summary.md`` with tables generated from those files.
Exit codes: 0 all episodes completed; 3 some episodes failed (listed); 2 invalid input.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from swarm.evaluation.statistics import BootstrapSpec
from swarm.experiments.experiment import (
    ExperimentResult,
    compare_experiments,
    parse_seeds,
    run_experiment,
)
from swarm.policies.registry import POLICIES
from swarm.scenarios.config import ConfigError, ScenarioConfig

# (baseline, candidate) pairs pre-registered in docs/16 (Questions 1-5).
PREREGISTERED_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("random_valid", "astar_waypoint"),
    ("astar_waypoint", "frontier"),
    ("frontier", "prioritized_frontier"),
    ("prioritized_frontier", "assignment"),
    ("frontier", "frontier_hazard_aware"),
    ("assignment", "assignment_hazard_aware"),
    ("assignment", "assignment_oracle"),
)
TABLE_METRICS: Final[tuple[str, ...]] = (
    "recall",
    "mission_success",
    "time_to_recall_target",
    "coverage_sensed",
    "energy_used",
    "move_conflicts",
    "invalid_actions",
    "agents_disabled_energy",
    "agents_disabled_hazard",
    "hazard_exposures",
    "precision",
)
COMPARISON_METRICS: Final[tuple[str, ...]] = (
    "recall",
    "mission_success",
    "coverage_sensed",
    "hazard_exposures",
    "agents_disabled_hazard",
    "energy_used",
)


def _fmt(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def summary_markdown(
    results: dict[str, ExperimentResult], comparisons: list[dict[str, Any]]
) -> str:
    """Tables generated from experiment and comparison records (no hand-typed numbers)."""
    first = next(iter(results.values()))
    lines = [
        f"Config hash `{first.config_hash[:16]}`, seeds {first.seeds[0]}-{first.seeds[-1]} "
        f"(n = {len(first.seeds)}), designation `{first.designation}`, commit "
        f"`{(first.metadata.get('code', {}).get('commit') or 'unknown')[:8]}` "
        f"(dirty: {first.metadata.get('code', {}).get('dirty')}).",
        "",
        "Mean [95% bootstrap CI]; n_defined shown when below n. Oracle rows are references only.",
        "",
        "| policy | information | complete | " + " | ".join(TABLE_METRICS) + " |",
        "|---|---|---|" + "---|" * len(TABLE_METRICS),
    ]
    for name, res in results.items():
        cells = []
        for metric in TABLE_METRICS:
            s = res.summary[metric]
            text = f"{_fmt(s.mean)} [{_fmt(s.ci_low)}, {_fmt(s.ci_high)}]"
            if s.n_defined < s.n_seeds:
                text += f" (n={s.n_defined})"
            cells.append(text)
        lines.append(
            f"| {name} | {res.policy['information']} | {res.complete} | " + " | ".join(cells) + " |"
        )
    lines += [
        "",
        "Paired differences (candidate - baseline), mean [95% bootstrap CI], n pairs. "
        "Exploratory (no multiplicity correction).",
        "",
        "| baseline → candidate | " + " | ".join(COMPARISON_METRICS) + " |",
        "|---|" + "---|" * len(COMPARISON_METRICS),
    ]
    for cmp in comparisons:
        cells = []
        for metric in COMPARISON_METRICS:
            p = cmp["metrics"][metric]
            text = f"{_fmt(p['mean_difference'])} [{_fmt(p['ci_low'])}, {_fmt(p['ci_high'])}]"
            cells.append(f"{text} (n={p['n_pairs']})")
        label = f"{cmp['baseline_policy']['name']} -> {cmp['candidate_policy']['name']}"
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m swarm.experiments.run_baselines")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args(argv)
    try:
        config = ScenarioConfig.from_json(args.config.read_text(encoding="utf-8"))
        seeds = parse_seeds(args.seeds)
        names = [n.strip() for n in args.policies.split(",") if n.strip()]
        unknown = [n for n in names if n not in POLICIES]
        if unknown:
            raise ValueError(f"unknown policies {unknown}; known: {sorted(POLICIES)}")
        spec = BootstrapSpec(args.resamples, 0.95, args.bootstrap_seed)
    except (OSError, ConfigError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, ExperimentResult] = {}
    for name in names:
        res = run_experiment(config, POLICIES[name], seeds, bootstrap=spec)
        (args.out_dir / f"{name}.json").write_text(res.to_json(), encoding="utf-8")
        results[name] = res
        failed = [e.seed for e in res.episodes if e.status == "failed"]
        print(f"{name}: complete={res.complete} failed={failed}")
    comparisons = [
        compare_experiments(results[a], results[b], bootstrap=spec)
        for a, b in PREREGISTERED_PAIRS
        if a in results and b in results
    ]
    (args.out_dir / "comparisons.json").write_text(
        json.dumps(comparisons, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "summary.md").write_text(
        summary_markdown(results, comparisons), encoding="utf-8"
    )
    return 0 if all(r.complete for r in results.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())

"""Multi-condition studies: one base config, condition overrides, policies and a seed list.

A study spec is a JSON file::

    {
      "name": "phase3",
      "base_config": "nominal.json",           # relative to the spec file
      "seeds": "200-249",
      "designation": "evaluation",
      "conditions": [{"id": "nominal", "overrides": {}, "policies": [...]}, ...],
      "hypotheses": [...],                     # read by swarm.experiments.study_analysis
      "exploratory": [...]
    }

``overrides`` are deep-merged into the base config's JSON form (a ``null`` value replaces the
key), so every condition is an ordinary validated :class:`ScenarioConfig`. Each (condition,
policy) cell is run with :func:`swarm.experiments.experiment.run_experiment` over the same seeds and
written to ``<out>/<condition>/<policy>.json``. ``seed_level.csv`` holds one row per episode.

Example::

    python -m swarm.experiments.study --spec configs/phase3/study.json \\
        --out-dir docs/results/phase3/runs --jobs 2
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import json
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swarm.evaluation.statistics import BootstrapSpec
from swarm.experiments.experiment import parse_seeds, run_experiment
from swarm.experiments.runner import DESIGNATIONS
from swarm.policies.registry import POLICIES
from swarm.scenarios.config import ConfigError, ScenarioConfig

SPEC_KEYS = {"name", "base_config", "seeds", "designation", "conditions"}
OPTIONAL_SPEC_KEYS = {
    "description",
    "hypotheses",
    "exploratory",
    "thresholds",
    "bootstrap",
    "learned_policies",
}


@dataclass(frozen=True, slots=True)
class Condition:
    id: str
    config: ScenarioConfig
    policies: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LearnedPolicy:
    """A trained actor (Phase 4): ``best_actor.npz`` in ``run_dir``, used greedily or sampled as
    selected on the validation seeds (``model.json``, ADR-0022)."""

    checkpoint: str
    sample: bool


@dataclass(frozen=True, slots=True)
class Study:
    name: str
    seeds: tuple[int, ...]
    designation: str
    conditions: tuple[Condition, ...]
    raw: dict[str, Any]
    learned: dict[str, LearnedPolicy] = field(default_factory=dict)

    def condition(self, cid: str) -> Condition:
        for c in self.conditions:
            if c.id == cid:
                return c
        raise KeyError(f"unknown condition {cid!r}")


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` with ``overrides`` merged in; nested dicts merge, other values replace."""
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_study(path: Path) -> Study:
    """Parse and validate a study spec. Raises ``ValueError`` (or ``ConfigError``) on errors."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("study: must be a JSON object")
    missing = SPEC_KEYS - raw.keys()
    unknown = raw.keys() - SPEC_KEYS - OPTIONAL_SPEC_KEYS
    if missing or unknown:
        raise ValueError(f"study: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")
    if raw["designation"] not in DESIGNATIONS:
        raise ValueError(f"study.designation: must be one of {sorted(DESIGNATIONS)}")
    learned: dict[str, LearnedPolicy] = {}
    for name, run_dir in raw.get("learned_policies", {}).items():
        if name in POLICIES:
            raise ValueError(f"learned_policies.{name}: clashes with a registered policy")
        directory = path.parent / run_dir
        model = json.loads((directory / "model.json").read_text("utf-8"))
        best = model.get("best")
        if not best:
            raise ValueError(f"learned_policies.{name}: {directory} has no validated best actor")
        learned[name] = LearnedPolicy(
            str(directory / "best_actor.npz"), best.get("mode", "greedy") == "sampled"
        )
    base = ScenarioConfig.from_json((path.parent / raw["base_config"]).read_text("utf-8"))
    base_dict = base.to_dict()
    conditions: list[Condition] = []
    for k, entry in enumerate(raw["conditions"]):
        cid = entry["id"]
        if not isinstance(cid, str) or not cid.replace("_", "").isalnum():
            raise ValueError(f"conditions[{k}].id: must be alphanumeric/underscore, got {cid!r}")
        policies = tuple(entry["policies"])
        unknown_policies = sorted(set(policies) - set(POLICIES) - set(learned))
        if unknown_policies or not policies:
            raise ValueError(f"conditions[{k}].policies: unknown or empty {unknown_policies}")
        merged = deep_merge(base_dict, entry.get("overrides", {}))
        if merged.get("comms") is not None or merged.get("failures") is not None:
            merged["schema_version"] = max(merged["schema_version"], 3)
            merged.setdefault("comms", None)
        try:
            config = ScenarioConfig.from_dict(merged)
        except ConfigError as exc:
            raise ConfigError(f"conditions[{k}] ({cid}): {exc}") from exc
        conditions.append(Condition(cid, config, policies))
    ids = [c.id for c in conditions]
    if len(set(ids)) != len(ids):
        raise ValueError(f"conditions: duplicate ids in {ids}")
    return Study(
        name=raw["name"],
        seeds=tuple(parse_seeds(raw["seeds"])),
        designation=raw["designation"],
        conditions=tuple(conditions),
        raw=raw,
        learned=learned,
    )


def _factory(policy: str, learned: LearnedPolicy | None) -> Any:
    if learned is None:
        return POLICIES[policy]
    from swarm.learning.sector import learned_policy

    return learned_policy(policy, Path(learned.checkpoint), sample=learned.sample)


def _run_cell(
    args: tuple[str, str, str, tuple[int, ...], str, str, LearnedPolicy | None],
) -> tuple[str, str, bool]:
    cid, policy, config_json, seeds, designation, out, learned = args
    config = ScenarioConfig.from_json(config_json)
    result = run_experiment(
        config,
        _factory(policy, learned),
        list(seeds),
        bootstrap=BootstrapSpec(),
        designation=designation,
    )
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result.to_json(), encoding="utf-8")
    return cid, policy, result.complete


def cell_path(out_dir: Path, cid: str, policy: str) -> Path:
    return out_dir / cid / f"{policy}.json"


def _is_current(path: Path, condition: Condition, seeds: tuple[int, ...]) -> bool:
    if not path.exists():
        return False
    record = json.loads(path.read_text(encoding="utf-8"))
    return bool(
        record.get("config_hash") == condition.config.config_hash()
        and tuple(record.get("seeds", ())) == seeds
    )


def run_study(study: Study, out_dir: Path, *, jobs: int = 1, resume: bool = False) -> bool:
    """Run every cell; return ``True`` if every episode completed.

    With ``resume``, a cell whose file already exists *for the same config hash and seeds* is not
    rerun. Episodes are deterministic, so this never changes a result; it only saves time.
    """
    tasks = []
    for c in study.conditions:
        for policy in c.policies:
            path = cell_path(out_dir, c.id, policy)
            if resume and _is_current(path, c, study.seeds):
                continue
            tasks.append(
                (
                    c.id,
                    policy,
                    c.config.to_json(),
                    study.seeds,
                    study.designation,
                    str(path),
                    study.learned.get(policy),
                )
            )
    complete = True
    if jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for cid, policy, ok in pool.map(_run_cell, tasks):
                complete &= ok
                print(f"{cid}/{policy}: {'complete' if ok else 'INCOMPLETE'}", file=sys.stderr)
    else:
        for task in tasks:
            cid, policy, ok = _run_cell(task)
            complete &= ok
            print(f"{cid}/{policy}: {'complete' if ok else 'INCOMPLETE'}", file=sys.stderr)
    write_seed_level(study, out_dir)
    return complete and all(
        json.loads(cell_path(out_dir, c.id, p).read_text("utf-8"))["complete"]
        for c in study.conditions
        for p in c.policies
    )


def write_seed_level(study: Study, out_dir: Path) -> Path:
    """One CSV row per (condition, policy, seed) with every metric (failed episodes included)."""
    rows: list[dict[str, Any]] = []
    for c in study.conditions:
        for policy in c.policies:
            record = json.loads(cell_path(out_dir, c.id, policy).read_text("utf-8"))
            for ep in record["episodes"]:
                row: dict[str, Any] = {
                    "condition": c.id,
                    "policy": policy,
                    "seed": ep["seed"],
                    "status": ep["status"],
                    "end_reason": ep["end_reason"],
                    "steps": ep["steps"],
                    "scenario_sha256": ep["scenario_sha256"],
                }
                row.update(ep.get("metrics") or {})
                row.update(ep.get("comm_metrics") or {})
                rows.append(row)
    columns: list[str] = []
    for row in rows:
        columns.extend(k for k in row if k not in columns)
    target = out_dir / "seed_level.csv"
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m swarm.experiments.study")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--seeds", help="override the spec's seeds (development checks only; forces development)"
    )
    args = parser.parse_args(argv)
    try:
        study = load_study(args.spec)
        if args.seeds is not None:
            study = dataclasses.replace(
                study, seeds=tuple(parse_seeds(args.seeds)), designation="development"
            )
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    ok = run_study(study, args.out_dir, jobs=args.jobs, resume=args.resume)
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())

"""Analysis of a study run (see :mod:`swarm.experiments.study`): the pre-registered tests.

For every comparison, values are paired by seed. Pairing is valid only if both cells generated
the same scenario for that seed, so this is checked (``scenario_sha256``) and a mismatch is an
error. A pair is used only when both values are defined. Failed episodes are never dropped
silently: they are counted and make a comparison's ``n_excluded`` non-zero.

Per comparison (difference = B - A):
- mean difference with a 95% percentile-bootstrap interval;
- Cohen's d_z;
- a two-sided paired sign-flip p-value.

Confirmatory hypotheses are Holm-adjusted as one family. Their verdicts:
- *supported*: adjusted p < alpha with the expected sign (for a two-sided hypothesis, any sign:
  "difference detected");
- *contradicted*: adjusted p < alpha with the opposite sign;
- *inconclusive*: otherwise.

Exploratory comparisons are Benjamini-Hochberg adjusted as one family and are descriptive only.
An exploratory comparison against nominal whose condition changes the scenario itself (for
example the map size) cannot be paired by seed; it is listed under ``exploratory_not_pairable``
instead of being tested. Confirmatory comparisons always raise on a pairing mismatch.

``time_censored`` is derived: ``time_to_recall_target`` for successful episodes, otherwise
``time_limit + 1`` (a pre-registered, conservative penalty for not reaching the target).
``agents_disabled_injected`` (descriptive) is ``agents_disabled`` minus the energy and hazard
counts: the scheduled failures that happened before the episode ended (ADR-0021).

Example::

    python -m swarm.experiments.study_analysis --spec configs/phase3/study.json \\
        --runs docs/results/phase3/runs --out docs/results/phase3/analysis.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from swarm.evaluation.statistics import (
    BootstrapSpec,
    bh_adjust,
    bootstrap_mean_ci,
    cohens_dz,
    holm_adjust,
    sign_flip_p_value,
    summarize,
)
from swarm.experiments.experiment import parse_seeds
from swarm.experiments.study import Study, cell_path, load_study

DERIVED = ("time_censored", "agents_disabled_injected")
SUMMARY_METRICS = (
    "recall",
    "mission_success",
    "time_censored",
    "time_to_recall_target",
    "coverage_sensed",
    "energy_used",
    "agents_disabled",
    "agents_disabled_injected",
    "precision",
    "messages_sent",
    "message_units_per_step",
    "message_delivery_rate",
    "survivor_info_delay_mean",
    "survivor_info_informed_fraction",
    "disabled_detection_delay_mean",
    "disabled_detection_unresolved",
    "suspicion_detection_delay_mean",
    "suspicion_detection_unresolved",
    "false_suspicions",
    "abandoned_sectors",
    "reassignment_delay_mean",
    "unresolved_sectors",
    "duplicate_ownership_steps",
)


class PairingError(ValueError):
    """Two cells did not generate the same scenario for a seed."""


def _value(episode: dict[str, Any], metric: str, time_limit: int) -> Any:
    if episode["status"] != "completed":
        return None
    metrics = episode.get("metrics") or {}
    if metric == "time_censored":
        reached = metrics.get("time_to_recall_target")
        return reached if metrics.get("mission_success") and reached is not None else time_limit + 1
    if metric == "agents_disabled_injected":
        return (
            metrics["agents_disabled"]
            - metrics["agents_disabled_energy"]
            - metrics["agents_disabled_hazard"]
        )
    if metric in metrics:
        return metrics[metric]
    return (episode.get("comm_metrics") or {}).get(metric)


class StudyData:
    """Cell results loaded once from a run directory."""

    def __init__(self, study: Study, runs: Path) -> None:
        self.study = study
        self._cells: dict[tuple[str, str], dict[str, Any]] = {}
        for c in study.conditions:
            for p in c.policies:
                record = json.loads(cell_path(runs, c.id, p).read_text("utf-8"))
                if record["config_hash"] != c.config.config_hash():
                    raise ValueError(f"{c.id}/{p}: result config hash does not match the spec")
                if tuple(record["seeds"]) != study.seeds:
                    raise ValueError(f"{c.id}/{p}: result seeds do not match the spec")
                self._cells[(c.id, p)] = record

        # Pooled policies (Phase 4): one value per scenario seed, the mean over member policies
        # (for example, one learned policy per training seed). Undefined if any member is.
        self.pooled: dict[str, list[str]] = dict(study.raw.get("pooled_policies", {}))

    def available(self, cid: str, policy: str) -> bool:
        """Whether ``policy`` (or every member of a pooled policy) ran in condition ``cid``."""
        return all((cid, m) in self._cells for m in self.pooled.get(policy, [policy]))

    def members(self, cid: str, policy: str) -> list[str]:
        if policy not in self.pooled:
            return [policy]
        members = self.pooled[policy]
        missing = [m for m in members if (cid, m) not in self._cells]
        if missing:
            raise ValueError(f"pooled policy {policy}: no {cid} results for {missing}")
        return members

    def cell(self, cid: str, policy: str) -> dict[str, Any]:
        return self._cells[(cid, policy)]

    def values(self, cid: str, policy: str, metric: str) -> list[Any]:
        limit = self.study.condition(cid).config.time_limit
        columns = [
            [_value(ep, metric, limit) for ep in self.cell(cid, m)["episodes"]]
            for m in self.members(cid, policy)
        ]
        if len(columns) == 1:
            return columns[0]
        return [
            None if any(v is None for v in row) else float(np.mean([float(v) for v in row]))
            for row in zip(*columns, strict=True)
        ]

    def failed(self, cid: str, policy: str) -> int:
        return sum(
            ep["status"] != "completed"
            for m in self.members(cid, policy)
            for ep in self.cell(cid, m)["episodes"]
        )

    def scenarios(self, cid: str, policy: str) -> list[str | None]:
        columns = [
            [ep["scenario_sha256"] for ep in self.cell(cid, m)["episodes"]]
            for m in self.members(cid, policy)
        ]
        for col in columns[1:]:
            if col != columns[0]:
                raise PairingError(f"pooled policy {policy}: members ran different scenarios")
        return columns[0]


def compare(
    data: StudyData,
    a: Sequence[str],
    b: Sequence[str],
    metric: str,
    *,
    spec: BootstrapSpec,
    resamples: int,
    perm_seed: int,
) -> dict[str, Any]:
    """Paired comparison B - A on one metric (see module docstring)."""
    sa, sb = data.scenarios(a[0], a[1]), data.scenarios(b[0], b[1])
    for seed, x, y in zip(data.study.seeds, sa, sb, strict=True):
        if x is not None and y is not None and x != y:
            raise PairingError(f"seed {seed}: {a} and {b} generated different scenarios")
    va, vb = data.values(a[0], a[1], metric), data.values(b[0], b[1], metric)
    diffs = np.array(
        [
            float(y) - float(x)
            for x, y in zip(va, vb, strict=True)
            if x is not None and y is not None
        ]
    )
    ci = bootstrap_mean_ci(diffs, spec, f"{a[0]}/{a[1]}|{b[0]}/{b[1]}|{metric}")
    return {
        "a": list(a),
        "b": list(b),
        "metric": metric,
        "n_pairs": int(diffs.size),
        "n_excluded": len(va) - int(diffs.size),
        "mean_a": float(np.mean([v for v in va if v is not None]))
        if any(v is not None for v in va)
        else None,
        "mean_b": float(np.mean([v for v in vb if v is not None]))
        if any(v is not None for v in vb)
        else None,
        "mean_difference": float(diffs.mean()) if diffs.size else None,
        "ci_low": None if ci is None else ci[0],
        "ci_high": None if ci is None else ci[1],
        "cohens_dz": cohens_dz(diffs.tolist()),
        "p_value": sign_flip_p_value(diffs.tolist(), resamples=resamples, seed=perm_seed),
    }


def verdict(result: dict[str, Any], expected: str, alpha: float) -> str:
    """Pre-registered decision rule for one confirmatory hypothesis."""
    p, diff = result.get("p_adjusted"), result.get("mean_difference")
    if p is None or diff is None or p >= alpha or diff == 0:
        return "inconclusive"
    if expected == "two-sided":
        return "difference detected"
    sign = 1 if diff > 0 else -1
    return "supported" if sign == (1 if expected == "+" else -1) else "contradicted"


def practical(result: dict[str, Any], threshold: float | None) -> str | None:
    """Whether the whole interval lies beyond the practical threshold (either direction)."""
    lo, hi = result.get("ci_low"), result.get("ci_high")
    if threshold is None or lo is None or hi is None:
        return None
    if lo >= threshold or hi <= -threshold:
        return "practically significant"
    if -threshold < lo and hi < threshold:
        return "within threshold"
    return "uncertain"


def analyze(study: Study, runs: Path) -> dict[str, Any]:
    """Run every pre-registered test and the descriptive summaries."""
    raw = study.raw
    opts = raw.get("bootstrap", {})
    spec = BootstrapSpec(
        opts.get("resamples", 10_000), opts.get("confidence", 0.95), opts.get("seed", 0)
    )
    resamples = opts.get("permutation_resamples", 20_000)
    perm_seed = opts.get("permutation_seed", 0)
    alpha = opts.get("alpha", 0.05)
    thresholds: dict[str, float] = raw.get("thresholds", {})
    data = StudyData(study, runs)

    confirmatory = []
    for h in raw.get("hypotheses", []):
        r = compare(
            data, h["a"], h["b"], h["metric"], spec=spec, resamples=resamples, perm_seed=perm_seed
        )
        confirmatory.append(h | r)
    holm = holm_adjust([r["p_value"] for r in confirmatory])
    for r, adjusted in zip(confirmatory, holm, strict=True):
        r["p_adjusted"] = adjusted
        r["verdict"] = verdict(r, r["expected"], alpha)
        r["practical"] = practical(r, thresholds.get(r["metric"]))

    exploratory = []
    skipped: list[dict[str, Any]] = []
    ex = raw.get("exploratory", {})
    for c in study.conditions:
        if c.id == "nominal":
            continue
        for policy in ex.get("vs_nominal_policies", []):
            if not data.available(c.id, policy):
                continue
            for metric in ex.get("vs_nominal_metrics", []):
                try:
                    result = compare(
                        data,
                        ["nominal", policy],
                        [c.id, policy],
                        metric,
                        spec=spec,
                        resamples=resamples,
                        perm_seed=perm_seed,
                    )
                except PairingError as exc:
                    # A condition that changes the scenario itself (map size, terrain, survivors)
                    # cannot be paired with nominal by seed: recorded, not tested.
                    skipped.append(
                        {
                            "a": ["nominal", policy],
                            "b": [c.id, policy],
                            "metric": metric,
                            "reason": str(exc),
                        }
                    )
                    continue
                exploratory.append({"kind": "vs_nominal"} | result)
    for c in study.conditions:
        for pa, pb in ex.get("pairs", []):
            if not (data.available(c.id, pa) and data.available(c.id, pb)):
                continue
            for metric in ex.get("pair_metrics", []):
                exploratory.append(
                    {"kind": "pair"}
                    | compare(
                        data,
                        [c.id, pa],
                        [c.id, pb],
                        metric,
                        spec=spec,
                        resamples=resamples,
                        perm_seed=perm_seed,
                    )
                )
    bh = bh_adjust([r["p_value"] for r in exploratory])
    for r, adjusted in zip(exploratory, bh, strict=True):
        r["p_adjusted_bh"] = adjusted
        r["practical"] = practical(r, thresholds.get(r["metric"]))

    summaries: dict[str, dict[str, Any]] = {}
    for c in study.conditions:
        for p in c.policies:
            cell = data.cell(c.id, p)
            n_failed = data.failed(c.id, p)
            per_metric = {}
            for metric in SUMMARY_METRICS:
                recorded = metric in DERIVED or any(
                    metric in (ep.get("metrics") or {}) or metric in (ep.get("comm_metrics") or {})
                    for ep in cell["episodes"]
                )
                if not recorded:
                    continue
                completed = [
                    v
                    for v, ep in zip(data.values(c.id, p, metric), cell["episodes"], strict=True)
                    if ep["status"] == "completed"
                ]
                per_metric[metric] = summarize(
                    completed, n_failed=n_failed, spec=spec, metric=metric
                ).to_dict()
            n = len(cell["episodes"])
            per_metric["runtime_seconds_per_episode"] = {
                "mean": cell["metadata"]["run"]["duration_seconds"] / n if n else None,
                "note": "wall clock, not deterministic; whole cell duration / episodes",
            }
            summaries[f"{c.id}/{p}"] = {
                "condition": c.id,
                "policy": p,
                "config_hash": cell["config_hash"],
                "n_failed": n_failed,
                "end_reasons": cell["end_reasons"],
                "metrics": per_metric,
            }
        for pooled, members in data.pooled.items():
            if not all(m in c.policies for m in members):
                continue
            per_metric = {}
            for metric in ("time_censored", "mission_success", "recall", "energy_used"):
                values = [v for v in data.values(c.id, pooled, metric) if v is not None]
                per_metric[metric] = summarize(
                    values, n_failed=data.failed(c.id, pooled), spec=spec, metric=metric
                ).to_dict()
            summaries[f"{c.id}/{pooled}"] = {
                "condition": c.id,
                "policy": pooled,
                "pooled_members": members,
                "n_failed": data.failed(c.id, pooled),
                "metrics": per_metric,
            }
    return {
        "study": study.name,
        "seeds": list(study.seeds),
        "designation": study.designation,
        "alpha": alpha,
        "bootstrap": spec.to_dict(),
        "permutation": {
            "test": "two-sided paired sign-flip",
            "resamples": resamples,
            "seed": perm_seed,
            "exact_when_nonzero_differences_at_most": 16,
        },
        "corrections": {"confirmatory": "Holm", "exploratory": "Benjamini-Hochberg"},
        "confirmatory": confirmatory,
        "exploratory": exploratory,
        "exploratory_not_pairable": skipped,
        "summaries": summaries,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m swarm.experiments.study_analysis")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", help="analyze a development run made with these seeds")
    args = parser.parse_args(argv)
    try:
        study = load_study(args.spec)
        if args.seeds is not None:
            study = dataclasses.replace(
                study, seeds=tuple(parse_seeds(args.seeds)), designation="development"
            )
        result = analyze(study, args.runs)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for r in result["confirmatory"]:
        print(f"{r['id']}: {r['verdict']} (diff {r['mean_difference']}, p_holm {r['p_adjusted']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

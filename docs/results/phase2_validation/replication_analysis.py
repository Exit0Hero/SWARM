"""Apply the pre-registered replication rules of docs/17_PHASE2_VALIDATION_STUDY.md.

Reads the archived corrected exploratory comparisons (seeds 0-49) and the validation comparisons
(seeds 100-149) and writes ``replication.json`` and ``replication.md`` next to this file. It reads
result files only: it runs no episodes and uses only the standard library.

Usage (from the repository root)::

    python docs/results/phase2_validation/replication_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXPLORATORY = HERE.parent / "phase2_corrected"
CONFIGS = ("m1", "m2")
POLICIES = (
    "random_valid",
    "astar_waypoint",
    "frontier",
    "frontier_hazard_aware",
    "prioritized_frontier",
    "assignment",
    "assignment_hazard_aware",
    "assignment_oracle",
)
PAIRS = {
    "P1": ("random_valid", "astar_waypoint"),
    "P2": ("astar_waypoint", "frontier"),
    "P3": ("frontier", "prioritized_frontier"),
    "P4": ("prioritized_frontier", "assignment"),
    "P5": ("frontier", "frontier_hazard_aware"),
    "P6": ("assignment", "assignment_hazard_aware"),
    "P7": ("assignment", "assignment_oracle"),
}
THRESHOLDS = {
    "recall": 0.02,
    "mission_success": 0.05,
    "time_to_recall_target": 5.0,
    "hazard_exposures": 1.0,
    "agents_disabled_hazard": 0.05,
    "agents_disabled_energy": 0.05,
    "energy_used": 50.0,
}
REPORTED = tuple(THRESHOLDS)
HEADLINE = (
    ("H1", "m1", "P1", "recall", 1),
    ("H2", "m2", "P1", "recall", 1),
    ("H3", "m2", "P1", "mission_success", 1),
    ("H4", "m1", "P2", "recall", 1),
    ("H5", "m1", "P2", "mission_success", 1),
    ("H6", "m2", "P2", "recall", 1),
    ("H7", "m2", "P2", "mission_success", 1),
    ("H8", "m1", "P3", "recall", 1),
    ("H9", "m1", "P3", "mission_success", 1),
    ("H10", "m1", "P3", "time_to_recall_target", -1),
    ("H11", "m2", "P3", "time_to_recall_target", -1),
    ("H12", "m2", "P5", "hazard_exposures", -1),
    ("H13", "m2", "P5", "agents_disabled_hazard", -1),
    ("H14", "m2", "P6", "hazard_exposures", -1),
    ("H15", "m2", "P6", "agents_disabled_hazard", -1),
    ("H16", "m1", "P7", "time_to_recall_target", -1),
    ("H17", "m2", "P7", "time_to_recall_target", -1),
)
NULLS = tuple(
    [
        ("N1", c, "P4", m)
        for c in CONFIGS
        for m in ("recall", "mission_success", "time_to_recall_target")
    ]
    + [("N2", "m2", p, m) for p in ("P5", "P6") for m in ("recall", "mission_success")]
)


def _comparisons(root: Path, config: str) -> dict[str, dict[str, Any]]:
    data = json.loads((root / config / "comparisons.json").read_text(encoding="utf-8"))
    by_names = {(c["baseline_policy"]["name"], c["candidate_policy"]["name"]): c for c in data}
    return {pid: by_names[names]["metrics"] for pid, names in PAIRS.items()}


def _sign(d: dict[str, Any]) -> int:
    """+1 / -1 if the 95% interval excludes 0 above / below, 0 if it includes 0."""
    if d["ci_low"] is None:
        return 0
    return 1 if d["ci_low"] > 0 else (-1 if d["ci_high"] < 0 else 0)


def _direction(d: dict[str, Any]) -> int:
    m = d["mean_difference"]
    return 0 if not m else (1 if m > 0 else -1)


def classify_headline(d: dict[str, Any], expected: int) -> str:
    """Pre-registered rule for an expected non-null effect."""
    sig = _sign(d)
    if sig == expected:
        return "replicated"
    if sig == -expected:
        return "contradicted"
    if _direction(d) == expected:
        return "direction consistent, not replicated"
    return "not replicated (opposite point estimate, interval includes 0)"


def classify_null(d: dict[str, Any], metric: str) -> str:
    """Pre-registered rule for an expected null."""
    if _sign(d) == 0:
        return "consistent"
    if abs(d["mean_difference"]) < THRESHOLDS[metric]:
        return "consistent (detectable but below practical threshold)"
    return "not consistent"


def practical(d: dict[str, Any], metric: str) -> str:
    """Whether an interval excluding 0 is also practically meaningful."""
    if _sign(d) == 0:
        return "not detectable"
    t = THRESHOLDS[metric]
    lo, hi = abs(d["ci_low"]), abs(d["ci_high"])
    if min(lo, hi) >= t:
        return "practically significant (whole interval beyond threshold)"
    if abs(d["mean_difference"]) >= t:
        return "point estimate beyond threshold, interval reaches below it"
    return "detectable but small"


def _fmt(d: dict[str, Any]) -> str:
    if d["mean_difference"] is None:
        return "undefined"
    return (
        f"{d['mean_difference']:+.3g} [{d['ci_low']:+.3g}, {d['ci_high']:+.3g}] (n={d['n_pairs']})"
    )


def _policy_rows(root: Path, config: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for name in POLICIES:
        d = json.loads((root / config / f"{name}.json").read_text(encoding="utf-8"))
        rows[name] = {
            "metrics": {k: d["summary"][k] for k in REPORTED},
            "failed_plans": d["policy_diagnostics_summary"].get("failed_plans"),
            "failed_plans_unreachable": d["policy_diagnostics_summary"].get(
                "failed_plans_unreachable"
            ),
            "failed_plans_temporary": d["policy_diagnostics_summary"].get("failed_plans_temporary"),
            "energy_route_aborts": d["policy_diagnostics_summary"].get("energy_route_aborts"),
            "end_reasons": d["end_reasons"],
            "failed_seeds": d["failed_seeds"],
        }
    return rows


def main() -> None:
    explo = {c: _comparisons(EXPLORATORY, c) for c in CONFIGS}
    valid = {c: _comparisons(HERE, c) for c in CONFIGS}
    headline = []
    for hid, config, pid, metric, expected in HEADLINE:
        v = valid[config][pid][metric]
        headline.append(
            {
                "id": hid,
                "config": config,
                "pair": pid,
                "metric": metric,
                "expected_sign": expected,
                "exploratory": explo[config][pid][metric],
                "validation": v,
                "verdict": classify_headline(v, expected),
                "practical": practical(v, metric),
            }
        )
    nulls = []
    for nid, config, pid, metric in NULLS:
        v = valid[config][pid][metric]
        nulls.append(
            {
                "id": nid,
                "config": config,
                "pair": pid,
                "metric": metric,
                "exploratory": explo[config][pid][metric],
                "validation": v,
                "verdict": classify_null(v, metric),
            }
        )
    all_pairs = []
    for config in CONFIGS:
        for pid in PAIRS:
            for metric in REPORTED:
                e, v = explo[config][pid][metric], valid[config][pid][metric]
                all_pairs.append(
                    {
                        "config": config,
                        "pair": pid,
                        "metric": metric,
                        "exploratory": e,
                        "validation": v,
                        "same_point_direction": _direction(e) == _direction(v),
                        "same_interval_conclusion": _sign(e) == _sign(v),
                        "practical": practical(v, metric),
                    }
                )
    # Per-policy detail for the validation run (the exploratory archive keeps summaries only).
    policies = {c: _policy_rows(HERE, c) for c in CONFIGS}
    crashes = sum(len(policies[c][p]["failed_seeds"]) for c in CONFIGS for p in POLICIES)
    contradicted = [h["id"] for h in headline if h["verdict"] == "contradicted"]
    not_replicated = [h["id"] for h in headline if h["verdict"] != "replicated"]
    bad_nulls = [
        f"{n['id']}:{n['config']}:{n['pair']}:{n['metric']}"
        for n in nulls
        if n["verdict"] == "not consistent"
    ]
    if crashes or contradicted:
        verdict = "failed"
    elif not_replicated or bad_nulls:
        verdict = "partially successful"
    else:
        verdict = "successful"
    out = {
        "verdict": verdict,
        "crashed_episodes": crashes,
        "headline_not_replicated": not_replicated,
        "headline_contradicted": contradicted,
        "nulls_not_consistent": bad_nulls,
        "thresholds": THRESHOLDS,
        "headline": headline,
        "nulls": nulls,
        "all_pairs": all_pairs,
        "policies": policies,
    }
    (HERE / "replication.json").write_text(
        json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )

    md = [
        "# Phase 2 validation: replication against the corrected exploratory run",
        "",
        "Generated by `replication_analysis.py` from `comparisons.json` files; rules from",
        "`docs/17_PHASE2_VALIDATION_STUDY.md`. Paired differences are candidate - baseline,",
        "mean [95% bootstrap interval] (n pairs). Exploratory = seeds 0-49 (corrected code);",
        "validation = seeds 100-149.",
        "",
        f"**Pre-registered verdict: {verdict}.** Crashed episodes: {crashes}. "
        f"Headline effects not replicated: {', '.join(not_replicated) or 'none'}. "
        f"Contradicted: {', '.join(contradicted) or 'none'}. "
        f"Expected nulls not consistent: {', '.join(bad_nulls) or 'none'}.",
        "",
        "## Headline effects",
        "",
        "| ID | Config | Pair | Metric | Exploratory | Validation | Verdict | Practical |",
        "|---|---|---|---|---|---|---|---|",
    ]
    md += [
        f"| {h['id']} | {h['config'].upper()} | {h['pair']} | {h['metric']} | "
        f"{_fmt(h['exploratory'])} | {_fmt(h['validation'])} | {h['verdict']} | {h['practical']} |"
        for h in headline
    ]
    md += [
        "",
        "## Expected nulls",
        "",
        "| ID | Config | Pair | Metric | Exploratory | Validation | Verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    md += [
        f"| {n['id']} | {n['config'].upper()} | {n['pair']} | {n['metric']} | "
        f"{_fmt(n['exploratory'])} | {_fmt(n['validation'])} | {n['verdict']} |"
        for n in nulls
    ]
    md += [
        "",
        "## All pre-registered pairs (descriptive)",
        "",
        "| Config | Pair | Metric | Exploratory | Validation | Same direction "
        "| Same interval conclusion |",
        "|---|---|---|---|---|---|---|",
    ]
    md += [
        f"| {r['config'].upper()} | {r['pair']} | {r['metric']} | {_fmt(r['exploratory'])} | "
        f"{_fmt(r['validation'])} | {'yes' if r['same_point_direction'] else 'no'} | "
        f"{'yes' if r['same_interval_conclusion'] else 'no'} |"
        for r in all_pairs
    ]
    (HERE / "replication.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(
        f"verdict={verdict} crashes={crashes} not_replicated={not_replicated} "
        f"contradicted={contradicted} nulls_not_consistent={bad_nulls}"
    )


if __name__ == "__main__":
    main()

"""M3.5: multi-condition study runner and pre-registered analysis."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from swarm.experiments.study import deep_merge, load_study, run_study
from swarm.experiments.study_analysis import PairingError, _value, analyze, verdict
from swarm.scenarios.config import CommsConfig
from tests.helpers import make_config

ROOT = Path(__file__).resolve().parents[1]


def _spec(tmp_path: Path, **extra: Any) -> Path:
    base = replace(
        make_config(drones=1, grounds=1, time_limit=25),
        comms=CommsConfig(ranges=(("drone", 8), ("ground", 5))),
    )
    (tmp_path / "base.json").write_text(base.to_json())
    spec = {
        "name": "tiny",
        "base_config": "base.json",
        "seeds": "1000-1002",
        "designation": "development",
        "conditions": [
            {"id": "nominal", "overrides": {}, "policies": ["frontier", "auction_comm"]},
            {
                "id": "lossy",
                "overrides": {"comms": {"p_drop": 0.5}},
                "policies": ["auction_comm"],
            },
        ],
        "hypotheses": [
            {
                "id": "H1",
                "a": ["nominal", "frontier"],
                "b": ["nominal", "auction_comm"],
                "metric": "time_censored",
                "expected": "-",
            }
        ],
        "exploratory": {
            "vs_nominal_policies": ["auction_comm"],
            "vs_nominal_metrics": ["recall", "message_delivery_rate"],
            "pairs": [["frontier", "auction_comm"]],
            "pair_metrics": ["recall"],
        },
        "bootstrap": {"resamples": 200, "permutation_resamples": 200},
    } | extra
    path = tmp_path / "study.json"
    path.write_text(json.dumps(spec))
    return path


def test_deep_merge_merges_nested_and_replaces_leaves() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": [1]}
    assert deep_merge(base, {"a": {"b": 5}, "d": None}) == {"a": {"b": 5, "c": 2}, "d": None}
    assert base == {"a": {"b": 1, "c": 2}, "d": [1]}  # not mutated


def test_study_runs_every_cell_writes_seed_rows_and_analyzes(tmp_path: Path) -> None:
    study = load_study(_spec(tmp_path))
    lossy = study.condition("lossy").config.comms
    assert lossy is not None
    assert lossy.p_drop == 0.5
    out = tmp_path / "runs"
    assert run_study(study, out, jobs=1)
    with (out / "seed_level.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3 * 3
    assert {r["condition"] for r in rows} == {"nominal", "lossy"}
    result = analyze(study, out)
    (h1,) = result["confirmatory"]
    assert h1["n_pairs"] == 3
    assert h1["verdict"] in {"supported", "contradicted", "inconclusive"}
    assert h1["p_adjusted"] >= h1["p_value"]
    kinds = {(r["kind"], r["metric"]) for r in result["exploratory"]}
    assert ("vs_nominal", "message_delivery_rate") in kinds
    assert ("pair", "recall") in kinds
    assert all("p_adjusted_bh" in r for r in result["exploratory"])
    assert "nominal/frontier" in result["summaries"]
    # resume skips current cells and gives byte-identical results
    before = (out / "nominal" / "frontier.json").read_text()
    assert run_study(study, out, jobs=1, resume=True)
    assert (out / "nominal" / "frontier.json").read_text() == before


def test_analysis_refuses_results_for_another_config(tmp_path: Path) -> None:
    study = load_study(_spec(tmp_path))
    out = tmp_path / "runs"
    run_study(study, out)
    record = json.loads((out / "lossy" / "auction_comm.json").read_text())
    record["config_hash"] = "0" * 64
    (out / "lossy" / "auction_comm.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="config hash"):
        analyze(study, out)


def test_pairing_requires_identical_scenarios(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    raw = json.loads(spec.read_text())
    raw["conditions"][1]["overrides"] = {"terrain": {"debris_fraction": 0.3}}
    raw["hypotheses"][0]["a"] = ["lossy", "auction_comm"]
    spec.write_text(json.dumps(raw))
    study = load_study(spec)
    out = tmp_path / "runs"
    run_study(study, out)
    with pytest.raises(PairingError, match="different scenarios"):
        analyze(study, out)


def test_spec_errors_are_reported(tmp_path: Path) -> None:
    spec = _spec(tmp_path, bogus=1)
    with pytest.raises(ValueError, match="unknown keys"):
        load_study(spec)
    spec = _spec(tmp_path)
    raw = json.loads(spec.read_text())
    raw["conditions"][0]["policies"] = ["no_such_policy"]
    spec.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="policies"):
        load_study(spec)


def test_time_censored_penalizes_unsuccessful_and_failed_episodes() -> None:
    ok = {"status": "completed", "metrics": {"mission_success": True, "time_to_recall_target": 40}}
    miss = {
        "status": "completed",
        "metrics": {"mission_success": False, "time_to_recall_target": None},
    }
    crashed = {"status": "failed", "metrics": None}
    assert _value(ok, "time_censored", 200) == 40
    assert _value(miss, "time_censored", 200) == 201
    assert _value(crashed, "time_censored", 200) is None


@pytest.mark.parametrize(
    ("diff", "p", "expected", "outcome"),
    [
        (-3.0, 0.01, "-", "supported"),
        (3.0, 0.01, "-", "contradicted"),
        (-3.0, 0.2, "-", "inconclusive"),
        (2.0, 0.01, "two-sided", "difference detected"),
        (0.0, 0.01, "+", "inconclusive"),
    ],
)
def test_verdict_rule(diff: float, p: float, expected: str, outcome: str) -> None:
    assert verdict({"mean_difference": diff, "p_adjusted": p}, expected, 0.05) == outcome


def test_the_phase3_spec_loads_with_reserved_evaluation_seeds() -> None:
    study = load_study(ROOT / "configs" / "phase3" / "study.json")
    assert study.seeds == tuple(range(200, 250))
    assert study.designation == "evaluation"
    oracle = [c for c in study.conditions if "assignment_oracle" in c.policies]
    assert [c.id for c in oracle] == ["nominal"]
    for h in study.raw["hypotheses"]:
        for cid, policy in (h["a"], h["b"]):
            assert policy in study.condition(cid).policies

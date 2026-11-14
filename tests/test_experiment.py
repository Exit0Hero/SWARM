"""Multi-seed experiments, failure accounting and paired comparisons."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from swarm.core.observation import Observation
from swarm.core.types import Action, AgentId
from swarm.evaluation.metrics import METRIC_DEFINITIONS
from swarm.evaluation.statistics import BootstrapSpec
from swarm.experiments import compare as compare_cli
from swarm.experiments import run_experiment as experiment_cli
from swarm.experiments.experiment import (
    ComparisonError,
    ExperimentResult,
    compare_experiments,
    parse_seeds,
    run_experiment,
)
from swarm.experiments.runner import run_episode
from swarm.policies.random_valid import random_valid_factory
from swarm.scenarios.config import AgentTypeSpec, ScenarioConfig
from tests.helpers import make_config

ROOT = Path(__file__).resolve().parents[1]
M2_CONFIG = ROOT / "configs" / "m2_default.json"
FAST = BootstrapSpec(resamples=500, seed=0)


def m2_config() -> ScenarioConfig:
    return ScenarioConfig.from_json(M2_CONFIG.read_text(encoding="utf-8"))


class StayPolicy:
    name = "stay"
    version = "1"
    uses_oracle = False

    def act(self, observation: Observation, rng: np.random.Generator) -> int:
        return int(Action.STAY)


def stay_factory(agent_id: AgentId, spec: AgentTypeSpec) -> StayPolicy:
    return StayPolicy()


class CrashOnStepPolicy(StayPolicy):
    """Raises on a chosen step: simulates a buggy policy for failure accounting."""

    name = "crash"

    def act(self, observation: Observation, rng: np.random.Generator) -> int:
        if observation.step == 2:
            raise RuntimeError("policy bug")
        return int(Action.STAY)


def crash_factory(agent_id: AgentId, spec: AgentTypeSpec) -> CrashOnStepPolicy:
    return CrashOnStepPolicy()


# ----------------------------------------------------------------- seeds


@pytest.mark.parametrize(
    ("text", "expected"),
    [("0-4", [0, 1, 2, 3, 4]), ("3", [3]), ("0-2,10,20-21", [0, 1, 2, 10, 20, 21])],
)
def test_parse_seeds(text: str, expected: list[int]) -> None:
    assert parse_seeds(text) == expected


@pytest.mark.parametrize("text", ["", "5-1", "a-3", "1,,2"])
def test_parse_seeds_rejects_bad_input(text: str) -> None:
    with pytest.raises(ValueError, match="seeds"):
        parse_seeds(text)


@pytest.mark.parametrize(
    ("seeds", "message"), [([], "at least one"), ([1, 2, 1], "duplicates"), ([-1], "seed")]
)
def test_invalid_seed_lists_raise(seeds: list[int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        run_experiment(make_config(), stay_factory, seeds, bootstrap=FAST)


# ----------------------------------------------------------------- running


def test_experiment_records_every_seed_and_summarizes_every_metric() -> None:
    result = run_experiment(m2_config(), random_valid_factory, [3, 1, 2], bootstrap=FAST)
    assert [e.seed for e in result.episodes] == [3, 1, 2]  # given order kept
    assert result.complete
    assert set(result.summary) == set(METRIC_DEFINITIONS)
    for s in result.summary.values():
        assert s.n_seeds == 3
        assert s.n_defined + s.n_undefined == s.n_completed == 3


def test_episode_records_match_single_episode_runs() -> None:
    """Multi-seed reproducibility: each record equals an independent run_episode."""
    config = m2_config()
    result = run_experiment(config, random_valid_factory, [0, 5], bootstrap=FAST)
    for record in result.episodes:
        single = run_episode(config, random_valid_factory, record.seed)
        assert record.event_stream_sha256 == single.event_stream_sha256
        assert record.scenario_sha256 == single.scenario_sha256
        assert record.metrics == single.metrics.to_dict()


def test_experiment_is_reproducible() -> None:
    a = run_experiment(m2_config(), random_valid_factory, range(4), bootstrap=FAST)
    b = run_experiment(m2_config(), random_valid_factory, range(4), bootstrap=FAST)
    assert a.episodes == b.episodes
    assert a.summary == b.summary


def test_failed_episodes_are_recorded_not_dropped() -> None:
    result = run_experiment(make_config(time_limit=5), crash_factory, [0, 1], bootstrap=FAST)
    assert not result.complete
    assert [e.status for e in result.episodes] == ["failed", "failed"]
    assert result.episodes[0].error is not None
    assert result.episodes[0].error["type"] == "RuntimeError"
    assert "policy bug" in result.episodes[0].error["message"]
    recall = result.summary["recall"]
    assert (recall.n_seeds, recall.n_failed, recall.n_completed, recall.mean) == (2, 2, 0, None)
    record = result.to_dict()
    assert record["complete"] is False
    assert record["failed_seeds"] == [0, 1]
    assert record["end_reasons"] == {"failed": 2}


def test_scenario_generation_failure_for_one_seed_is_recorded() -> None:
    # A 2x2 grid with 3 survivors fails whenever a structure is generated.
    config = make_config(height=2, width=2, survivors=3, structure=0.5, debris=0.0)
    result = run_experiment(config, stay_factory, range(12), bootstrap=FAST)
    statuses = {e.status for e in result.episodes}
    assert statuses == {"completed", "failed"}
    failed = [e for e in result.episodes if e.status == "failed"]
    assert all(e.error is not None and e.error["type"] == "ScenarioError" for e in failed)
    assert result.summary["recall"].n_failed == len(failed)


def test_undefined_metrics_are_counted_not_averaged() -> None:
    result = run_experiment(make_config(survivors=0), stay_factory, [0, 1, 2], bootstrap=FAST)
    recall = result.summary["recall"]
    assert (recall.n_completed, recall.n_defined, recall.n_undefined) == (3, 0, 3)
    assert recall.mean is None
    assert result.summary["mission_success"].n_undefined == 3


def test_per_agent_oracle_policy_rejected_up_front() -> None:
    class Oracle(StayPolicy):
        uses_oracle = True

    with pytest.raises(ValueError, match="cannot use the oracle"):
        run_experiment(make_config(), lambda a, s: Oracle(), [0], bootstrap=FAST)


def test_result_json_round_trip() -> None:
    result = run_experiment(m2_config(), random_valid_factory, [0, 1], bootstrap=FAST)
    restored = ExperimentResult.from_dict(json.loads(result.to_json()))
    assert restored.episodes == result.episodes
    assert restored.summary == result.summary
    assert restored.config == result.config
    assert restored.seeds == result.seeds


def test_result_json_rejects_tampered_config() -> None:
    record = json.loads(run_experiment(make_config(), stay_factory, [0], bootstrap=FAST).to_json())
    record["config"]["time_limit"] += 1
    with pytest.raises(ValueError, match="config_hash"):
        ExperimentResult.from_dict(record)


# ----------------------------------------------------------------- comparisons


def test_comparison_uses_identical_seeds_and_pairs_by_seed() -> None:
    config = m2_config()
    seeds = [0, 1, 2, 3, 4]
    a = run_experiment(config, stay_factory, seeds, bootstrap=FAST)
    b = run_experiment(config, random_valid_factory, seeds, bootstrap=FAST)
    cmp = compare_experiments(a, b, bootstrap=FAST)
    assert cmp["seeds"] == seeds
    assert cmp["excluded_failed_seeds"] == []
    diff = cmp["metrics"]["coverage_visited"]
    per_seed = [
        (eb.metrics or {})["coverage_visited"] - (ea.metrics or {})["coverage_visited"]
        for ea, eb in zip(a.episodes, b.episodes, strict=True)
    ]
    assert diff["n_pairs"] == 5
    assert diff["mean_difference"] == pytest.approx(float(np.mean(per_seed)))


def test_comparison_rejects_different_seed_lists() -> None:
    a = run_experiment(make_config(), stay_factory, [0, 1], bootstrap=FAST)
    b = run_experiment(make_config(), stay_factory, [1, 0], bootstrap=FAST)
    with pytest.raises(ComparisonError, match="seed lists differ"):
        compare_experiments(a, b)


def test_comparison_rejects_different_configs() -> None:
    a = run_experiment(make_config(), stay_factory, [0], bootstrap=FAST)
    b = run_experiment(replace(make_config(), time_limit=31), stay_factory, [0], bootstrap=FAST)
    with pytest.raises(ComparisonError, match="config hashes differ"):
        compare_experiments(a, b)


def test_comparison_excludes_and_lists_failed_seeds() -> None:
    config = make_config(time_limit=5)
    a = run_experiment(config, stay_factory, [0, 1], bootstrap=FAST)
    b = run_experiment(config, crash_factory, [0, 1], bootstrap=FAST)
    cmp = compare_experiments(a, b, bootstrap=FAST)
    assert cmp["excluded_failed_seeds"] == [0, 1]
    assert cmp["metrics"]["recall"]["n_pairs"] == 0


# ----------------------------------------------------------------- CLIs


def test_experiment_cli_and_compare_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_a, out_b = tmp_path / "a.json", tmp_path / "b.json"
    base = ["--config", str(M2_CONFIG), "--seeds", "0-2", "--resamples", "200"]
    assert experiment_cli.main([*base, "--out", str(out_a)]) == 0
    assert experiment_cli.main([*base, "--out", str(out_b)]) == 0
    capsys.readouterr()
    a, b = json.loads(out_a.read_text()), json.loads(out_b.read_text())
    assert a["summary"] == b["summary"]
    assert a["seeds"] == [0, 1, 2]
    out_c = tmp_path / "c.json"
    assert (
        compare_cli.main(["--baseline", str(out_a), "--candidate", str(out_b), "--out", str(out_c)])
        == 0
    )
    cmp = json.loads(out_c.read_text())
    assert cmp["metrics"]["recall"]["mean_difference"] == 0.0


def test_experiment_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        experiment_cli.main(
            ["--config", str(M2_CONFIG), "--seeds", "2-1", "--out", str(tmp_path / "x.json")]
        )
        == 2
    )
    assert "descending" in capsys.readouterr().err


def test_compare_cli_refuses_mismatched_seeds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    common = ["--config", str(M2_CONFIG), "--resamples", "200"]
    experiment_cli.main([*common, "--seeds", "0-1", "--out", str(a)])
    experiment_cli.main([*common, "--seeds", "0-2", "--out", str(b)])
    capsys.readouterr()
    code = compare_cli.main(
        ["--baseline", str(a), "--candidate", str(b), "--out", str(tmp_path / "c.json")]
    )
    assert code == 2
    assert "seed lists differ" in capsys.readouterr().err

"""Random baseline, episode runner, result records and the CLI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from swarm.core.observation import Observation
from swarm.core.types import Action, AgentId, AgentStatus, Layer
from swarm.evaluation.metrics import METRIC_DEFINITIONS
from swarm.experiments import run_episode as cli
from swarm.experiments.runner import RESULT_SCHEMA_VERSION, run_episode, write_result
from swarm.policies.random_valid import RandomValidPolicy, random_valid_factory
from swarm.scenarios.config import AgentTypeSpec, ScenarioConfig
from tests.helpers import make_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "m1_default.json"


class StayPolicy:
    name = "stay"
    version = "1"
    uses_oracle = False

    def act(self, observation: Observation, rng: np.random.Generator) -> int:
        return int(Action.STAY)


class OraclePolicy(StayPolicy):
    name = "oracle"
    uses_oracle = True


def stay_factory(agent_id: AgentId, spec: AgentTypeSpec) -> StayPolicy:
    return StayPolicy()


def oracle_factory(agent_id: AgentId, spec: AgentTypeSpec) -> OraclePolicy:
    return OraclePolicy()


def default_config() -> ScenarioConfig:
    return ScenarioConfig.from_json(DEFAULT_CONFIG.read_text(encoding="utf-8"))


# ----------------------------------------------------------------- random baseline


@given(mask=st.lists(st.booleans(), min_size=4, max_size=4), seed=st.integers(0, 10_000))
def test_random_policy_only_picks_valid_actions(mask: list[bool], seed: int) -> None:
    full_mask = (True, *mask)  # STAY is always valid
    obs = _obs_with_mask(full_mask)
    rng = np.random.default_rng(seed)
    picks = {RandomValidPolicy().act(obs, rng) for _ in range(30)}
    assert all(full_mask[a] for a in picks)


def test_random_policy_covers_all_valid_actions() -> None:
    obs = _obs_with_mask((True, True, False, True, False))
    rng = np.random.default_rng(0)
    picks = [RandomValidPolicy().act(obs, rng) for _ in range(600)]
    counts = np.bincount(picks, minlength=5)
    assert counts[2] == counts[4] == 0
    assert all(150 < counts[a] < 250 for a in (0, 1, 3))  # ~200 each


def _obs_with_mask(mask: tuple[bool, ...]) -> Observation:
    return Observation(
        agent_id="a_0",
        agent_type="a",
        layer=Layer.AIR,
        step=0,
        position=(0, 0),
        energy=1,
        energy_capacity=1,
        status=AgentStatus.ACTIVE,
        sensor_radius=1,
        grid_shape=(1, 1),
        base_position=(0, 0),
        local_terrain=np.zeros((3, 3), dtype=np.int8),
        local_hazard=np.zeros((3, 3), dtype=np.bool_),
        visible_agents=(),
        detections=(),
        action_mask=mask,
    )


# ----------------------------------------------------------------- reproducibility


def test_same_seed_reproduces_events_and_metrics_exactly() -> None:
    config = default_config()
    a = run_episode(config, random_valid_factory, 11)
    b = run_episode(config, random_valid_factory, 11)
    assert a.event_stream_sha256 == b.event_stream_sha256
    assert a.events == b.events
    assert a.metrics == b.metrics
    assert a.scenario_sha256 == b.scenario_sha256
    assert a.metadata["run"]["run_id"] != b.metadata["run"]["run_id"]


def test_different_seeds_give_different_scenarios_and_streams() -> None:
    config = default_config()
    results = [run_episode(config, random_valid_factory, s) for s in range(5)]
    assert len({r.scenario_sha256 for r in results}) == 5
    assert len({r.event_stream_sha256 for r in results}) == 5


def test_policy_swap_does_not_change_the_scenario() -> None:
    config = default_config()
    random_run = run_episode(config, random_valid_factory, 4)
    stay_run = run_episode(config, stay_factory, 4)
    assert random_run.scenario_sha256 == stay_run.scenario_sha256
    assert random_run.event_stream_sha256 != stay_run.event_stream_sha256


def test_stay_policy_episode_is_fully_predictable() -> None:
    config = make_config(survivors=0, time_limit=7)
    result = run_episode(config, stay_factory, 0)
    assert result.end_reason == "time_limit"
    assert result.steps == 7
    assert result.metrics.recall is None
    assert result.metrics.coverage_visited > 0
    assert result.metrics.move_conflicts == 0


# ----------------------------------------------------------------- result record


def test_result_record_contains_required_metadata() -> None:
    result = run_episode(default_config(), random_valid_factory, 0)
    record = json.loads(result.to_json())
    assert record["result_schema_version"] == RESULT_SCHEMA_VERSION
    assert record["designation"] == "evaluation"
    assert record["seed"] == 0
    assert record["policy"] == {"name": "random_valid", "version": "1", "uses_oracle": False}
    assert ScenarioConfig.from_dict(record["config"]) == default_config()
    assert record["config_hash"] == default_config().config_hash()
    assert set(record["metrics"]) == set(METRIC_DEFINITIONS)
    assert set(record["metric_definitions"]) == set(METRIC_DEFINITIONS)
    meta = record["metadata"]
    assert meta["seed_derivation"]["root_seed"] == 0
    assert set(meta["schema_versions"]) == {"config", "events", "metrics", "result"}
    assert meta["software"]["numpy"] == np.__version__
    assert set(meta["code"]) == {"commit", "dirty"}
    assert {"run_id", "started_at", "duration_seconds", "platform"} <= set(meta["run"])
    assert record["episode"]["event_count"] == len(result.events)


def test_written_event_stream_matches_recorded_hash(tmp_path: Path) -> None:
    result = run_episode(default_config(), random_valid_factory, 2)
    write_result(result, tmp_path / "r.json", tmp_path / "e.jsonl")
    stream = (tmp_path / "e.jsonl").read_bytes()
    assert hashlib.sha256(stream).hexdigest() == result.event_stream_sha256
    assert json.loads((tmp_path / "r.json").read_text())["event_stream_sha256"] == (
        result.event_stream_sha256
    )


def test_oracle_policies_are_rejected_until_phase_2() -> None:
    with pytest.raises(ValueError, match="OracleView"):
        run_episode(make_config(), oracle_factory, 0)


def test_invalid_designation_and_seed_rejected() -> None:
    with pytest.raises(ValueError, match="designation"):
        run_episode(make_config(), stay_factory, 0, designation="training")
    with pytest.raises(ValueError, match="seed"):
        run_episode(make_config(), stay_factory, -1)


# ----------------------------------------------------------------- CLI


def test_cli_writes_result_and_events(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out, events = tmp_path / "r.json", tmp_path / "e.jsonl"
    code = cli.main(
        ["--config", str(DEFAULT_CONFIG), "--seed", "3", "--out", str(out), "--events", str(events)]
    )
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["event_stream_sha256"] == json.loads(out.read_text())["event_stream_sha256"]
    assert events.read_text().count("\n") == json.loads(out.read_text())["episode"]["event_count"]


def test_cli_render_hides_survivors_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["--config", str(DEFAULT_CONFIG), "--seed", "0", "--out", str(tmp_path / "r.json")]
    assert cli.main([*args, "--render"]) == 0
    hidden = capsys.readouterr().out
    assert cli.main([*args, "--render", "--show-survivors"]) == 0
    shown = capsys.readouterr().out
    grid_hidden = hidden.split("legend:")[0]
    grid_shown = shown.split("legend:")[0]
    assert "S" not in grid_hidden.replace("step", "")
    assert "S" in grid_shown.replace("step", "")


def test_cli_reports_bad_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 1}))
    code = cli.main(["--config", str(bad), "--seed", "0", "--out", str(tmp_path / "r.json")])
    assert code == 2
    assert "missing required key" in capsys.readouterr().err
    code = cli.main(["--config", str(tmp_path / "nope.json"), "--seed", "0", "--out", "x.json"])
    assert code == 2

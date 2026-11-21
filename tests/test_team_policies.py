"""Team-policy infrastructure, oracle gating and information labels (ADR-0014)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from swarm.core.observation import Observation
from swarm.core.oracle import OracleView
from swarm.core.types import Action, AgentId, AgentStatus, Layer
from swarm.experiments.runner import describe_policy, run_episode
from swarm.policies.base import (
    MissionBrief,
    TeamFactory,
    build_team,
)
from swarm.policies.random_valid import random_valid_factory
from swarm.scenarios.config import HazardConfig, ScenarioConfig
from tests.helpers import make_config


class StayTeam:
    """Minimal team policy that records what it was given."""

    name = "stay_team"
    version = "1"

    def __init__(self, brief: MissionBrief, information: str = "centralized_pooled") -> None:
        self.brief = brief
        self.information = information
        self.uses_oracle = information == "oracle"
        self.oracle: OracleView | None = None
        self.seen_steps: list[set[str]] = []

    def act(
        self,
        observations: Mapping[AgentId, Observation],
        rngs: Mapping[AgentId, np.random.Generator],
    ) -> dict[AgentId, int]:
        self.seen_steps.append(set(observations))
        return {
            a: int(Action.STAY) for a, o in observations.items() if o.status is AgentStatus.ACTIVE
        }

    def attach_oracle(self, view: OracleView) -> None:
        self.oracle = view

    def diagnostics(self) -> dict[str, int | float | None]:
        return {"steps_seen": len(self.seen_steps)}

    def decision_log(self) -> list[dict[str, Any]]:
        return [{"step": 0, "note": "started"}]


def _config() -> ScenarioConfig:
    return make_config(survivors=2, time_limit=5)


def test_brief_is_an_allow_list_without_ground_truth() -> None:
    config = make_config(survivors=3)
    from dataclasses import replace

    config = replace(
        config,
        hazards=HazardConfig(0.3, (Layer.GROUND,), 4, 0.1),
    )
    brief = MissionBrief.from_config(config)
    fields = set(MissionBrief.__dataclass_fields__)
    assert fields == {
        "grid_shape",
        "base",
        "time_limit",
        "agents",
        "hazard",
        "p_detect",
        "p_false_positive",
        "recall_target",
        "comms",  # public channel parameters only (ADR-0018); None without a channel
        "perception",  # public sensor-model parameters only (ADR-0025); None by default
    }
    assert brief.comms is None
    assert brief.perception is None
    assert brief.hazard is not None
    assert set(type(brief.hazard).__dataclass_fields__) == {
        "affected_layers",
        "energy_cost",
        "disable_probability",
    }
    assert brief.hazard_energy_cost == 4


def test_team_factory_runs_and_records_identity_and_diagnostics() -> None:
    teams: list[StayTeam] = []

    def build(brief: MissionBrief) -> StayTeam:
        teams.append(StayTeam(brief))
        return teams[-1]

    result = run_episode(_config(), TeamFactory("stay_team", build), 0)
    assert result.policy == {
        "name": "stay_team",
        "version": "1",
        "uses_oracle": False,
        "information": "centralized_pooled",
    }
    assert result.policy_diagnostics == {"steps_seen": 5}
    assert result.decision_log == ({"step": 0, "note": "started"},)
    assert teams[0].oracle is None  # non-oracle teams never receive ground truth


def test_oracle_team_receives_ground_truth_and_is_labelled() -> None:
    teams: list[StayTeam] = []

    def build(brief: MissionBrief) -> StayTeam:
        teams.append(StayTeam(brief, information="oracle"))
        return teams[-1]

    result = run_episode(_config(), TeamFactory("oracle_stay", build), 3)
    view = teams[0].oracle
    assert view is not None
    assert view.survivors == result.final_snapshot.scenario.survivors
    assert result.policy["information"] == "oracle"
    assert result.policy["uses_oracle"] is True
    with pytest.raises(ValueError, match="read-only"):
        view.terrain[0, 0] = 0


def test_inconsistent_oracle_label_is_rejected() -> None:
    class Liar(StayTeam):
        def __init__(self, brief: MissionBrief) -> None:
            super().__init__(brief)
            self.uses_oracle = True  # claims oracle access without the oracle label

    with pytest.raises(ValueError, match="uses_oracle must be True exactly"):
        run_episode(_config(), TeamFactory("liar", Liar), 0)


def test_team_must_act_for_exactly_the_active_agents() -> None:
    class Lazy(StayTeam):
        def act(
            self,
            observations: Mapping[AgentId, Observation],
            rngs: Mapping[AgentId, np.random.Generator],
        ) -> dict[AgentId, int]:
            return {}

    with pytest.raises(ValueError, match="returned actions"):
        run_episode(_config(), TeamFactory("lazy", Lazy), 0)


def test_random_policy_through_independent_team_is_unchanged() -> None:
    """IndependentTeam reproduces the Phase 1 call order (golden hashes cover this too)."""
    team = build_team(random_valid_factory, MissionBrief.from_config(_config()))
    assert team.information == "decentralized"
    assert describe_policy(random_valid_factory, _config())["information"] == "decentralized"


def test_mission_success_episode_is_recorded_as_terminated() -> None:
    """Regression: M2 recorded mission_success episodes as neither terminated nor truncated."""
    from swarm.scenarios.config import AgentSpec
    from tests.helpers import config_for, spec

    config, _ = config_for(["BS"], (spec("gnd", Layer.GROUND),), (AgentSpec("gnd"),))
    for seed in range(10):
        result = run_episode(config, random_valid_factory, seed)
        if result.end_reason == "mission_success":
            assert result.terminated is True
            assert result.truncated is False
            return
    pytest.fail("no mission_success episode in 10 seeds")

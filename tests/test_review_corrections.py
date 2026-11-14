"""Regression tests for the PR #5 review corrections (ADR-0016 rules 1, 15 and 16).

Each test failed on the reviewed implementation (commit d7c461f) and passes after the correction.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import swarm.policies.centralized as centralized
from swarm.coordination.assignment import min_cost_assignment
from swarm.core.env import SwarmEnv
from swarm.core.rng import episode_streams
from swarm.core.types import Action, Layer, Position
from swarm.navigation.astar import UNREACHABLE
from swarm.navigation.energy import EnergyModel
from swarm.navigation.known_map import KnownMap
from swarm.navigation.prioritized import PlanRequest, plan_prioritized
from swarm.policies.base import MissionBrief, build_team
from swarm.policies.decentralized import NavigatorAgent
from swarm.policies.planning import Counters
from swarm.policies.registry import POLICIES
from swarm.scenarios.config import AgentSpec, HazardConfig, ScenarioConfig
from tests.helpers import config_for, spec

ROOT = Path(__file__).resolve().parents[1]
M2 = ScenarioConfig.from_json((ROOT / "configs" / "m2_default.json").read_text(encoding="utf-8"))
POOLED = ["prioritized_frontier", "assignment", "assignment_hazard_aware", "assignment_oracle"]
GROUND_HAZARD = HazardConfig(0.0, (Layer.GROUND,), 5, 0.0)


# ----------------------------------------------------------------- independent energy accounting


def _env_energy(model: EnergyModel, known: KnownMap, path: list[Position]) -> int:
    """Energy the *environment* charges for a timed path (ADR-0011): per step, move or stay cost,
    plus exposure whenever the step ends in a hazard. Written independently of EnergyModel."""
    total = 0
    for a, b in itertools.pairwise(path):
        total += model.move_cost if a != b else model.stay_cost
        total += model.hazard_cost if bool(known.hazards[b]) else 0
    return total


class _Recorder:
    """Wraps ``plan_prioritized`` in the centralized module and keeps the latest round."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[PlanRequest], Any]] = []

    def __call__(self, requests: list[PlanRequest], *args: Any) -> Any:
        result = plan_prioritized(requests, *args)
        self.calls.append((list(requests), result))
        return result


def _check_round(team: Any, recorder: _Recorder) -> tuple[int, int]:
    """Assert the commit rule on the executed round; return (#explorer plans, #return plans)."""
    requests, plans = recorder.calls[-1]
    explorers = returns = 0
    for req in requests:
        st = team.states[req.agent_id]
        if st.returning:
            returns += 1
            # Returning agents plan on energy costs, matching the energy-to-base field.
            assert req.enter_cost is not None
            assert np.array_equal(req.enter_cost, st.model.enter_energy(team.known))
            continue
        path = plans.paths[req.agent_id]
        if st.target is None or path is None:
            continue
        explorers += 1
        to_base = int(team.cache.to_base_for(team.known, st.model)[path[-1]])
        assert to_base != UNREACHABLE
        needed = _env_energy(st.model, team.known, path) + to_base
        assert st.energy - st.model.reserve >= needed, (req.agent_id, st.energy, needed)
    return explorers, returns


def _run_with_recorder(
    team: Any, env: SwarmEnv, obs: Any, seed: int, recorder: _Recorder, steps: int
) -> tuple[int, int]:
    rngs = episode_streams(seed, env.possible_agents).policies
    explorers = returns = 0
    for _ in range(steps):
        if env.done:
            break
        actions = team.act(obs, rngs)
        e, r = _check_round(team, recorder)
        explorers, returns = explorers + e, returns + r
        obs = env.step(actions).observations
    return explorers, returns


# ----------------------------------------------------------------- 1. pooled route feasibility


def test_oracle_does_not_commit_to_a_hazard_shortcut(monkeypatch: pytest.MonkeyPatch) -> None:
    """The step-cost shortest route to the survivor crosses four known hazards (24 energy) while
    the energy-cheapest route avoids them (15). With 30 energy, reserve 3 and 10 back to base,
    only the cheap route is feasible, but the planner executes the shortest one. Before the
    correction the oracle committed on the cheap route and executed the hazard route."""
    gnd = spec("gnd", Layer.GROUND, radius=1, capacity=30, move=1, stay=0)
    config, scenario = config_for(
        ["B....S", ".####.", "......"],
        (gnd,),
        (AgentSpec("gnd"),),
        time_limit=40,
        hazard_cells=((0, 1), (0, 2), (0, 3), (0, 4)),
        hazards=GROUND_HAZARD,
    )
    recorder = _Recorder()
    monkeypatch.setattr(centralized, "plan_prioritized", recorder)
    team = build_team(POLICIES["assignment_oracle"], MissionBrief.from_config(config))
    env = SwarmEnv(config)
    obs, _ = env.reset(0, scenario=scenario)
    team.attach_oracle(env.oracle_view())
    _run_with_recorder(team, env, obs, 0, recorder, 40)
    assert len(recorder.calls) > 0


@pytest.mark.parametrize(
    ("name", "capacity", "expect_returns"),
    [
        ("prioritized_frontier", 20, True),
        ("assignment", 20, True),
        ("assignment_hazard_aware", 20, True),
        ("assignment_oracle", 40, False),
    ],
)
def test_two_route_map_commit_rule_and_energy_returns(
    name: str, capacity: int, expect_returns: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Top row: a six-cell hazard shortcut. Bottom row: a longer safe route. Small batteries force
    repeated returns. Every executed exploring plan must satisfy the commit rule with its own
    energy, and returns must plan on energy costs. All four failed before the correction."""
    gnd = spec("gnd", Layer.GROUND, radius=1, capacity=capacity, move=1, stay=0, recharge=4)
    config, scenario = config_for(
        ["B......S", ".######.", "........"],
        (gnd,),
        (AgentSpec("gnd"),),
        time_limit=120,
        hazard_cells=tuple((0, c) for c in range(1, 7)),
        hazards=GROUND_HAZARD,
    )
    recorder = _Recorder()
    monkeypatch.setattr(centralized, "plan_prioritized", recorder)
    team = build_team(POLICIES[name], MissionBrief.from_config(config))
    env = SwarmEnv(config)
    obs, _ = env.reset(0, scenario=scenario)
    if team.uses_oracle:
        team.attach_oracle(env.oracle_view())
    explorers, returns = _run_with_recorder(team, env, obs, 0, recorder, 120)
    if expect_returns:
        assert explorers > 0
        assert returns > 0


@pytest.mark.parametrize("name", POOLED)
def test_pooled_plans_satisfy_the_commit_rule_on_the_executed_route(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On M2 development seeds (ground hazards cost 5 energy per step), every executed exploring
    plan satisfies the commit rule with its own energy, and every return plan uses energy costs."""
    recorder = _Recorder()
    monkeypatch.setattr(centralized, "plan_prioritized", recorder)
    explorers = returns = 0
    for seed in (1000, 1001):
        team = build_team(POLICIES[name], MissionBrief.from_config(M2))
        env = SwarmEnv(M2)
        obs, _ = env.reset(seed)
        if team.uses_oracle:
            team.attach_oracle(env.oracle_view())
        e, r = _run_with_recorder(team, env, obs, seed, recorder, M2.time_limit)
        explorers, returns = explorers + e, returns + r
    assert explorers > 0  # the check is not vacuous


# ----------------------------------------------------------------- 2. hazard waits cost energy


def test_route_energy_charges_hazard_exposure_while_waiting() -> None:
    """Move into a hazard, wait there twice, move on: the model must match the environment."""
    gnd = spec("gnd", Layer.GROUND, radius=1, capacity=50, move=1, stay=0)
    config, scenario = config_for(
        ["B..."],
        (gnd,),
        (AgentSpec("gnd"),),
        hazard_cells=((0, 1),),
        hazards=GROUND_HAZARD,
    )
    env = SwarmEnv(config)
    obs, _ = env.reset(0, scenario=scenario)
    known = KnownMap(scenario.shape, scenario.base)
    known.update(obs["gnd_0"])
    start = obs["gnd_0"].energy
    for action in (Action.EAST, Action.STAY, Action.STAY, Action.EAST):
        obs = env.step({"gnd_0": int(action)}).observations
    spent = start - obs["gnd_0"].energy
    model = EnergyModel.for_agent(gnd, (Layer.GROUND,), 5)
    path = [(0, 0), (0, 1), (0, 1), (0, 1), (0, 2)]
    assert spent == 1 + 5 + 5 + 5 + 1 == _env_energy(model, known, path)
    assert model.route_energy(path, model.enter_energy(known)) == spent
    assert int(model.wait_energy(known)[0, 1]) == 5
    assert int(model.wait_energy(known)[0, 2]) == 0


# ----------------------------------------------------------------- 3 + 5. disabled agents


def _disable_first_robot() -> tuple[ScenarioConfig, SwarmEnv, dict[str, Any]]:
    """gnd_0 (2 energy) moves east twice and is disabled at (0, 2); gnd_1 watches from the base."""
    gnd = spec("gnd", Layer.GROUND, radius=6, capacity=10, move=1, stay=0)
    config, scenario = config_for(
        ["B....."], (gnd,), (AgentSpec("gnd", initial_energy=2), AgentSpec("gnd")), time_limit=20
    )
    env = SwarmEnv(config)
    env.reset(0, scenario=scenario)
    obs = env.step({"gnd_0": int(Action.EAST), "gnd_1": 0}).observations
    assert obs["gnd_0"].status.value == "active"
    obs = env.step({"gnd_0": int(Action.EAST), "gnd_1": 0}).observations
    assert obs["gnd_0"].status.value == "disabled"
    assert obs["gnd_0"].position == (0, 2)
    return config, env, obs


def test_known_map_blocks_visibly_disabled_agents_only() -> None:
    config, _, obs = _disable_first_robot()
    known = KnownMap((config.height, config.width), config.base)
    known.update(obs["gnd_1"])  # gnd_1's own observation shows gnd_0 disabled
    assert not known.passable(Layer.GROUND)[0, 2]
    assert known.passable(Layer.AIR)[0, 2]  # a ground wreck does not block the air layer
    assert known.passable(Layer.GROUND)[0, 1]
    version = known.version
    known.update(obs["gnd_1"])
    assert known.version == version  # no new information, no cache invalidation


def test_pooled_return_field_treats_disabled_agents_as_obstacles() -> None:
    config, env, obs = _disable_first_robot()
    team = build_team(POLICIES["assignment"], MissionBrief.from_config(config))
    team.act(obs, episode_streams(0, env.possible_agents).policies)
    st = team.states["gnd_1"]  # type: ignore[attr-defined]
    to_base = team.cache.to_base_for(team.known, st.model)  # type: ignore[attr-defined]
    assert int(to_base[0, 1]) == 1
    assert int(to_base[0, 4]) == UNREACHABLE  # the only corridor is blocked at (0, 2)


# ----------------------------------------------------------------- 4. stale path at the base


def test_idle_agent_at_base_drops_a_stale_path() -> None:
    gnd = spec("gnd", Layer.GROUND, radius=5, capacity=100)
    config, scenario = config_for(["B...."], (gnd,), (AgentSpec("gnd"),))
    env = SwarmEnv(config)
    obs, _ = env.reset(0, scenario=scenario)
    nav = NavigatorAgent(
        "gnd_0",
        gnd,
        MissionBrief.from_config(config),
        target_rule="frontier",
        hazard_aware=False,
        counters=Counters(),
    )
    nav.path = [(0, 1), (0, 0)]  # left over from a return that just ended on the base
    action = nav.act(obs["gnd_0"], np.random.default_rng(0))  # fully known map: no frontier
    assert action == int(Action.STAY)
    assert nav.path == []


# ----------------------------------------------------------------- 6. assignment tie-breaking


def test_assignment_ties_are_deterministic() -> None:
    cost = np.zeros((3, 3), dtype=np.int64)
    first = min_cost_assignment(cost)
    assert first == min_cost_assignment(cost.copy())
    assert sorted(r for r, _ in first) == [0, 1, 2]
    assert sorted(c for _, c in first) == [0, 1, 2]


# ----------------------------------------------------------------- 7. failure classification


def test_plan_failures_are_classified() -> None:
    open3 = np.ones((3, 3), dtype=np.bool_)
    walled = open3.copy()
    walled[0, 1] = walled[1, 1] = walled[2, 1] = False
    requests = [
        # a reaches (1, 0) first and holds it, so b's goal is never free: temporary failure.
        PlanRequest("a", Layer.GROUND, (0, 0), (1, 0), open3),
        PlanRequest("b", Layer.GROUND, (2, 0), (1, 0), open3),
        # u's goal is behind a wall on its own map: unreachable.
        PlanRequest("u", Layer.AIR, (0, 0), (0, 2), walled),
    ]
    plans = plan_prioritized(requests, (2, 2), horizon=12, round_index=0)
    assert plans.paths["a"] is not None
    assert plans.paths["b"] is None
    assert plans.paths["u"] is None
    assert plans.unreachable == frozenset({"u"})


def test_failure_counters_are_reported() -> None:
    keys = Counters().as_dict()
    for key in ("failed_plans_unreachable", "failed_plans_temporary", "energy_route_aborts"):
        assert keys[key] == 0

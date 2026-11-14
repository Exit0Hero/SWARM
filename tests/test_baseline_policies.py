"""Phase 2 baseline policies (ADR-0014, ADR-0016): behaviour, safety, leakage, reproducibility."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from swarm.core.env import SwarmEnv
from swarm.core.events import AgentDisabled
from swarm.core.oracle import OracleView
from swarm.core.rng import episode_streams
from swarm.core.types import Layer, Terrain
from swarm.evaluation.statistics import BootstrapSpec
from swarm.experiments.experiment import run_experiment
from swarm.experiments.runner import run_episode
from swarm.navigation.known_map import KnownMap
from swarm.policies.base import MissionBrief, build_team
from swarm.policies.decentralized import NavigatorAgent
from swarm.policies.planning import Counters
from swarm.policies.registry import POLICIES
from swarm.scenarios.config import AgentSpec, HazardConfig, ScenarioConfig
from swarm.scenarios.generator import Scenario
from tests.helpers import config_for, make_config, spec

ROOT = Path(__file__).resolve().parents[1]
M2 = ScenarioConfig.from_json((ROOT / "configs" / "m2_default.json").read_text(encoding="utf-8"))
DEV_SEEDS = [1000, 1001, 1002, 1003]
NON_ORACLE = [n for n in POLICIES if n != "assignment_oracle"]
PLANNERS = [n for n in POLICIES if n != "random_valid"]
CENTRALIZED = ["prioritized_frontier", "assignment", "assignment_hazard_aware", "assignment_oracle"]


# ----------------------------------------------------------------- registry and labels


def test_registry_labels() -> None:
    expected = {
        "random_valid": "decentralized",
        "astar_waypoint": "decentralized",
        "frontier": "decentralized",
        "frontier_hazard_aware": "decentralized",
        "prioritized_frontier": "centralized_pooled",
        "assignment": "centralized_pooled",
        "assignment_hazard_aware": "centralized_pooled",
        "assignment_oracle": "oracle",
    }
    brief = MissionBrief.from_config(M2)
    for name, info in expected.items():
        team = build_team(POLICIES[name], brief)
        assert team.name == name
        assert team.information == info
        assert team.uses_oracle == (info == "oracle")


@pytest.mark.parametrize("name", [n for n in PLANNERS if n != "assignment_oracle"])
def test_non_oracle_teams_refuse_the_oracle(name: str) -> None:
    team = build_team(POLICIES[name], MissionBrief.from_config(M2))
    view = OracleView(np.zeros((1, 1), dtype=np.int8), np.zeros((1, 1), dtype=np.bool_), (), (0, 0))
    with pytest.raises(RuntimeError):
        team.attach_oracle(view)


# ----------------------------------------------------------------- frontier selection


def _navigator(
    rows: list[str], radius: int = 1, capacity: int = 100
) -> tuple[SwarmEnv, NavigatorAgent]:
    gnd = spec("gnd", Layer.GROUND, radius=radius, capacity=capacity)
    config, scenario = config_for(rows, (gnd,), (AgentSpec("gnd"),), time_limit=200)
    env = SwarmEnv(config)
    env.reset(0, scenario=scenario)
    brief = MissionBrief.from_config(config)
    nav = NavigatorAgent(
        "gnd_0", gnd, brief, target_rule="frontier", hazard_aware=False, counters=Counters()
    )
    return env, nav


def test_frontier_selects_nearest_reachable_frontier() -> None:
    env, nav = _navigator(["....", "B...", "...."])
    observation = env.step({"gnd_0": 0}).observations["gnd_0"]
    action = nav.act(observation, np.random.default_rng(0))
    assert nav.target is not None
    known_frontier = nav.known.frontier(Layer.GROUND)
    assert known_frontier[nav.target]
    # The base (1, 0) itself is a frontier only if next to unknown; the chosen target is at
    # distance 1 (the nearest possible) and ties break by (row, col).
    assert abs(nav.target[0] - 1) + abs(nav.target[1]) == 1
    assert nav.target == (0, 0)
    assert action in (1, 2, 3, 4)


def test_frontier_behind_a_wall_is_never_selected() -> None:
    # Radius 2 lets the robot see open cells past the wall at column 1; they are unreachable.
    env, nav = _navigator(["B#..", "##..", "...."], radius=2)
    observation = env.step({"gnd_0": 0}).observations["gnd_0"]
    nav.act(observation, np.random.default_rng(0))
    assert nav.target is None  # every frontier it knows of is unreachable
    assert nav.known.frontier(Layer.GROUND)[0, 2]


def test_exhausted_target_triggers_replanning() -> None:
    env, nav = _navigator(["......", "B.....", "......"])
    rng = np.random.default_rng(0)
    obs = env.step({"gnd_0": 0}).observations["gnd_0"]
    for _ in range(12):
        action = nav.act(obs, rng)
        obs = env.step({"gnd_0": action}).observations["gnd_0"]
    assert nav.counters.targets_selected >= 3
    assert nav.counters.targets_reached + nav.counters.target_invalidations >= 2
    assert not (nav.known.terrain == -1).all()


def test_fully_explored_map_returns_and_idles_at_base() -> None:
    env, nav = _navigator(["B.", ".."], radius=2)
    rng = np.random.default_rng(0)
    obs = env.step({"gnd_0": 0}).observations["gnd_0"]
    actions = []
    for _ in range(5):
        a = nav.act(obs, rng)
        actions.append(a)
        obs = env.step({"gnd_0": a}).observations["gnd_0"]
    assert set(actions) == {0}
    assert obs.position == (0, 0)


# ----------------------------------------------------------------- energy safety


@pytest.mark.parametrize(
    "name", ["astar_waypoint", "frontier", "prioritized_frontier", "assignment"]
)
def test_commit_rule_prevents_energy_depletion_on_a_long_corridor(name: str) -> None:
    # Capacity 30 cannot cover the 60-cell corridor out and back; agents must turn around.
    gnd = spec("gnd", Layer.GROUND, radius=1, capacity=30, move=1, stay=0, recharge=5)
    rows = ["B" + "." * 59]
    config, scenario = config_for(rows, (gnd,), (AgentSpec("gnd"),), time_limit=150)
    team = build_team(POLICIES[name], MissionBrief.from_config(config))
    env = SwarmEnv(config)
    obs, _ = env.reset(0, scenario=scenario)
    rngs = episode_streams(0, env.possible_agents).policies
    events: list[object] = []
    far = 0
    while not env.done:
        result = env.step(team.act(obs, rngs))
        obs = result.observations
        events.extend(result.events)
        far = max(far, env.snapshot().agents[0].position[1])
    assert not [e for e in events if isinstance(e, AgentDisabled)]
    assert 5 <= far <= 14  # explored outwards but never beyond the safe radius ((30 - 3) / 2)


def test_agent_waits_at_base_when_no_target_is_feasible() -> None:
    gnd = spec("gnd", Layer.GROUND, radius=1, capacity=30, move=1, stay=0, recharge=0)
    config, scenario = config_for(
        ["B" + "." * 10], (gnd,), (AgentSpec("gnd", initial_energy=3),), time_limit=10
    )
    team = build_team(POLICIES["frontier"], MissionBrief.from_config(config))
    env = SwarmEnv(config)
    obs, _ = env.reset(0, scenario=scenario)
    rngs = episode_streams(0, env.possible_agents).policies
    # energy 3 - reserve 3 = 0: nothing is feasible, so the agent stays on the base.
    for _ in range(10):
        actions = team.act(obs, rngs)
        assert actions == {"gnd_0": 0}
        obs = env.step(actions).observations
        if env.done:
            break


# ----------------------------------------------------------------- assignment


def _hazard_trap_config() -> ScenarioConfig:
    return replace(
        make_config(height=10, width=10, survivors=2, time_limit=60, structure=0.0, debris=0.0),
        hazards=HazardConfig(0.25, (Layer.GROUND,), 0, 1.0),
    )


def test_assignment_targets_are_distinct_per_decision_round() -> None:
    for seed in DEV_SEEDS:
        result = run_episode(M2, POLICIES["assignment"], seed)
        rounds: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for entry in result.decision_log:
            if entry["target"] is not None:
                rounds[entry["step"]].append(tuple(entry["target"]))
        for targets in rounds.values():
            assert len(targets) == len(set(targets))


def test_reassignment_after_agent_failure_is_logged() -> None:
    """Ground robots die in hazards (disable prob. 1); the planner reassigns the survivors."""
    found = False
    for seed in range(20):
        result = run_episode(_hazard_trap_config(), POLICIES["assignment"], seed)
        disabled = [e for e in result.events if isinstance(e, AgentDisabled)]
        if not disabled:
            continue
        first = disabled[0]
        after = [e for e in result.decision_log if e["step"] > first.step]
        assert all(e["agent"] != first.agent_id for e in after)
        assert any(e["trigger"] == "agent_set_changed" for e in result.decision_log)
        found = True
        break
    assert found, "no seed produced an agent failure"


def test_oracle_targets_survivors_and_is_labelled() -> None:
    result = run_episode(M2, POLICIES["assignment_oracle"], 1000)
    assert result.policy["information"] == "oracle"
    survivors = result.final_snapshot.scenario.survivors
    for entry in result.decision_log:
        if entry["target"] is not None:
            target = tuple(entry["target"])
            # Target lies inside a sensor footprint (Euclidean radius <= 3) around a survivor.
            assert any((target[0] - s[0]) ** 2 + (target[1] - s[1]) ** 2 <= 9 for s in survivors)


# ----------------------------------------------------------------- conflicts and invalid actions


@pytest.mark.parametrize("name", CENTRALIZED)
def test_centralized_planners_cause_no_conflicts_or_invalid_actions(name: str) -> None:
    for seed in DEV_SEEDS:
        result = run_episode(M2, POLICIES[name], seed)
        assert result.metrics.move_conflicts == 0
        assert result.metrics.invalid_actions == 0


@pytest.mark.parametrize("name", PLANNERS)
def test_planners_never_submit_masked_actions(name: str) -> None:
    for seed in DEV_SEEDS:
        assert run_episode(M2, POLICIES[name], seed).metrics.invalid_actions == 0


# ----------------------------------------------------------------- hidden-state leakage


def _lockstep_until_observations_differ(name: str, seed: int) -> tuple[int, bool]:
    """Run two scenarios that differ only in survivor positions with the same policy.

    Returns (steps compared, whether actions ever differed while observations were equal).
    """
    config = M2
    env_a, env_b = SwarmEnv(config), SwarmEnv(config)
    env_a.reset(seed)
    base_scenario = env_a.snapshot().scenario
    # Move every survivor to a different open/debris cell far from the base.
    free = np.argwhere(
        (base_scenario.terrain == Terrain.OPEN) | (base_scenario.terrain == Terrain.DEBRIS)
    )
    far = sorted(
        ((int(r), int(c)) for r, c in free),
        key=lambda p: -abs(p[0] - base_scenario.base[0]) - abs(p[1] - base_scenario.base[1]),
    )
    moved = tuple(p for p in far if p not in base_scenario.survivors)[
        : len(base_scenario.survivors)
    ]
    other = Scenario(
        terrain=base_scenario.terrain,
        base=base_scenario.base,
        survivors=moved,
        hazards=base_scenario.hazard_grid,
    )
    obs_a, _ = env_a.reset(seed, scenario=base_scenario)
    obs_b, _ = env_b.reset(seed, scenario=other)
    brief = MissionBrief.from_config(config)
    team_a, team_b = build_team(POLICIES[name], brief), build_team(POLICIES[name], brief)
    if team_a.uses_oracle:
        team_a.attach_oracle(env_a.oracle_view())
        team_b.attach_oracle(env_b.oracle_view())
    rngs_a = episode_streams(seed, env_a.possible_agents).policies
    rngs_b = episode_streams(seed, env_b.possible_agents).policies
    steps = 0
    while not (env_a.done or env_b.done) and obs_a == obs_b:
        act_a, act_b = team_a.act(obs_a, rngs_a), team_b.act(obs_b, rngs_b)
        if act_a != act_b:
            return steps, True
        obs_a, obs_b = env_a.step(act_a).observations, env_b.step(act_b).observations
        steps += 1
    return steps, False


@pytest.mark.parametrize("name", NON_ORACLE)
def test_non_oracle_actions_depend_only_on_observations(name: str) -> None:
    for seed in DEV_SEEDS[:2]:
        steps, leaked = _lockstep_until_observations_differ(name, seed)
        assert not leaked, f"{name} acted differently on identical observations (seed {seed})"
        assert steps >= 1


def test_positive_control_oracle_acts_on_hidden_survivors() -> None:
    """The lockstep test can detect leakage: the oracle diverges before observations do."""
    assert any(_lockstep_until_observations_differ("assignment_oracle", s)[1] for s in DEV_SEEDS)


# ----------------------------------------------------------------- reproducibility and metrics


@pytest.mark.parametrize("name", list(POLICIES))
def test_multi_seed_reproducibility_and_metric_compatibility(name: str) -> None:
    spec_bs = BootstrapSpec(resamples=200)
    a = run_experiment(M2, POLICIES[name], DEV_SEEDS[:3], bootstrap=spec_bs)
    b = run_experiment(M2, POLICIES[name], DEV_SEEDS[:3], bootstrap=spec_bs)
    assert a.complete
    assert a.episodes == b.episodes
    assert a.summary == b.summary
    assert a.diagnostics_summary == b.diagnostics_summary
    assert (
        a.policy["information"]
        == build_team(POLICIES[name], MissionBrief.from_config(M2)).information
    )


@settings(max_examples=15)
@given(seed=st.integers(0, 10_000), name=st.sampled_from(PLANNERS))
def test_planners_on_random_small_maps_are_safe(seed: int, name: str) -> None:
    config = make_config(height=8, width=8, survivors=2, time_limit=40)
    result = run_episode(config, POLICIES[name], seed)
    assert result.metrics.invalid_actions == 0
    if name in CENTRALIZED:
        assert result.metrics.move_conflicts == 0


def test_known_map_of_a_policy_matches_only_observed_cells() -> None:
    """A decentralized agent's map never contains cells outside its footprint history."""
    gnd = spec("gnd", Layer.GROUND, radius=1, capacity=100)
    config, scenario = config_for(["......", "B.....", "......"], (gnd,), (AgentSpec("gnd"),))
    env = SwarmEnv(config)
    obs = env.reset(0, scenario=scenario)[0]
    brief = MissionBrief.from_config(config)
    team = build_team(POLICIES["frontier"], brief)
    seen: set[tuple[int, int]] = {(1, 0)}
    rngs = episode_streams(0, env.possible_agents).policies
    for _ in range(6):
        for o in obs.values():
            r0, c0 = o.position
            for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                if 0 <= r0 + dr < 3 and 0 <= c0 + dc < 6:
                    seen.add((r0 + dr, c0 + dc))
        actions = team.act(obs, rngs)
        obs = env.step(actions).observations
    nav = team.agents["gnd_0"]  # type: ignore[attr-defined]
    known_cells = {(int(r), int(c)) for r, c in np.argwhere(nav.known.known)}
    assert known_cells <= seen
    assert isinstance(nav.known, KnownMap)


# ----------------------------------------------------------------- livelock regressions (docs/16)

M1 = ScenarioConfig.from_json((ROOT / "configs" / "m1_default.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("name", "config_name", "seed"),
    [
        ("frontier", "m1", 9),  # two drones oscillating at the base entrance (void run)
        ("frontier", "m1", 42),  # depart/return collision at the base exit
        ("astar_waypoint", "m1", 1000),  # mirrored detours between returning and exploring drones
    ],
)
def test_decentralized_livelock_regressions(name: str, config_name: str, seed: int) -> None:
    config = M1 if config_name == "m1" else M2
    result = run_episode(config, POLICIES[name], seed)
    assert result.metrics.agents_disabled_energy == 0
    assert result.metrics.invalid_actions == 0


def test_corridor_head_on_meeting_resolves() -> None:
    """Two ground robots meeting head-on in a one-cell corridor must not wait forever."""
    result = run_episode(M1, POLICIES["frontier"], 9)
    diagnostics = result.policy_diagnostics or {}
    assert diagnostics["deadlock_abandons"] >= 1
    assert diagnostics["planned_waits"] < 200  # was ~380 before the right-of-way rule


@pytest.mark.parametrize("name", ["astar_waypoint", "frontier", "frontier_hazard_aware"])
def test_decentralized_planners_never_strand_agents_on_dev_seeds(name: str) -> None:
    for seed in range(1000, 1008):
        for config in (M1, M2):
            assert run_episode(config, POLICIES[name], seed).metrics.agents_disabled_energy == 0

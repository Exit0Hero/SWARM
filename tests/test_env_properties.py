"""Hypothesis invariants and determinism over random configs, seeds and actions."""

from __future__ import annotations

import itertools
from dataclasses import replace
from fractions import Fraction

import numpy as np
from hypothesis import assume, example, given
from hypothesis import strategies as st

from swarm.core.env import EnvSnapshot, SwarmEnv
from swarm.core.events import (
    AgentDisabled,
    Detection,
    EpisodeEnd,
    Event,
    HazardExposure,
    event_stream_sha256,
)
from swarm.core.types import AgentStatus, Layer, is_traversable
from swarm.scenarios.config import AgentSpec, HazardConfig, ScenarioConfig
from swarm.scenarios.generator import ScenarioError
from tests.helpers import make_config, mask_random_actions, spec


@st.composite
def configs(draw: st.DrawFn) -> ScenarioConfig:
    h, w = draw(st.integers(1, 9)), draw(st.integers(1, 9))
    structure = draw(st.sampled_from([0.0, 0.1, 0.3, 1.0]))
    debris = draw(st.sampled_from([0.0, 0.2, 0.4])) if structure < 1.0 else 0.0
    debris = min(debris, 1.0 - structure)
    air = spec(
        "air",
        Layer.AIR,
        radius=draw(st.integers(1, 3)),
        capacity=draw(st.integers(1, 12)),
        stay=draw(st.integers(0, 2)),
        recharge=draw(st.integers(0, 3)),
    )
    gnd = spec("gnd", Layer.GROUND, capacity=draw(st.integers(1, 12)), stay=draw(st.integers(0, 1)))
    roster = []
    for name, cap in (("air", air.energy_capacity), ("gnd", gnd.energy_capacity)):
        for _ in range(draw(st.integers(0, 3))):
            roster.append(AgentSpec(name, draw(st.none() | st.integers(0, cap))))
    assume(roster)
    hazards = None
    if draw(st.integers(0, 2)) > 0:  # hazards in about two thirds of the examples
        hazards = HazardConfig(
            fraction=draw(st.sampled_from([0.0, 0.5, 1.0])),
            affected_layers=tuple(
                draw(st.sampled_from([(Layer.AIR,), (Layer.GROUND,), (Layer.AIR, Layer.GROUND)]))
            ),
            energy_cost=draw(st.integers(0, 4)),
            disable_probability=draw(st.sampled_from([0.0, 0.3, 1.0])),
        )
    config = make_config(
        height=h,
        width=w,
        structure=structure,
        debris=debris,
        survivors=draw(st.integers(0, min(3, h * w - 1))),
        agent_types=(air, gnd),
        roster=tuple(roster),
        time_limit=draw(st.integers(1, 25)),
        p_detect=draw(st.sampled_from([0.0, 0.5, 1.0])),
    )
    return replace(
        config,
        hazards=hazards,
        p_false_positive=draw(st.sampled_from([0.0, 0.05, 0.5])),
        recall_target=draw(st.sampled_from([0.3, 0.5, 1.0])),
    )


def run(
    config: ScenarioConfig, seed: int, invalid_rate: float
) -> tuple[list[Event], list[EnvSnapshot]]:
    """Random episode; with probability ``invalid_rate`` an agent submits any action 0..4."""
    env = SwarmEnv(config)
    observations, _ = env.reset(seed)
    rng = np.random.default_rng(seed + 1)
    events: list[Event] = []
    snapshots = [env.snapshot()]
    while not env.done:
        actions = mask_random_actions({a: observations[a] for a in env.agents}, rng)
        for aid in actions:
            if rng.random() < invalid_rate:
                actions[aid] = int(rng.integers(5))
        result = env.step(actions)
        for aid, obs in result.observations.items():
            assert obs.action_mask == env.action_mask(aid)
        observations = result.observations
        events.extend(result.events)
        snapshots.append(env.snapshot())
    return events, snapshots


# Regression example: a hazard disablement leaves energy > 0 (ADR-0011). Found by Hypothesis when
# the M1 invariant "disabled => energy == 0" was still in place; pinned so it always runs.
_HAZARD_KEEPS_ENERGY = replace(
    make_config(
        height=1,
        width=3,
        structure=0.0,
        debris=0.0,
        survivors=0,
        agent_types=(
            spec("air", Layer.AIR, capacity=1),
            spec("gnd", Layer.GROUND, capacity=2),
        ),
        roster=(AgentSpec("gnd"),) * 3,
        time_limit=1,
        p_detect=0.0,
    ),
    hazards=HazardConfig(
        fraction=0.5, affected_layers=(Layer.GROUND,), energy_cost=0, disable_probability=1.0
    ),
    recall_target=0.3,
)


@example(config=_HAZARD_KEEPS_ENERGY, seed=0, invalid_rate=0.0)
@given(config=configs(), seed=st.integers(0, 10_000), invalid_rate=st.sampled_from([0.0, 0.3]))
def test_invariants_hold_every_step(config: ScenarioConfig, seed: int, invalid_rate: float) -> None:
    try:
        events, snapshots = run(config, seed, invalid_rate)
    except ScenarioError:
        assume(False)
    scenario = snapshots[0].scenario
    h, w = scenario.shape
    hazard_disabled = {
        e.agent_id for e in events if isinstance(e, AgentDisabled) and e.reason == "hazard"
    }
    for before, after in itertools.pairwise(snapshots):
        cells: dict[tuple[Layer, tuple[int, int]], str] = {}
        for prev, agent in zip(before.agents, after.agents, strict=True):
            r, c = agent.position
            assert 0 <= r < h
            assert 0 <= c < w
            assert is_traversable(agent.layer, int(scenario.terrain[r, c]))
            assert 0 <= agent.energy <= agent.energy_capacity
            assert abs(r - prev.position[0]) + abs(c - prev.position[1]) <= 1
            if prev.status is AgentStatus.DISABLED:
                assert agent.status is AgentStatus.DISABLED
                assert agent.position == prev.position
            if agent.status is AgentStatus.DISABLED:
                # Depletion leaves 0 energy; only a hazard can disable an agent with energy left.
                assert agent.energy == 0 or agent.agent_id in hazard_disabled
                assert agent.position != scenario.base
            if agent.position != scenario.base:
                key = (agent.layer, agent.position)
                assert key not in cells, f"{agent.agent_id} shares {key} with {cells.get(key)}"
                cells[key] = agent.agent_id
    assert events[-1].kind == "episode_end"
    assert sum(e.kind == "episode_end" for e in events) == 1
    assert snapshots[-1].step <= config.time_limit
    _check_m2_event_invariants(config, events, snapshots)


def _check_m2_event_invariants(
    config: ScenarioConfig, events: list[Event], snapshots: list[EnvSnapshot]
) -> None:
    """Hazard, detection and mission-success invariants (ADR-0009, ADR-0011, ADR-0012)."""
    scenario = snapshots[0].scenario
    survivors = set(scenario.survivors)
    layer = {a.agent_id: a.layer for a in snapshots[0].agents}
    affected = set() if config.hazards is None else set(config.hazards.affected_layers)
    disabled: set[str] = set()
    found: set[int] = set()
    target = Fraction(str(config.recall_target))
    n = len(survivors)
    for e in events:
        if isinstance(e, AgentDisabled):
            assert e.agent_id not in disabled, "an agent is disabled at most once"
            disabled.add(e.agent_id)
        elif isinstance(e, HazardExposure):
            assert scenario.hazard_grid[e.cell]
            assert layer[e.agent_id] in affected
            assert e.agent_id not in disabled
        elif isinstance(e, Detection):
            assert e.agent_id not in disabled
            assert (e.cell in survivors) == e.true_positive
            if e.true_positive:
                assert scenario.survivors[e.survivor_index or 0] == e.cell
                found.add(e.survivor_index or 0)
        elif isinstance(e, EpisodeEnd):
            reached = n > 0 and Fraction(len(found), n) >= target
            assert (e.reason == "mission_success") == reached
            assert e.terminated == (e.reason != "time_limit")
            assert e.truncated == (e.reason == "time_limit")


@given(config=configs(), seed=st.integers(0, 10_000))
def test_same_seed_same_event_stream(config: ScenarioConfig, seed: int) -> None:
    try:
        first, _ = run(config, seed, 0.2)
    except ScenarioError:
        assume(False)
    second, _ = run(config, seed, 0.2)
    assert event_stream_sha256(first) == event_stream_sha256(second)
    assert first == second


@st.composite
def hazard_heavy_configs(draw: st.DrawFn) -> ScenarioConfig:
    """Dense hazards affecting both layers, with enough energy to move into them."""
    size = draw(st.integers(3, 8))
    air = spec("air", Layer.AIR, radius=draw(st.integers(1, 2)), capacity=40, stay=0, recharge=2)
    gnd = spec("gnd", Layer.GROUND, capacity=40, stay=0, recharge=2)
    roster = (AgentSpec("air"),) * draw(st.integers(1, 3)) + (AgentSpec("gnd"),) * draw(
        st.integers(1, 3)
    )
    config = make_config(
        height=size,
        width=size,
        structure=0.0,
        debris=draw(st.sampled_from([0.0, 0.2])),
        survivors=draw(st.integers(0, 3)),
        agent_types=(air, gnd),
        roster=roster,
        time_limit=draw(st.integers(5, 30)),
        p_detect=0.5,
    )
    return replace(
        config,
        hazards=HazardConfig(
            fraction=draw(st.sampled_from([0.5, 1.0])),
            affected_layers=(Layer.AIR, Layer.GROUND),
            energy_cost=draw(st.integers(0, 15)),
            disable_probability=draw(st.sampled_from([0.0, 0.1, 0.5, 1.0])),
        ),
        p_false_positive=draw(st.sampled_from([0.0, 0.2])),
        recall_target=1.0,
    )


@given(config=hazard_heavy_configs(), seed=st.integers(0, 10_000))
def test_hazard_invariants_and_determinism(config: ScenarioConfig, seed: int) -> None:
    events, snapshots = run(config, seed, 0.0)
    _check_m2_event_invariants(config, events, snapshots)
    for snap in snapshots:
        assert all(0 <= a.energy <= a.energy_capacity for a in snap.agents)
    again, _ = run(config, seed, 0.0)
    assert event_stream_sha256(events) == event_stream_sha256(again)

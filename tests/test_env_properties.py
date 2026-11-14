"""Hypothesis invariants and determinism over random configs, seeds and actions."""

from __future__ import annotations

import itertools

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

from swarm.core.env import EnvSnapshot, SwarmEnv
from swarm.core.events import Event, event_stream_sha256
from swarm.core.types import AgentStatus, Layer, is_traversable
from swarm.scenarios.config import AgentSpec, ScenarioConfig
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
    return make_config(
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


@given(config=configs(), seed=st.integers(0, 10_000), invalid_rate=st.sampled_from([0.0, 0.3]))
def test_invariants_hold_every_step(config: ScenarioConfig, seed: int, invalid_rate: float) -> None:
    try:
        events, snapshots = run(config, seed, invalid_rate)
    except ScenarioError:
        assume(False)
    scenario = snapshots[0].scenario
    h, w = scenario.shape
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
                assert agent.energy == 0
                assert agent.position != scenario.base
            if agent.position != scenario.base:
                key = (agent.layer, agent.position)
                assert key not in cells, f"{agent.agent_id} shares {key} with {cells.get(key)}"
                cells[key] = agent.agent_id
    assert events[-1].kind == "episode_end"
    assert sum(e.kind == "episode_end" for e in events) == 1
    assert snapshots[-1].step <= config.time_limit


@given(config=configs(), seed=st.integers(0, 10_000))
def test_same_seed_same_event_stream(config: ScenarioConfig, seed: int) -> None:
    try:
        first, _ = run(config, seed, 0.2)
    except ScenarioError:
        assume(False)
    second, _ = run(config, seed, 0.2)
    assert event_stream_sha256(first) == event_stream_sha256(second)
    assert first == second

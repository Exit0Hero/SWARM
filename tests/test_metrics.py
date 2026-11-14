"""Metric definitions on hand-built event streams, and replay versus direct tracking."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from swarm.core.env import SwarmEnv
from swarm.core.events import (
    AgentDisabled,
    Detection,
    EnergySpent,
    Event,
    InvalidAction,
    MoveConflict,
    Moved,
)
from swarm.core.observation import footprint_offsets
from swarm.core.types import Layer
from swarm.evaluation.metrics import METRIC_DEFINITIONS, EpisodeMetrics, compute_metrics
from swarm.scenarios.config import AgentSpec
from swarm.scenarios.generator import ScenarioError, eligible_cells
from tests.helpers import config_for, make_config, mask_random_actions, spec

GND = spec("gnd", Layer.GROUND, radius=1)
AIR = spec("air", Layer.AIR, radius=1)


def test_every_metric_has_a_definition_and_unit() -> None:
    assert set(METRIC_DEFINITIONS) == set(EpisodeMetrics.__dataclass_fields__)
    assert all(d.definition and d.unit for d in METRIC_DEFINITIONS.values())


def test_hand_built_stream() -> None:
    # 1x5 corridor, base at (0,0). Survivors at (0,2) [index 0] and (0,4) [index 1].
    config, scenario = config_for(["B.S.S"], (GND,), (AgentSpec("gnd"), AgentSpec("gnd")))
    events: list[Event] = [
        Moved(1, "gnd_0", (0, 0), (0, 1)),
        EnergySpent(1, "gnd_0", 1, 9),
        InvalidAction(2, "gnd_1", 4, "out_of_bounds"),
        MoveConflict(2, "gnd_1", (0, 0), (0, 1), "blocked"),
        Moved(2, "gnd_0", (0, 1), (0, 2)),
        EnergySpent(2, "gnd_0", 1, 8),
        Detection(2, "gnd_0", (0, 2), True, 0),
        Detection(3, "gnd_0", (0, 2), True, 0),
        AgentDisabled(3, "gnd_1", (0, 0), "energy_depleted"),
    ]
    m = compute_metrics(config, scenario, events)
    assert m.recall == 0.5
    assert m.time_to_first_detection == 2
    # Eligible cells: all 5. Visited: (0,0),(0,1),(0,2). Sensed (radius 1): (0,0)..(0,3).
    assert m.coverage_visited == pytest.approx(3 / 5)
    assert m.coverage_sensed == pytest.approx(4 / 5)
    assert m.energy_used == 2
    assert m.move_conflicts == 1
    assert m.invalid_actions == 1
    assert m.agents_disabled == 1


def test_zero_survivors_gives_null_recall_and_detection_time() -> None:
    config, scenario = config_for(["B.."], (GND,), (AgentSpec("gnd"),))
    m = compute_metrics(config, scenario, [])
    assert m.recall is None
    assert m.time_to_first_detection is None
    assert m.coverage_visited == pytest.approx(1 / 3)
    assert m.coverage_sensed == pytest.approx(2 / 3)


def test_repeated_detections_count_once() -> None:
    config, scenario = config_for(["BS"], (GND,), (AgentSpec("gnd"),))
    events = [Detection(t, "gnd_0", (0, 1), True, 0) for t in (1, 2, 3)]
    m = compute_metrics(config, scenario, events)
    assert m.recall == 1.0
    assert m.time_to_first_detection == 1


def test_coverage_denominator_excludes_structures_and_unreachable_cells() -> None:
    # Ground-only roster: debris and the cell behind it are not eligible.
    config, scenario = config_for(["B~.", "##."], (GND,), (AgentSpec("gnd"),))
    assert int(eligible_cells(scenario, [Layer.GROUND]).sum()) == 1
    assert compute_metrics(config, scenario, []).coverage_visited == 1.0


def test_disabled_agents_stop_sensing() -> None:
    config, scenario = config_for(["B...."], (GND,), (AgentSpec("gnd"),))
    events: list[Event] = [
        Moved(1, "gnd_0", (0, 0), (0, 1)),
        AgentDisabled(1, "gnd_0", (0, 1), "energy_depleted"),
    ]
    m = compute_metrics(config, scenario, events)
    # Sensed only at reset from the base: (0,0),(0,1). Not from (0,1) after disabling.
    assert m.coverage_sensed == pytest.approx(2 / 5)
    assert m.coverage_visited == pytest.approx(2 / 5)


@given(seed=st.integers(0, 5000), size=st.integers(2, 10))
def test_replayed_coverage_matches_direct_tracking(seed: int, size: int) -> None:
    """Metrics replayed from events equal coverage tracked directly from snapshots."""
    config = make_config(height=size, width=size, survivors=min(2, size * size - 1), time_limit=25)
    env = SwarmEnv(config)
    try:
        observations, _ = env.reset(seed)
    except ScenarioError:
        assume(False)
    rng = np.random.default_rng(seed)
    scenario = env.snapshot().scenario
    sensed = np.zeros(scenario.shape, dtype=np.bool_)
    visited = np.zeros(scenario.shape, dtype=np.bool_)
    events: list[Event] = []

    def track() -> None:
        for agent in env.snapshot().agents:
            visited[agent.position] = True
            if agent.status.value == "active":
                for dr, dc in footprint_offsets(agent.sensor_radius):
                    r, c = agent.position[0] + dr, agent.position[1] + dc
                    if 0 <= r < size and 0 <= c < size:
                        sensed[r, c] = True

    track()
    while not env.done:
        result = env.step(mask_random_actions({a: observations[a] for a in env.agents}, rng))
        observations = result.observations
        events.extend(result.events)
        track()
    eligible = eligible_cells(scenario, [Layer.AIR, Layer.GROUND])
    m = compute_metrics(config, scenario, events)
    assert m.coverage_sensed == pytest.approx((sensed & eligible).sum() / eligible.sum())
    assert m.coverage_visited == pytest.approx((visited & eligible).sum() / eligible.sum())
    final = env.snapshot()
    assert m.agents_disabled == sum(a.status.value == "disabled" for a in final.agents)

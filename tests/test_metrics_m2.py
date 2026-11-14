"""Phase 1 M2 metric registry (ADR-0009 revision 2): exact values on hand-built streams."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from swarm.core.env import SwarmEnv
from swarm.core.events import (
    AgentDisabled,
    Detection,
    EnergySpent,
    EpisodeEnd,
    Event,
    HazardExposure,
    Moved,
    Recharged,
)
from swarm.core.types import Layer
from swarm.evaluation.metrics import (
    METRIC_DEFINITIONS,
    EpisodeMetrics,
    EvaluationError,
    compute_metrics,
)
from swarm.scenarios.config import AgentSpec, HazardConfig
from swarm.scenarios.generator import eligible_cells
from tests.helpers import config_for, make_config, mask_random_actions, spec

GND = spec("gnd", Layer.GROUND, capacity=10)
AIR = spec("air", Layer.AIR, capacity=10)


def test_registry_covers_every_metric_with_role() -> None:
    assert set(METRIC_DEFINITIONS) == set(EpisodeMetrics.__dataclass_fields__)
    assert {d.role for d in METRIC_DEFINITIONS.values()} == {"primary", "diagnostic"}
    assert METRIC_DEFINITIONS["agents_disabled"].role == "diagnostic"
    assert METRIC_DEFINITIONS["reachable_recall"].role == "diagnostic"


# ----------------------------------------------------------------- precision and duplicates


def test_precision_counts_locations_and_reports_separately() -> None:
    # Survivor at (0, 2). Reports: (0,2) true twice (two agents), (0,1) false twice, (0,3) false.
    config, scenario = config_for(["B.S."], (GND,), (AgentSpec("gnd"), AgentSpec("gnd")))
    events: list[Event] = [
        Detection(1, "gnd_0", (0, 2), True, 0),
        Detection(1, "gnd_1", (0, 2), True, 0),
        Detection(1, "gnd_0", (0, 1), False, None),
        Detection(2, "gnd_1", (0, 1), False, None),
        Detection(2, "gnd_1", (0, 3), False, None),
    ]
    m = compute_metrics(config, scenario, events)
    assert m.precision == pytest.approx(1 / 3)  # distinct cells {(0,1),(0,2),(0,3)}, one true
    assert m.report_precision == pytest.approx(2 / 5)
    assert m.false_positive_reports == 3
    assert m.recall == 1.0


def test_precision_is_undefined_without_reports() -> None:
    config, scenario = config_for(["B.S."], (GND,), (AgentSpec("gnd"),))
    m = compute_metrics(config, scenario, [])
    assert m.precision is None
    assert m.report_precision is None
    assert m.false_positive_reports == 0


def test_only_false_reports_give_zero_precision_and_recall() -> None:
    config, scenario = config_for(["B.S."], (GND,), (AgentSpec("gnd"),))
    m = compute_metrics(config, scenario, [Detection(1, "gnd_0", (0, 1), False, None)])
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.time_to_first_detection is None


@pytest.mark.parametrize(
    "event",
    [
        Detection(1, "gnd_0", (0, 1), True, 0),  # labelled true on an empty cell
        Detection(1, "gnd_0", (0, 2), False, None),  # labelled false on a survivor
        Detection(1, "gnd_0", (0, 2), True, 7),  # wrong survivor index
    ],
)
def test_evaluator_uses_ground_truth_and_rejects_mislabelled_reports(event: Detection) -> None:
    config, scenario = config_for(["B.S."], (GND,), (AgentSpec("gnd"),))
    with pytest.raises(EvaluationError):
        compute_metrics(config, scenario, [event])


# ----------------------------------------------------------------- recall and success


def test_reachable_recall_excludes_unreachable_survivors_but_recall_does_not() -> None:
    # (0, 4) is enclosed by structures: not eligible for a ground-only roster.
    config, scenario = config_for(["BS.#S"], (GND,), (AgentSpec("gnd"),))
    assert not eligible_cells(scenario, [Layer.GROUND])[0, 4]
    m = compute_metrics(config, scenario, [Detection(1, "gnd_0", (0, 1), True, 0)])
    assert m.recall == 0.5
    assert m.reachable_recall == 1.0


def test_reachable_recall_undefined_when_no_survivor_is_reachable() -> None:
    config, scenario = config_for(["B#S"], (GND,), (AgentSpec("gnd"),))
    assert compute_metrics(config, scenario, []).reachable_recall is None


def test_mission_success_and_time_to_target() -> None:
    config, scenario = config_for(["BS.S"], (GND,), (AgentSpec("gnd"),), recall_target=0.5)
    events = [
        Detection(3, "gnd_0", (0, 1), True, 0),
        EpisodeEnd(3, "mission_success", True, False),
    ]
    m = compute_metrics(config, scenario, events)
    assert m.mission_success is True
    assert m.time_to_recall_target == 3


def test_mission_failure_reports_false_and_null_time() -> None:
    config, scenario = config_for(["BS.S"], (GND,), (AgentSpec("gnd"),))
    events = [Detection(1, "gnd_0", (0, 1), True, 0), EpisodeEnd(5, "time_limit", False, True)]
    m = compute_metrics(config, scenario, events)
    assert m.mission_success is False
    assert m.time_to_recall_target is None


def test_mission_success_undefined_for_zero_survivors_and_incomplete_streams() -> None:
    config, scenario = config_for(["B.."], (GND,), (AgentSpec("gnd"),))
    assert (
        compute_metrics(
            config, scenario, [EpisodeEnd(2, "time_limit", False, True)]
        ).mission_success
        is None
    )
    config, scenario = config_for(["BS."], (GND,), (AgentSpec("gnd"),))
    assert compute_metrics(config, scenario, []).mission_success is None


def test_unsupported_success_claim_is_rejected() -> None:
    config, scenario = config_for(["BS.S"], (GND,), (AgentSpec("gnd"),))
    with pytest.raises(EvaluationError, match="claims mission_success"):
        compute_metrics(config, scenario, [EpisodeEnd(1, "mission_success", True, False)])


def test_unclaimed_success_is_rejected() -> None:
    config, scenario = config_for(["BS"], (GND,), (AgentSpec("gnd"),))
    events = [Detection(1, "gnd_0", (0, 1), True, 0), EpisodeEnd(4, "time_limit", False, True)]
    with pytest.raises(EvaluationError, match="reached"):
        compute_metrics(config, scenario, events)


# ----------------------------------------------------------------- resources and hazards


def test_energy_efficiency_and_hazard_metrics() -> None:
    config, scenario = config_for(["BS.S"], (GND,), (AgentSpec("gnd"),))
    events: list[Event] = [
        Moved(1, "gnd_0", (0, 0), (0, 1)),
        EnergySpent(1, "gnd_0", 4, 6),
        HazardExposure(1, "gnd_0", (0, 1), 3, 3),
        Detection(1, "gnd_0", (0, 1), True, 0),
        HazardExposure(2, "gnd_0", (0, 1), 3, 0),
        AgentDisabled(2, "gnd_0", (0, 1), "energy_depleted"),
    ]
    m = compute_metrics(config, scenario, events)
    assert m.energy_used == 4  # hazard damage is excluded
    assert m.energy_efficiency == pytest.approx(1 * 1000 / 4)
    assert m.hazard_exposures == 2
    assert m.hazard_energy_lost == 6
    assert (m.agents_disabled, m.agents_disabled_energy, m.agents_disabled_hazard) == (1, 1, 0)


def test_energy_efficiency_undefined_without_energy() -> None:
    config, scenario = config_for(["BS"], (GND,), (AgentSpec("gnd"),))
    assert compute_metrics(config, scenario, []).energy_efficiency is None


def test_disablement_breakdown_and_double_disable_rejected() -> None:
    config, scenario = config_for(["B..."], (GND,), (AgentSpec("gnd"), AgentSpec("gnd")))
    events: list[Event] = [
        AgentDisabled(1, "gnd_0", (0, 1), "hazard"),
        AgentDisabled(2, "gnd_1", (0, 2), "energy_depleted"),
    ]
    m = compute_metrics(config, scenario, events)
    assert (m.agents_disabled, m.agents_disabled_energy, m.agents_disabled_hazard) == (2, 1, 1)
    with pytest.raises(EvaluationError, match="disabled twice"):
        compute_metrics(config, scenario, [*events, AgentDisabled(3, "gnd_0", (0, 1), "hazard")])


# ----------------------------------------------------------------- utilization


def test_agent_utilization_hand_built() -> None:
    # Two agents, 4 steps. gnd_0 idles on base at full energy the whole time (0 active steps).
    # gnd_1 moves at step 1 and is disabled at step 3 (active at steps 1 and 2 only).
    config, scenario = config_for(["B..."], (GND,), (AgentSpec("gnd"), AgentSpec("gnd")))
    events: list[Event] = [
        Moved(1, "gnd_1", (0, 0), (0, 1)),
        EnergySpent(1, "gnd_1", 1, 9),
        AgentDisabled(3, "gnd_1", (0, 1), "hazard"),
        EpisodeEnd(4, "time_limit", False, True),
    ]
    assert compute_metrics(config, scenario, events).agent_utilization == pytest.approx(2 / 8)


def test_recharging_on_base_counts_as_active() -> None:
    config, scenario = config_for(["B."], (GND,), (AgentSpec("gnd", initial_energy=5),))
    events: list[Event] = [
        Recharged(1, "gnd_0", 3, 8),
        Recharged(2, "gnd_0", 2, 10),
        EpisodeEnd(3, "time_limit", False, True),
    ]
    # Steps 1 and 2 start below capacity (active, recharging). Step 3 starts full: idle.
    assert compute_metrics(config, scenario, events).agent_utilization == pytest.approx(2 / 3)


def test_utilization_undefined_for_an_empty_stream() -> None:
    config, scenario = config_for(["B."], (GND,), (AgentSpec("gnd"),))
    assert compute_metrics(config, scenario, []).agent_utilization is None


# ----------------------------------------------------------------- replay versus direct tracking


@given(seed=st.integers(0, 3000), size=st.integers(3, 9), p_fp=st.sampled_from([0.0, 0.1]))
def test_replayed_metrics_match_direct_tracking_with_hazards(
    seed: int, size: int, p_fp: float
) -> None:
    config = replace(
        make_config(height=size, width=size, survivors=2, time_limit=30, structure=0.1),
        hazards=HazardConfig(
            fraction=0.4,
            affected_layers=(Layer.AIR, Layer.GROUND),
            energy_cost=2,
            disable_probability=0.1,
        ),
        p_false_positive=p_fp,
        recall_target=1.0,
    )
    env = SwarmEnv(config)
    observations, _ = env.reset(seed)
    rng = np.random.default_rng(seed)
    events: list[Event] = []
    active_steps = 0
    steps = 0
    while not env.done:
        before = {a.agent_id: (a.position, a.energy) for a in env.snapshot().agents}
        result = env.step(mask_random_actions({a: observations[a] for a in env.agents}, rng))
        observations = result.observations
        events.extend(result.events)
        steps += 1
        snap = env.snapshot()
        for a in snap.agents:
            start_position, start_energy = before[a.agent_id]
            idle = (
                a.position == snap.scenario.base
                and start_position == a.position
                and start_energy == a.energy_capacity
            )
            active_steps += int(a.status.value == "active" and not idle)
    final = env.snapshot()
    m = compute_metrics(config, final.scenario, events)
    assert m.agent_utilization == pytest.approx(active_steps / (len(final.agents) * steps))
    assert m.agents_disabled == sum(a.status.value == "disabled" for a in final.agents)
    assert m.mission_success == (final.end_reason == "mission_success")

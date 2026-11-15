"""Phase 3 M3.2: belief maps, information sharing and propagation metrics (ADR-0018)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from swarm.coordination.belief import SWEEP_RECORDS, BeliefMap
from swarm.core.env import SwarmEnv
from swarm.core.events import (
    AgentDisabled,
    Detection,
    Event,
    SurvivorReportReceived,
)
from swarm.core.messages import Message, Record
from swarm.core.observation import UNKNOWN
from swarm.core.rng import episode_streams
from swarm.core.types import Action, AgentId, Layer, Terrain
from swarm.evaluation.comms_metrics import compute_comm_metrics
from swarm.experiments.runner import run_episode
from swarm.policies.base import MissionBrief, build_team
from swarm.policies.registry import POLICIES
from swarm.scenarios.config import AgentSpec, CommsConfig, ScenarioConfig
from tests.helpers import config_for, make_config, spec

GND = spec("gnd", Layer.GROUND, radius=1, capacity=100, move=1, stay=0)
STAY = int(Action.STAY)
OPEN, DEBRIS = int(Terrain.OPEN), int(Terrain.DEBRIS)
SHAPE, BASE = (4, 6), (0, 0)


def _env(**comms: Any) -> tuple[SwarmEnv, dict[AgentId, Any]]:
    config, scenario = config_for(
        ["B.....", "......", "......", "......"],
        (GND,),
        (AgentSpec("gnd"), AgentSpec("gnd")),
        time_limit=30,
    )
    env = SwarmEnv(replace(config, comms=CommsConfig(**({"ranges": {"gnd": 10}} | comms))))
    obs, _ = env.reset(0, scenario=scenario)
    return env, obs


def _known_cells(belief: BeliefMap) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(belief.terrain != UNKNOWN)}


# ----------------------------------------------------------------- local observations


def test_own_observation_updates_only_the_owner() -> None:
    _, obs = _env()
    b0, b1 = BeliefMap("gnd_0", SHAPE, BASE), BeliefMap("gnd_1", SHAPE, BASE)
    b0.update(obs["gnd_0"])
    assert _known_cells(b0) == {(0, 0), (0, 1), (1, 0)}  # radius-1 footprint at the corner
    assert _known_cells(b1) == {BASE}  # only the public base
    assert b1.records() == ()
    with pytest.raises(ValueError, match="cannot take"):
        b0.update(obs["gnd_1"])


def test_unknown_cells_stay_unknown_without_evidence() -> None:
    b = BeliefMap("a", SHAPE, BASE)
    b.merge_records((("det", 2, 3, "z", 4, 4, 1),))  # survivor evidence is not terrain
    assert _known_cells(b) == {BASE}
    assert b.observed_step[2, 3] == -1
    assert b.cell_record((2, 3)) is None
    assert set(b.survivor_evidence()) == {(2, 3)}


# ----------------------------------------------------------------- propagation and merge rules


def test_records_propagate_one_observers_knowledge_to_another() -> None:
    _, obs = _env()
    b0, b1 = BeliefMap("gnd_0", SHAPE, BASE), BeliefMap("gnd_1", SHAPE, BASE)
    b0.update(obs["gnd_0"])
    b1.merge_records(b0.outbox_records())
    assert _known_cells(b1) == _known_cells(b0)
    assert b1.records() == b0.records()
    assert b1.cell_record((0, 1)) == ("cell", 0, 1, OPEN, False, 0, "gnd_0")


def test_duplicates_change_nothing() -> None:
    b = BeliefMap("a", SHAPE, BASE)
    records: tuple[Record, ...] = (("cell", 1, 1, OPEN, False, 3, "z"), ("det", 1, 1, "z", 3, 3, 1))
    b.merge_records(records)
    snapshot, version = b.records(), b.version
    b.merge_records(records)
    b.merge_records(records)
    assert b.records() == snapshot
    assert b.version == version
    assert b.stats["duplicates"] == 4


@pytest.mark.parametrize("newer_first", [True, False])
def test_older_information_never_overwrites_newer(newer_first: bool) -> None:
    new: Record = ("cell", 1, 2, OPEN, False, 5, "a")
    old: Record = ("cell", 1, 2, DEBRIS, True, 3, "b")
    b = BeliefMap("x", SHAPE, BASE)
    b.merge_records((new, old) if newer_first else (old, new))
    assert b.cell_record((1, 2)) == new
    assert int(b.terrain[1, 2]) == OPEN
    assert not b.hazards[1, 2]
    assert b.observed_step[1, 2] == 5
    assert b.stats["conflicts"] == 1


def test_equal_steps_resolve_by_observer_then_value_on_every_agent() -> None:
    a: Record = ("cell", 0, 3, OPEN, False, 4, "drone_0")
    z: Record = ("cell", 0, 3, DEBRIS, False, 4, "gnd_1")
    x, y = BeliefMap("x", SHAPE, BASE), BeliefMap("y", SHAPE, BASE)
    x.merge_records((a, z))
    y.merge_records((z, a))
    assert x.cell_record((0, 3)) == y.cell_record((0, 3)) == z


def test_detection_evidence_is_not_double_counted_through_relays() -> None:
    source = BeliefMap("z", SHAPE, BASE)
    source.merge_records((("det", 2, 2, "z", 3, 7, 2),))
    relay, sink = BeliefMap("r", SHAPE, BASE), BeliefMap("s", SHAPE, BASE)
    relay.merge_records(source.records())
    sink.merge_records(source.records())  # direct copy
    sink.merge_records(relay.records())  # and a relayed copy
    sink.merge_records((("det", 2, 2, "z", 3, 5, 1),))  # an older, smaller copy
    assert sink.survivor_evidence()[(2, 2)] == {"z": (3, 7, 2)}


def test_own_detections_are_counted_once_per_step() -> None:
    _, obs = _env()
    b = BeliefMap("gnd_0", SHAPE, BASE)
    detecting = replace(obs["gnd_0"], detections=((0, 1),))
    b.update(detecting)
    b.update(detecting)  # the same observation again: no extra count
    assert b.survivor_evidence()[(0, 1)] == {"gnd_0": (0, 0, 1)}
    b.update(replace(detecting, step=1))
    assert b.survivor_evidence()[(0, 1)] == {"gnd_0": (0, 1, 2)}


def test_disabled_sightings_block_the_cell_and_keep_the_earliest() -> None:
    b = BeliefMap("x", SHAPE, BASE)
    b.merge_records((("dis", "gnd_3", int(Layer.GROUND), 2, 2, 9, "b"),))
    b.merge_records((("dis", "gnd_3", int(Layer.GROUND), 2, 2, 4, "a"),))
    assert b.blocked[int(Layer.GROUND)][2, 2]
    assert not b.blocked[int(Layer.AIR)][2, 2]
    assert ("dis", "gnd_3", int(Layer.GROUND), 2, 2, 4, "a") in b.records()


def test_unknown_tags_are_ignored_and_malformed_records_raise() -> None:
    b = BeliefMap("x", SHAPE, BASE)
    b.merge_records((("future_kind", 1, 2),))
    assert b.records() == ()
    with pytest.raises(ValueError, match="malformed"):
        b.merge_records((("cell", 1, 2, OPEN),))
    with pytest.raises(ValueError, match="expected int"):
        b.merge_records((("cell", 1, 2, OPEN, False, "3", "a"),))
    with pytest.raises(ValueError, match="outside the grid"):
        b.merge_records((("cell", 9, 9, OPEN, False, 3, "a"),))


_cells = st.tuples(
    st.just("cell"),
    st.integers(0, 3),
    st.integers(0, 5),
    st.sampled_from([OPEN, DEBRIS]),
    st.booleans(),
    st.integers(0, 6),
    st.sampled_from(["a", "b", "c"]),
)
_dets = st.tuples(
    st.just("det"),
    st.integers(0, 3),
    st.integers(0, 5),
    st.sampled_from(["a", "b"]),
    st.integers(0, 3),
    st.integers(3, 6),
    st.integers(1, 4),
)
_dis = st.tuples(
    st.just("dis"),
    st.sampled_from(["p", "q"]),
    st.just(int(Layer.GROUND)),
    st.integers(1, 3),
    st.integers(1, 5),
    st.integers(0, 6),
    st.sampled_from(["a", "b"]),
)


@settings(max_examples=60)
@given(records=st.lists(st.one_of(_cells, _dets, _dis), max_size=25), data=st.data())
def test_merging_is_independent_of_arrival_order_and_grouping(
    records: list[Record], data: st.DataObject
) -> None:
    reference = BeliefMap("ref", SHAPE, BASE)
    reference.merge_records(tuple(records))
    shuffled = data.draw(st.permutations(records))
    other = BeliefMap("other", SHAPE, BASE)
    other.merge_records(tuple(shuffled))
    # Grouping: two partial beliefs merged into a third.
    cut = data.draw(st.integers(0, len(records)))
    left, right, joined = (BeliefMap(n, SHAPE, BASE) for n in ("l", "r", "j"))
    left.merge_records(tuple(shuffled[:cut]))
    right.merge_records(tuple(shuffled[cut:]))
    joined.merge_records(right.records())
    joined.merge_records(left.records())
    for belief in (other, joined):
        assert belief.records() == reference.records()
        assert np.array_equal(belief.terrain, reference.terrain)
        assert np.array_equal(belief.hazards, reference.hazards)
        assert np.array_equal(belief.blocked, reference.blocked)


# ----------------------------------------------------------------- sharing (outbox)


def test_outbox_sends_changes_first_and_eventually_everything() -> None:
    b = BeliefMap("x", SHAPE, BASE)
    many = tuple(("cell", r, c, OPEN, False, 1, "z") for r in range(4) for c in range(6))
    b.merge_records(many)
    first = b.outbox_records(max_records=10)
    assert len(first) == 10
    assert first == tuple(sorted(many))[:10]  # fresh records, in key order
    seen = set(first)
    for _ in range(10):
        seen |= set(b.outbox_records(max_records=10))
    assert seen == set(many)  # leftovers stay fresh; the sweep resends the rest
    quiet = b.outbox_records()
    assert 0 < len(quiet) <= SWEEP_RECORDS  # nothing new: anti-entropy slice only


# ----------------------------------------------------------------- channel instrumentation


def test_survivor_reports_are_logged_once_per_receiver_and_cell() -> None:
    env, _ = _env()
    det: Record = ("det", 2, 3, "gnd_0", 1, 1, 1)
    r1 = env.step(
        {"gnd_0": STAY, "gnd_1": STAY},
        {"gnd_0": [Message("gnd_0", 0, "belief", (det, ("det", "x", 1), ("cell", 0, 1)))]},
    )
    r2 = env.step({"gnd_0": STAY, "gnd_1": STAY}, {"gnd_0": [Message("gnd_0", 1, "b", (det,))]})
    logged = [e for e in (*r1.events, *r2.events) if isinstance(e, SurvivorReportReceived)]
    assert logged == [SurvivorReportReceived(1, "gnd_1", (2, 3), "gnd_0")]


# ----------------------------------------------------------------- propagation metric


def test_propagation_delay_counts_informed_and_unresolved_pairs() -> None:
    config = make_config(survivors=2)
    env = SwarmEnv(config)
    env.reset(0)
    scenario = env.snapshot().scenario
    s0, s1 = scenario.survivors
    agents = sorted(config.agents())
    origin, a, b, c = agents[0], agents[1], agents[2], agents[3]
    events: list[Event] = [
        AgentDisabled(1, c, (0, 0), "energy_depleted"),  # disabled before t0: no pair
        Detection(2, origin, s0, True, 0),  # t0 = 2 for s0
        Detection(2, a, s0, True, 0),  # simultaneous own detection: delay 0
        SurvivorReportReceived(5, b, s0, origin),  # delay 3
        Detection(3, origin, (9, 9), False, None),  # false positive: ignored
        Detection(6, b, s1, True, 1),  # t0 = 6 for s1, origin b
        AgentDisabled(7, a, (0, 0), "hazard"),  # a never learns of s1: unresolved
    ]
    m = compute_comm_metrics(config, scenario, events)
    # s0: pairs (a, 0), (b, 3); c excluded. s1: pairs origin (unresolved), a (unresolved).
    assert m["survivor_info_pairs"] == 4
    assert m["survivor_info_informed"] == 2
    assert m["survivor_info_unresolved"] == 2
    assert m["survivor_info_informed_fraction"] == 0.5
    assert m["survivor_info_delay_mean"] == 1.5
    assert m["survivor_info_delay_max"] == 3


def test_no_detections_means_no_pairs() -> None:
    config = make_config(survivors=1)
    env = SwarmEnv(config)
    env.reset(0)
    m = compute_comm_metrics(config, env.snapshot().scenario, [])
    assert m["survivor_info_pairs"] == 0
    assert m["survivor_info_delay_mean"] is None
    assert m["survivor_info_informed_fraction"] is None


# ----------------------------------------------------------------- policy level


def _comms(**kw: Any) -> ScenarioConfig:
    settings_: dict[str, Any] = {"ranges": (("drone", 8), ("ground", 5))} | kw
    return replace(make_config(survivors=6, time_limit=120), comms=CommsConfig(**settings_))


_MESSAGE_KINDS = {
    "message_sent",
    "message_dropped",
    "message_delivered",
    "survivor_report_received",
}


def test_without_deliveries_frontier_shared_behaves_exactly_like_frontier() -> None:
    """No information arrives (every message dropped), so beliefs equal Phase 2 known maps."""
    config = _comms(p_drop=1.0)
    for seed in (0, 1):
        shared = run_episode(config, POLICIES["frontier_shared"], seed)
        plain = run_episode(config, POLICIES["frontier"], seed)
        assert [e for e in shared.events if e.kind not in _MESSAGE_KINDS] == list(plain.events)
        assert shared.policy_diagnostics is not None
        assert shared.policy_diagnostics["belief_records_received"] == 0


def test_shared_beliefs_propagate_and_runs_are_deterministic() -> None:
    config = _comms()
    a = run_episode(config, POLICIES["frontier_shared"], 2)
    b = run_episode(config, POLICIES["frontier_shared"], 2)
    c = run_episode(config, POLICIES["frontier_shared"], 3)
    assert a.event_stream_sha256 == b.event_stream_sha256 != c.event_stream_sha256
    assert a.policy["information"] == "decentralized_comm"
    assert a.policy_diagnostics is not None
    assert a.policy_diagnostics["belief_records_changed"] > 0
    assert a.comm_metrics is not None
    assert a.comm_metrics["survivor_info_informed"] > 0
    assert a.to_dict()["comm_metrics"] == a.comm_metrics


def test_only_active_agents_send_and_beliefs_hold_only_observed_cells() -> None:
    config = _comms(latency=1, jitter=2, p_drop=0.3)
    team = build_team(POLICIES["frontier_shared"], MissionBrief.from_config(config))
    env = SwarmEnv(config)
    obs, _ = env.reset(4)
    scenario = env.snapshot().scenario
    rngs = episode_streams(4, env.possible_agents).policies
    sensed: set[tuple[int, int]] = {scenario.base}
    while not env.done:
        actions = team.act(obs, rngs)
        outbox = team.outbox()  # type: ignore[attr-defined]
        assert set(outbox) <= set(env.agents)
        result = env.step(actions, outbox)
        for o in result.observations.values():
            r0, c0 = o.position
            rad = o.sensor_radius
            sensed |= {
                (r0 + dr, c0 + dc)
                for dr in range(-rad, rad + 1)
                for dc in range(-rad, rad + 1)
                if o.local_terrain[dr + rad, dc + rad] >= 0
            }
        for agent in team.agents.values():  # type: ignore[attr-defined]
            assert _known_cells(agent.belief) <= sensed  # no knowledge without an observation
            for cell in _known_cells(agent.belief) - {scenario.base}:
                assert int(agent.belief.terrain[cell]) == int(scenario.terrain[cell])
        obs = result.observations


def test_frontier_shared_requires_a_channel() -> None:
    with pytest.raises(ValueError, match="comms block"):
        build_team(POLICIES["frontier_shared"], MissionBrief.from_config(make_config()))


def test_no_channel_results_keep_the_phase2_layout() -> None:
    result = run_episode(make_config(), POLICIES["frontier"], 0)
    assert result.comm_metrics is None
    assert "comm_metrics" not in result.to_dict()

"""Phase 3 M3.4: failure detection, leases and reassignment (ADR-0020)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from swarm.coordination.auction import LEASE_STEPS, LeaseAuction
from swarm.coordination.failure import FailureDetector, PeerState
from swarm.core.events import AgentDisabled, EnergySpent, EpisodeEnd, Event
from swarm.evaluation.comms_metrics import compute_failure_metrics
from swarm.experiments.runner import run_episode
from swarm.policies.registry import POLICIES
from swarm.scenarios.config import CommsConfig, ScenarioConfig
from tests.helpers import make_config

ROOT = Path(__file__).resolve().parents[1]
NONE: frozenset[Any] = frozenset()
A, P = PeerState.ACTIVE, PeerState
TEAM = ["a", "b", "c"]


# ----------------------------------------------------------------- detector state machine


def test_suspicion_boundary_and_states() -> None:
    d = FailureDetector("a", TEAM, suspicion_steps=10, contact_steps=2)
    assert d.classify("b", 12, {"b": 10}, NONE) is P.ACTIVE  # age 2 <= contact
    assert d.classify("b", 13, {"b": 10}, NONE) is P.DISCONNECTED  # temporary gap
    assert d.classify("b", 20, {"b": 10}, NONE) is P.DISCONNECTED  # age 10: not yet suspected
    assert d.classify("b", 21, {"b": 10}, NONE) is P.SUSPECTED  # age 11
    assert d.classify("b", 21, {"b": 10}, frozenset({"b"})) is P.DISABLED
    assert d.classify("c", 5, {}, NONE) is P.DISCONNECTED  # unheard since the joint start


def test_missing_messages_never_mean_disabled_and_suspicion_recovers() -> None:
    d = FailureDetector("a", TEAM, suspicion_steps=3)
    assert d.update(1, {"b": 1, "c": 1}, NONE) == []
    assert d.update(5, {"b": 1, "c": 5}, NONE) == [(5, "b", A, P.SUSPECTED)]
    assert d.update(500, {"b": 1, "c": 500}, NONE) == []  # still only suspected
    assert d.update(501, {"b": 501, "c": 501}, NONE) == [(501, "b", P.SUSPECTED, A)]
    assert (d.suspicions, d.recoveries) == (1, 1)


def test_visible_disablement_is_certain_and_absorbing() -> None:
    d = FailureDetector("a", TEAM)
    d.update(4, {"b": 4}, frozenset({"b"}))
    assert d.states["b"] is P.DISABLED
    changes = d.update(9, {"b": 9, "c": 9}, NONE)
    assert [c for c in changes if c[1] == "b"] == []  # a later heartbeat cannot revive it
    assert d.states["b"] is P.DISABLED


def test_detector_rejects_bad_timeouts() -> None:
    with pytest.raises(ValueError, match="contact_steps"):
        FailureDetector("a", TEAM, suspicion_steps=2, contact_steps=3)


@settings(max_examples=40)
@given(st.lists(st.dictionaries(st.sampled_from(["b", "c"]), st.integers(0, 40)), max_size=8))
def test_detector_transitions_are_deterministic(beats: list[dict[str, int]]) -> None:
    runs = []
    for _ in range(2):
        d = FailureDetector("a", TEAM, suspicion_steps=5, contact_steps=2)
        heard: dict[str, int] = {}
        log = []
        for now, update in enumerate(beats, start=1):
            for peer, step in update.items():
                heard[peer] = max(heard.get(peer, 0), min(step, now))
            log += d.update(now * 3, heard, NONE)
        runs.append(log)
    assert runs[0] == runs[1]


# ----------------------------------------------------------------- leases and reassignment


def test_configurable_suspicion_controls_the_incumbency_allowance() -> None:
    holder = LeaseAuction("a")
    holder.decide(1, {0: 9}, NONE, NONE)
    for timeout, expect in ((10, None), (3, 0)):
        rival = LeaseAuction("b", suspicion_steps=timeout)
        rival.table.merge(holder.table.records())
        # At step 6, a is unheard for 5 steps: alive under 10, suspected under 3.
        # Bid 5 + allowance 5 = 10 does not beat 9; bid 5 with no allowance does.
        assert rival.decide(6, {0: 5}, NONE, NONE) == expect


def test_a_suspected_holder_keeps_its_lease_until_it_expires() -> None:
    holder, rival = LeaseAuction("a"), LeaseAuction("b")
    holder.decide(1, {0: 2}, NONE, NONE)
    rival.table.merge(holder.table.records())
    assert rival.decide(1 + 11, {0: 3}, NONE, NONE) is None  # suspected, lease still valid
    assert rival.decide(2 + LEASE_STEPS, {0: 3}, NONE, NONE) == 0  # lease expired


def _run(
    auctions: list[LeaseAuction],
    bids: list[dict[int, int]],
    steps: range,
    groups: list[set[int]] | None,
) -> None:
    for step in steps:
        for auction, own in zip(auctions, bids, strict=True):
            auction.decide(step, own, NONE, NONE)
        for group in groups or [set(range(len(auctions)))]:
            members = [auctions[i] for i in sorted(group)]
            records = [m.table.records() for m in members]
            for m in members:
                for r in records:
                    m.table.merge(r)


@settings(max_examples=40)
@given(
    bids=st.lists(
        st.dictionaries(st.integers(0, 3), st.integers(0, 20), min_size=1, max_size=4),
        min_size=2,
        max_size=4,
    ),
    split=st.integers(1, 3),
    partition_steps=st.integers(0, 3 * LEASE_STEPS),
)
def test_agents_converge_after_a_partition_heals(
    bids: list[dict[int, int]], split: int, partition_steps: int
) -> None:
    n = len(bids)
    auctions = [LeaseAuction(f"agent_{i}") for i in range(n)]
    groups = [set(range(min(split, n - 1))), set(range(min(split, n - 1), n))]
    _run(auctions, bids, range(1, 1 + partition_steps), groups)  # partitioned
    end = 1 + partition_steps + 4
    _run(auctions, bids, range(1 + partition_steps, end), None)  # reconnected
    held = [a.holding for a in auctions if a.holding is not None]
    assert len(held) == len(set(held))  # no duplicate ownership survives reconnection
    views = [a.winners(end - 1, NONE) for a in auctions]
    assert all(v == views[0] for v in views)
    assert all(views[0][a.holding][1] == a.owner for a in auctions if a.holding is not None)


def test_a_recovered_holder_and_its_replacement_settle_on_one_owner() -> None:
    a, b = LeaseAuction("a"), LeaseAuction("b")
    _run([a, b], [{0: 4}, {0: 6}], range(1, 3), None)
    assert (a.holding, b.holding) == (0, None)
    _run([a, b], [{0: 4}, {0: 6}], range(3, 3 + LEASE_STEPS + 2), [{0}, {1}])  # partition
    assert (a.holding, b.holding) == (0, 0)  # b took over the lapsed claim; a never knew
    _run([a, b], [{0: 4}, {0: 6}], range(3 + LEASE_STEPS + 2, 3 + LEASE_STEPS + 5), None)
    assert (a.holding, b.holding) == (0, None)  # lower bid keeps it; no permanent duplicate


# ----------------------------------------------------------------- evaluation metrics


def _log(*entries: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"kind": "detector", "step": 0, "suspicion_steps": 10}, *entries]


def test_failure_metrics_on_a_hand_built_episode() -> None:
    config = make_config(drones=1, grounds=2)  # agents drone_0, ground_0, ground_1
    events: list[Event] = [
        EnergySpent(3, "ground_0", 4, 96),
        AgentDisabled(5, "ground_0", (1, 1), "hazard"),
        EpisodeEnd(20, "time_limit", False, True),
    ]
    log = _log(
        {"kind": "hold", "step": 1, "agent": "ground_0", "sector": 2},
        {"kind": "hold", "step": 1, "agent": "drone_0", "sector": 3},
        {
            "kind": "peer_state",
            "step": 3,
            "observer": "drone_0",
            "peer": "ground_1",
            "state": "suspected",
        },
        {
            "kind": "peer_state",
            "step": 4,
            "observer": "drone_0",
            "peer": "ground_1",
            "state": "active",
        },
        {
            "kind": "peer_state",
            "step": 9,
            "observer": "drone_0",
            "peer": "ground_0",
            "state": "suspected",
        },
        {
            "kind": "peer_state",
            "step": 12,
            "observer": "drone_0",
            "peer": "ground_0",
            "state": "disabled",
        },
        {"kind": "hold", "step": 8, "agent": "ground_1", "sector": 2},
    )
    m = compute_failure_metrics(config, events, log)
    assert m["failure_pairs"] == 2  # (ground_0; drone_0) and (ground_0; ground_1)
    assert m["disabled_detection_delay_mean"] == 7  # drone_0 at 12; ground_1 never
    assert m["disabled_detection_unresolved"] == 1
    assert m["suspicion_detection_delay_mean"] == 4  # drone_0 suspected at 9
    assert m["suspicion_detection_unresolved"] == 1
    assert m["false_suspicions"] == 1  # ground_1 was alive at step 3
    assert m["recoveries"] == 1
    assert m["abandoned_sectors"] == 1  # sector 2
    assert m["reassignment_delay_mean"] == 3  # ground_1 holds it from step 8
    assert m["unresolved_sectors"] == 0
    assert m["duplicate_ownership_steps"] == 0
    assert m["failed_agent_steps_lost"] == 15
    assert m["failed_agent_energy_spent"] == 4


def test_unreassigned_sectors_and_duplicates_are_explicit() -> None:
    config = make_config(drones=1, grounds=2)
    events: list[Event] = [
        AgentDisabled(4, "ground_0", (1, 1), "hazard"),
        EpisodeEnd(10, "time_limit", False, True),
    ]
    log = _log(
        {"kind": "hold", "step": 1, "agent": "ground_0", "sector": 5},
        {"kind": "hold", "step": 2, "agent": "drone_0", "sector": 7},
        {"kind": "hold", "step": 2, "agent": "ground_1", "sector": 7},
        {"kind": "hold", "step": 4, "agent": "ground_1", "sector": None},
    )
    m = compute_failure_metrics(config, events, log)
    assert m["abandoned_sectors"] == 1
    assert m["unresolved_sectors"] == 1  # never reassigned: not counted as a success
    assert m["reassignment_delay_mean"] is None
    assert m["duplicate_ownership_steps"] == 2  # steps 2 and 3
    assert m["disabled_detection_unresolved"] == 2  # nobody marked it


def test_an_observer_already_suspecting_a_peer_when_it_fails_is_aware_at_once() -> None:
    config = make_config(drones=1, grounds=2)
    events: list[Event] = [
        AgentDisabled(8, "ground_0", (1, 1), "hazard"),
        AgentDisabled(8, "ground_1", (2, 2), "hazard"),  # simultaneous failures
        EpisodeEnd(20, "time_limit", False, True),
    ]
    suspicion = {"kind": "peer_state", "step": 5, "observer": "drone_0", "peer": "ground_0"}
    log = _log(suspicion | {"state": "suspected"})  # before the failure; no change after it
    m = compute_failure_metrics(config, events, log)
    assert m["failure_pairs"] == 2  # only drone_0 observes; each failed agent excludes the other
    assert m["false_suspicions"] == 1  # ground_0 was alive at step 5
    assert m["suspicion_detection_delay_mean"] == 0  # still suspected at the failure
    assert m["suspicion_detection_unresolved"] == 1  # nobody became aware of ground_1


def test_policies_without_a_detector_report_none() -> None:
    config = make_config()
    m = compute_failure_metrics(config, [EpisodeEnd(5, "time_limit", False, True)], [])
    assert m["failure_pairs"] is None
    assert m["reassignment_delay_mean"] is None
    assert m["failed_agent_steps_lost"] == 0


# ----------------------------------------------------------------- policies


def _hazard_config() -> ScenarioConfig:
    m2 = ScenarioConfig.from_json((ROOT / "configs" / "m2_default.json").read_text("utf-8"))
    assert m2.hazards is not None
    return replace(
        m2,
        recall_target=1.0,
        hazards=replace(m2.hazards, disable_probability=0.3),
        comms=CommsConfig(ranges=(("drone", 8), ("ground", 5)), p_drop=0.2, latency=1, jitter=1),
    )


@pytest.mark.parametrize("name", ["auction_comm", "auction_no_reassign"])
def test_logged_disablement_is_never_wrong_and_runs_reproduce(name: str) -> None:
    config = _hazard_config()
    for seed in (1000, 1002):
        result = run_episode(config, POLICIES[name], seed)
        again = run_episode(config, POLICIES[name], seed)
        assert result.event_stream_sha256 == again.event_stream_sha256
        assert result.decision_log == again.decision_log
        assert result.comm_metrics == again.comm_metrics
        failed = {e.agent_id: e.step for e in result.events if isinstance(e, AgentDisabled)}
        assert failed  # development check: these seeds include failures
        for entry in result.decision_log:
            if entry.get("kind") == "peer_state" and entry["state"] == "disabled":
                assert entry["peer"] in failed  # no false certainty
                assert entry["step"] >= failed[entry["peer"]]
        m = result.comm_metrics
        assert m is not None
        assert m["failure_pairs"] >= m["disabled_detection_unresolved"]
        assert m["abandoned_sectors"] >= m["unresolved_sectors"]


def test_decision_logs_hold_only_observable_facts() -> None:
    result = run_episode(_hazard_config(), POLICIES["auction_comm"], 1000)
    allowed = {
        "detector": {"kind", "step", "suspicion_steps"},
        "peer_state": {"kind", "step", "observer", "peer", "state"},
        "hold": {"kind", "step", "agent", "sector"},
    }
    for entry in result.decision_log:
        assert set(entry) == allowed[entry["kind"]]


def test_no_reassign_leaves_a_dead_holders_sector_to_lower_bidders_only() -> None:
    a = LeaseAuction("a", honor_failures=False)
    a.decide(1, {0: 5}, NONE, NONE)
    b = LeaseAuction("b", honor_failures=False)
    b.table.merge(a.table.records())
    dead = frozenset({"a"})
    assert all(b.decide(t, {0: 6}, NONE, dead) is None for t in range(2, 80))
    assert b.decide(80, {0: 4}, NONE, dead) == 0


def test_channel_free_episodes_have_no_failure_metrics() -> None:
    result = run_episode(make_config(), POLICIES["frontier"], 0)
    assert result.comm_metrics is None

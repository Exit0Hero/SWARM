"""Phase 3 M3.3: sectors, claim tables and the lease-based auction (ADR-0019)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from swarm.coordination.auction import (
    LEASE_STEPS,
    ClaimTable,
    LeaseAuction,
    sector_masks,
    sector_of,
)
from swarm.core.env import SwarmEnv
from swarm.core.messages import Record
from swarm.core.observation import UNKNOWN
from swarm.core.rng import episode_streams
from swarm.experiments.runner import run_episode
from swarm.policies.base import MissionBrief, build_team
from swarm.policies.registry import POLICIES
from swarm.scenarios.config import CommsConfig, ScenarioConfig
from tests.helpers import make_config

NONE: frozenset[Any] = frozenset()  # no done sectors / no disabled agents


def _comms(**kw: Any) -> ScenarioConfig:
    settings_: dict[str, Any] = {"ranges": (("drone", 8), ("ground", 5))} | kw
    return replace(
        make_config(height=12, width=12, survivors=6, time_limit=150),
        comms=CommsConfig(**settings_),
    )


def _exchange(*auctions: LeaseAuction) -> None:
    """Ideal channel: everyone receives everyone's records."""
    records = [a.table.records() for a in auctions]
    for a in auctions:
        for r in records:
            a.table.merge(r)


# ----------------------------------------------------------------- sectors


def test_sectors_partition_the_grid() -> None:
    masks = sector_masks((20, 20))
    assert len(masks) == 25
    assert all(int(m.sum()) == 16 for m in masks)
    assert np.array_equal(np.sum(masks, axis=0), np.ones((20, 20)))
    edge = sector_masks((6, 10))
    assert len(edge) == 6
    assert np.array_equal(np.sum(edge, axis=0), np.ones((6, 10)))
    for cell in [(0, 0), (3, 3), (4, 0), (5, 9), (2, 8)]:
        assert edge[sector_of(cell, (6, 10))][cell]


# ----------------------------------------------------------------- claim table merging


def test_newer_records_win_and_delayed_older_ones_are_ignored() -> None:
    t = ClaimTable()
    release: Record = ("claim", 3, "a", 7, 10, False)
    older_claim: Record = ("claim", 3, "a", 7, 8, True)
    t.merge((release,))
    t.merge((older_claim,))  # arrives late
    assert t.valid(10, LEASE_STEPS, NONE) == {}
    t.merge((older_claim, release, older_claim))  # duplicates
    assert t.records() == (release,)


def test_malformed_records_raise_and_unknown_tags_are_ignored() -> None:
    t = ClaimTable()
    t.merge((("cell", 0, 0, 1, False, 0, "a"),))
    assert t.records() == ()
    with pytest.raises(ValueError, match="claim"):
        t.merge((("claim", 1, "a", 2, 3),))
    with pytest.raises(ValueError, match="beat"):
        t.merge((("beat", "a", "3"),))


_claims = st.tuples(
    st.just("claim"),
    st.integers(0, 3),
    st.sampled_from(["a", "b", "c"]),
    st.integers(0, 9),
    st.integers(0, 20),
    st.booleans(),
)
_beats = st.tuples(st.just("beat"), st.sampled_from(["a", "b", "c"]), st.integers(0, 20))


@settings(max_examples=60)
@given(records=st.lists(st.one_of(_claims, _beats), max_size=25), data=st.data())
def test_claim_merging_is_independent_of_order_grouping_and_duplicates(
    records: list[Record], data: st.DataObject
) -> None:
    reference = ClaimTable()
    reference.merge(records)
    shuffled = data.draw(st.permutations(records))
    cut = data.draw(st.integers(0, len(records)))
    left, right, joined = ClaimTable(), ClaimTable(), ClaimTable()
    left.merge(shuffled[:cut])
    right.merge(shuffled[cut:] + shuffled[:cut])  # overlap = duplicates
    joined.merge(right.records())
    joined.merge(left.records())
    assert joined.records() == reference.records()
    for now in (5, 20, 40):
        assert joined.valid(now, LEASE_STEPS, NONE) == reference.valid(now, LEASE_STEPS, NONE)


# ----------------------------------------------------------------- winner, lease, failures


def test_lowest_bid_wins_with_ties_broken_by_agent_id() -> None:
    t = ClaimTable()
    t.merge((("claim", 0, "b", 4, 1, True), ("claim", 0, "a", 4, 1, True)))
    t.merge((("claim", 0, "c", 9, 1, True),))
    assert t.valid(1, LEASE_STEPS, NONE)[0] == [(4, "a"), (4, "b"), (9, "c")]


def test_leases_expire_after_15_steps_unless_infinite() -> None:
    t = ClaimTable()
    t.merge((("claim", 2, "a", 3, 10, True),))
    assert t.valid(10 + LEASE_STEPS, LEASE_STEPS, NONE) == {2: [(3, "a")]}
    assert t.valid(11 + LEASE_STEPS, LEASE_STEPS, NONE) == {}
    assert t.valid(500, None, NONE) == {2: [(3, "a")]}  # auction_no_reassign


def test_conflicting_claims_resolve_to_the_lower_bid() -> None:
    a, b = LeaseAuction("a"), LeaseAuction("b")
    assert a.decide(1, {0: 5, 1: 9}, NONE, NONE) == 0
    assert b.decide(1, {0: 3, 1: 8}, NONE, NONE) == 0  # simultaneous: both claim sector 0
    _exchange(a, b)
    assert a.decide(2, {0: 5, 1: 9}, NONE, NONE) == 1  # outbid: releases 0, takes 1
    assert b.decide(2, {0: 3, 1: 8}, NONE, NONE) == 0  # renews
    assert a.stats["released_outbid"] == 1
    _exchange(a, b)
    assert a.winners(2, NONE) == b.winners(2, NONE) == {0: (3, "b"), 1: (9, "a")}


@settings(max_examples=30)
@given(
    bids=st.lists(
        st.dictionaries(st.integers(0, 5), st.integers(0, 30), min_size=1, max_size=6),
        min_size=2,
        max_size=4,
    )
)
def test_allocation_converges_under_an_ideal_channel(bids: list[dict[int, int]]) -> None:
    auctions = [LeaseAuction(f"agent_{i}") for i in range(len(bids))]
    for step in range(1, 8):
        for auction, own in zip(auctions, bids, strict=True):
            auction.decide(step, own, NONE, NONE)
        _exchange(*auctions)
    held = [a.holding for a in auctions if a.holding is not None]
    assert len(held) == len(set(held))  # no sector held twice
    views = [a.winners(7, NONE) for a in auctions]
    assert all(v == views[0] for v in views)  # everyone agrees on the winners
    assert all(views[0][a.holding][1] == a.owner for a in auctions if a.holding is not None)


def test_holders_converge_when_both_bids_keep_falling() -> None:
    """Both approach the sector (bids fall each step). Each must judge the conflict from the
    shared claim table, not its own fresh bid against the other's stale one."""
    a, b = LeaseAuction("a"), LeaseAuction("b")
    for step in range(1, 9):
        a.decide(step, {0: 20 - step}, NONE, NONE)
        b.decide(step, {0: 20 - step}, NONE, NONE)
        _exchange(a, b)
    assert [x.holding for x in (a, b)] == [0, None]  # equal bids: lower agent ID keeps it
    assert b.stats["released_outbid"] == 1


def test_disabled_holders_release_at_once_and_silent_ones_after_the_lease() -> None:
    a, b = LeaseAuction("a"), LeaseAuction("b")
    a.decide(1, {0: 2}, NONE, NONE)
    _exchange(a, b)
    assert b.decide(2, {0: 9}, NONE, NONE) is None  # a still wins
    assert b.decide(3, {0: 9}, NONE, frozenset({"a"})) == 0  # a seen disabled: taken over
    assert b.stats["takeovers_after_failure"] == 1
    c = LeaseAuction("c")
    c.table.merge(a.table.records())  # a's last renewal was at step 1
    assert c.decide(1 + LEASE_STEPS, {0: 9}, NONE, NONE) is None
    assert c.decide(2 + LEASE_STEPS, {0: 9}, NONE, NONE) == 0  # lease expired


def test_no_reassign_keeps_a_failed_holders_sector_unless_outbid() -> None:
    a = LeaseAuction("a", honor_failures=False)
    a.decide(1, {0: 5}, NONE, NONE)
    b, c = LeaseAuction("b", honor_failures=False), LeaseAuction("c", honor_failures=False)
    for x in (b, c):
        x.table.merge(a.table.records())
    assert b.decide(100, {0: 9}, NONE, frozenset({"a"})) is None  # no expiry, no release
    assert b.stats["takeovers_after_failure"] == 0
    assert c.decide(100, {0: 4}, NONE, frozenset({"a"})) == 0  # a lower bid still wins


def test_holders_release_done_and_infeasible_sectors() -> None:
    a = LeaseAuction("a")
    a.decide(1, {0: 2, 1: 3}, NONE, NONE)
    assert a.decide(2, {1: 3}, frozenset({0}), NONE) == 1
    assert a.stats["released_done"] == 1
    assert a.decide(3, {}, NONE, NONE) is None
    assert a.stats["released_infeasible"] == 1


# ----------------------------------------------------------------- policies


@pytest.mark.parametrize("name", ["auction_comm", "auction_no_reassign"])
def test_runs_are_deterministic_and_decentralized(name: str) -> None:
    config = _comms(latency=1, jitter=1, p_drop=0.2)
    a = run_episode(config, POLICIES[name], 1)
    b = run_episode(config, POLICIES[name], 1)
    c = run_episode(config, POLICIES[name], 2)
    assert a.event_stream_sha256 == b.event_stream_sha256 != c.event_stream_sha256
    assert a.policy["information"] == "decentralized_comm"
    assert a.policy["uses_oracle"] is False
    assert a.policy_diagnostics is not None
    assert a.policy_diagnostics["auction_claims"] > 0


def test_total_communication_loss_leaves_agents_independent() -> None:
    config = _comms(p_drop=1.0)
    team = build_team(POLICIES["auction_comm"], MissionBrief.from_config(config))
    env = SwarmEnv(config)
    obs, _ = env.reset(3)
    rngs = episode_streams(3, env.possible_agents).policies
    while not env.done:
        actions = team.act(obs, rngs)
        result = env.step(actions, team.outbox())  # type: ignore[attr-defined]
        obs = result.observations
    for aid, agent in team.agents.items():  # type: ignore[attr-defined]
        holders = {r[2] for r in agent.auction.table.records() if r[0] == "claim"}
        assert holders <= {aid}  # nothing was ever received
        assert agent.belief.stats["records_received"] == 0


def test_beliefs_hold_only_observed_truth_and_only_active_agents_send() -> None:
    config = _comms(latency=1, jitter=2, p_drop=0.3)
    team = build_team(POLICIES["auction_comm"], MissionBrief.from_config(config))
    env = SwarmEnv(config)
    obs, _ = env.reset(5)
    scenario = env.snapshot().scenario
    rngs = episode_streams(5, env.possible_agents).policies
    while not env.done:
        actions = team.act(obs, rngs)
        outbox = team.outbox()  # type: ignore[attr-defined]
        assert set(outbox) <= set(env.agents)
        obs = env.step(actions, outbox).observations
        for agent in team.agents.values():  # type: ignore[attr-defined]
            known = agent.belief.terrain != UNKNOWN
            assert np.array_equal(agent.belief.terrain[known], scenario.terrain[known])


def test_auction_policies_require_a_channel_and_refuse_the_oracle() -> None:
    with pytest.raises(ValueError, match="comms block"):
        build_team(POLICIES["auction_comm"], MissionBrief.from_config(make_config()))
    team = build_team(POLICIES["auction_comm"], MissionBrief.from_config(_comms()))
    env = SwarmEnv(_comms())
    env.reset(0)
    with pytest.raises(RuntimeError, match="never uses the oracle"):
        team.attach_oracle(env.oracle_view())

"""Lease-based auction policies over the communication channel (ADR-0019).

``auction_comm``: each agent keeps its own belief map (ADR-0018) and claim table, bids on 4 x 4
search sectors, and explores frontier cells inside the sector it holds, falling back to the Phase 2
frontier rule when it holds none. ``auction_no_reassign`` is the H5 control: identical bidding,
renewal and release, but leases never expire and disabled holders are not released, and its
fallback skips sectors held by others. Agents see only their own observation and inbox.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from swarm.coordination.auction import SUSPICION_STEPS, LeaseAuction, sector_masks
from swarm.core.messages import Message
from swarm.core.observation import Observation
from swarm.core.types import AgentId
from swarm.navigation.astar import UNREACHABLE, cost_field
from swarm.policies.base import Diagnostics, MissionBrief, TeamFactory
from swarm.policies.decentralized import NavigatorAgent
from swarm.policies.planning import Counters, feasible_mask
from swarm.policies.shared import MESSAGE_KIND, SharedFrontierTeam, SharedNavigatorAgent
from swarm.scenarios.config import AgentTypeSpec


class AuctionAgent(SharedNavigatorAgent):
    """A shared-belief frontier navigator that searches inside the sector it has won."""

    def __init__(
        self,
        agent_id: AgentId,
        spec: AgentTypeSpec,
        brief: MissionBrief,
        counters: Counters,
        *,
        honor_failures: bool,
    ) -> None:
        super().__init__(agent_id, spec, brief, counters)
        self.auction = LeaseAuction(agent_id, honor_failures=honor_failures)
        self.sectors = sector_masks(brief.grid_shape)
        self._held_by_others: npt.NDArray[np.bool_] | None = None
        self.suspected: set[AgentId] = set()
        self.suspicions = 0

    def act(self, obs: Observation, rng: np.random.Generator) -> int:
        for delivery in obs.inbox:
            self.belief.merge_records(delivery.message.records)
            self.auction.table.merge(delivery.message.records)
        self.belief.update(obs)  # own observation first, so bids use it
        self._allocate(obs)
        return NavigatorAgent.act(self, obs, rng)

    def _allocate(self, obs: Observation) -> None:
        now, here = obs.step, obs.position
        passable = self.cache.passable_for(self.known, self.layer)
        to_base = self.cache.to_base_for(self.known, self.model)
        feasible = feasible_mask(self.model, obs.energy, passable, here, self.known, to_base)
        dist = cost_field(passable, here, self._costs())
        targetable = feasible & self.cache.frontier_for(self.known, self.layer)
        targetable &= dist != UNREACHABLE
        known = self.known.known
        bids: dict[int, int] = {}
        done: set[int] = set()
        for k, mask in enumerate(self.sectors):
            if bool(known[mask].all()):
                done.add(k)
                continue
            cells = targetable & mask
            if bool(cells.any()):
                bids[k] = int(dist[cells].min())
        disabled = self.belief.disabled_agents()
        self.auction.decide(now, bids, frozenset(done), disabled)
        for peer, last in self.auction.table.heard().items():
            if peer == self.agent_id or peer in disabled:
                continue
            if now - last > SUSPICION_STEPS:
                if peer not in self.suspected:
                    self.suspected.add(peer)
                    self.suspicions += 1
            else:
                self.suspected.discard(peer)
        self._held_by_others = None
        if self.auction.holding is None and not self.auction.honor_failures:
            held = np.zeros(self.known.shape, dtype=np.bool_)
            for k, (_, holder) in self.auction.winners(now, disabled).items():
                if holder != self.agent_id:
                    held |= self.sectors[k]
            self._held_by_others = held

    def _scope(self) -> npt.NDArray[np.bool_] | None:
        if self.auction.holding is not None:
            return self.sectors[self.auction.holding]
        if self._held_by_others is not None:
            return ~self._held_by_others
        return None

    def compose(self) -> list[Message]:
        if self.max_records == 0:
            return []
        claims = self.auction.table.outbox_records(self.max_records)
        room = None if self.max_records is None else self.max_records - len(claims)
        beliefs = self.belief.outbox_records(room) if room is None or room > 0 else ()
        records = claims + beliefs
        if not records:
            return []
        message = Message(self.agent_id, self.seq, MESSAGE_KIND, records)
        self.seq += 1
        return [message]


class AuctionTeam(SharedFrontierTeam):
    """One :class:`AuctionAgent` per agent; communicates only through the channel."""

    def __init__(self, name: str, brief: MissionBrief, *, honor_failures: bool) -> None:
        super().__init__(name, brief)
        self.agents: dict[AgentId, AuctionAgent] = {  # type: ignore[assignment]
            aid: AuctionAgent(aid, spec, brief, self.counters, honor_failures=honor_failures)
            for aid, spec in brief.agents.items()
        }
        self.duplicate_claim_steps = 0

    def act(
        self,
        observations: Mapping[AgentId, Observation],
        rngs: Mapping[AgentId, np.random.Generator],
    ) -> dict[AgentId, int]:
        actions = super().act(observations, rngs)
        held = [self.agents[a].auction.holding for a in actions]
        held_sectors = [k for k in held if k is not None]
        if len(held_sectors) != len(set(held_sectors)):
            self.duplicate_claim_steps += 1  # diagnostic only: no agent reads it
        return actions

    def diagnostics(self) -> Diagnostics:
        out = super().diagnostics()
        for key in (
            "claims",
            "renewals",
            "released_done",
            "released_outbid",
            "released_infeasible",
            "takeovers_after_failure",
        ):
            out[f"auction_{key}"] = sum(a.auction.stats[key] for a in self.agents.values())
        out["auction_suspicions"] = sum(a.suspicions for a in self.agents.values())
        out["auction_duplicate_claim_steps"] = self.duplicate_claim_steps
        return out


def _factory(name: str, honor_failures: bool) -> TeamFactory:
    return TeamFactory(name, lambda brief: AuctionTeam(name, brief, honor_failures=honor_failures))


AUCTION_COMM = _factory("auction_comm", honor_failures=True)
AUCTION_NO_REASSIGN = _factory("auction_no_reassign", honor_failures=False)

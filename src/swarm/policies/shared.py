"""``frontier_shared``: Phase 2 frontier exploration on beliefs shared over the channel (ADR-0018).

Each agent runs the Phase 2 :class:`~swarm.policies.decentralized.NavigatorAgent` unchanged, except
that its map is a :class:`~swarm.coordination.belief.BeliefMap`: before deciding, it merges the
records in its own inbox; after deciding, it broadcasts its changed records plus an anti-entropy
slice. There are no claims, no auction and no reassignment: the only difference from ``frontier``
is the information each agent holds. An agent sees only its own observation and inbox.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from swarm.coordination.belief import BeliefMap
from swarm.core.messages import HEADER_UNITS, Message
from swarm.core.observation import Observation
from swarm.core.oracle import OracleView
from swarm.core.types import AgentId, AgentStatus
from swarm.policies.base import DECENTRALIZED_COMM, Diagnostics, MissionBrief, TeamFactory
from swarm.policies.decentralized import NavigatorAgent
from swarm.policies.planning import Counters
from swarm.scenarios.config import AgentTypeSpec

MESSAGE_KIND = "belief"


class SharedNavigatorAgent(NavigatorAgent):
    """A frontier navigator whose map also takes records received over the channel."""

    def __init__(
        self, agent_id: AgentId, spec: AgentTypeSpec, brief: MissionBrief, counters: Counters
    ) -> None:
        super().__init__(
            agent_id, spec, brief, target_rule="frontier", hazard_aware=False, counters=counters
        )
        self.belief = BeliefMap(agent_id, brief.grid_shape, brief.base)
        self.known = self.belief  # the Phase 2 planner reads and updates the belief map
        bandwidth = None if brief.comms is None else brief.comms.bandwidth
        self.max_records = None if bandwidth is None else max(0, bandwidth - HEADER_UNITS)
        self.seq = 0

    def act(self, obs: Observation, rng: np.random.Generator) -> int:
        self.belief.merge_inbox(obs.inbox)
        return super().act(obs, rng)  # updates the belief with ``obs`` itself, then decides

    def compose(self) -> list[Message]:
        """This step's outgoing message (empty when there is nothing to send or no room)."""
        if self.max_records == 0:
            return []
        records = self.belief.outbox_records(self.max_records)
        if not records:
            return []
        message = Message(self.agent_id, self.seq, MESSAGE_KIND, records)
        self.seq += 1
        return [message]


class SharedFrontierTeam:
    """One :class:`SharedNavigatorAgent` per agent; communicates only through the channel."""

    information = DECENTRALIZED_COMM
    uses_oracle = False
    version = "1"

    def __init__(self, name: str, brief: MissionBrief) -> None:
        if brief.comms is None:
            raise ValueError(f"{name}: needs a scenario with a comms block (ADR-0017)")
        self.name = name
        self.counters = Counters()
        self.agents = {
            aid: SharedNavigatorAgent(aid, spec, brief, self.counters)
            for aid, spec in brief.agents.items()
        }
        self._outbox: dict[AgentId, list[Message]] = {}

    def act(
        self,
        observations: Mapping[AgentId, Observation],
        rngs: Mapping[AgentId, np.random.Generator],
    ) -> dict[AgentId, int]:
        actions: dict[AgentId, int] = {}
        self._outbox = {}
        for aid, obs in sorted(observations.items()):
            if obs.status is not AgentStatus.ACTIVE:
                continue
            agent = self.agents[aid]
            actions[aid] = agent.act(obs, rngs[aid])
            messages = agent.compose()
            if messages:
                self._outbox[aid] = messages
        return actions

    def outbox(self) -> Mapping[AgentId, Sequence[Message]]:
        return self._outbox

    def attach_oracle(self, view: OracleView) -> None:
        raise RuntimeError(f"{self.name} is decentralized and never uses the oracle")

    def diagnostics(self) -> Diagnostics:
        out: Diagnostics = dict(self.counters.as_dict())
        for key in (
            "records_sent",
            "records_received",
            "records_changed",
            "duplicates",
            "stale_ignored",
            "conflicts",
        ):
            out[f"belief_{key}"] = sum(a.belief.stats[key] for a in self.agents.values())
        return out

    def decision_log(self) -> list[dict[str, Any]]:
        return []


FRONTIER_SHARED = TeamFactory(
    "frontier_shared", lambda brief: SharedFrontierTeam("frontier_shared", brief)
)

"""Failure detection from communication evidence (ADR-0020).

Each agent classifies every teammate from what it can observe, never from simulator state:

* ``active``: heard within ``contact_steps`` (the normal heartbeat age on this channel);
* ``disconnected``: not heard for longer, but for at most ``suspicion_steps``: a temporary gap;
* ``suspected``: not heard for more than ``suspicion_steps``. Missing messages are not proof of
  failure, so this is reversible: hearing the peer again recovers it;
* ``disabled``: seen disabled (a tombstone: the agent's own observation or a relayed sighting).
  This is the only certain state and is absorbing.

"Heard" means any heartbeat, direct or relayed, so evidence of life travels over several hops. All
teammates start at the base together, so every peer counts as heard at step 0. Transitions are a
deterministic function of ``(now, last_heard, disabled)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Final

from swarm.core.types import AgentId

DEFAULT_SUSPICION_STEPS: Final[int] = 10


class PeerState(StrEnum):
    ACTIVE = "active"
    DISCONNECTED = "disconnected"
    SUSPECTED = "suspected"
    DISABLED = "disabled"


Transition = tuple[int, AgentId, PeerState, PeerState]  # (step, peer, old, new)


class FailureDetector:
    """One agent's view of its teammates' availability."""

    def __init__(
        self,
        owner: AgentId,
        teammates: Iterable[AgentId],
        *,
        suspicion_steps: int = DEFAULT_SUSPICION_STEPS,
        contact_steps: int = 1,
    ) -> None:
        if suspicion_steps < 1 or contact_steps < 1 or contact_steps > suspicion_steps:
            raise ValueError(
                "need 1 <= contact_steps <= suspicion_steps, "
                f"got {contact_steps} and {suspicion_steps}"
            )
        self.owner = owner
        self.suspicion_steps = suspicion_steps
        self.contact_steps = contact_steps
        self.states: dict[AgentId, PeerState] = {
            p: PeerState.ACTIVE for p in sorted(teammates) if p != owner
        }
        self.suspicions = 0
        self.recoveries = 0

    def classify(
        self, peer: AgentId, now: int, last_heard: Mapping[AgentId, int], disabled: frozenset[str]
    ) -> PeerState:
        """The state ``peer`` should be in now (without the absorbing rule)."""
        if peer in disabled:
            return PeerState.DISABLED
        age = now - last_heard.get(peer, 0)
        if age <= self.contact_steps:
            return PeerState.ACTIVE
        if age <= self.suspicion_steps:
            return PeerState.DISCONNECTED
        return PeerState.SUSPECTED

    def update(
        self, now: int, last_heard: Mapping[AgentId, int], disabled: frozenset[str]
    ) -> list[Transition]:
        """Advance every peer's state; return this step's transitions, sorted by peer."""
        changes: list[Transition] = []
        for peer, old in self.states.items():
            if old is PeerState.DISABLED:
                continue  # certain and permanent
            new = self.classify(peer, now, last_heard, disabled)
            if new is old:
                continue
            if new is PeerState.SUSPECTED:
                self.suspicions += 1
            elif old is PeerState.SUSPECTED and new is not PeerState.DISABLED:
                self.recoveries += 1
            self.states[peer] = new
            changes.append((now, peer, old, new))
        return changes

    def suspected(self) -> frozenset[AgentId]:
        return frozenset(p for p, s in self.states.items() if s is PeerState.SUSPECTED)

"""Typed, immutable simulation events (ADR-0005).

Each ``step()`` returns events in step-phase order (validation, movement, energy, sensing,
mission state) and, within a phase, sorted by agent ID. Events are the only input metrics need,
and the episode's event stream serializes to canonical JSON Lines whose SHA-256 is the
determinism fingerprint.

Some fields (``Detection.true_positive``, ``Detection.survivor_index``) are ground truth for the
evaluator. Events are never given to policies (ADR-0008).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, ClassVar, Final

from swarm.core.types import AgentId, Position

EVENT_SCHEMA_VERSION: Final[int] = 1


@dataclass(frozen=True, slots=True)
class Event:
    """Base class. ``step`` is the time *after* the ``step()`` call that produced the event."""

    kind: ClassVar[str] = "event"
    step: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict with ``kind`` first; tuples become lists, enums their values."""
        out: dict[str, Any] = {"kind": self.kind}
        for f in fields(self):
            out[f.name] = _jsonable(getattr(self, f.name))
        return out


@dataclass(frozen=True, slots=True)
class InvalidAction(Event):
    """A masked action was submitted; it was executed as ``STAY`` (ADR-0004)."""

    kind: ClassVar[str] = "invalid_action"
    agent_id: AgentId
    action: int
    reason: str  # "out_of_bounds" | "impassable_terrain" | "insufficient_energy"


@dataclass(frozen=True, slots=True)
class Moved(Event):
    kind: ClassVar[str] = "moved"
    agent_id: AgentId
    from_position: Position
    to_position: Position


@dataclass(frozen=True, slots=True)
class MoveConflict(Event):
    """A statically valid move failed in conflict resolution (ADR-0003)."""

    kind: ClassVar[str] = "move_conflict"
    agent_id: AgentId
    position: Position
    target: Position
    conflict: str  # "vertex" | "swap" | "blocked"


@dataclass(frozen=True, slots=True)
class EnergySpent(Event):
    kind: ClassVar[str] = "energy_spent"
    agent_id: AgentId
    amount: int
    remaining: int


@dataclass(frozen=True, slots=True)
class Recharged(Event):
    kind: ClassVar[str] = "recharged"
    agent_id: AgentId
    amount: int
    remaining: int


@dataclass(frozen=True, slots=True)
class AgentDisabled(Event):
    kind: ClassVar[str] = "agent_disabled"
    agent_id: AgentId
    position: Position
    reason: str  # "energy_depleted" | "hazard" | "injected" (scheduled failure, ADR-0021)


@dataclass(frozen=True, slots=True)
class HazardExposure(Event):
    """An affected agent ended a step in a hazardous cell (ADR-0011). Emitted even at 0 damage."""

    kind: ClassVar[str] = "hazard_exposure"
    agent_id: AgentId
    cell: Position
    energy_lost: int
    remaining: int


@dataclass(frozen=True, slots=True)
class Detection(Event):
    """A detection report (true or false, ADR-0012).

    ``true_positive`` and ``survivor_index`` are evaluator-only; policies only ever see the cell.
    """

    kind: ClassVar[str] = "detection"
    agent_id: AgentId
    cell: Position
    true_positive: bool
    survivor_index: int | None


@dataclass(frozen=True, slots=True)
class MessageSent(Event):
    """A message left its sender (ADR-0017). ``recipients`` = agents in range, sorted; empty when
    the sender was disabled during the step or nobody was in range."""

    kind: ClassVar[str] = "message_sent"
    sender: AgentId
    seq: int
    message_kind: str
    size: int
    recipients: tuple[AgentId, ...]


@dataclass(frozen=True, slots=True)
class MessageDropped(Event):
    """A message to one in-range receiver was lost (ADR-0017).

    ``reason``: ``"random"`` (the drop draw) or ``"receiver_disabled"`` (disabled before delivery).
    """

    kind: ClassVar[str] = "message_dropped"
    receiver: AgentId
    sender: AgentId
    seq: int
    reason: str


@dataclass(frozen=True, slots=True)
class MessageDelivered(Event):
    """A message reached a receiver's inbox (ADR-0017). ``latency`` = ``step - sent_step``."""

    kind: ClassVar[str] = "message_delivered"
    receiver: AgentId
    sender: AgentId
    seq: int
    sent_step: int
    latency: int


@dataclass(frozen=True, slots=True)
class SurvivorReportReceived(Event):
    """``receiver`` first got a survivor-report record for ``cell`` (ADR-0018), via ``sender``.

    Logged once per (receiver, cell), after the ``message_delivered`` event that carried it.
    Evaluator-only: used for the survivor-information propagation metric.
    """

    kind: ClassVar[str] = "survivor_report_received"
    receiver: AgentId
    cell: Position
    sender: AgentId


@dataclass(frozen=True, slots=True)
class EpisodeEnd(Event):
    kind: ClassVar[str] = "episode_end"
    reason: str  # "mission_success" | "all_disabled" | "time_limit" (that precedence, ADR-0009)
    terminated: bool
    truncated: bool


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple | list):
        return [_jsonable(v) for v in value]
    return value


def events_to_jsonl(events: Iterable[Event]) -> str:
    """Canonical JSON Lines: sorted keys, compact separators, one event per line."""
    return "".join(
        json.dumps(e.to_dict(), sort_keys=True, separators=(",", ":")) + "\n" for e in events
    )


def event_stream_sha256(events: Iterable[Event]) -> str:
    """SHA-256 of :func:`events_to_jsonl`: the episode's determinism fingerprint (ADR-0005)."""
    return hashlib.sha256(events_to_jsonl(events).encode("utf-8")).hexdigest()

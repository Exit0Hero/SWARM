"""Messages carried by the communication channel (ADR-0017).

A :class:`Message` is a broadcast from one agent. Its ``records`` are flat tuples of ``int``,
``str`` and ``bool`` values only, so a message can never carry an array, an object reference or a
float: whatever an agent sends is plain data derived from its own observations and inbox
(ADR-0008). Typed payload kinds (beacons, map and evidence deltas, claims) are layered on top of
records in later Phase 3 milestones.

Size accounting: a message costs :data:`HEADER_UNITS` plus one unit per record.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from swarm.core.types import AgentId

HEADER_UNITS: Final[int] = 4
"""Size of a message with no records, in channel units."""

MAX_RECORD_FIELDS: Final[int] = 16
MAX_TEXT_LENGTH: Final[int] = 64
MAX_KIND_LENGTH: Final[int] = 32

Scalar = int | str | bool
Record = tuple[Scalar, ...]


@dataclass(frozen=True, slots=True)
class Message:
    """One broadcast message.

    Attributes:
        sender: the sending agent; must match the outbox it is submitted in.
        seq: per-sender sequence number, strictly increasing over the episode.
        kind: short label for logs and events (for example ``"beacon"``).
        records: flat tuples of ``int``, ``str`` or ``bool`` values.
    """

    sender: AgentId
    seq: int
    kind: str
    records: tuple[Record, ...] = ()

    @property
    def size(self) -> int:
        """Channel units this message costs: header plus one unit per record."""
        return HEADER_UNITS + len(self.records)


@dataclass(frozen=True, slots=True)
class Delivery:
    """A message as it arrives in a receiver's inbox."""

    message: Message
    sent_step: int
    """The step during which the message was sent."""


def validate_message(message: object, sender: AgentId) -> Message:
    """Return ``message`` if it is a well-formed message from ``sender``, else raise.

    Raises:
        TypeError: if ``message`` is not a :class:`Message` or a record holds a disallowed type.
        ValueError: on a sender mismatch, a negative ``seq``, or a malformed kind or record.
    """
    # Exact types throughout: a subclass could carry extra attributes (arrays, references).
    if type(message) is not Message:
        raise TypeError(f"messages[{sender!r}]: expected Message, got {type(message).__name__}")
    where = f"messages[{sender!r}] seq={message.seq!r}"
    if message.sender != sender:
        raise ValueError(f"{where}: sender {message.sender!r} does not match outbox {sender!r}")
    if isinstance(message.seq, bool) or not isinstance(message.seq, int) or message.seq < 0:
        raise ValueError(f"{where}: seq must be a non-negative int")
    if type(message.kind) is not str or not 0 < len(message.kind) <= MAX_KIND_LENGTH:
        raise ValueError(f"{where}: kind must be a string of 1..{MAX_KIND_LENGTH} characters")
    if type(message.records) is not tuple:
        raise TypeError(f"{where}: records must be a tuple, got {type(message.records).__name__}")
    for k, record in enumerate(message.records):
        _validate_record(record, f"{where} records[{k}]")
    return message


def _validate_record(record: object, where: str) -> None:
    if type(record) is not tuple:
        raise TypeError(f"{where}: must be a tuple, got {type(record).__name__}")
    if len(record) > MAX_RECORD_FIELDS:
        raise ValueError(f"{where}: at most {MAX_RECORD_FIELDS} fields, got {len(record)}")
    for value in record:
        # Exact types only: rejects NumPy scalars, floats, arrays and arbitrary objects.
        if type(value) not in (int, str, bool):
            raise TypeError(f"{where}: values must be int, str or bool, got {type(value).__name__}")
        if type(value) is str and len(value) > MAX_TEXT_LENGTH:
            raise ValueError(f"{where}: strings are limited to {MAX_TEXT_LENGTH} characters")


def outbox_size(messages: Sequence[Message]) -> int:
    """Total channel units of ``messages``."""
    return sum(m.size for m in messages)

"""The communication channel: step 5b of the environment step (ADR-0017).

The channel decides which messages arrive, when, and to whom. Policies only compose messages.

Per step ``t``:

1. **Transmit.** Senders in sorted ID order, each sender's messages in ``seq`` order. A sender
   disabled earlier in the step reaches no one. Otherwise the receivers are every other active
   agent whose Euclidean distance (after movement) is at most ``min(range_sender,
   range_receiver)``. For every (message, receiver) pair in range, in sorted receiver order, the
   channel always draws exactly two uniforms from the ``comms`` stream: ``u`` for the drop test
   (dropped when ``u < p_drop``) and ``v`` for the jitter, ``floor(v * (jitter + 1))`` in
   ``0..jitter``. The draw count never depends on ``p_drop``, ``jitter`` or drop outcomes, so the
   stream stays aligned across channel settings. Dropped pairs emit
   ``message_dropped(reason="random")``; the rest are queued for step ``t + latency + jitter``.
2. **Deliver.** Every queued pair due at or before ``t``, sorted by (receiver, sent step, sender,
   seq), goes to the receiver's inbox, or is dropped with reason ``receiver_disabled`` if the
   receiver was disabled meanwhile.

A message sent during step ``t`` with delay ``d`` is in the observation returned by step
``t + d``, so with ``latency = 0`` it can first influence the action chosen for step ``t + 1``.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from swarm.core.events import Event, MessageDelivered, MessageDropped, MessageSent
from swarm.core.messages import Delivery, Message
from swarm.core.types import AgentId, Position
from swarm.scenarios.config import CommsConfig


@dataclass(frozen=True, slots=True, order=True)
class _Pending:
    due_step: int
    receiver: AgentId
    sent_step: int
    sender: AgentId
    seq: int
    message: Message = field(compare=False)


class Channel:
    """Range-limited, lossy, delayed broadcast channel for one episode.

    Args:
        config: the channel parameters.
        agent_ranges: communication range per agent ID (``None`` = unlimited).
    """

    def __init__(self, config: CommsConfig, agent_ranges: Mapping[AgentId, int | None]) -> None:
        self._config = config
        self._ranges = dict(agent_ranges)
        self._pending: list[_Pending] = []
        self._rng: np.random.Generator | None = None

    def reset(self, rng: np.random.Generator) -> None:
        """Start an episode with an empty queue and the episode's ``comms`` stream."""
        self._rng = rng
        self._pending = []

    @property
    def in_flight(self) -> int:
        """Number of (message, receiver) pairs queued for later delivery."""
        return len(self._pending)

    def in_range(self, a: AgentId, pos_a: Position, b: AgentId, pos_b: Position) -> bool:
        """Whether a link between ``a`` and ``b`` exists at these positions (symmetric)."""
        ra, rb = self._ranges[a], self._ranges[b]
        reach = rb if ra is None else ra if rb is None else min(ra, rb)
        if reach is None:
            return True
        dr, dc = pos_a[0] - pos_b[0], pos_a[1] - pos_b[1]
        return dr * dr + dc * dc <= reach * reach

    def transmit(
        self,
        step: int,
        outbox: Mapping[AgentId, Sequence[Message]],
        positions: Mapping[AgentId, Position],
        active: frozenset[AgentId],
    ) -> list[Event]:
        """Send every message in ``outbox`` (already validated) and queue survivors."""
        rng = self._rng
        if rng is None:
            raise RuntimeError("Channel.reset(rng) must be called before transmit")
        events: list[Event] = []
        receivers_all = sorted(active)
        for sender in sorted(outbox):
            for message in sorted(outbox[sender], key=lambda m: m.seq):
                recipients: tuple[AgentId, ...] = ()
                if sender in active:
                    recipients = tuple(
                        r
                        for r in receivers_all
                        if r != sender and self.in_range(sender, positions[sender], r, positions[r])
                    )
                events.append(
                    MessageSent(step, sender, message.seq, message.kind, message.size, recipients)
                )
                for receiver in recipients:
                    u = float(rng.random())
                    # A uniform, not ``integers``: ``integers(0, 1)`` draws nothing when
                    # jitter is 0, which would shift every later draw (ADR-0017 rule 4).
                    jitter = int(float(rng.random()) * (self._config.jitter + 1))
                    if u < self._config.p_drop:
                        events.append(MessageDropped(step, receiver, sender, message.seq, "random"))
                        continue
                    due = step + self._config.latency + jitter
                    bisect.insort(
                        self._pending, _Pending(due, receiver, step, sender, message.seq, message)
                    )
        return events

    def deliver(
        self, step: int, active: frozenset[AgentId]
    ) -> tuple[dict[AgentId, tuple[Delivery, ...]], list[Event]]:
        """Deliver every queued message due at or before ``step``."""
        due = [p for p in self._pending if p.due_step <= step]
        self._pending = [p for p in self._pending if p.due_step > step]
        due.sort(key=lambda p: (p.receiver, p.sent_step, p.sender, p.seq))
        inboxes: dict[AgentId, list[Delivery]] = {}
        events: list[Event] = []
        for p in due:
            if p.receiver not in active:
                events.append(
                    MessageDropped(step, p.receiver, p.sender, p.seq, "receiver_disabled")
                )
                continue
            inboxes.setdefault(p.receiver, []).append(Delivery(p.message, p.sent_step))
            events.append(
                MessageDelivered(step, p.receiver, p.sender, p.seq, p.sent_step, step - p.sent_step)
            )
        return {r: tuple(d) for r, d in inboxes.items()}, events

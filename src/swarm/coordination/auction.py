"""Search sectors and the lease-based consensus auction (ADR-0019).

**Sectors.** The grid is cut into fixed 4 x 4-cell sectors, numbered row-major (edge sectors may be
smaller). A sector is *done* for an agent when every cell in it is known in that agent's belief.

**Claims.** A claim record ``("claim", sector, holder, bid, stamp, active)`` is published by its
holder only. Per (sector, holder) the record with the greatest ``(stamp, active, bid)`` is kept, so
merging is commutative, associative and idempotent, and a delayed older record never overrides a
newer one (a release is a newer record with ``active = False``). A claim is *valid* at step
``now`` when it is active, ``stamp + lease >= now`` and its holder is not known to be disabled.
The winner of a sector is the valid claim with the smallest ``(bid, holder)``.

**Heartbeats.** ``("beat", agent, step)``: the latest step known for each agent, relayed like any
record. A peer is *suspected* when nothing newer than ``SUSPICION_STEPS`` has been heard. Suspicion
is local and diagnostic only; leases decide claims, so an unreachable agent and a failed one are
treated alike.

**Per step** (:meth:`LeaseAuction.decide`): renew the held sector while it is undone, feasible and
still won according to the claim table (own last published claim included, so all agents with the
same table agree); otherwise release it. With no sector held, claim the sector with the lowest own
bid among those this agent would win: no valid claim, or a worse one. An *alive* holder (heard
within ``SUSPICION_STEPS``) keeps an incumbency allowance of ``now - stamp``, so a claimant must
beat the holder's published bid by more than the steps since it was stamped; otherwise a
claimant's fresh bid would always look better than the holder's stale one and the two would
flip-flop. A suspected holder has no allowance. ``honor_failures = False``
(``auction_no_reassign``) makes leases infinite and ignores disabled holders, so a failed agent's
sector is only taken by a live agent that bids lower.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Final

import numpy as np
import numpy.typing as npt

from swarm.core.messages import Record
from swarm.core.types import AgentId, Position

SECTOR_SIZE: Final[int] = 4
LEASE_STEPS: Final[int] = 15
SUSPICION_STEPS: Final[int] = 10
SWEEP_CLAIMS: Final[int] = 8
CLAIM_TAG: Final[str] = "claim"
BEAT_TAG: Final[str] = "beat"

Bid = tuple[int, AgentId]
_ClaimValue = tuple[int, bool, int]  # (stamp, active, bid)


def sector_masks(shape: tuple[int, int], size: int = SECTOR_SIZE) -> list[npt.NDArray[np.bool_]]:
    """Boolean mask per sector, row-major; edge sectors cover what remains."""
    h, w = shape
    masks: list[npt.NDArray[np.bool_]] = []
    for r0 in range(0, h, size):
        for c0 in range(0, w, size):
            mask = np.zeros(shape, dtype=np.bool_)
            mask[r0 : r0 + size, c0 : c0 + size] = True
            masks.append(mask)
    return masks


def sector_of(cell: Position, shape: tuple[int, int], size: int = SECTOR_SIZE) -> int:
    """Index of the sector containing ``cell``."""
    per_row = math.ceil(shape[1] / size)
    return (cell[0] // size) * per_row + cell[1] // size


class ClaimTable:
    """Claims and heartbeats known to one agent; merged order-independently."""

    def __init__(self) -> None:
        self._claims: dict[tuple[int, AgentId], _ClaimValue] = {}
        self._beats: dict[AgentId, int] = {}
        self._fresh: set[tuple[int, AgentId]] = set()
        self._sweep = 0

    # ----------------------------------------------------------------- merging

    def merge(self, records: Iterable[Record]) -> None:
        """Merge claim and beat records; other tags are ignored, malformed ones raise."""
        for record in records:
            if not record:
                continue
            if record[0] == CLAIM_TAG:
                if len(record) != 6:
                    raise ValueError(f"malformed claim record: {record!r}")
                _, sector, holder, bid, stamp, active = record
                if not (
                    type(sector) is int
                    and type(holder) is str
                    and type(bid) is int
                    and type(stamp) is int
                    and type(active) is bool
                ):
                    raise ValueError(f"malformed claim record: {record!r}")
                self._put(sector, holder, (stamp, active, bid))
            elif record[0] == BEAT_TAG:
                if len(record) != 3 or type(record[1]) is not str or type(record[2]) is not int:
                    raise ValueError(f"malformed beat record: {record!r}")
                self.beat(record[1], record[2])

    def publish(self, sector: int, holder: AgentId, bid: int, stamp: int, active: bool) -> None:
        """Record this agent's own claim, renewal or release."""
        self._put(sector, holder, (stamp, active, bid))

    def beat(self, agent: AgentId, step: int) -> None:
        if step > self._beats.get(agent, -1):
            self._beats[agent] = step

    def _put(self, sector: int, holder: AgentId, value: _ClaimValue) -> None:
        stored = self._claims.get((sector, holder))
        if stored is None or value > stored:
            self._claims[(sector, holder)] = value
            self._fresh.add((sector, holder))

    # ----------------------------------------------------------------- queries

    def stamp(self, sector: int, holder: AgentId) -> int | None:
        value = self._claims.get((sector, holder))
        return None if value is None else value[0]

    def last_heard(self, agent: AgentId) -> int | None:
        return self._beats.get(agent)

    def heard(self) -> Mapping[AgentId, int]:
        return dict(self._beats)

    def valid(
        self, now: int, lease: int | None, excluded: frozenset[AgentId]
    ) -> dict[int, list[Bid]]:
        """Valid claims per sector, sorted best first."""
        out: dict[int, list[Bid]] = {}
        for (sector, holder), (stamp, active, bid) in self._claims.items():
            if not active or holder in excluded:
                continue
            if lease is not None and stamp + lease < now:
                continue
            out.setdefault(sector, []).append((bid, holder))
        for bids in out.values():
            bids.sort()
        return out

    def lapsed_holders(
        self, sector: int, now: int, lease: int | None, excluded: frozenset[AgentId]
    ) -> list[AgentId]:
        """Holders whose active claim on ``sector`` is no longer valid (expired or disabled)."""
        lapsed = []
        for (s, holder), (stamp, active, _) in sorted(self._claims.items()):
            if s != sector or not active:
                continue
            if holder in excluded or (lease is not None and stamp + lease < now):
                lapsed.append(holder)
        return lapsed

    def records(self) -> tuple[Record, ...]:
        """Every claim and beat held, sorted (canonical snapshot)."""
        claims = tuple(self._encode(k) for k in sorted(self._claims))
        beats = tuple((BEAT_TAG, a, s) for a, s in sorted(self._beats.items()))
        return claims + beats

    # ----------------------------------------------------------------- sharing

    def outbox_records(self, max_records: int | None = None) -> tuple[Record, ...]:
        """Heartbeats, then changed claims, then a rotating slice of all claims."""
        beats: list[Record] = [(BEAT_TAG, a, s) for a, s in sorted(self._beats.items())]
        if max_records is not None:
            beats = beats[:max_records]
        room = None if max_records is None else max_records - len(beats)
        fresh = sorted(self._fresh)
        chosen = fresh if room is None else fresh[:room]
        self._fresh.difference_update(chosen)
        left = SWEEP_CLAIMS if room is None else min(SWEEP_CLAIMS, room - len(chosen))
        taken = set(chosen)
        rest = [k for k in sorted(self._claims) if k not in taken]
        sweep: list[tuple[int, AgentId]] = []
        if rest and left > 0:
            start = self._sweep % len(rest)
            sweep = (rest[start:] + rest[:start])[:left]
            self._sweep = start + len(sweep)
        return tuple(beats + [self._encode(k) for k in [*chosen, *sweep]])

    def _encode(self, key: tuple[int, AgentId]) -> Record:
        stamp, active, bid = self._claims[key]
        return (CLAIM_TAG, key[0], key[1], bid, stamp, active)


class LeaseAuction:
    """One agent's side of the auction: which sector it holds, and its claims."""

    def __init__(
        self,
        owner: AgentId,
        *,
        honor_failures: bool = True,
        suspicion_steps: int = SUSPICION_STEPS,
    ) -> None:
        self.owner = owner
        self.honor_failures = honor_failures
        self.suspicion_steps = suspicion_steps
        self.lease: int | None = LEASE_STEPS if honor_failures else None
        self.table = ClaimTable()
        self.holding: int | None = None
        self.stats = {
            "claims": 0,
            "renewals": 0,
            "released_done": 0,
            "released_outbid": 0,
            "released_infeasible": 0,
            "takeovers_after_failure": 0,
        }

    def winners(self, now: int, disabled: frozenset[AgentId]) -> dict[int, Bid]:
        """Current winner of every sector with a valid claim, in this agent's view."""
        excluded = disabled if self.honor_failures else frozenset()
        return {s: bids[0] for s, bids in self.table.valid(now, self.lease, excluded).items()}

    def decide(
        self,
        now: int,
        bids: Mapping[int, int],
        done: frozenset[int],
        disabled: frozenset[AgentId],
    ) -> int | None:
        """Renew, release or claim; return the sector held after this step (or ``None``).

        ``bids`` maps each sector this agent can feasibly search to its integer bid.
        """
        self.table.beat(self.owner, now)
        excluded = disabled if self.honor_failures else frozenset()
        valid = self.table.valid(now, self.lease, excluded)
        if self.holding is not None:
            k = self.holding
            # Judge the conflict from the claim table only (own last published claim included),
            # never own fresh bid against others' stale ones: agents with the same table then
            # agree on the winner, so duplicate holders converge.
            ranked = valid.get(k, [])
            mine = bids.get(k)
            reason = None
            if k in done:
                reason = "released_done"
            elif mine is None:
                reason = "released_infeasible"
            elif ranked and ranked[0][1] != self.owner:
                reason = "released_outbid"
            if reason is None:
                assert mine is not None
                self.table.publish(k, self.owner, mine, now, True)
                self.stats["renewals"] += 1
                return k
            self.table.publish(k, self.owner, mine if mine is not None else 0, now, False)
            self.stats[reason] += 1
            self.holding = None
        best: tuple[int, int] | None = None
        for k in sorted(bids):
            if k in done:
                continue
            others = [b for b in valid.get(k, []) if b[1] != self.owner]
            if others and others[0] < (bids[k] + self._allowance(k, others[0][1], now), self.owner):
                continue  # someone else validly wins it
            if best is None or (bids[k], k) < best:
                best = (bids[k], k)
        if best is None:
            return None
        bid, k = best
        lapsed = self.table.lapsed_holders(k, now, self.lease, excluded)
        if any(holder != self.owner for holder in lapsed):
            self.stats["takeovers_after_failure"] += 1
        self.table.publish(k, self.owner, bid, now, True)
        self.stats["claims"] += 1
        self.holding = k
        return k

    def _allowance(self, sector: int, holder: AgentId, now: int) -> int:
        """Incumbency allowance of a valid holder: steps since its claim was stamped, or 0 if the
        holder is suspected (unheard for more than ``suspicion_steps``)."""
        stamp = self.table.stamp(sector, holder)
        assert stamp is not None
        heard = max(stamp, self.table.last_heard(holder) or stamp)
        if now - heard > self.suspicion_steps:
            return 0
        return max(0, now - stamp)

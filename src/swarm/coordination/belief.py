"""Per-agent belief maps and order-independent merging (ADR-0018).

A :class:`BeliefMap` is a :class:`~swarm.navigation.known_map.KnownMap` (so the Phase 2 planners
use it unchanged) whose contents are kept as provenance-stamped records:

* **cell** ``("cell", row, col, terrain, hazard, step, observer)``: what ``observer`` saw in a
  cell at ``step``. Per cell the record with the greatest ``(step, observer, terrain, hazard)``
  wins, so newer information is never overwritten by older and ties resolve the same way on
  every agent.
* **det** ``("det", row, col, observer, first_step, last_step, count)``: survivor evidence, one
  record per (cell, observer). Merged field-wise with ``min``/``max``/``max``; only the observer
  ever increments its own count, so relayed copies are never double-counted.
* **dis** ``("dis", agent, layer, row, col, step, observer)``: an agent seen disabled. The
  earliest sighting (smallest record) is kept, and blocked cells are derived from the kept
  sightings only, so they too depend only on which records arrived.

Every merge is commutative, associative and idempotent, so the final belief depends only on
which records arrived, never on their order, and duplicates or delayed copies are harmless.
Cells become known only through a cell record, which only an observation creates; survivor
evidence never makes terrain known.

Sharing: :meth:`BeliefMap.outbox_records` returns records changed since the last call (sorted
by key), then a rotating anti-entropy slice of :data:`SWEEP_RECORDS` records over the whole
store, so information lost to dropped messages is eventually resent.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from swarm.core.messages import SURVIVOR_REPORT_TAG, Delivery, Record
from swarm.core.observation import Observation
from swarm.core.types import AgentId, AgentStatus, Layer, Position
from swarm.navigation.known_map import KnownMap

CELL_TAG: Final[str] = "cell"
DETECTION_TAG: Final[str] = SURVIVOR_REPORT_TAG
DISABLED_TAG: Final[str] = "dis"
SWEEP_RECORDS: Final[int] = 16
"""Records of the whole store resent per call, in rotation (anti-entropy)."""

_CellValue = tuple[int, str, int, bool]  # (step, observer, terrain, hazard)
_DetValue = tuple[int, int, int]  # (first_step, last_step, count)
_DisValue = tuple[int, str, int, int, int]  # (step, observer, layer, row, col)
_Key = tuple[str | int, ...]


class BeliefMap(KnownMap):
    """One agent's knowledge, built from its own observations and received records."""

    def __init__(self, owner: AgentId, shape: tuple[int, int], base: Position) -> None:
        super().__init__(shape, base)
        self.owner = owner
        self.observed_step = np.full(shape, -1, dtype=np.int64)
        """Step of the winning cell record (``-1`` = never observed by anyone known to us)."""
        self._cells: dict[Position, _CellValue] = {}
        self._detections: dict[tuple[Position, str], _DetValue] = {}
        self._disabled: dict[AgentId, _DisValue] = {}
        self._fresh: set[_Key] = set()
        self._sweep = 0
        self._last_own_step = -1
        self.stats = {
            "records_received": 0,
            "records_changed": 0,
            "duplicates": 0,
            "stale_ignored": 0,
            "conflicts": 0,
            "records_sent": 0,
        }

    # ----------------------------------------------------------------- queries

    def cell_record(self, cell: Position) -> Record | None:
        """The winning record for ``cell``, or ``None`` if nobody's observation reached us."""
        value = self._cells.get(cell)
        return None if value is None else _cell_record(cell, value)

    def survivor_evidence(self) -> dict[Position, dict[str, _DetValue]]:
        """Detections known to this agent: cell -> observer -> (first, last, count)."""
        out: dict[Position, dict[str, _DetValue]] = {}
        for (cell, observer), value in sorted(self._detections.items()):
            out.setdefault(cell, {})[observer] = value
        return out

    def disabled_agents(self) -> frozenset[AgentId]:
        """Agents this agent knows to be disabled (own sightings or received records)."""
        return frozenset(self._disabled)

    def records(self) -> tuple[Record, ...]:
        """Every record held, sorted by key (a canonical snapshot of the belief)."""
        return tuple(self._encode(k) for k in sorted(self._all_keys()))

    # ----------------------------------------------------------------- own observations

    def update(self, observation: Observation) -> None:
        """Record what the owner observed (footprint cells, detections, disabled agents)."""
        if observation.agent_id != self.owner:
            raise ValueError(
                f"BeliefMap({self.owner!r}) cannot take {observation.agent_id!r}'s observation"
            )
        step = observation.step
        radius = observation.sensor_radius
        r0, c0 = observation.position
        h, w = self.shape
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = r0 + dr, c0 + dc
                code = int(observation.local_terrain[dr + radius, dc + radius])
                if code < 0 or not (0 <= r < h and 0 <= c < w):
                    continue  # outside the footprint or the grid: stays unknown
                hazard = bool(observation.local_hazard[dr + radius, dc + radius])
                self._merge_cell((r, c), (step, self.owner, code, hazard))
        if step > self._last_own_step:  # count each step's own detections once
            self._last_own_step = step
            for cell in observation.detections:
                first, last, count = self._detections.get((cell, self.owner), (step, step, 0))
                own = (min(first, step), max(last, step), count + 1)
                self._merge_detection(cell, self.owner, own)
        for v in observation.visible_agents:
            if v.status is AgentStatus.DISABLED:
                self._merge_disabled(
                    v.agent_id, (step, self.owner, int(v.layer), v.position[0], v.position[1])
                )

    # ----------------------------------------------------------------- received information

    def merge_inbox(self, inbox: tuple[Delivery, ...]) -> None:
        """Merge every record of every delivered message (any order gives the same result)."""
        for delivery in inbox:
            self.merge_records(delivery.message.records)

    def merge_records(self, records: tuple[Record, ...]) -> None:
        """Merge records; unknown tags are ignored, malformed known ones raise ``ValueError``."""
        for record in records:
            self.stats["records_received"] += 1
            tag = record[0] if record else None
            if tag == CELL_TAG:
                _, r, c, terrain, hazard, step, observer = _fields(record, 7)
                self._merge_cell(
                    (_int(r), _int(c)), (_int(step), _str(observer), _int(terrain), _bool(hazard))
                )
            elif tag == DETECTION_TAG:
                _, r, c, observer, first, last, count = _fields(record, 7)
                self._merge_detection(
                    (_int(r), _int(c)), _str(observer), (_int(first), _int(last), _int(count))
                )
            elif tag == DISABLED_TAG:
                _, agent, layer, r, c, step, observer = _fields(record, 7)
                self._merge_disabled(
                    _str(agent), (_int(step), _str(observer), _int(layer), _int(r), _int(c))
                )

    # ----------------------------------------------------------------- sharing

    def outbox_records(self, max_records: int | None = None) -> tuple[Record, ...]:
        """Records to send now: changed since the last call, then an anti-entropy slice.

        ``max_records`` caps the total; fresh records that do not fit stay fresh.
        """
        cap = max_records if max_records is not None else None
        fresh = sorted(self._fresh)
        chosen = fresh if cap is None else fresh[:cap]
        self._fresh.difference_update(chosen)
        room = SWEEP_RECORDS if cap is None else min(SWEEP_RECORDS, cap - len(chosen))
        taken = set(chosen)
        everything = [k for k in sorted(self._all_keys()) if k not in taken]
        sweep: list[_Key] = []
        if everything and room > 0:
            start = self._sweep % len(everything)
            sweep = (everything[start:] + everything[:start])[:room]
            self._sweep = start + len(sweep)
        out = tuple(self._encode(k) for k in [*chosen, *sweep])
        self.stats["records_sent"] += len(out)
        return out

    # ----------------------------------------------------------------- merge rules

    def _merge_cell(self, cell: Position, value: _CellValue) -> None:
        h, w = self.shape
        if not (0 <= cell[0] < h and 0 <= cell[1] < w):
            raise ValueError(f"cell record outside the grid: {cell}")
        stored = self._cells.get(cell)
        if stored is not None:
            if value == stored:
                self.stats["duplicates"] += 1
                return
            if (value[2], value[3]) != (stored[2], stored[3]):
                self.stats["conflicts"] += 1
            if value < stored:
                self.stats["stale_ignored"] += 1
                return
        self._cells[cell] = value
        self._changed((CELL_TAG, cell[0], cell[1]))
        self.observed_step[cell] = value[0]
        terrain, hazard = value[2], value[3]
        if int(self.terrain[cell]) != terrain or bool(self.hazards[cell]) != hazard:
            self.terrain[cell] = terrain
            self.hazards[cell] = hazard
            self.version += 1

    def _merge_detection(self, cell: Position, observer: str, value: _DetValue) -> None:
        stored = self._detections.get((cell, observer))
        merged = value if stored is None else _merge_det(stored, value)
        if merged == stored:
            self.stats["duplicates"] += 1
            return
        self._detections[(cell, observer)] = merged
        self._changed((DETECTION_TAG, cell[0], cell[1], observer))

    def _merge_disabled(self, agent: AgentId, value: _DisValue) -> None:
        stored = self._disabled.get(agent)
        if stored is not None and stored <= value:
            self.stats["duplicates" if stored == value else "stale_ignored"] += 1
            return
        self._disabled[agent] = value
        self._changed((DISABLED_TAG, agent))
        blocked = np.zeros_like(self.blocked)
        for _, _, layer, r, c in self._disabled.values():
            if (r, c) != self.base:
                blocked[Layer(layer)][r, c] = True
        if not np.array_equal(blocked, self.blocked):
            self.blocked = blocked
            self.version += 1

    def _changed(self, key: _Key) -> None:
        self.stats["records_changed"] += 1
        self._fresh.add(key)

    def _all_keys(self) -> list[_Key]:
        keys: list[_Key] = [(CELL_TAG, r, c) for (r, c) in self._cells]
        keys += [(DETECTION_TAG, cell[0], cell[1], obs) for (cell, obs) in self._detections]
        keys += [(DISABLED_TAG, agent) for agent in self._disabled]
        return keys

    def _encode(self, key: _Key) -> Record:
        tag = key[0]
        if tag == CELL_TAG:
            cell = (int(key[1]), int(key[2]))
            return _cell_record(cell, self._cells[cell])
        if tag == DETECTION_TAG:
            cell, observer = (int(key[1]), int(key[2])), str(key[3])
            first, last, count = self._detections[(cell, observer)]
            return (DETECTION_TAG, cell[0], cell[1], observer, first, last, count)
        agent = str(key[1])
        step, observer, layer, r, c = self._disabled[agent]
        return (DISABLED_TAG, agent, layer, r, c, step, observer)


def _merge_det(a: _DetValue, b: _DetValue) -> _DetValue:
    return (min(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2]))


def _cell_record(cell: Position, value: _CellValue) -> Record:
    step, observer, terrain, hazard = value
    return (CELL_TAG, cell[0], cell[1], terrain, hazard, step, observer)


def _fields(record: Record, n: int) -> Record:
    if len(record) != n:
        raise ValueError(f"malformed {record[0]!r} record (expected {n} fields): {record!r}")
    return record


def _int(value: object) -> int:
    if type(value) is not int:
        raise ValueError(f"expected int in record, got {value!r}")
    return value


def _str(value: object) -> str:
    if type(value) is not str:
        raise ValueError(f"expected str in record, got {value!r}")
    return value


def _bool(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(f"expected bool in record, got {value!r}")
    return value

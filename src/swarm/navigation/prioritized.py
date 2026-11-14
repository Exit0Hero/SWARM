"""Prioritized multi-agent planning with space-time reservation tables (ADR-0015 rules 7-13).

Agents are planned one at a time in a rotating priority order. Each accepted plan reserves its
cells per time step (vertex), its moves (edge, to forbid swaps) and its goal from arrival onwards
(hold). Lower-priority agents plan around those reservations with space-time A*. Layers are
planned in separate tables, and the base cell is exempt from every reservation, as in the
environment (ADR-0003).
"""

from __future__ import annotations

import heapq
import itertools
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from swarm.core.types import Action, AgentId, Layer, Position
from swarm.navigation.astar import UNREACHABLE, cost_field, manhattan

TimedPath = list[Position]
"""``path[t]`` is the planned cell at time ``t`` (``t = 0`` is now)."""

_MOVES: tuple[tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1), (0, 0))
_ACTION_FOR_DELTA: dict[tuple[int, int], Action] = {
    (0, 0): Action.STAY,
    (-1, 0): Action.NORTH,
    (0, 1): Action.EAST,
    (1, 0): Action.SOUTH,
    (0, -1): Action.WEST,
}


def default_horizon(shape: tuple[int, int]) -> int:
    """``H = 2 x (height + width)`` (ADR-0015 rule 7)."""
    return 2 * (shape[0] + shape[1])


def priority_order(agent_ids: Sequence[AgentId], round_index: int) -> tuple[AgentId, ...]:
    """Sorted IDs rotated left by ``round_index mod n`` (ADR-0015 rule 13)."""
    ids = sorted(agent_ids)
    if not ids:
        return ()
    k = round_index % len(ids)
    return tuple(ids[k:] + ids[:k])


def first_action(path: TimedPath) -> Action:
    """The action that moves from ``path[0]`` to ``path[1]`` (STAY for a one-cell path)."""
    if len(path) < 2:
        return Action.STAY
    delta = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    return _ACTION_FOR_DELTA[delta]


@dataclass
class ReservationTable:
    """Reservations for one layer. The base is never reserved or checked."""

    base: Position
    horizon: int
    vertex: dict[tuple[Position, int], AgentId] = field(default_factory=dict)
    edge: dict[tuple[Position, Position, int], AgentId] = field(default_factory=dict)
    hold: dict[Position, tuple[int, AgentId]] = field(default_factory=dict)
    provisional: dict[Position, AgentId] = field(default_factory=dict)
    uses: dict[Position, list[tuple[int, AgentId]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def reserve_provisional(self, owner: AgentId, cell: Position) -> None:
        """An unplanned agent may still be in ``cell`` at ``t = 1`` (ADR-0015 rule 11)."""
        if cell != self.base:
            self.provisional[cell] = owner

    def reserve_path(self, owner: AgentId, path: TimedPath) -> None:
        """Reserve every step of ``path`` and hold its last cell until the horizon."""
        self.provisional = {c: o for c, o in self.provisional.items() if o != owner}
        for t, cell in enumerate(path):
            if cell == self.base:
                continue
            self.vertex[(cell, t)] = owner
            self.uses[cell].append((t, owner))
        for t, (a, b) in enumerate(itertools.pairwise(path)):
            if a != b and a != self.base and b != self.base:
                self.edge[(a, b, t)] = owner
        goal = path[-1]
        if goal != self.base:
            self.hold[goal] = (len(path) - 1, owner)
            self.uses[goal].append((self.horizon + 1, owner))

    def reserve_stationary(self, owner: AgentId, cell: Position) -> None:
        """An agent without a plan stays put for the whole horizon."""
        self.reserve_path(owner, [cell])
        if cell != self.base:
            self.hold[cell] = (0, owner)

    def vertex_blocked(self, cell: Position, t: int, owner: AgentId) -> bool:
        if cell == self.base:
            return False
        other = self.vertex.get((cell, t))
        if other is not None and other != owner:
            return True
        held = self.hold.get(cell)
        if held is not None and held[1] != owner and held[0] <= t:
            return True
        prov = self.provisional.get(cell)
        return t == 1 and prov is not None and prov != owner

    def edge_blocked(self, a: Position, b: Position, t: int, owner: AgentId) -> bool:
        """Moving ``a -> b`` between ``t`` and ``t + 1`` swaps with a reserved ``b -> a``."""
        if a == self.base or b == self.base or a == b:
            return False
        other = self.edge.get((b, a, t))
        return other is not None and other != owner

    def free_after(self, cell: Position, t: int, owner: AgentId) -> bool:
        """No other agent uses ``cell`` at any time after ``t`` (goal acceptance)."""
        if cell == self.base:
            return True
        return all(o == owner or ut <= t for ut, o in self.uses.get(cell, ()))


def plan_space_time(
    passable: npt.NDArray[np.bool_],
    start: Position,
    goal: Position,
    table: ReservationTable,
    owner: AgentId,
    enter_cost: npt.NDArray[np.int64] | None = None,
    wait_cost: npt.NDArray[np.int64] | None = None,
) -> TimedPath | None:
    """Space-time A* from ``start`` to ``goal`` around ``table``'s reservations, or ``None``.

    A move costs ``enter_cost`` of the entered cell (1 if omitted); waiting in a cell costs
    ``wait_cost`` of that cell (1 if omitted). The goal is accepted at time ``T`` only when no
    other agent uses it after ``T``. The search is bounded by ``table.horizon`` and preceded by a
    static reachability check (:func:`statically_reachable`).
    """
    shape = passable.shape
    if not statically_reachable(passable, start, goal):
        return None
    cost = np.ones(shape, dtype=np.int64) if enter_cost is None else enter_cost
    waiting = np.ones(shape, dtype=np.int64) if wait_cost is None else wait_cost
    if bool((waiting < 0).any()):
        raise ValueError("wait_cost must be non-negative")
    h0 = manhattan(start, goal)
    heap: list[tuple[int, int, int, int, int]] = [(h0, h0, 0, start[0], start[1])]
    g: dict[tuple[Position, int], int] = {(start, 0): 0}
    parent: dict[tuple[Position, int], tuple[Position, int]] = {}
    closed: set[tuple[Position, int]] = set()
    while heap:
        _, _, t, r, c = heapq.heappop(heap)
        state = ((r, c), t)
        if state in closed:
            continue
        closed.add(state)
        if (r, c) == goal and table.free_after(goal, t, owner):
            return _timed_path(parent, state)
        if t >= table.horizon:
            continue
        for dr, dc in _MOVES:
            nxt = (r + dr, c + dc)
            if not (0 <= nxt[0] < shape[0] and 0 <= nxt[1] < shape[1]):
                continue
            if nxt != (r, c) and not passable[nxt]:
                continue
            nt = t + 1
            if table.vertex_blocked(nxt, nt, owner) or table.edge_blocked((r, c), nxt, t, owner):
                continue
            nstate = (nxt, nt)
            if nstate in closed:
                continue
            step = int(waiting[nxt]) if nxt == (r, c) else int(cost[nxt])
            tentative = g[state] + step
            if tentative < g.get(nstate, tentative + 1):
                g[nstate] = tentative
                parent[nstate] = state
                h = manhattan(nxt, goal)
                heapq.heappush(heap, (tentative + h, h, nt, nxt[0], nxt[1]))
    return None


def statically_reachable(passable: npt.NDArray[np.bool_], start: Position, goal: Position) -> bool:
    """Whether ``goal`` is reachable from ``start`` ignoring reservations (no time bound)."""
    if start == goal:
        return True
    return bool(passable[goal]) and cost_field(passable, start)[goal] != UNREACHABLE


def _timed_path(
    parent: dict[tuple[Position, int], tuple[Position, int]], end: tuple[Position, int]
) -> TimedPath:
    states = [end]
    while states[-1] in parent:
        states.append(parent[states[-1]])
    states.reverse()
    return [cell for cell, _ in states]


@dataclass(frozen=True, slots=True)
class PlanRequest:
    """One agent's planning problem for this round."""

    agent_id: AgentId
    layer: Layer
    start: Position
    goal: Position
    passable: npt.NDArray[np.bool_]
    enter_cost: npt.NDArray[np.int64] | None = None
    wait_cost: npt.NDArray[np.int64] | None = None


@dataclass(frozen=True, slots=True)
class PlanRound:
    """Plans for one round. ``paths[a]`` is ``None`` when agent ``a``'s plan failed.

    ``unreachable`` lists failed agents whose goal is statically unreachable on their map; every
    other failure is temporary (reservations or the planning horizon).
    """

    order: tuple[AgentId, ...]
    paths: dict[AgentId, TimedPath | None]
    unreachable: frozenset[AgentId] = frozenset()

    @property
    def failed(self) -> tuple[AgentId, ...]:
        return tuple(a for a in self.order if self.paths[a] is None)


def plan_prioritized(
    requests: Sequence[PlanRequest], base: Position, horizon: int, round_index: int
) -> PlanRound:
    """Plan all ``requests`` in rotating priority order (ADR-0015 rules 7-13)."""
    by_id = {r.agent_id: r for r in requests}
    if len(by_id) != len(requests):
        raise ValueError("plan_prioritized: duplicate agent IDs")
    tables = {layer: ReservationTable(base=base, horizon=horizon) for layer in Layer}
    for r in requests:
        tables[r.layer].reserve_provisional(r.agent_id, r.start)
    order = priority_order(list(by_id), round_index)
    paths: dict[AgentId, TimedPath | None] = {}
    unreachable: set[AgentId] = set()
    for aid in order:
        r = by_id[aid]
        table = tables[r.layer]
        path = plan_space_time(r.passable, r.start, r.goal, table, aid, r.enter_cost, r.wait_cost)
        paths[aid] = path
        if path is None:
            if not statically_reachable(r.passable, r.start, r.goal):
                unreachable.add(aid)
            table.reserve_stationary(aid, r.start)
        else:
            table.reserve_path(aid, path)
    return PlanRound(order=order, paths=paths, unreachable=frozenset(unreachable))


def find_conflicts(
    paths: dict[AgentId, TimedPath], layers: dict[AgentId, Layer], base: Position
) -> list[str]:
    """Vertex and swap conflicts between timed paths (agents park at their last cell)."""
    problems: list[str] = []
    ids = sorted(paths)
    length = max((len(p) for p in paths.values()), default=0)

    def at(aid: AgentId, t: int) -> Position:
        p = paths[aid]
        return p[min(t, len(p) - 1)]

    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if layers[a] != layers[b]:
                continue
            for t in range(length):
                if at(a, t) == at(b, t) and at(a, t) != base:
                    problems.append(f"vertex {a}/{b} at {at(a, t)} t={t}")
                if (
                    t + 1 < length
                    and at(a, t) == at(b, t + 1)
                    and at(b, t) == at(a, t + 1)
                    and at(a, t) != at(a, t + 1)
                    and base not in (at(a, t), at(b, t))
                ):
                    problems.append(f"swap {a}/{b} t={t}")
    return problems

"""Single-agent grid search: A*, Dijkstra fields and route costs (ADR-0015).

All functions take a boolean ``passable`` grid (the planner's view, never ground truth unless the
caller is the oracle) and an optional integer ``enter_cost`` grid (cost of *entering* a cell,
``>= 1``; all ones when omitted). Moves are 4-connected; neighbours are expanded N, E, S, W.
"""

from __future__ import annotations

import heapq
import itertools
from typing import Final

import numpy as np
import numpy.typing as npt

from swarm.core.types import Position

UNREACHABLE: Final[int] = -1
_NEIGHBOURS: Final[tuple[tuple[int, int], ...]] = ((-1, 0), (0, 1), (1, 0), (0, -1))

Path = list[Position]


def manhattan(a: Position, b: Position) -> int:
    """Admissible heuristic for 4-connected moves with every step costing at least 1."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _costs(
    passable: npt.NDArray[np.bool_], enter_cost: npt.NDArray[np.int64] | None
) -> npt.NDArray[np.int64]:
    if enter_cost is None:
        return np.ones(passable.shape, dtype=np.int64)
    if enter_cost.shape != passable.shape:
        raise ValueError(f"enter_cost shape {enter_cost.shape} != passable {passable.shape}")
    if bool((enter_cost[passable] < 1).any()):
        raise ValueError("enter_cost must be >= 1 on passable cells (heuristic admissibility)")
    return np.asarray(enter_cost, dtype=np.int64)


def _in_grid(cell: Position, shape: tuple[int, ...]) -> bool:
    return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]


def astar(
    passable: npt.NDArray[np.bool_],
    start: Position,
    goal: Position,
    enter_cost: npt.NDArray[np.int64] | None = None,
) -> Path | None:
    """Minimum-cost path from ``start`` to ``goal`` (both included), or ``None``.

    The start is always allowed even if not passable. ``start == goal`` returns ``[start]``.
    Ties are broken deterministically by ``(f, h, row, col)``.
    """
    if not (_in_grid(start, passable.shape) and _in_grid(goal, passable.shape)):
        raise ValueError(f"start {start} or goal {goal} outside grid {passable.shape}")
    if start == goal:
        return [start]
    if not passable[goal]:
        return None
    cost = _costs(passable, enter_cost)
    g: dict[Position, int] = {start: 0}
    parent: dict[Position, Position] = {}
    h0 = manhattan(start, goal)
    open_heap: list[tuple[int, int, int, int]] = [(h0, h0, start[0], start[1])]
    closed: set[Position] = set()
    while open_heap:
        _, _, r, c = heapq.heappop(open_heap)
        cell = (r, c)
        if cell in closed:
            continue
        if cell == goal:
            return _reconstruct(parent, start, goal)
        closed.add(cell)
        for dr, dc in _NEIGHBOURS:
            nxt = (r + dr, c + dc)
            if not _in_grid(nxt, passable.shape) or not passable[nxt] or nxt in closed:
                continue
            tentative = g[cell] + int(cost[nxt])
            if tentative < g.get(nxt, tentative + 1):
                g[nxt] = tentative
                parent[nxt] = cell
                h = manhattan(nxt, goal)
                heapq.heappush(open_heap, (tentative + h, h, nxt[0], nxt[1]))
    return None


def _reconstruct(parent: dict[Position, Position], start: Position, goal: Position) -> Path:
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def path_cost(path: Path, enter_cost: npt.NDArray[np.int64] | None = None) -> int:
    """Sum of entry costs of every cell after the first (moves only)."""
    if enter_cost is None:
        return len(path) - 1
    return int(sum(int(enter_cost[cell]) for cell in path[1:]))


def cost_field(
    passable: npt.NDArray[np.bool_],
    source: Position,
    enter_cost: npt.NDArray[np.int64] | None = None,
) -> npt.NDArray[np.int64]:
    """Minimum cost from ``source`` to every cell (Dijkstra). ``UNREACHABLE`` where no path."""
    cost = _costs(passable, enter_cost)
    dist = np.full(passable.shape, UNREACHABLE, dtype=np.int64)
    dist[source] = 0
    heap: list[tuple[int, int, int]] = [(0, source[0], source[1])]
    done = np.zeros(passable.shape, dtype=np.bool_)
    while heap:
        d, r, c = heapq.heappop(heap)
        if done[r, c]:
            continue
        done[r, c] = True
        for dr, dc in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not _in_grid((nr, nc), passable.shape) or not passable[nr, nc] or done[nr, nc]:
                continue
            nd = d + int(cost[nr, nc])
            if dist[nr, nc] == UNREACHABLE or nd < dist[nr, nc]:
                dist[nr, nc] = nd
                heapq.heappush(heap, (nd, nr, nc))
    return dist


def cost_to_target_field(
    passable: npt.NDArray[np.bool_],
    target: Position,
    enter_cost: npt.NDArray[np.int64] | None = None,
) -> npt.NDArray[np.int64]:
    """Minimum cost from every cell *to* ``target`` (reverse Dijkstra; ADR-0016 rule 2).

    Moving from ``v`` into ``u`` costs ``enter_cost[u]``, so the field is computed backwards from
    the target, charging the cost of the cell being *left towards the target*.
    """
    cost = _costs(passable, enter_cost)
    dist = np.full(passable.shape, UNREACHABLE, dtype=np.int64)
    dist[target] = 0
    heap: list[tuple[int, int, int]] = [(0, target[0], target[1])]
    done = np.zeros(passable.shape, dtype=np.bool_)
    while heap:
        d, r, c = heapq.heappop(heap)
        if done[r, c]:
            continue
        done[r, c] = True
        for dr, dc in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not _in_grid((nr, nc), passable.shape) or not passable[nr, nc] or done[nr, nc]:
                continue
            nd = d + int(cost[r, c])  # forward move (nr, nc) -> (r, c) enters (r, c)
            if dist[nr, nc] == UNREACHABLE or nd < dist[nr, nc]:
                dist[nr, nc] = nd
                heapq.heappush(heap, (nd, nr, nc))
    return dist


def is_valid_path(passable: npt.NDArray[np.bool_], path: Path) -> bool:
    """Adjacent consecutive cells, all after the first passable."""
    if not path:
        return False
    return all(not (manhattan(a, b) != 1 or not passable[b]) for a, b in itertools.pairwise(path))

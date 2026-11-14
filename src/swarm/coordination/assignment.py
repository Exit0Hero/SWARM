"""Minimum-cost assignment with forbidden pairs: the Hungarian method (ADR-0016 rule 9).

Kuhn-Munkres with row/column potentials and shortest augmenting paths, ``O(n^2 m)`` for an
``n x m`` matrix with ``n <= m`` (the matrix is transposed otherwise). Pure Python: the problems
are small (agents x candidate tasks), so no SciPy dependency is needed.

Semantics: among assignments that use only feasible pairs, the result has the **maximum number
of assigned pairs** and, among those, the **minimum total cost**. This is achieved by giving
infeasible pairs a cost larger than any feasible total and dropping them from the result.
Ties are resolved deterministically by row then column order of the input matrix, so callers
order rows by agent ID and columns by task ID.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def min_cost_assignment(
    cost: npt.NDArray[np.int64], feasible: npt.NDArray[np.bool_] | None = None
) -> list[tuple[int, int]]:
    """Return ``(row, col)`` pairs of an optimal assignment, sorted by row.

    Args:
        cost: non-negative integer costs, shape ``(n, m)``.
        feasible: ``False`` marks pairs that must not be assigned (all feasible if omitted).
    """
    cost = np.asarray(cost, dtype=np.int64)
    if cost.ndim != 2:
        raise ValueError(f"cost: must be a 2D matrix, got shape {cost.shape}")
    if bool((cost < 0).any()):
        raise ValueError("cost: must be non-negative")
    allowed = np.ones(cost.shape, dtype=np.bool_) if feasible is None else np.asarray(feasible)
    if allowed.shape != cost.shape:
        raise ValueError(f"feasible shape {allowed.shape} != cost shape {cost.shape}")
    n, m = cost.shape
    if n == 0 or m == 0 or not bool(allowed.any()):
        return []
    if n > m:
        pairs = min_cost_assignment(cost.T.copy(), allowed.T.copy())
        return sorted((r, c) for c, r in pairs)

    big = int(cost[allowed].sum()) + 1  # larger than any feasible total
    a = np.where(allowed, cost, big).astype(np.int64)
    inf = int(a.sum()) * 2 + 1

    # e-maxx formulation, 1-indexed: u (rows), v (cols), p[j] = row matched to column j.
    u = [0] * (n + 1)
    v = [0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = int(a[i0 - 1, j - 1]) - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]
    return sorted((r, c) for r, c in pairs if allowed[r, c])

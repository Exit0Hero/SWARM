"""Hungarian assignment against brute force (ADR-0016 rule 9)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from swarm.coordination.assignment import min_cost_assignment


def brute_force(cost: npt.NDArray[np.int64], feasible: npt.NDArray[np.bool_]) -> tuple[int, int]:
    """(max number of feasible pairs, min total cost among those) by exhaustive search."""
    n, m = cost.shape
    best = (0, 0)

    def rec(row: int, used: frozenset[int], count: int, total: int) -> None:
        nonlocal best
        if row == n:
            if count > best[0] or (count == best[0] and total < best[1]):
                best = (count, total)
            return
        rec(row + 1, used, count, total)  # leave this row unassigned
        for col in range(m):
            if col not in used and feasible[row, col]:
                rec(row + 1, used | {col}, count + 1, total + int(cost[row, col]))

    rec(0, frozenset(), 0, 0)
    return best


@st.composite
def problems(draw: st.DrawFn) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    n, m = draw(st.integers(1, 5)), draw(st.integers(1, 5))
    cost = np.array(draw(st.lists(st.integers(0, 20), min_size=n * m, max_size=n * m)))
    feas = np.array(draw(st.lists(st.booleans(), min_size=n * m, max_size=n * m)))
    return cost.reshape(n, m).astype(np.int64), feas.reshape(n, m)


@settings(max_examples=300)
@given(problems())
def test_matches_brute_force(problem: tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]) -> None:
    cost, feasible = problem
    pairs = min_cost_assignment(cost, feasible)
    rows = [r for r, _ in pairs]
    cols = [c for _, c in pairs]
    assert len(set(rows)) == len(rows)
    assert len(set(cols)) == len(cols)
    assert all(feasible[r, c] for r, c in pairs)
    count, total = brute_force(cost, feasible)
    assert len(pairs) == count
    assert sum(int(cost[r, c]) for r, c in pairs) == total


@given(problems())
def test_deterministic(problem: tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]) -> None:
    cost, feasible = problem
    assert min_cost_assignment(cost, feasible) == min_cost_assignment(cost, feasible)


def test_known_optimum() -> None:
    cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]], dtype=np.int64)
    assert min_cost_assignment(cost) == [(0, 1), (1, 0), (2, 2)]  # 1 + 2 + 2 = 5


def test_more_agents_than_tasks() -> None:
    cost = np.array([[5], [1], [3]], dtype=np.int64)
    assert min_cost_assignment(cost) == [(1, 0)]


def test_infeasible_pairs_are_never_assigned() -> None:
    cost = np.array([[1, 100], [1, 100]], dtype=np.int64)
    feasible = np.array([[True, False], [True, True]])
    # Max cardinality first: both agents assigned even though (1, 1) is expensive.
    assert min_cost_assignment(cost, feasible) == [(0, 0), (1, 1)]
    assert min_cost_assignment(cost, np.zeros((2, 2), dtype=np.bool_)) == []


def test_ties_resolved_by_input_order() -> None:
    cost = np.zeros((2, 2), dtype=np.int64)
    assert min_cost_assignment(cost) == min_cost_assignment(cost.copy())


def test_empty_and_invalid_inputs() -> None:
    assert min_cost_assignment(np.zeros((0, 3), dtype=np.int64)) == []
    with pytest.raises(ValueError, match="non-negative"):
        min_cost_assignment(np.array([[-1]], dtype=np.int64))
    with pytest.raises(ValueError, match="2D"):
        min_cost_assignment(np.zeros(3, dtype=np.int64))

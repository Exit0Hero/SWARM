"""Reservation tables and prioritized planning (ADR-0015)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import numpy.typing as npt
from hypothesis import assume, given
from hypothesis import strategies as st

from swarm.core.env import SwarmEnv
from swarm.core.events import MoveConflict
from swarm.core.types import Action, Layer, Position
from swarm.navigation.known_map import KnownMap
from swarm.navigation.prioritized import (
    PlanRequest,
    ReservationTable,
    default_horizon,
    find_conflicts,
    first_action,
    plan_prioritized,
    plan_space_time,
    priority_order,
)
from swarm.scenarios.config import AgentSpec
from tests.helpers import config_for, spec

BASE: Position = (9, 9)


def grid(rows: list[str]) -> npt.NDArray[np.bool_]:
    return np.array([[ch != "#" for ch in row] for row in rows], dtype=np.bool_)


def req(
    aid: str,
    start: Position,
    goal: Position,
    passable: npt.NDArray[np.bool_],
    layer: Layer = Layer.GROUND,
) -> PlanRequest:
    return PlanRequest(aid, layer, start, goal, passable)


# ----------------------------------------------------------------- reservation table


def test_vertex_edge_and_hold_reservations() -> None:
    table = ReservationTable(base=BASE, horizon=10)
    table.reserve_path("a", [(0, 0), (0, 1), (0, 2)])
    assert table.vertex_blocked((0, 1), 1, "b")
    assert not table.vertex_blocked((0, 1), 1, "a")  # own reservation
    assert not table.vertex_blocked((0, 1), 2, "b")
    assert table.edge_blocked((0, 1), (0, 0), 0, "b")  # swap with a's move (0,0)->(0,1) at t=0
    assert table.vertex_blocked((0, 2), 7, "b")  # goal held from t=2 on
    assert not table.vertex_blocked((0, 2), 1, "b")
    assert not table.free_after((0, 2), 5, "b")
    assert table.free_after((0, 0), 1, "b")


def test_base_is_never_reserved() -> None:
    table = ReservationTable(base=(0, 0), horizon=10)
    table.reserve_path("a", [(0, 1), (0, 0)])
    assert not table.vertex_blocked((0, 0), 1, "b")
    assert not table.edge_blocked((0, 0), (0, 1), 0, "b")  # swap through the base allowed
    assert table.free_after((0, 0), 0, "b")


def test_provisional_reservation_only_at_t1_and_cleared_when_planned() -> None:
    table = ReservationTable(base=BASE, horizon=10)
    table.reserve_provisional("b", (1, 1))
    assert table.vertex_blocked((1, 1), 1, "a")
    assert not table.vertex_blocked((1, 1), 2, "a")
    table.reserve_path("b", [(1, 1), (1, 2)])
    assert not table.vertex_blocked((1, 1), 1, "a")


# ----------------------------------------------------------------- space-time planning


def test_waits_for_a_crossing_agent() -> None:
    open_grid = grid(["...", "...", "..."])
    table = ReservationTable(base=BASE, horizon=12)
    table.reserve_path("a", [(1, 0), (1, 1), (1, 2)])
    path = plan_space_time(open_grid, (0, 1), (2, 1), table, "b")
    assert path is not None
    assert path[1] != (1, 1)  # cannot be in the centre at t=1
    assert (
        find_conflicts(
            {"a": [(1, 0), (1, 1), (1, 2)], "b": path}, {"a": Layer.GROUND, "b": Layer.GROUND}, BASE
        )
        == []
    )


def test_head_on_corridor_uses_a_side_pocket() -> None:
    # Corridor with a pocket at (1, 3), one step from b's start: b can yield in time.
    passable = grid([".....", "###.#"])
    result = plan_prioritized(
        [req("a", (0, 0), (0, 4), passable), req("b", (0, 4), (0, 0), passable)],
        base=BASE,
        horizon=20,
        round_index=0,
    )
    paths = result.paths
    assert all(p is not None for p in paths.values())
    clean = {k: v for k, v in paths.items() if v is not None}
    assert find_conflicts(clean, {"a": Layer.GROUND, "b": Layer.GROUND}, BASE) == []
    assert any((1, 3) in p for p in clean.values())  # someone stepped into the pocket


def test_prioritized_planning_is_incomplete_when_the_pocket_is_too_far() -> None:
    """Known limitation (ADR-0015): a higher-priority straight plan leaves no time to yield.

    The pocket at (1, 2) is two steps from b's start, but a reaches (0, 2) at t = 2.
    """
    passable = grid([".....", "##.##"])
    result = plan_prioritized(
        [req("a", (0, 0), (0, 4), passable), req("b", (0, 4), (0, 0), passable)],
        base=BASE,
        horizon=20,
        round_index=0,
    )
    assert result.failed == ("b",)


def test_corridor_without_pocket_fails_one_plan() -> None:
    passable = grid(["....."])
    result = plan_prioritized(
        [req("a", (0, 0), (0, 4), passable), req("b", (0, 4), (0, 0), passable)],
        base=BASE,
        horizon=20,
        round_index=0,
    )
    assert result.paths["a"] is not None
    assert result.failed == ("b",)


def test_goal_hold_blocks_passing_through_a_parked_agent() -> None:
    passable = grid(["....."])
    table = ReservationTable(base=BASE, horizon=20)
    table.reserve_path("a", [(0, 1), (0, 2)])  # a parks at (0, 2) from t=1
    assert plan_space_time(passable, (0, 0), (0, 4), table, "b") is None


def test_base_allows_simultaneous_arrival() -> None:
    passable = grid(["..."])
    base = (0, 1)
    result = plan_prioritized(
        [req("a", (0, 0), base, passable), req("b", (0, 2), base, passable)],
        base=base,
        horizon=10,
        round_index=0,
    )
    assert result.paths == {"a": [(0, 0), (0, 1)], "b": [(0, 2), (0, 1)]}


def test_layers_do_not_constrain_each_other() -> None:
    passable = grid(["..."])
    result = plan_prioritized(
        [
            req("air_0", (0, 0), (0, 2), passable, Layer.AIR),
            req("gnd_0", (0, 2), (0, 0), passable, Layer.GROUND),
        ],
        base=BASE,
        horizon=10,
        round_index=0,
    )
    assert result.paths["air_0"] == [(0, 0), (0, 1), (0, 2)]
    assert result.paths["gnd_0"] == [(0, 2), (0, 1), (0, 0)]


def test_unreachable_goal_fails_fast_and_holds_start() -> None:
    passable = grid([".#."])
    result = plan_prioritized(
        [req("a", (0, 0), (0, 2), passable), req("b", (0, 2), (0, 2), passable)],
        base=BASE,
        horizon=10,
        round_index=0,
    )
    assert result.paths["a"] is None
    assert result.paths["b"] == [(0, 2)]


def test_first_action() -> None:
    assert first_action([(1, 1), (0, 1)]) is Action.NORTH
    assert first_action([(1, 1), (1, 1)]) is Action.STAY
    assert first_action([(1, 1)]) is Action.STAY


def test_default_horizon() -> None:
    assert default_horizon((20, 20)) == 80


# ----------------------------------------------------------------- priority fairness


@given(n=st.integers(1, 9), start=st.integers(0, 1000))
def test_priority_rotation_gives_every_rank_equally(n: int, start: int) -> None:
    ids = [f"a{i}" for i in range(n)]
    ranks: Counter[tuple[str, int]] = Counter()
    for k in range(start, start + n):
        for rank, aid in enumerate(priority_order(ids, k)):
            ranks[(aid, rank)] += 1
    assert all(ranks[(aid, rank)] == 1 for aid in ids for rank in range(n))


def test_priority_order_is_deterministic_and_ignores_input_order() -> None:
    assert priority_order(["c", "a", "b"], 1) == ("b", "c", "a")
    assert priority_order(["b", "c", "a"], 1) == ("b", "c", "a")
    assert priority_order([], 3) == ()


# ----------------------------------------------------------------- execution in the environment


@st.composite
def team_problems(draw: st.DrawFn) -> tuple[list[str], list[Position], int]:
    h, w = draw(st.integers(3, 7)), draw(st.integers(3, 7))
    rows = [["." for _ in range(w)] for _ in range(h)]
    for _ in range(draw(st.integers(0, (h * w) // 5))):
        rows[draw(st.integers(0, h - 1))][draw(st.integers(0, w - 1))] = "#"
    base = (h // 2, w // 2)
    rows[base[0]][base[1]] = "B"
    n = draw(st.integers(2, 5))
    open_cells = [(r, c) for r in range(h) for c in range(w) if rows[r][c] == "."]
    assume(len(open_cells) >= n)
    goals = draw(st.lists(st.sampled_from(open_cells), min_size=n, max_size=n, unique=True))
    return ["".join(r) for r in rows], goals, draw(st.integers(0, 50))


@given(team_problems())
def test_executing_prioritized_plans_never_causes_conflicts(
    problem: tuple[list[str], list[Position], int],
) -> None:
    rows, goals, round_offset = problem
    gnd = spec("gnd", Layer.GROUND, capacity=500)
    config, scenario = config_for(
        rows, (gnd,), tuple(AgentSpec("gnd") for _ in goals), time_limit=60
    )
    env = SwarmEnv(config)
    env.reset(0, scenario=scenario)
    known = KnownMap.from_ground_truth(scenario.terrain, scenario.hazard_grid, scenario.base)
    passable = known.passable(Layer.GROUND)
    goal_of = dict(zip(env.possible_agents, goals, strict=True))
    horizon = default_horizon(scenario.shape)
    for step in range(30):
        snap = env.snapshot()
        requests = [
            PlanRequest(a.agent_id, a.layer, a.position, goal_of[a.agent_id], passable)
            for a in snap.agents
        ]
        plans = plan_prioritized(requests, scenario.base, horizon, round_offset + step)
        clean = {k: v for k, v in plans.paths.items() if v is not None}
        assert find_conflicts(clean, {k: Layer.GROUND for k in clean}, scenario.base) == []
        actions = {
            aid: int(first_action(p)) if (p := plans.paths[aid]) else 0 for aid in env.agents
        }
        result = env.step(actions)
        assert not [e for e in result.events if isinstance(e, MoveConflict)]
        if env.done:
            break

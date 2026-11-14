"""Single-agent navigation: A*, cost fields, known map, frontiers, energy (ADR-0015, ADR-0016)."""

from __future__ import annotations

from collections import deque

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st

from swarm.core.env import SwarmEnv
from swarm.core.types import Action, Layer, Terrain
from swarm.navigation.astar import (
    UNREACHABLE,
    astar,
    cost_field,
    cost_to_target_field,
    is_valid_path,
    manhattan,
    path_cost,
)
from swarm.navigation.energy import EnergyModel
from swarm.navigation.known_map import UNKNOWN, KnownMap
from swarm.scenarios.config import AgentSpec
from tests.helpers import config_for, spec


def grid(rows: list[str]) -> npt.NDArray[np.bool_]:
    """'.' passable, '#' blocked."""
    return np.array([[ch == "." for ch in row] for row in rows], dtype=np.bool_)


def bfs_distance(
    passable: npt.NDArray[np.bool_], start: tuple[int, int], goal: tuple[int, int]
) -> int:
    if start == goal:
        return 0
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        (r, c), d = queue.popleft()
        for nr, nc in ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)):
            if (
                0 <= nr < passable.shape[0]
                and 0 <= nc < passable.shape[1]
                and passable[nr, nc]
                and (nr, nc) not in seen
            ):
                if (nr, nc) == goal:
                    return d + 1
                seen.add((nr, nc))
                queue.append(((nr, nc), d + 1))
    return -1


# ----------------------------------------------------------------- A* edge cases


def test_straight_line_and_detour() -> None:
    assert astar(grid(["...."]), (0, 0), (0, 3)) == [(0, 0), (0, 1), (0, 2), (0, 3)]
    detour = astar(grid(["...", ".#.", "..."]), (1, 0), (1, 2))
    assert detour is not None
    assert len(detour) == 5
    assert (1, 1) not in detour


def test_start_equals_goal() -> None:
    assert astar(grid(["#"]), (0, 0), (0, 0)) == [(0, 0)]


def test_blocked_goal_and_disconnected_regions() -> None:
    g = grid([".#.", ".#.", ".#."])
    assert astar(g, (0, 0), (0, 1)) is None  # goal blocked
    assert astar(g, (0, 0), (2, 2)) is None  # other side of a wall
    assert cost_field(g, (0, 0))[2, 2] == UNREACHABLE


def test_start_cell_may_be_impassable() -> None:
    g = grid(["#.."])
    assert astar(g, (0, 0), (0, 2)) == [(0, 0), (0, 1), (0, 2)]


def test_boundary_conditions() -> None:
    corridor = grid(["." * 7])
    assert astar(corridor, (0, 6), (0, 0)) == [(0, c) for c in range(6, -1, -1)]
    column = grid(["."] * 4)
    assert astar(column, (3, 0), (0, 0)) == [(3, 0), (2, 0), (1, 0), (0, 0)]
    with pytest.raises(ValueError, match="outside grid"):
        astar(corridor, (0, 0), (0, 7))


def test_deterministic_tie_breaking() -> None:
    """Many equal-length paths: the (f, h, row, col) order and N,E,S,W expansion fix the choice."""
    open_grid = grid(["...", "...", "..."])
    first = astar(open_grid, (0, 0), (2, 2))
    assert first == astar(open_grid, (0, 0), (2, 2))
    assert first == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]


def test_layer_specific_terrain() -> None:
    _config, scenario = config_for(
        ["B~."], (spec("air", Layer.AIR), spec("gnd", Layer.GROUND)), (AgentSpec("air"),)
    )
    known = KnownMap.from_ground_truth(scenario.terrain, scenario.hazard_grid, scenario.base)
    assert astar(known.passable(Layer.AIR), (0, 0), (0, 2)) == [(0, 0), (0, 1), (0, 2)]
    assert astar(known.passable(Layer.GROUND), (0, 0), (0, 2)) is None


def test_weighted_costs_prefer_cheaper_detour() -> None:
    g = grid(["...", "...", "..."])
    cost = np.ones((3, 3), dtype=np.int64)
    cost[0, 1] = cost[1, 1] = 10  # make the middle column expensive
    path = astar(g, (0, 0), (0, 2), cost)
    assert path is not None
    assert path_cost(path, cost) == 6  # around through row 2: 6 moves at cost 1
    with pytest.raises(ValueError, match=">= 1"):
        astar(g, (0, 0), (0, 2), np.zeros((3, 3), dtype=np.int64))


# ----------------------------------------------------------------- properties


@st.composite
def random_grids(draw: st.DrawFn) -> tuple[npt.NDArray[np.bool_], tuple[int, int], tuple[int, int]]:
    h, w = draw(st.integers(1, 8)), draw(st.integers(1, 8))
    cells = draw(st.lists(st.booleans(), min_size=h * w, max_size=h * w))
    passable = np.array(cells, dtype=np.bool_).reshape(h, w) | (
        np.arange(h * w).reshape(h, w) % 3 == 0
    )
    start = (draw(st.integers(0, h - 1)), draw(st.integers(0, w - 1)))
    goal = (draw(st.integers(0, h - 1)), draw(st.integers(0, w - 1)))
    return passable, start, goal


@given(random_grids())
def test_astar_is_optimal_against_bfs(
    case: tuple[npt.NDArray[np.bool_], tuple[int, int], tuple[int, int]],
) -> None:
    passable, start, goal = case
    path = astar(passable, start, goal)
    expected = bfs_distance(passable, start, goal) if (passable[goal] or start == goal) else -1
    if expected == -1:
        assert path is None
    else:
        assert path is not None
        assert is_valid_path(passable, path)
        assert path[0] == start
        assert path[-1] == goal
        assert len(path) - 1 == expected


@given(random_grids(), st.data())
def test_weighted_astar_matches_dijkstra(
    case: tuple[npt.NDArray[np.bool_], tuple[int, int], tuple[int, int]], data: st.DataObject
) -> None:
    passable, start, goal = case
    h, w = passable.shape
    costs = np.array(data.draw(st.lists(st.integers(1, 6), min_size=h * w, max_size=h * w)))
    cost = costs.reshape(h, w).astype(np.int64)
    path = astar(passable, start, goal, cost)
    field = cost_field(passable, start, cost)
    if start == goal:
        assert path == [start]
    elif field[goal] == UNREACHABLE:
        assert path is None
    else:
        assert path is not None
        assert path_cost(path, cost) == field[goal]


@given(random_grids())
def test_manhattan_heuristic_is_admissible(
    case: tuple[npt.NDArray[np.bool_], tuple[int, int], tuple[int, int]],
) -> None:
    passable, start, _ = case
    field = cost_field(passable, start)
    for r in range(passable.shape[0]):
        for c in range(passable.shape[1]):
            if field[r, c] != UNREACHABLE:
                assert manhattan(start, (r, c)) <= field[r, c]


@given(random_grids(), st.data())
def test_reverse_field_equals_forward_costs(
    case: tuple[npt.NDArray[np.bool_], tuple[int, int], tuple[int, int]], data: st.DataObject
) -> None:
    passable, _, target = case
    h, w = passable.shape
    cost = np.array(data.draw(st.lists(st.integers(1, 4), min_size=h * w, max_size=h * w)))
    cost = cost.reshape(h, w).astype(np.int64)
    passable = passable.copy()
    passable[target] = True
    to_target = cost_to_target_field(passable, target, cost)
    for r in range(h):
        for c in range(w):
            if not passable[r, c]:
                continue
            forward = cost_field(passable, (r, c), cost)[target]
            assert to_target[r, c] == forward


# ----------------------------------------------------------------- known map and frontiers


def _env(rows: list[str], radius: int = 1) -> SwarmEnv:
    config, scenario = config_for(
        rows, (spec("gnd", Layer.GROUND, radius=radius, capacity=50),), (AgentSpec("gnd"),)
    )
    env = SwarmEnv(config)
    env.reset(0, scenario=scenario)
    return env


def test_known_map_records_only_the_footprint() -> None:
    env = _env(["....", ".B~.", "...."])
    obs = env.step({"gnd_0": int(Action.STAY)}).observations["gnd_0"]
    known = KnownMap(obs.grid_shape, obs.base_position)
    known.update(obs)
    expected_known = {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}
    assert {(int(r), int(c)) for r, c in np.argwhere(known.known)} == expected_known
    assert known.terrain[1, 2] == Terrain.DEBRIS
    assert known.terrain[0, 0] == UNKNOWN


def test_known_map_hazards_from_observations() -> None:
    config, scenario = config_for(
        ["B.."],
        (spec("gnd", Layer.GROUND, radius=1),),
        (AgentSpec("gnd"),),
        hazard_cells=((0, 1),),
    )
    env = SwarmEnv(config)
    obs = env.reset(0, scenario=scenario)[0]["gnd_0"]
    known = KnownMap(obs.grid_shape, obs.base_position)
    known.update(obs)
    assert known.hazards.tolist() == [[False, True, False]]


def test_frontier_definition() -> None:
    known = KnownMap((3, 3), (1, 1))
    known.terrain[:] = Terrain.OPEN
    known.terrain[1, 1] = Terrain.BASE
    known.terrain[0, :] = UNKNOWN
    known.terrain[2, 2] = Terrain.DEBRIS
    ground = known.frontier(Layer.GROUND)
    air = known.frontier(Layer.AIR)
    assert {(int(r), int(c)) for r, c in np.argwhere(ground)} == {(1, 0), (1, 1), (1, 2)}
    assert air.sum() == ground.sum()  # debris (2,2) is not next to unknown cells
    known.terrain[:] = Terrain.OPEN
    assert not known.frontier(Layer.GROUND).any()  # fully known: no frontier


def test_unknown_cells_are_impassable() -> None:
    known = KnownMap((1, 3), (0, 0))
    assert known.passable(Layer.AIR).tolist() == [[True, False, False]]


# ----------------------------------------------------------------- energy feasibility


def _model(**kw: int) -> EnergyModel:
    base = {"move": 2, "stay": 1, "capacity": 100}
    base.update(kw)
    return EnergyModel.for_agent(
        spec("air", Layer.AIR, capacity=base["capacity"], move=base["move"], stay=base["stay"]),
        hazard_layers=(Layer.AIR,),
        hazard_energy_cost=5,
        reserve_fraction=0.1,
    )


def test_reserve_is_ceil_of_fraction_of_capacity() -> None:
    assert _model(capacity=95).reserve == 10
    assert _model(capacity=100).reserve == 10


def test_route_energy_counts_moves_hazards_and_waits() -> None:
    known = KnownMap((1, 4), (0, 0))
    known.terrain[:] = Terrain.OPEN
    known.terrain[0, 0] = Terrain.BASE
    known.hazards[0, 2] = True
    model = _model()
    enter = model.enter_energy(known)
    assert model.route_energy([(0, 0), (0, 1), (0, 2), (0, 3)], enter) == 2 + 7 + 2
    assert model.route_energy([(0, 0), (0, 1)], enter, waits=3) == 2 + 3


def test_unaffected_layer_pays_no_hazard_energy() -> None:
    gnd = EnergyModel.for_agent(spec("gnd", Layer.GROUND, move=1), (Layer.AIR,), 5)
    assert gnd.hazard_cost == 0


def test_commit_rule_boundary_is_exact() -> None:
    model = _model()  # reserve 10
    assert model.can_commit(energy=40, route_energy=20, energy_to_base_at_target=10)
    assert not model.can_commit(energy=39, route_energy=20, energy_to_base_at_target=10)
    assert not model.can_commit(energy=100, route_energy=1, energy_to_base_at_target=UNREACHABLE)


def test_geometrically_reachable_but_energetically_infeasible() -> None:
    # A long corridor: the far cell is reachable, but out and back exceeds the budget.
    known = KnownMap((1, 30), (0, 0))
    known.terrain[:] = Terrain.OPEN
    known.terrain[0, 0] = Terrain.BASE
    model = _model(capacity=100)  # move 2, reserve 10
    enter = model.enter_energy(known)
    to_base = model.energy_to_base(known)
    path = astar(known.passable(Layer.AIR), (0, 0), (0, 29), enter)
    assert path is not None  # geometrically reachable
    out = model.route_energy(path, enter)
    assert out == 58
    assert to_base[0, 29] == 58
    assert not model.can_commit(100, out, int(to_base[0, 29]))  # 90 < 116
    near = astar(known.passable(Layer.AIR), (0, 0), (0, 22), enter)
    assert near is not None
    assert model.can_commit(100, model.route_energy(near, enter), int(to_base[0, 22]))  # 90 >= 88


def test_energy_to_base_includes_known_hazards() -> None:
    known = KnownMap((1, 3), (0, 0))
    known.terrain[:] = Terrain.OPEN
    known.terrain[0, 0] = Terrain.BASE
    known.hazards[0, 1] = True
    to_base = _model().energy_to_base(known)
    # From (0,2): enter (0,1) [2 + 5 hazard] then the base [2].
    assert to_base.tolist() == [[0, 2, 9]]


def test_invalid_reserve_fraction() -> None:
    with pytest.raises(ValueError, match="reserve_fraction"):
        EnergyModel.for_agent(spec("air", Layer.AIR), (), 0, reserve_fraction=1.0)

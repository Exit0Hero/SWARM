"""Simultaneous movement resolution (ADR-0003)."""

from __future__ import annotations

import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from swarm.core.movement import ConflictKind, MoveRequest, MoveResolution, resolve_moves
from swarm.core.types import Layer, Position

BASE: Position = (9, 9)
A, G = Layer.AIR, Layer.GROUND


def req(agent: str, pos: Position, target: Position, layer: Layer = A) -> MoveRequest:
    return MoveRequest(agent_id=agent, layer=layer, position=pos, target=target)


def resolve(*requests: MoveRequest, base: Position = BASE) -> MoveResolution:
    return resolve_moves(requests, base)


def test_free_move_succeeds() -> None:
    out = resolve(req("a", (0, 0), (0, 1)))
    assert out.positions == {"a": (0, 1)}
    assert out.moved == ("a",)
    assert out.conflicts == {}


def test_vertex_conflict_all_contenders_stay() -> None:
    out = resolve(req("a", (0, 0), (0, 1)), req("b", (0, 2), (0, 1)), req("c", (1, 1), (0, 1)))
    assert out.positions == {"a": (0, 0), "b": (0, 2), "c": (1, 1)}
    assert out.conflicts == dict.fromkeys(("a", "b", "c"), ConflictKind.VERTEX)
    assert out.moved == ()


def test_swap_conflict_both_stay() -> None:
    out = resolve(req("a", (0, 0), (0, 1)), req("b", (0, 1), (0, 0)))
    assert out.positions == {"a": (0, 0), "b": (0, 1)}
    assert out.conflicts == {"a": ConflictKind.SWAP, "b": ConflictKind.SWAP}


def test_blocked_by_stationary_agent() -> None:
    out = resolve(req("a", (0, 0), (0, 1)), req("b", (0, 1), (0, 1)))
    assert out.positions == {"a": (0, 0), "b": (0, 1)}
    assert out.conflicts == {"a": ConflictKind.BLOCKED}


def test_chain_behind_a_stationary_agent_all_blocked() -> None:
    # c stays; b is blocked by c; a is blocked by b (cascade).
    out = resolve(req("a", (0, 0), (0, 1)), req("b", (0, 1), (0, 2)), req("c", (0, 2), (0, 2)))
    assert out.positions == {"a": (0, 0), "b": (0, 1), "c": (0, 2)}
    assert out.conflicts == {"a": ConflictKind.BLOCKED, "b": ConflictKind.BLOCKED}


def test_chain_into_free_cell_all_move() -> None:
    out = resolve(req("a", (0, 0), (0, 1)), req("b", (0, 1), (0, 2)), req("c", (0, 2), (0, 3)))
    assert out.positions == {"a": (0, 1), "b": (0, 2), "c": (0, 3)}
    assert out.moved == ("a", "b", "c")


def test_chain_behind_a_vertex_conflict_cascades() -> None:
    # b and d collide at (0,2) so both stay; a is then blocked by b.
    out = resolve(req("a", (0, 0), (0, 1)), req("b", (0, 1), (0, 2)), req("d", (1, 2), (0, 2)))
    assert out.conflicts == {
        "a": ConflictKind.BLOCKED,
        "b": ConflictKind.VERTEX,
        "d": ConflictKind.VERTEX,
    }


def test_three_cycle_rotates() -> None:
    # Four cells around a 2x2 square; a->b's cell, b->c's cell, c->d's cell, d->a's cell.
    out = resolve(
        req("a", (0, 0), (0, 1)),
        req("b", (0, 1), (1, 1)),
        req("c", (1, 1), (1, 0)),
        req("d", (1, 0), (0, 0)),
    )
    assert out.positions == {"a": (0, 1), "b": (1, 1), "c": (1, 0), "d": (0, 0)}
    assert out.conflicts == {}


def test_base_has_unlimited_capacity() -> None:
    out = resolve(
        req("a", (8, 9), BASE),
        req("b", (9, 8), BASE),
        req("c", BASE, BASE),
    )
    assert out.positions == {"a": BASE, "b": BASE, "c": BASE}
    assert out.conflicts == {}


def test_agents_leaving_base_to_the_same_cell_conflict() -> None:
    out = resolve(req("a", BASE, (8, 9)), req("b", BASE, (8, 9)))
    assert out.conflicts == {"a": ConflictKind.VERTEX, "b": ConflictKind.VERTEX}
    assert out.positions == {"a": BASE, "b": BASE}


def test_swap_through_the_base_is_allowed() -> None:
    out = resolve(req("a", BASE, (8, 9)), req("b", (8, 9), BASE))
    assert out.positions == {"a": (8, 9), "b": BASE}
    assert out.conflicts == {}


def test_layers_do_not_interact() -> None:
    out = resolve(
        req("drone", (0, 0), (0, 1), A),
        req("robot", (0, 2), (0, 1), G),
        req("drone2", (1, 1), (1, 1), A),
        req("robot2", (1, 0), (1, 1), G),
    )
    assert out.positions["drone"] == (0, 1)
    assert out.positions["robot"] == (0, 1)
    assert out.positions["robot2"] == (1, 1)
    assert out.conflicts == {}


@pytest.mark.parametrize(
    ("requests", "message"),
    [
        ((req("a", (0, 0), (0, 2)),), "not adjacent"),
        ((req("a", (0, 0), (1, 1)),), "not adjacent"),
        ((req("a", (0, 0), (0, 0)), req("a", (1, 1), (1, 1))), "duplicate"),
        ((req("a", (0, 0), (0, 0)), req("b", (0, 0), (0, 1))), "share non-base cell"),
    ],
)
def test_contract_violations_raise(requests: tuple[MoveRequest, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve(*requests)


# ------------------------------------------------------------------- property tests

DELTAS = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
SIZE = 5
SMALL_BASE: Position = (2, 2)


@st.composite
def scenes(draw: st.DrawFn) -> list[MoveRequest]:
    """Random agents on a 5x5 grid: distinct non-base cells per layer, any number on the base."""
    cells = [(r, c) for r in range(SIZE) for c in range(SIZE)]
    requests: list[MoveRequest] = []
    for layer in Layer:
        n = draw(st.integers(0, 8))
        on_base = draw(st.integers(0, 2))
        spots = draw(
            st.lists(
                st.sampled_from([c for c in cells if c != SMALL_BASE]),
                min_size=n,
                max_size=n,
                unique=True,
            )
        )
        for i, pos in enumerate([SMALL_BASE] * on_base + spots):
            dr, dc = draw(st.sampled_from(DELTAS))
            target = (pos[0] + dr, pos[1] + dc)  # may leave the grid: resolver doesn't care
            requests.append(req(f"{layer.name.lower()}_{i}", pos, target, layer))
    return requests


@given(scenes())
def test_no_two_agents_share_a_non_base_cell_per_layer(requests: list[MoveRequest]) -> None:
    out = resolve_moves(requests, SMALL_BASE)
    for layer in Layer:
        cells = [out.positions[r.agent_id] for r in requests if r.layer == layer]
        non_base = [c for c in cells if c != SMALL_BASE]
        assert len(non_base) == len(set(non_base))


@given(scenes())
def test_no_successful_swap_outside_the_base(requests: list[MoveRequest]) -> None:
    out = resolve_moves(requests, SMALL_BASE)
    moved = [r for r in requests if r.agent_id in out.moved]
    for a in moved:
        for b in moved:
            if a.layer == b.layer and a.target == b.position and b.target == a.position:
                assert SMALL_BASE in (a.position, b.position)


@given(scenes())
def test_each_agent_ends_at_start_or_target(requests: list[MoveRequest]) -> None:
    out = resolve_moves(requests, SMALL_BASE)
    for r in requests:
        final = out.positions[r.agent_id]
        assert final in (r.position, r.target)
        assert (final == r.target and r.target != r.position) == (r.agent_id in out.moved)
        if r.agent_id in out.conflicts:
            assert r.target != r.position
            assert final == r.position


@given(scenes())
def test_conflict_reasons_are_justified(requests: list[MoveRequest]) -> None:
    out = resolve_moves(requests, SMALL_BASE)
    by_id = {r.agent_id: r for r in requests}
    for agent, kind in out.conflicts.items():
        r = by_id[agent]
        same_layer = [o for o in requests if o.layer == r.layer and o.agent_id != agent]
        assert r.target != SMALL_BASE
        if kind is ConflictKind.VERTEX:
            assert any(o.target == r.target and o.target != o.position for o in same_layer)
        elif kind is ConflictKind.SWAP:
            assert any(o.position == r.target and o.target == r.position for o in same_layer)
        else:
            assert any(out.positions[o.agent_id] == r.target for o in same_layer)


@given(scenes(), st.randoms(use_true_random=False))
def test_permutation_and_relabelling_invariance(
    requests: list[MoveRequest], rnd: random.Random
) -> None:
    baseline = resolve_moves(requests, SMALL_BASE)

    shuffled = list(requests)
    rnd.shuffle(shuffled)
    assert resolve_moves(shuffled, SMALL_BASE) == baseline

    # Rename agents with a random bijection: outcomes must follow the agents, not the names.
    names = [r.agent_id for r in requests]
    new_names = [f"x{i}" for i in range(len(names))]
    rnd.shuffle(new_names)
    rename = dict(zip(names, new_names, strict=True))
    renamed = [req(rename[r.agent_id], r.position, r.target, r.layer) for r in requests]
    out = resolve_moves(renamed, SMALL_BASE)
    for old, new in rename.items():
        assert out.positions[new] == baseline.positions[old]
        assert out.conflicts.get(new) == baseline.conflicts.get(old)

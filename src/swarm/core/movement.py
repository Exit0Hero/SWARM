"""Simultaneous movement and conflict resolution (ADR-0003).

:func:`resolve_moves` is a pure function. It takes every agent's current cell and intended
target (equal for agents that stay) and returns where each agent ends up. Per layer:

1. **Vertex:** two or more movers targeting the same non-base cell all stay.
2. **Swap:** two movers exchanging cells both stay, unless either cell is the base
   (the base has unlimited capacity, so agents pass each other there).
3. **Blocked:** a mover whose non-base target is occupied by a stationary agent stays; this
   repeats until no further move fails (failed moves can block other moves).
4. Every remaining move succeeds, including rotations of three or more agents.

Agents on different layers never interact. The result depends only on the *set* of requests,
not on their order (permutation invariance), and no two agents of one layer end in the same
non-base cell.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from swarm.core.types import AgentId, Layer, Position


class ConflictKind(StrEnum):
    """Why a move failed (ADR-0003)."""

    VERTEX = "vertex"
    SWAP = "swap"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class MoveRequest:
    """One agent's intended move. ``target == position`` means the agent stays."""

    agent_id: AgentId
    layer: Layer
    position: Position
    target: Position


@dataclass(frozen=True, slots=True)
class MoveResolution:
    """Outcome of :func:`resolve_moves`.

    ``positions`` maps every agent to its final cell. ``moved`` lists agents whose move
    succeeded and ``conflicts`` maps agents whose move failed to the reason. Both are sorted by
    agent ID.
    """

    positions: dict[AgentId, Position]
    moved: tuple[AgentId, ...]
    conflicts: dict[AgentId, ConflictKind]


def resolve_moves(requests: Iterable[MoveRequest], base: Position) -> MoveResolution:
    """Resolve simultaneous moves on all layers. See the module docstring for the rules.

    Raises:
        ValueError: on duplicate agent IDs, a target more than one 4-connected step away, or two
            agents of the same layer starting in the same non-base cell.
    """
    reqs = sorted(requests, key=lambda r: r.agent_id)
    ids = [r.agent_id for r in reqs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"resolve_moves: duplicate agent IDs in {ids}")
    for r in reqs:
        if abs(r.target[0] - r.position[0]) + abs(r.target[1] - r.position[1]) > 1:
            raise ValueError(
                f"resolve_moves: {r.agent_id} target {r.target} is not adjacent to {r.position}"
            )

    conflicts: dict[AgentId, ConflictKind] = {}
    for layer in Layer:
        conflicts.update(_resolve_layer([r for r in reqs if r.layer == layer], base))

    positions: dict[AgentId, Position] = {}
    moved: list[AgentId] = []
    for r in reqs:
        if r.target != r.position and r.agent_id not in conflicts:
            positions[r.agent_id] = r.target
            moved.append(r.agent_id)
        else:
            positions[r.agent_id] = r.position
    return MoveResolution(
        positions=positions,
        moved=tuple(moved),
        conflicts=dict(sorted(conflicts.items())),
    )


def _resolve_layer(reqs: list[MoveRequest], base: Position) -> dict[AgentId, ConflictKind]:
    occupant: dict[Position, MoveRequest] = {}
    for r in reqs:
        if r.position == base:
            continue
        if r.position in occupant:
            raise ValueError(
                f"resolve_moves: {occupant[r.position].agent_id} and {r.agent_id} share "
                f"non-base cell {r.position} on layer {r.layer.name}"
            )
        occupant[r.position] = r

    movers = [r for r in reqs if r.target != r.position]
    failed: dict[AgentId, ConflictKind] = {}

    # 1. Vertex conflicts.
    by_target: dict[Position, list[MoveRequest]] = {}
    for r in movers:
        if r.target != base:
            by_target.setdefault(r.target, []).append(r)
    for group in by_target.values():
        if len(group) > 1:
            for r in group:
                failed[r.agent_id] = ConflictKind.VERTEX

    # 2. Swap conflicts (exempt when either cell is the base).
    for r in movers:
        if r.agent_id in failed or r.position == base or r.target == base:
            continue
        other = occupant.get(r.target)
        if (
            other is not None
            and other.target == r.position
            and other.agent_id not in failed
            and other.target != other.position
        ):
            failed[r.agent_id] = ConflictKind.SWAP
            failed[other.agent_id] = ConflictKind.SWAP

    # 3. Blocked by a stationary agent, iterated to a fixed point.
    stationary: set[Position] = {
        r.position
        for r in reqs
        if r.position != base and (r.target == r.position or r.agent_id in failed)
    }
    changed = True
    while changed:
        changed = False
        for r in movers:
            if r.agent_id in failed or r.target == base:
                continue
            if r.target in stationary:
                failed[r.agent_id] = ConflictKind.BLOCKED
                if r.position != base:
                    stationary.add(r.position)
                changed = True
    return failed

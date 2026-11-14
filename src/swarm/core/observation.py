"""Per-agent observations: the only thing a policy sees (ADR-0008).

An :class:`Observation` contains the agent's own state, the terrain and agents inside its sensor
footprint, this step's detection reports, static public facts (grid shape, base position) and
the action mask. It never contains survivor ground truth, other agents' energy, RNG state or a
reference to the environment.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import cache
from typing import Final

import numpy as np
import numpy.typing as npt

from swarm.core.types import AgentId, AgentStatus, Layer, Position

UNKNOWN: Final[int] = -1
"""``local_terrain`` code for a cell inside the grid but outside the sensor footprint."""

OUT_OF_BOUNDS: Final[int] = -2
"""``local_terrain`` code for a cell outside the grid."""


@cache
def footprint_offsets(radius: int) -> tuple[tuple[int, int], ...]:
    """Offsets ``(dr, dc)`` with ``dr² + dc² ≤ radius²``, sorted (Euclidean disk, ADR-0001)."""
    if radius < 0:
        raise ValueError(f"radius: must be >= 0, got {radius}")
    r2 = radius * radius
    return tuple(
        (dr, dc)
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
        if dr * dr + dc * dc <= r2
    )


def in_footprint(center: Position, cell: Position, radius: int) -> bool:
    """Whether ``cell`` is within Euclidean ``radius`` of ``center``."""
    dr, dc = cell[0] - center[0], cell[1] - center[1]
    return dr * dr + dc * dc <= radius * radius


@dataclass(frozen=True, slots=True)
class VisibleAgent:
    """Another agent inside the observer's footprint. Its energy is not visible."""

    agent_id: AgentId
    agent_type: str
    layer: Layer
    position: Position
    status: AgentStatus


@dataclass(frozen=True, slots=True, eq=False)
class Observation:
    """What one agent knows at one step.

    ``local_terrain`` is a read-only ``int8`` array of shape ``(2r+1, 2r+1)`` centred on the
    agent (``r = sensor_radius``), holding :class:`~swarm.core.types.Terrain` codes inside the
    footprint, :data:`UNKNOWN` outside it and :data:`OUT_OF_BOUNDS` beyond the grid.
    ``detections`` are the cells reported this step (possibly noisy; empty on reset).
    ``action_mask[a]`` is ``True`` when action ``a`` is valid (ADR-0004).
    """

    agent_id: AgentId
    agent_type: str
    layer: Layer
    step: int
    position: Position
    energy: int
    energy_capacity: int
    status: AgentStatus
    sensor_radius: int
    grid_shape: tuple[int, int]
    base_position: Position
    local_terrain: npt.NDArray[np.int8]
    visible_agents: tuple[VisibleAgent, ...]
    detections: tuple[Position, ...]
    action_mask: tuple[bool, ...]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Observation):
            return NotImplemented
        names = [f.name for f in fields(self) if f.name != "local_terrain"]
        return all(getattr(self, n) == getattr(other, n) for n in names) and np.array_equal(
            self.local_terrain, other.local_terrain
        )

    __hash__ = None  # type: ignore[assignment]

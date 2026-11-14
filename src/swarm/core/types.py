"""Enumerations and value types shared by the simulator (ADR-0001, ADR-0002, ADR-0004)."""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Final

import numpy as np
import numpy.typing as npt

AgentId = str
"""Stable agent identifier, for example ``"drone_0"`` (ADR-0010)."""

Position = tuple[int, int]
"""Grid cell as ``(row, col)`` with ``(0, 0)`` at the top-left (ADR-0001)."""


class Layer(IntEnum):
    """Occupancy layer. Agents on different layers never conflict (ADR-0002)."""

    AIR = 0
    GROUND = 1


class Terrain(IntEnum):
    """Terrain type of a cell (ADR-0002)."""

    OPEN = 0
    DEBRIS = 1
    STRUCTURE = 2
    BASE = 3


class Action(IntEnum):
    """The five primitive actions (ADR-0004). North is decreasing row."""

    STAY = 0
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4


class AgentStatus(StrEnum):
    """Agent status. Returning to base is a policy behaviour, not a status (ADR-0004)."""

    ACTIVE = "active"
    DISABLED = "disabled"


NUM_ACTIONS: Final[int] = len(Action)

ACTION_DELTAS: Final[dict[Action, tuple[int, int]]] = {
    Action.STAY: (0, 0),
    Action.NORTH: (-1, 0),
    Action.EAST: (0, 1),
    Action.SOUTH: (1, 0),
    Action.WEST: (0, -1),
}

# TRAVERSABLE[layer, terrain] — the traversability table from ADR-0002, kept as data.
_TRAVERSABLE: Final[npt.NDArray[np.bool_]] = np.array(
    [
        # OPEN  DEBRIS STRUCTURE BASE
        [True, True, False, True],  # AIR
        [True, False, False, True],  # GROUND
    ],
    dtype=np.bool_,
)
_TRAVERSABLE.flags.writeable = False


def is_traversable(layer: Layer, terrain: Terrain | int) -> bool:
    """Return whether an agent on ``layer`` may occupy a cell of type ``terrain``."""
    return bool(_TRAVERSABLE[int(layer), int(terrain)])


def traversable_grid(layer: Layer, terrain: npt.NDArray[np.int8]) -> npt.NDArray[np.bool_]:
    """Return a boolean grid of cells that ``layer`` may occupy."""
    result: npt.NDArray[np.bool_] = _TRAVERSABLE[int(layer)][terrain]
    return result

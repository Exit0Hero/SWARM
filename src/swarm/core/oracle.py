"""The explicit oracle channel to ground truth (ADR-0008 rule 5, ADR-0014 rule 5).

Only team policies that declare ``uses_oracle = True`` ever receive an :class:`OracleView`, and
their results are labelled ``information = "oracle"``. Arrays are read-only copies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from swarm.core.types import Position


@dataclass(frozen=True, slots=True, eq=False)
class OracleView:
    """Static ground truth of one episode: terrain, hazards, survivors and base."""

    terrain: npt.NDArray[np.int8]
    hazards: npt.NDArray[np.bool_]
    survivors: tuple[Position, ...]
    base: Position

    def __post_init__(self) -> None:
        terrain = np.array(self.terrain, dtype=np.int8, copy=True)
        hazards = np.array(self.hazards, dtype=np.bool_, copy=True)
        terrain.flags.writeable = False
        hazards.flags.writeable = False
        object.__setattr__(self, "terrain", terrain)
        object.__setattr__(self, "hazards", hazards)

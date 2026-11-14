"""Deterministic known-map memory built only from observations (ADR-0015 rule 1).

A :class:`KnownMap` is the union of everything observed: terrain codes and hazard flags inside
sensor footprints, plus the cells of agents *seen* to be disabled (ADR-0016 rule 15), which are
permanent obstacles for their layer. It is not a probabilistic belief map (that is Phase 3); a cell
is either known exactly or unknown.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from swarm.core.observation import Observation
from swarm.core.types import AgentStatus, Layer, Position, Terrain, traversable_grid

UNKNOWN: int = -1


class KnownMap:
    """What a planner knows: terrain (``UNKNOWN`` where unseen) and seen hazards."""

    def __init__(self, shape: tuple[int, int], base: Position) -> None:
        self.shape = shape
        self.base = base
        self.terrain: npt.NDArray[np.int8] = np.full(shape, UNKNOWN, dtype=np.int8)
        self.hazards: npt.NDArray[np.bool_] = np.zeros(shape, dtype=np.bool_)
        self.terrain[base] = Terrain.BASE  # the base location is public (mission brief)
        self.blocked: npt.NDArray[np.bool_] = np.zeros((len(Layer), *shape), dtype=np.bool_)
        """``blocked[layer]``: cells of agents observed disabled on that layer (never the base)."""
        self.version = 0
        """Incremented whenever an update reveals new information (for cache invalidation)."""

    @classmethod
    def from_ground_truth(
        cls,
        terrain: npt.NDArray[np.int8],
        hazards: npt.NDArray[np.bool_],
        base: Position,
    ) -> KnownMap:
        """A fully known map. Only for the oracle reference (ADR-0014)."""
        shape = (int(terrain.shape[0]), int(terrain.shape[1]))
        known = cls(shape, base)
        known.terrain = np.array(terrain, dtype=np.int8, copy=True)
        known.hazards = np.array(hazards, dtype=np.bool_, copy=True)
        return known

    def update(self, observation: Observation) -> None:
        """Merge one observation's footprint view (terrain and hazards) into the map."""
        radius = observation.sensor_radius
        r0, c0 = observation.position
        h, w = self.shape
        top, left = r0 - radius, c0 - radius
        rs, re = max(0, top), min(h, r0 + radius + 1)
        cs, ce = max(0, left), min(w, c0 + radius + 1)
        view = observation.local_terrain[rs - top : re - top, cs - left : ce - left]
        hz = observation.local_hazard[rs - top : re - top, cs - left : ce - left]
        seen = view >= 0
        window_t = self.terrain[rs:re, cs:ce]
        window_h = self.hazards[rs:re, cs:ce]
        if bool((window_t[seen] != view[seen]).any() or (window_h[seen] != hz[seen]).any()):
            window_t[seen] = view[seen]
            window_h[seen] = hz[seen]
            self.version += 1
        for v in observation.visible_agents:
            if v.status is AgentStatus.DISABLED:
                self.mark_blocked(v.layer, v.position)

    def mark_blocked(self, layer: Layer, cell: Position) -> None:
        """Record a disabled agent at ``cell`` as a permanent obstacle for ``layer``."""
        if cell != self.base and not self.blocked[int(layer)][cell]:
            self.blocked[int(layer)][cell] = True
            self.version += 1

    def update_all(self, observations: Iterable[Observation]) -> None:
        for obs in observations:
            self.update(obs)

    @property
    def known(self) -> npt.NDArray[np.bool_]:
        result: npt.NDArray[np.bool_] = self.terrain != UNKNOWN
        return result

    def passable(self, layer: Layer) -> npt.NDArray[np.bool_]:
        """Known cells the layer may occupy. Unknown cells and cells of observed disabled agents
        of the same layer are impassable (pessimistic)."""
        grid = np.where(self.known, self.terrain, Terrain.STRUCTURE).astype(np.int8)
        result: npt.NDArray[np.bool_] = traversable_grid(layer, grid) & ~self.blocked[int(layer)]
        return result

    def frontier(self, layer: Layer) -> npt.NDArray[np.bool_]:
        """Frontier cells for ``layer`` (ADR-0016 rule 7): known, traversable, next to unknown."""
        unknown = ~self.known
        near_unknown = np.zeros(self.shape, dtype=np.bool_)
        near_unknown[1:, :] |= unknown[:-1, :]
        near_unknown[:-1, :] |= unknown[1:, :]
        near_unknown[:, 1:] |= unknown[:, :-1]
        near_unknown[:, :-1] |= unknown[:, 1:]
        result: npt.NDArray[np.bool_] = self.passable(layer) & near_unknown
        return result

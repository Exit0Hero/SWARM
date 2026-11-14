"""Energy model and the return-to-base commit rule (ADR-0016 rules 1-6).

Planner assumptions, stated once and used everywhere:

* a move into cell ``x`` costs ``move_cost``, plus the hazard ``energy_cost`` when ``x`` is a
  *known* hazard and the agent's layer is affected;
* a planned wait costs ``stay_cost``;
* recharge happens only on the base and is never assumed en route;
* hazard *disablement* risk is not converted into energy (hazard-aware variants avoid hazards).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from swarm.core.types import Layer
from swarm.navigation.astar import UNREACHABLE, Path, cost_to_target_field
from swarm.navigation.known_map import KnownMap
from swarm.scenarios.config import AgentTypeSpec

DEFAULT_RESERVE_FRACTION: float = 0.1


@dataclass(frozen=True, slots=True)
class EnergyModel:
    """One agent's energy costs under planner assumptions."""

    layer: Layer
    move_cost: int
    stay_cost: int
    capacity: int
    hazard_cost: int
    reserve: int

    @classmethod
    def for_agent(
        cls,
        spec: AgentTypeSpec,
        hazard_layers: tuple[Layer, ...],
        hazard_energy_cost: int,
        reserve_fraction: float = DEFAULT_RESERVE_FRACTION,
    ) -> EnergyModel:
        if not 0.0 <= reserve_fraction < 1.0:
            raise ValueError(f"reserve_fraction: must be in [0, 1), got {reserve_fraction}")
        affected = spec.layer in hazard_layers
        return cls(
            layer=spec.layer,
            move_cost=spec.move_cost,
            stay_cost=spec.stay_cost,
            capacity=spec.energy_capacity,
            hazard_cost=hazard_energy_cost if affected else 0,
            reserve=math.ceil(reserve_fraction * spec.energy_capacity),
        )

    def enter_energy(self, known: KnownMap) -> npt.NDArray[np.int64]:
        """Energy of entering each cell (>= move_cost >= 1)."""
        grid = np.full(known.shape, self.move_cost, dtype=np.int64)
        if self.hazard_cost:
            grid += self.hazard_cost * known.hazards.astype(np.int64)
        return grid

    def route_energy(self, path: Path, enter: npt.NDArray[np.int64], waits: int = 0) -> int:
        """Energy of following ``path`` (moves) plus ``waits`` planned waits."""
        return int(sum(int(enter[cell]) for cell in path[1:])) + waits * self.stay_cost

    def energy_to_base(self, known: KnownMap) -> npt.NDArray[np.int64]:
        """Minimum energy from every known cell back to the base; ``UNREACHABLE`` if none."""
        return cost_to_target_field(
            known.passable(self.layer), known.base, self.enter_energy(known)
        )

    def can_commit(self, energy: int, route_energy: int, energy_to_base_at_target: int) -> bool:
        """Commit rule: ``energy - reserve >= route + return`` (ADR-0016 rule 4)."""
        if energy_to_base_at_target == UNREACHABLE or route_energy < 0:
            return False
        return energy - self.reserve >= route_energy + energy_to_base_at_target

"""Shared planning helpers for the Phase 2 baselines (ADR-0015, ADR-0016)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import numpy.typing as npt

from swarm.core.types import Layer, Position
from swarm.navigation.astar import UNREACHABLE, cost_field
from swarm.navigation.energy import EnergyModel
from swarm.navigation.known_map import KnownMap

DEFAULT_HAZARD_PENALTY: Final[int] = 10
BLACKLIST_STEPS: Final[int] = 5
"""Steps a target that failed to plan is skipped before it may be selected again."""


def planning_costs(
    known: KnownMap, layer: Layer, hazard_layers: tuple[Layer, ...], hazard_aware: bool
) -> npt.NDArray[np.int64] | None:
    """Enter costs for target choice and paths: ``None`` (unit) for shortest-path baselines;
    ``1 + DEFAULT_HAZARD_PENALTY`` on known hazards for hazard-aware variants (ADR-0015 rule 6).
    """
    if not hazard_aware or layer not in hazard_layers:
        return None
    return 1 + DEFAULT_HAZARD_PENALTY * known.hazards.astype(np.int64)


@dataclass
class FieldCache:
    """Caches per-layer fields keyed by the map version (fields only change with the map)."""

    version: int = -1
    to_base: dict[tuple[Layer, int], npt.NDArray[np.int64]] = field(default_factory=dict)
    frontier: dict[Layer, npt.NDArray[np.bool_]] = field(default_factory=dict)
    passable: dict[Layer, npt.NDArray[np.bool_]] = field(default_factory=dict)

    def refresh(self, known: KnownMap) -> None:
        if known.version != self.version:
            self.version = known.version
            self.to_base.clear()
            self.frontier.clear()
            self.passable.clear()

    def passable_for(self, known: KnownMap, layer: Layer) -> npt.NDArray[np.bool_]:
        self.refresh(known)
        if layer not in self.passable:
            self.passable[layer] = known.passable(layer)
        return self.passable[layer]

    def frontier_for(self, known: KnownMap, layer: Layer) -> npt.NDArray[np.bool_]:
        self.refresh(known)
        if layer not in self.frontier:
            self.frontier[layer] = known.frontier(layer)
        return self.frontier[layer]

    def to_base_for(self, known: KnownMap, model: EnergyModel) -> npt.NDArray[np.int64]:
        self.refresh(known)
        key = (model.layer, model.hazard_cost * 1_000_000 + model.move_cost)
        if key not in self.to_base:
            self.to_base[key] = model.energy_to_base(known)
        return self.to_base[key]


def block_cells(
    passable: npt.NDArray[np.bool_], cells: list[Position], base: Position
) -> npt.NDArray[np.bool_]:
    """Copy of ``passable`` with ``cells`` (except the base) made impassable."""
    if not cells:
        return passable
    out = passable.copy()
    for cell in cells:
        if cell != base:
            out[cell] = False
    return out


def feasible_mask(
    model: EnergyModel,
    energy: int,
    passable: npt.NDArray[np.bool_],
    here: Position,
    known: KnownMap,
    to_base: npt.NDArray[np.int64],
) -> npt.NDArray[np.bool_]:
    """Cells the commit rule allows *via the minimum-energy route* from ``here`` (ADR-0016)."""
    out_energy = cost_field(passable, here, model.enter_energy(known))
    reachable = (out_energy != UNREACHABLE) & (to_base != UNREACHABLE)
    budget = energy - model.reserve
    result: npt.NDArray[np.bool_] = reachable & (out_energy + to_base <= budget)
    return result


@dataclass
class Counters:
    """Deterministic policy diagnostics (ADR-0016 rule 11)."""

    targets_selected: int = 0
    targets_reached: int = 0
    target_invalidations: int = 0
    failed_plans: int = 0
    failed_plans_unreachable: int = 0
    failed_plans_temporary: int = 0
    energy_route_aborts: int = 0
    planned_waits: int = 0
    returns_to_base: int = 0
    reassignments: int = 0
    move_failures: int = 0
    backoff_waits: int = 0
    detours: int = 0
    deadlock_abandons: int = 0
    depot_yields: int = 0
    completion_times: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, int | float | None]:
        times = self.completion_times
        return {
            "targets_selected": self.targets_selected,
            "targets_reached": self.targets_reached,
            "target_invalidations": self.target_invalidations,
            "failed_plans": self.failed_plans,
            "failed_plans_unreachable": self.failed_plans_unreachable,
            "failed_plans_temporary": self.failed_plans_temporary,
            "energy_route_aborts": self.energy_route_aborts,
            "planned_waits": self.planned_waits,
            "returns_to_base": self.returns_to_base,
            "reassignments": self.reassignments,
            "move_failures": self.move_failures,
            "backoff_waits": self.backoff_waits,
            "detours": self.detours,
            "deadlock_abandons": self.deadlock_abandons,
            "depot_yields": self.depot_yields,
            "mean_completion_time": (sum(times) / len(times)) if times else None,
        }

"""Replaceable perception models (Phase 5, ADR-0025).

Without a ``perception`` block the environment keeps its original synthetic sensor (ADR-0012), so
every earlier event stream and golden hash is unchanged. With one, a :class:`SensorModel` picked
by name from :data:`SENSOR_MODELS` decides:

- the probability of detecting a survivor at a given distance (range limitation);
- the confidence reported with each detection (a Bayes posterior under the model, computed from
  public parameters, never from ground truth);
- how terrain and hazard readings in the footprint are corrupted (obstacle and hazard perception
  uncertainty).

All randomness comes from the episode's ``perception`` stream with a fixed number of draws per
in-grid footprint cell (two), so changing an error rate never shifts later draws (ADR-0006).
Errors are logged as evaluator-only :class:`~swarm.core.events.PerceptionError` events.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np
import numpy.typing as npt

from swarm.core.types import Terrain

# Misread terrain: what a corrupted reading reports instead of the true type.
MISREAD: Final[dict[int, int]] = {
    int(Terrain.OPEN): int(Terrain.DEBRIS),
    int(Terrain.DEBRIS): int(Terrain.OPEN),
    int(Terrain.STRUCTURE): int(Terrain.DEBRIS),
}


class SensorModel(Protocol):
    """Interface every perception model implements."""

    def detect_probability(self, p_detect: float, distance: float, radius: int) -> float:
        """Probability of detecting a survivor ``distance`` cells away."""
        ...

    def confidence(self, p_detect: float, p_false: float, distance: float, radius: int) -> float:
        """Confidence in [0, 1] attached to a report at ``distance``."""
        ...

    def corrupt_terrain(self, true_code: int, u: float) -> int:
        """Reported terrain code for one cell, given one uniform draw ``u``."""
        ...

    def corrupt_hazard(self, true_hazard: bool, u: float) -> bool:
        """Reported hazard flag for one cell, given one uniform draw ``u``."""
        ...


@dataclass(frozen=True, slots=True)
class NoisySensor:
    """Range-limited detection with confidence and independent terrain/hazard misreads.

    - Detection probability falls linearly with distance:
      ``p_detect * (1 - range_falloff * distance / (radius + 1))``.
    - Confidence is the posterior that a report is a true survivor, assuming a survivor prior of
      ``prior`` per cell: ``q * prior / (q * prior + p_false * (1 - prior))`` with ``q`` the
      detection probability at that distance.
    - A cell's terrain is misread with probability ``terrain_error`` (open ↔ debris, structure →
      debris; the base is never misread); a true hazard is missed with ``hazard_miss`` and a
      safe cell flagged with ``hazard_false``.
    """

    range_falloff: float = 0.0
    terrain_error: float = 0.0
    hazard_miss: float = 0.0
    hazard_false: float = 0.0
    prior: float = 0.02

    def detect_probability(self, p_detect: float, distance: float, radius: int) -> float:
        return max(0.0, p_detect * (1.0 - self.range_falloff * distance / (radius + 1)))

    def confidence(self, p_detect: float, p_false: float, distance: float, radius: int) -> float:
        q = self.detect_probability(p_detect, distance, radius)
        num = q * self.prior
        den = num + p_false * (1.0 - self.prior)
        return 1.0 if den == 0.0 else num / den

    def corrupt_terrain(self, true_code: int, u: float) -> int:
        if u < self.terrain_error and true_code in MISREAD:
            return MISREAD[true_code]
        return true_code

    def corrupt_hazard(self, true_hazard: bool, u: float) -> bool:
        if true_hazard:
            return not (u < self.hazard_miss)
        return u < self.hazard_false


SENSOR_MODELS: Final[dict[str, Callable[..., SensorModel]]] = {"noisy": NoisySensor}


def make_sensor(model: str, **params: float) -> SensorModel:
    """Build a registered sensor model; raises ``ValueError`` for an unknown name."""
    if model not in SENSOR_MODELS:
        raise ValueError(f"perception.model: unknown {model!r}; known: {sorted(SENSOR_MODELS)}")
    return SENSOR_MODELS[model](**params)


def corrupt_view(
    sensor: SensorModel,
    terrain: npt.NDArray[np.int8],
    hazard: npt.NDArray[np.bool_],
    in_grid: list[tuple[int, int]],
    draws: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.bool_], list[tuple[int, int, str]]]:
    """Apply the sensor to the view cells ``in_grid`` (two draws each, row-major).

    Returns the corrupted copies and the errors as ``(view_row, view_col, kind)`` with kind
    ``terrain``, ``hazard_missed`` or ``hazard_false``.
    """
    t, h = terrain.copy(), hazard.copy()
    errors: list[tuple[int, int, str]] = []
    for k, (vr, vc) in enumerate(in_grid):
        true_t, true_h = int(terrain[vr, vc]), bool(hazard[vr, vc])
        new_t = sensor.corrupt_terrain(true_t, float(draws[2 * k]))
        new_h = sensor.corrupt_hazard(true_h, float(draws[2 * k + 1]))
        if new_t != true_t:
            t[vr, vc] = new_t
            errors.append((vr, vc, "terrain"))
        if new_h != true_h:
            h[vr, vc] = new_h
            errors.append((vr, vc, "hazard_missed" if true_h else "hazard_false"))
    t.flags.writeable = False
    h.flags.writeable = False
    return t, h, errors


def report_confidence(
    sensor: SensorModel,
    p_detect: float,
    p_false: float,
    observer: tuple[int, int],
    cell: tuple[int, int],
    radius: int,
) -> float:
    """Confidence an agent can attach to its own report of ``cell`` from ``observer``.

    Uses only public parameters (the mission brief's sensor model), never ground truth, so it is
    computed by policies rather than delivered in the observation.
    """
    distance = float(np.hypot(cell[0] - observer[0], cell[1] - observer[1]))
    return sensor.confidence(p_detect, p_false, distance, radius)

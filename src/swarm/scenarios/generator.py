"""Scenarios (ground truth) and seeded scenario generation (ADR-0002, ADR-0006).

A :class:`Scenario` holds the hidden ground truth of an episode: terrain, base and survivor
positions. Only the environment and the evaluator read it; policies never receive it (ADR-0008).
"""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from swarm.core.types import Layer, Position, Terrain, traversable_grid
from swarm.scenarios.config import ScenarioConfig


class ScenarioError(ValueError):
    """A scenario cannot be generated from a config, or a given scenario is inconsistent."""


SURVIVOR_TERRAIN: Final[frozenset[Terrain]] = frozenset({Terrain.OPEN, Terrain.DEBRIS})
"""Survivors are placed only on open ground or debris (ADR-0002)."""

_ASCII_TERRAIN: Final[dict[str, Terrain]] = {
    ".": Terrain.OPEN,
    "~": Terrain.DEBRIS,
    "#": Terrain.STRUCTURE,
    "B": Terrain.BASE,
    "S": Terrain.OPEN,  # survivor on open ground
    "s": Terrain.DEBRIS,  # survivor under debris
}


@dataclass(frozen=True, slots=True, eq=False)
class Scenario:
    """Ground truth for one episode.

    ``terrain`` is a read-only ``int8`` array of :class:`Terrain` codes, shape ``(height, width)``.
    ``survivors`` is sorted by ``(row, col)``; a survivor's index in this tuple is its identity.
    """

    terrain: npt.NDArray[np.int8]
    base: Position
    survivors: tuple[Position, ...]

    def __post_init__(self) -> None:
        grid = np.array(self.terrain, dtype=np.int8, copy=True)
        if grid.ndim != 2 or grid.size == 0:
            raise ScenarioError(f"terrain: must be a non-empty 2D array, got shape {grid.shape}")
        valid_codes = {int(t) for t in Terrain}
        if not set(np.unique(grid).tolist()) <= valid_codes:
            raise ScenarioError(f"terrain: codes must be in {sorted(valid_codes)}")
        grid.flags.writeable = False
        object.__setattr__(self, "terrain", grid)

        h, w = grid.shape
        r, c = self.base
        if not (0 <= r < h and 0 <= c < w):
            raise ScenarioError(f"base: {self.base} is outside the {h}x{w} grid")
        base_cells = np.argwhere(grid == Terrain.BASE)
        if len(base_cells) != 1 or tuple(int(v) for v in base_cells[0]) != self.base:
            raise ScenarioError(f"terrain: must contain exactly one BASE cell, at {self.base}")

        survivors = tuple(sorted((int(sr), int(sc)) for sr, sc in self.survivors))
        if len(set(survivors)) != len(survivors):
            raise ScenarioError("survivors: positions must be unique")
        for sr, sc in survivors:
            if not (0 <= sr < h and 0 <= sc < w):
                raise ScenarioError(f"survivors: {(sr, sc)} is outside the grid")
            if Terrain(int(grid[sr, sc])) not in SURVIVOR_TERRAIN:
                raise ScenarioError(f"survivors: {(sr, sc)} is not on OPEN or DEBRIS terrain")
        object.__setattr__(self, "survivors", survivors)

    @property
    def shape(self) -> tuple[int, int]:
        h, w = self.terrain.shape
        return (int(h), int(w))

    def fingerprint(self) -> str:
        """SHA-256 over terrain, base and survivors: equal fingerprints mean equal scenarios."""
        digest = hashlib.sha256()
        digest.update(np.asarray(self.terrain.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(self.terrain).tobytes())
        digest.update(repr((self.base, self.survivors)).encode("ascii"))
        return digest.hexdigest()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Scenario):
            return NotImplemented
        return (
            self.base == other.base
            and self.survivors == other.survivors
            and np.array_equal(self.terrain, other.terrain)
        )

    __hash__ = None  # type: ignore[assignment]


def generate_scenario(config: ScenarioConfig, rng: np.random.Generator) -> Scenario:
    """Generate terrain and survivor placement from ``config`` using the ``scenario`` stream.

    Each non-base cell draws one uniform number ``u``: ``u < structure_fraction`` gives
    ``STRUCTURE``; ``u < structure_fraction + debris_fraction`` gives ``DEBRIS``; otherwise
    ``OPEN``. Survivors are then drawn without replacement from all ``OPEN``/``DEBRIS`` cells in
    row-major order. Reachability is *not* enforced: a survivor may be unreachable, which the
    coverage and recall metrics account for (ADR-0009).

    Raises:
        ScenarioError: if there are fewer eligible cells than ``config.survivor_count``.
    """
    h, w = config.height, config.width
    structure = config.terrain.structure_fraction
    debris = config.terrain.debris_fraction
    u = rng.random((h, w))
    terrain = np.full((h, w), Terrain.OPEN, dtype=np.int8)
    terrain[u < structure + debris] = Terrain.DEBRIS
    terrain[u < structure] = Terrain.STRUCTURE
    terrain[config.base] = Terrain.BASE

    candidates = np.flatnonzero((terrain == Terrain.OPEN) | (terrain == Terrain.DEBRIS))
    if config.survivor_count > len(candidates):
        raise ScenarioError(
            f"survivor_count={config.survivor_count} but only {len(candidates)} OPEN/DEBRIS "
            f"cells were generated; lower survivor_count or the structure/debris fractions"
        )
    survivors: list[Position] = []
    if config.survivor_count > 0:
        chosen = rng.choice(candidates, size=config.survivor_count, replace=False)
        survivors = [(int(i) // w, int(i) % w) for i in chosen]
    return Scenario(terrain=terrain, base=config.base, survivors=tuple(survivors))


def check_scenario_matches(config: ScenarioConfig, scenario: Scenario) -> None:
    """Raise :class:`ScenarioError` if ``scenario`` cannot be used with ``config``."""
    if scenario.shape != (config.height, config.width):
        raise ScenarioError(
            f"scenario shape {scenario.shape} != config ({config.height}, {config.width})"
        )
    if scenario.base != config.base:
        raise ScenarioError(f"scenario base {scenario.base} != config base {config.base}")


def parse_ascii_map(rows: Sequence[str]) -> Scenario:
    """Build a :class:`Scenario` from a text map, for tests and examples.

    ``.`` open, ``~`` debris, ``#`` structure, ``B`` base, ``S`` survivor on open ground,
    ``s`` survivor under debris. All rows must have equal length.
    """
    if not rows or len({len(r) for r in rows}) != 1:
        raise ScenarioError("ascii map: rows must be non-empty and of equal length")
    terrain = np.zeros((len(rows), len(rows[0])), dtype=np.int8)
    survivors: list[Position] = []
    base: Position | None = None
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            if ch not in _ASCII_TERRAIN:
                raise ScenarioError(f"ascii map: unknown character {ch!r} at {(r, c)}")
            terrain[r, c] = _ASCII_TERRAIN[ch]
            if ch in "Ss":
                survivors.append((r, c))
            if ch == "B":
                base = (r, c)
    if base is None:
        raise ScenarioError("ascii map: no base 'B'")
    return Scenario(terrain=terrain, base=base, survivors=tuple(survivors))


def reachable_from_base(scenario: Scenario, layer: Layer) -> npt.NDArray[np.bool_]:
    """Cells reachable from the base by 4-connected moves through ``layer``-traversable cells."""
    passable = traversable_grid(layer, scenario.terrain)
    h, w = scenario.shape
    seen = np.zeros((h, w), dtype=np.bool_)
    seen[scenario.base] = True
    queue: deque[Position] = deque([scenario.base])
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return seen


def eligible_cells(scenario: Scenario, layers: Iterable[Layer]) -> npt.NDArray[np.bool_]:
    """Cells that are not ``STRUCTURE`` and are reachable from the base by any given layer.

    This is the coverage denominator from ADR-0009.
    """
    result = np.zeros(scenario.shape, dtype=np.bool_)
    for layer in sorted(set(layers)):
        result |= reachable_from_base(scenario, layer)
    return result

"""3D scene construction from a 2D episode (ADR-0027). NumPy-free and dependency-free.

The 2D simulator stays the authority on movement, sensing and communication. A scene lifts one
recorded episode into three dimensions:

- every structure and debris cell becomes a box (2 m and 0.4 m tall);
- hazards become flat markers;
- each robot becomes a track of ``(x, y, z)`` points, one per step.

Ground robots stay at ``GROUND_Z``. A drone is at ``BASE_Z`` while on the base cell and at
``FLIGHT_ALTITUDE`` elsewhere: it takes off when leaving the base and lands when returning. The
altitude profile is a kinematic convention, not a flight model. Coordinates: x = column, y = row,
z = up, one cell = 1 m.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

CELL: Final[float] = 1.0
STRUCTURE_HEIGHT: Final[float] = 2.0
DEBRIS_HEIGHT: Final[float] = 0.4
FLIGHT_ALTITUDE: Final[float] = 3.0
GROUND_Z: Final[float] = 0.25
BASE_Z: Final[float] = 0.15
TERRAIN_HEIGHT: Final[dict[int, float]] = {1: DEBRIS_HEIGHT, 2: STRUCTURE_HEIGHT}
TERRAIN_NAME: Final[dict[int, str]] = {1: "debris", 2: "structure"}


def agent_z(layer: str, cell: Sequence[int], base: Sequence[int]) -> float:
    """Altitude of an agent at ``cell`` (see module docstring)."""
    if layer != "air":
        return GROUND_Z
    return BASE_Z if list(cell) == list(base) else FLIGHT_ALTITUDE


def build_scene(scenario: Mapping[str, Any], frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A JSON-ready 3D scene from a scenario view and recorded frames (dashboard format)."""
    terrain, base = scenario["terrain"], scenario["base"]
    boxes = [
        {"x": c, "y": r, "h": TERRAIN_HEIGHT[code], "kind": TERRAIN_NAME[code]}
        for r, row in enumerate(terrain)
        for c, code in enumerate(row)
        if code in TERRAIN_HEIGHT
    ]
    hazards = [
        [c, r] for r, row in enumerate(scenario["hazard"]) for c, flag in enumerate(row) if flag
    ]
    tracks: dict[str, dict[str, Any]] = {}
    for k, frame in enumerate(frames):
        for a in frame["agents"]:
            track = tracks.setdefault(
                a["id"], {"layer": a["layer"], "points": [], "disabled_at": None}
            )
            r, c = a["pos"]
            track["points"].append([c, r, agent_z(a["layer"], a["pos"], base)])
            if not a["active"] and track["disabled_at"] is None:
                track["disabled_at"] = k
    return {
        "units": "1 cell = 1 m; x = column, y = row, z = up",
        "shape": scenario["shape"],
        "base": [base[1], base[0]],
        "boxes": boxes,
        "hazards": hazards,
        "survivors_ground_truth": [[c, r] for r, c in scenario["survivors_ground_truth"]],
        "tracks": tracks,
        "flight_altitude": FLIGHT_ALTITUDE,
    }

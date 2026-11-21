"""Packaged example scenarios for ``swarm demo`` and ``swarm run --config <name>``.

- ``basic``: the Phase 3 nominal scenario (20 x 20, 2 drones + 2 ground robots, 8 survivors,
  hazards, noisy detection, communication range 8 / 5 cells).
- ``advanced``: the same map with communication loss and delay, two scheduled robot failures,
  more hazards and noisy perception.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

NAMES: tuple[str, ...] = ("basic", "advanced")


def example_text(name: str) -> str:
    """The JSON text of a packaged example; raises ``KeyError`` for an unknown name."""
    if name not in NAMES:
        raise KeyError(f"unknown example {name!r}; examples: {', '.join(NAMES)}")
    return resources.files("swarm.examples").joinpath(f"{name}.json").read_text("utf-8")


def resolve_config_text(spec: str) -> str:
    """A config given as an example name or a path to a JSON file."""
    if spec in NAMES:
        return example_text(spec)
    path = Path(spec)
    if not path.is_file():
        raise FileNotFoundError(
            f"config {spec!r}: not an example ({', '.join(NAMES)}) and no such file"
        )
    return path.read_text(encoding="utf-8")

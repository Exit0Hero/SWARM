"""Dependency-free text rendering of an :class:`~swarm.core.env.EnvSnapshot` for debugging.

The renderer only reads a snapshot. It never touches the environment, so it cannot affect
simulation outcomes (docs/12). Survivor ground truth is hidden unless ``show_survivors=True``.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from swarm.core.env import EnvSnapshot
from swarm.core.types import AgentStatus, Layer, Terrain

TERRAIN_CHARS: Final[dict[int, str]] = {
    Terrain.OPEN: ".",
    Terrain.DEBRIS: "~",
    Terrain.STRUCTURE: "#",
    Terrain.BASE: "B",
}
LEGEND: Final[str] = (
    "legend: . open  ~ debris  # structure  B base  ! hazard  S survivor (ground truth)  "
    "A air agent  G ground agent  * air+ground  x disabled"
)


def render_snapshot_text(
    snapshot: EnvSnapshot, *, show_survivors: bool = False, legend: bool = True
) -> str:
    """Render a snapshot as a character grid with a status header.

    Precedence per cell: agents (``x`` if any agent there is disabled, ``*`` if both layers are
    present, otherwise ``A``/``G``), then survivors if shown, then hazards (``!``), then terrain.
    Agents on the base are drawn over the ``B``.
    """
    scenario = snapshot.scenario
    h, w = scenario.shape
    grid = [[TERRAIN_CHARS[int(scenario.terrain[r, c])] for c in range(w)] for r in range(h)]
    for r, c in zip(*np.nonzero(scenario.hazard_grid), strict=True):
        grid[int(r)][int(c)] = "!"
    if show_survivors:
        for r, c in scenario.survivors:
            grid[r][c] = "S"

    layers_at: dict[tuple[int, int], set[Layer]] = {}
    disabled_at: set[tuple[int, int]] = set()
    for agent in snapshot.agents:
        layers_at.setdefault(agent.position, set()).add(agent.layer)
        if agent.status is AgentStatus.DISABLED:
            disabled_at.add(agent.position)
    for (r, c), layers in layers_at.items():
        if (r, c) in disabled_at:
            grid[r][c] = "x"
        elif len(layers) == 2:
            grid[r][c] = "*"
        else:
            grid[r][c] = "A" if Layer.AIR in layers else "G"

    status = "done" if snapshot.done else "running"
    reason = f" ({snapshot.end_reason})" if snapshot.end_reason else ""
    lines = [f"step {snapshot.step} {status}{reason}"]
    lines.extend("".join(row) for row in grid)
    for agent in snapshot.agents:
        lines.append(
            f"{agent.agent_id:<10} {agent.layer.name.lower():<6} {agent.position!s:<9} "
            f"energy {agent.energy}/{agent.energy_capacity} {agent.status.value}"
        )
    if legend:
        lines.append(LEGEND)
    return "\n".join(lines) + "\n"

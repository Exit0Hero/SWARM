"""Episode metrics computed from the event stream (ADR-0005, ADR-0009).

The evaluator is the only component that combines events with ground truth (the
:class:`~swarm.scenarios.generator.Scenario`). Agent positions and footprints are *replayed* from
``moved`` and ``agent_disabled`` events, so metrics depend only on the recorded stream and can be
recomputed later from a stored event log.

This is the Phase 1 M1 subset. Precision, mission success, time to recall target, agent
utilization and energy efficiency arrive in M2 (docs/14_ROADMAP.md).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from swarm.core.events import (
    AgentDisabled,
    Detection,
    EnergySpent,
    Event,
    InvalidAction,
    MoveConflict,
    Moved,
)
from swarm.core.observation import footprint_offsets
from swarm.core.types import AgentId, Position
from swarm.scenarios.config import ScenarioConfig
from swarm.scenarios.generator import Scenario, eligible_cells

METRIC_SCHEMA_VERSION: Final[int] = 1


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    definition: str
    unit: str
    undefined_when: str | None = None


METRIC_DEFINITIONS: Final[dict[str, MetricDefinition]] = {
    "recall": MetricDefinition(
        "unique survivors with at least one true detection report / total survivors",
        "fraction in [0, 1]",
        "null when the scenario has zero survivors",
    ),
    "time_to_first_detection": MetricDefinition(
        "step index of the first true detection report",
        "simulation steps",
        "null when no survivor was detected",
    ),
    "coverage_sensed": MetricDefinition(
        "eligible cells ever inside an active agent's sensor footprint (including at reset) "
        "/ eligible cells; eligible = not STRUCTURE and reachable from the base by at least one "
        "agent type in the roster",
        "fraction in [0, 1]",
    ),
    "coverage_visited": MetricDefinition(
        "eligible cells ever occupied by any agent / eligible cells",
        "fraction in [0, 1]",
    ),
    "energy_used": MetricDefinition(
        "sum of energy spent by all agents; recharge is not subtracted",
        "energy units",
    ),
    "move_conflicts": MetricDefinition(
        "number of move_conflict events (vertex, swap or blocked moves; ADR-0003)",
        "count",
    ),
    "invalid_actions": MetricDefinition(
        "number of masked actions submitted by policies (executed as STAY; ADR-0004)",
        "count",
    ),
    "agents_disabled": MetricDefinition(
        "number of agents disabled by energy depletion during the episode",
        "count",
    ),
}


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """Metric values for one episode. ``None`` means undefined (see METRIC_DEFINITIONS)."""

    recall: float | None
    time_to_first_detection: int | None
    coverage_sensed: float
    coverage_visited: float
    energy_used: int
    move_conflicts: int
    invalid_actions: int
    agents_disabled: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_metrics(
    config: ScenarioConfig, scenario: Scenario, events: Iterable[Event]
) -> EpisodeMetrics:
    """Compute :class:`EpisodeMetrics` by replaying ``events`` over ``scenario``."""
    agents = config.agents()
    radius = {aid: spec.sensor_radius for aid, (spec, _) in agents.items()}
    eligible = eligible_cells(scenario, [spec.layer for spec, _ in agents.values()])
    n_eligible = int(eligible.sum())  # >= 1: the base itself is always eligible

    positions: dict[AgentId, Position] = dict.fromkeys(agents, scenario.base)
    active: set[AgentId] = set(agents)
    visited = np.zeros(scenario.shape, dtype=np.bool_)
    sensed = np.zeros(scenario.shape, dtype=np.bool_)
    visited[scenario.base] = True
    for aid in sorted(active):
        _mark_footprint(sensed, positions[aid], radius[aid])

    found: set[int] = set()
    first_detection: int | None = None
    energy_used = conflicts = invalid = disabled = 0
    current_step: int | None = None

    def sense_step() -> None:
        for aid in sorted(active):
            _mark_footprint(sensed, positions[aid], radius[aid])

    for event in events:
        if current_step is not None and event.step != current_step:
            sense_step()
        current_step = event.step
        if isinstance(event, Moved):
            positions[event.agent_id] = event.to_position
            visited[event.to_position] = True
        elif isinstance(event, AgentDisabled):
            active.discard(event.agent_id)
            disabled += 1
        elif isinstance(event, EnergySpent):
            energy_used += event.amount
        elif isinstance(event, MoveConflict):
            conflicts += 1
        elif isinstance(event, InvalidAction):
            invalid += 1
        elif isinstance(event, Detection) and event.true_positive:
            assert event.survivor_index is not None
            found.add(event.survivor_index)
            if first_detection is None:
                first_detection = event.step
    if current_step is not None:
        sense_step()

    total = len(scenario.survivors)
    return EpisodeMetrics(
        recall=None if total == 0 else len(found) / total,
        time_to_first_detection=first_detection,
        coverage_sensed=int((sensed & eligible).sum()) / n_eligible,
        coverage_visited=int((visited & eligible).sum()) / n_eligible,
        energy_used=energy_used,
        move_conflicts=conflicts,
        invalid_actions=invalid,
        agents_disabled=disabled,
    )


def _mark_footprint(grid: npt.NDArray[np.bool_], center: Position, radius: int) -> None:
    h, w = grid.shape
    for dr, dc in footprint_offsets(radius):
        r, c = center[0] + dr, center[1] + dc
        if 0 <= r < h and 0 <= c < w:
            grid[r, c] = True

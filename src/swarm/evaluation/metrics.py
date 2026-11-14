"""Episode metrics computed from the event stream (ADR-0005, ADR-0009 revision 2).

The evaluator is the only component that combines events with ground truth (the
:class:`~swarm.scenarios.generator.Scenario`). Agent positions, energy and footprints are
*replayed* step by step from the event stream, so metrics depend only on the recorded stream and
can be recomputed later from a stored event log.

Ground truth decides whether a report is true (ADR-0012 rule 4). The environment's
``true_positive`` labels and its ``mission_success`` claim are cross-checked, and a disagreement
raises :class:`EvaluationError` rather than being silently trusted.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from swarm.core.events import (
    AgentDisabled,
    Detection,
    EnergySpent,
    EpisodeEnd,
    Event,
    HazardExposure,
    InvalidAction,
    MoveConflict,
    Moved,
    Recharged,
)
from swarm.core.observation import footprint_offsets
from swarm.core.types import AgentId, Position
from swarm.scenarios.config import ScenarioConfig
from swarm.scenarios.generator import Scenario, eligible_cells

METRIC_SCHEMA_VERSION: Final[int] = 2


class EvaluationError(ValueError):
    """The event stream is inconsistent with the scenario ground truth or the config."""


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    definition: str
    unit: str
    undefined_when: str | None = None
    role: str = "primary"  # "primary" | "diagnostic"


METRIC_DEFINITIONS: Final[dict[str, MetricDefinition]] = {
    "recall": MetricDefinition(
        "|found survivors| / |all survivors|; found = at least one true report; unreachable "
        "survivors stay in the denominator",
        "fraction in [0, 1]",
        "zero survivors",
    ),
    "reachable_recall": MetricDefinition(
        "|found survivors on eligible cells| / |survivors on eligible cells|",
        "fraction in [0, 1]",
        "no survivor on an eligible cell",
        "diagnostic",
    ),
    "precision": MetricDefinition(
        "distinct reported cells containing a survivor / distinct reported cells (all agents, "
        "all steps; duplicates collapse to one location)",
        "fraction in [0, 1]",
        "no reports",
    ),
    "report_precision": MetricDefinition(
        "true reports / all reports (every step-agent-cell report counts)",
        "fraction in [0, 1]",
        "no reports",
        "diagnostic",
    ),
    "false_positive_reports": MetricDefinition(
        "number of reports of cells without a survivor", "count"
    ),
    "mission_success": MetricDefinition(
        "episode ended with end_reason mission_success (global recall reached recall_target)",
        "boolean",
        "zero survivors, or the stream has no episode_end",
    ),
    "time_to_first_detection": MetricDefinition(
        "step index of the first true report", "simulation steps", "no true report"
    ),
    "time_to_recall_target": MetricDefinition(
        "step index at which the episode ended in mission success",
        "simulation steps",
        "no mission success",
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
        "sum of action energy spent (energy_spent events); recharge not subtracted; hazard "
        "damage excluded",
        "energy units",
    ),
    "energy_efficiency": MetricDefinition(
        "|found survivors| * 1000 / energy_used",
        "survivors per 1000 energy units",
        "energy_used = 0",
    ),
    "hazard_exposures": MetricDefinition(
        "number of hazard_exposure events (agent-steps ending in a hazard)", "count"
    ),
    "hazard_energy_lost": MetricDefinition(
        "sum of energy_lost over hazard exposures", "energy units"
    ),
    "move_conflicts": MetricDefinition(
        "number of move_conflict events (vertex, swap or blocked moves; ADR-0003)", "count"
    ),
    "invalid_actions": MetricDefinition(
        "number of masked actions submitted by policies (executed as STAY; ADR-0004)", "count"
    ),
    "agent_utilization": MetricDefinition(
        "active agent-steps / (agents * episode steps); an agent-step is active if the agent is "
        "active at the end of the step and not idle on base (stayed on the base and started the "
        "step at full energy)",
        "fraction in [0, 1]",
        "the stream contains no steps",
    ),
    "agents_disabled": MetricDefinition(
        "distinct agents disabled during the episode (equals disablement events: an agent is "
        "disabled at most once)",
        "count",
        None,
        "diagnostic",
    ),
    "agents_disabled_energy": MetricDefinition(
        "agents disabled with reason energy_depleted", "count", None, "diagnostic"
    ),
    "agents_disabled_hazard": MetricDefinition(
        "agents disabled with reason hazard", "count", None, "diagnostic"
    ),
}


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """Metric values for one episode. ``None`` means undefined (see METRIC_DEFINITIONS)."""

    recall: float | None
    reachable_recall: float | None
    precision: float | None
    report_precision: float | None
    false_positive_reports: int
    mission_success: bool | None
    time_to_first_detection: int | None
    time_to_recall_target: int | None
    coverage_sensed: float
    coverage_visited: float
    energy_used: int
    energy_efficiency: float | None
    hazard_exposures: int
    hazard_energy_lost: int
    move_conflicts: int
    invalid_actions: int
    agent_utilization: float | None
    agents_disabled: int
    agents_disabled_energy: int
    agents_disabled_hazard: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_metrics(
    config: ScenarioConfig, scenario: Scenario, events: Iterable[Event]
) -> EpisodeMetrics:
    """Compute :class:`EpisodeMetrics` by replaying ``events`` over ``scenario``.

    Raises:
        EvaluationError: if a report's label or survivor index disagrees with ground truth, or
            the stream claims mission success that global recall does not support.
    """
    agents = config.agents()
    ids = sorted(agents)
    radius = {aid: spec.sensor_radius for aid, (spec, _) in agents.items()}
    capacity = {aid: spec.energy_capacity for aid, (spec, _) in agents.items()}
    energy = {aid: e for aid, (_, e) in agents.items()}
    eligible = eligible_cells(scenario, [spec.layer for spec, _ in agents.values()])
    n_eligible = int(eligible.sum())  # >= 1: the base itself is always eligible
    survivor_at = {pos: k for k, pos in enumerate(scenario.survivors)}

    by_step: dict[int, list[Event]] = defaultdict(list)
    for event in events:
        by_step[event.step].append(event)
    end = next((e for es in by_step.values() for e in es if isinstance(e, EpisodeEnd)), None)
    n_steps = end.step if end is not None else max(by_step, default=0)

    positions: dict[AgentId, Position] = dict.fromkeys(ids, scenario.base)
    active: set[AgentId] = set(ids)
    visited = np.zeros(scenario.shape, dtype=np.bool_)
    sensed = np.zeros(scenario.shape, dtype=np.bool_)
    visited[scenario.base] = True
    for aid in ids:
        _mark_footprint(sensed, positions[aid], radius[aid])

    found: set[int] = set()
    reported_cells: set[Position] = set()
    true_reports = false_reports = 0
    first_detection: int | None = None
    energy_used = conflicts = invalid = exposures = hazard_lost = 0
    disabled_reason: dict[AgentId, str] = {}
    active_agent_steps = 0

    for t in range(1, n_steps + 1):
        moved_now: set[AgentId] = set()
        full_at_start = {aid for aid in ids if energy[aid] == capacity[aid]}
        for event in by_step.get(t, ()):
            if isinstance(event, Moved):
                positions[event.agent_id] = event.to_position
                visited[event.to_position] = True
                moved_now.add(event.agent_id)
            elif isinstance(event, EnergySpent):
                energy_used += event.amount
                energy[event.agent_id] = event.remaining
            elif isinstance(event, Recharged):
                energy[event.agent_id] = event.remaining
            elif isinstance(event, HazardExposure):
                exposures += 1
                hazard_lost += event.energy_lost
                energy[event.agent_id] = event.remaining
            elif isinstance(event, AgentDisabled):
                if event.agent_id in disabled_reason:
                    raise EvaluationError(f"{event.agent_id} disabled twice (step {t})")
                active.discard(event.agent_id)
                disabled_reason[event.agent_id] = event.reason
            elif isinstance(event, MoveConflict):
                conflicts += 1
            elif isinstance(event, InvalidAction):
                invalid += 1
            elif isinstance(event, Detection):
                is_true = event.cell in survivor_at
                if is_true != event.true_positive:
                    raise EvaluationError(
                        f"detection at {event.cell} (step {t}) labelled "
                        f"true_positive={event.true_positive} but ground truth says {is_true}"
                    )
                reported_cells.add(event.cell)
                if is_true:
                    k = survivor_at[event.cell]
                    if event.survivor_index != k:
                        raise EvaluationError(
                            f"detection at {event.cell} has survivor_index "
                            f"{event.survivor_index}, expected {k}"
                        )
                    true_reports += 1
                    found.add(k)
                    if first_detection is None:
                        first_detection = t
                else:
                    false_reports += 1
        for aid in ids:
            if aid not in active:
                continue
            _mark_footprint(sensed, positions[aid], radius[aid])
            idle = positions[aid] == scenario.base and aid not in moved_now and aid in full_at_start
            active_agent_steps += 0 if idle else 1

    total = len(scenario.survivors)
    reachable = {k for k, pos in enumerate(scenario.survivors) if eligible[pos]}
    n_reports = true_reports + false_reports
    true_cells = sum(1 for cell in reported_cells if cell in survivor_at)
    success = _mission_success(config, end, total, len(found))

    return EpisodeMetrics(
        recall=None if total == 0 else len(found) / total,
        reachable_recall=None if not reachable else len(found & reachable) / len(reachable),
        precision=None if not reported_cells else true_cells / len(reported_cells),
        report_precision=None if n_reports == 0 else true_reports / n_reports,
        false_positive_reports=false_reports,
        mission_success=success,
        time_to_first_detection=first_detection,
        time_to_recall_target=end.step if success and end is not None else None,
        coverage_sensed=int((sensed & eligible).sum()) / n_eligible,
        coverage_visited=int((visited & eligible).sum()) / n_eligible,
        energy_used=energy_used,
        energy_efficiency=None if energy_used == 0 else len(found) * 1000 / energy_used,
        hazard_exposures=exposures,
        hazard_energy_lost=hazard_lost,
        move_conflicts=conflicts,
        invalid_actions=invalid,
        agent_utilization=None if n_steps == 0 else active_agent_steps / (len(ids) * n_steps),
        agents_disabled=len(disabled_reason),
        agents_disabled_energy=sum(r == "energy_depleted" for r in disabled_reason.values()),
        agents_disabled_hazard=sum(r == "hazard" for r in disabled_reason.values()),
    )


def _mission_success(
    config: ScenarioConfig, end: EpisodeEnd | None, total: int, found: int
) -> bool | None:
    """Mission success per ADR-0009, cross-checked against global recall."""
    if total == 0 or end is None:
        return None
    reached = Fraction(found, total) >= Fraction(str(config.recall_target))
    claimed = end.reason == "mission_success"
    if claimed and not reached:
        raise EvaluationError(
            f"episode claims mission_success but recall {found}/{total} < {config.recall_target}"
        )
    if reached and not claimed:
        raise EvaluationError(
            f"recall {found}/{total} reached {config.recall_target} but the episode ended with "
            f"{end.reason!r}"
        )
    return claimed


def _mark_footprint(grid: npt.NDArray[np.bool_], center: Position, radius: int) -> None:
    h, w = grid.shape
    for dr, dc in footprint_offsets(radius):
        r, c = center[0] + dr, center[1] + dc
        if 0 <= r < h and 0 <= c < w:
            grid[r, c] = True

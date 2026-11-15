"""Communication metrics, replayed from the event stream (ADR-0018).

Survivor-information propagation. For each survivor that is detected, the *origin* is its
earliest true detection (ties: lowest agent ID) at step ``t0``. Every other agent still active at
``t0`` forms a pair. The pair is *informed* at the first step at which that agent either detects
the survivor itself or receives a survivor-report record for its cell
(``survivor_report_received``). Its delay is that step minus ``t0`` (0 is possible: a
simultaneous own detection, or zero latency). Pairs never informed by the episode end, including
agents disabled before learning, are *unresolved*: they are counted, never given a delay.

These metrics exist only for scenarios with a communication channel; they are not part of the
ADR-0009 registry, so Phase 1 and 2 results are unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Final

from swarm.core.events import AgentDisabled, Detection, Event, SurvivorReportReceived
from swarm.core.types import AgentId, Position
from swarm.scenarios.config import ScenarioConfig
from swarm.scenarios.generator import Scenario


@dataclass(frozen=True, slots=True)
class CommMetricDefinition:
    definition: str
    unit: str
    undefined_when: str | None = None


COMM_METRIC_DEFINITIONS: Final[dict[str, CommMetricDefinition]] = {
    "survivor_info_pairs": CommMetricDefinition(
        "(detected survivor, other agent active at its first detection) pairs", "pairs"
    ),
    "survivor_info_informed": CommMetricDefinition(
        "pairs whose agent learned of the survivor (own detection or report) by the end", "pairs"
    ),
    "survivor_info_unresolved": CommMetricDefinition(
        "pairs whose agent never learned of the survivor by the end (no delay assigned)", "pairs"
    ),
    "survivor_info_informed_fraction": CommMetricDefinition(
        "informed / pairs", "fraction", "no pairs"
    ),
    "survivor_info_delay_mean": CommMetricDefinition(
        "mean steps from first detection to the agent learning of it, informed pairs only",
        "steps",
        "no informed pairs",
    ),
    "survivor_info_delay_max": CommMetricDefinition(
        "largest delay among informed pairs", "steps", "no informed pairs"
    ),
}


def compute_comm_metrics(
    config: ScenarioConfig, scenario: Scenario, events: Iterable[Event]
) -> dict[str, Any]:
    """Survivor-information propagation metrics for one episode (see module docstring)."""
    agents = sorted(config.agents())
    survivors = set(scenario.survivors)
    first_detection: dict[Position, tuple[int, AgentId]] = {}
    own_detection: dict[tuple[AgentId, Position], int] = {}
    reported: dict[tuple[AgentId, Position], int] = {}
    disabled_at: dict[AgentId, int] = {}
    for e in events:
        if isinstance(e, Detection) and e.true_positive and e.cell in survivors:
            key = (e.step, e.agent_id)
            if e.cell not in first_detection or key < first_detection[e.cell]:
                first_detection[e.cell] = key
            own_detection.setdefault((e.agent_id, e.cell), e.step)
        elif isinstance(e, SurvivorReportReceived):
            reported.setdefault((e.receiver, e.cell), e.step)
        elif isinstance(e, AgentDisabled):
            disabled_at.setdefault(e.agent_id, e.step)

    delays: list[int] = []
    unresolved = 0
    for cell, (t0, origin) in sorted(first_detection.items()):
        for agent in agents:
            if agent == origin or disabled_at.get(agent, t0 + 1) <= t0:
                continue
            learned = [
                s
                for s in (own_detection.get((agent, cell)), reported.get((agent, cell)))
                if s is not None
            ]
            if learned:
                delays.append(min(learned) - t0)
            else:
                unresolved += 1
    pairs = len(delays) + unresolved
    return {
        "survivor_info_pairs": pairs,
        "survivor_info_informed": len(delays),
        "survivor_info_unresolved": unresolved,
        "survivor_info_informed_fraction": len(delays) / pairs if pairs else None,
        "survivor_info_delay_mean": sum(delays) / len(delays) if delays else None,
        "survivor_info_delay_max": max(delays) if delays else None,
    }

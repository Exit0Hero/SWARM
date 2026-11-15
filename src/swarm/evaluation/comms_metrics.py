"""Communication metrics, replayed from the event stream (ADR-0018).

Survivor-information propagation. For each survivor that is detected, the *origin* is its
earliest true detection (ties: lowest agent ID) at step ``t0``. Every other agent still active at
``t0`` forms a pair. The pair is *informed* at the first step at which that agent either detects
the survivor itself or receives a survivor-report record for its cell
(``survivor_report_received``). Its delay is that step minus ``t0`` (0 is possible: a
simultaneous own detection, or zero latency). Pairs never informed by the episode end, including
agents disabled before learning, are *unresolved*: they are counted, never given a delay.

Failure detection and reassignment (ADR-0020). Simulator truth (``agent_disabled`` events) is
used here only, never by policies. It is joined with what agents *logged* (detector transitions
and sector holdings in the team's decision log). Pairs are (failed agent, observer still active
when it failed). Awareness that is never reached and abandoned sectors that are never reassigned
are counted as unresolved, never as successes. An observer already suspecting an agent when it
actually fails is aware at once (delay 0). Detection and reassignment metrics are ``None``
for policies without a failure detector.

These metrics exist only for scenarios with a communication channel; they are not part of the
ADR-0009 registry, so Phase 1 and 2 results are unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from swarm.core.events import (
    AgentDisabled,
    Detection,
    EnergySpent,
    EpisodeEnd,
    Event,
    SurvivorReportReceived,
)
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
    "failure_pairs": CommMetricDefinition(
        "(failed agent, observer active when it failed) pairs", "pairs", "no failure detector"
    ),
    "disabled_detection_delay_mean": CommMetricDefinition(
        "mean steps from a failure to the observer marking the agent disabled (resolved pairs)",
        "steps",
        "no resolved pairs",
    ),
    "disabled_detection_unresolved": CommMetricDefinition(
        "pairs whose observer never marked the failed agent disabled", "pairs", "no detector"
    ),
    "suspicion_detection_delay_mean": CommMetricDefinition(
        "mean steps from a failure to the observer first suspecting it or marking it disabled",
        "steps",
        "no resolved pairs",
    ),
    "suspicion_detection_unresolved": CommMetricDefinition(
        "pairs whose observer never suspected nor marked the failed agent", "pairs", "no detector"
    ),
    "false_suspicions": CommMetricDefinition(
        "suspicions of an agent that was actually active at that step", "count", "no detector"
    ),
    "recoveries": CommMetricDefinition(
        "suspected peers later heard again (suspected -> active or disconnected)",
        "count",
        "no detector",
    ),
    "abandoned_sectors": CommMetricDefinition(
        "sectors held by an agent when it failed", "sectors", "no detector"
    ),
    "reassignment_delay_mean": CommMetricDefinition(
        "mean steps from a failure until another active agent holds its abandoned sector",
        "steps",
        "no sector reassigned",
    ),
    "unresolved_sectors": CommMetricDefinition(
        "abandoned sectors never held by another active agent by the episode end",
        "sectors",
        "no detector",
    ),
    "duplicate_ownership_steps": CommMetricDefinition(
        "steps on which some sector is held by two or more active agents", "steps", "no detector"
    ),
    "duplicate_ownership_sector_steps": CommMetricDefinition(
        "sum over steps of extra active holders of each sector", "sector-steps", "no detector"
    ),
    "failed_agent_steps_lost": CommMetricDefinition(
        "sum over failed agents of steps from failure to the episode end", "agent-steps"
    ),
    "failed_agent_energy_spent": CommMetricDefinition(
        "energy spent moving and waiting by agents that later failed (search effort lost)",
        "energy",
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


def compute_failure_metrics(
    config: ScenarioConfig, events: Iterable[Event], decision_log: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Failure-detection and reassignment metrics (see module docstring)."""
    agents = sorted(config.agents())
    failed: dict[AgentId, int] = {}
    spent: dict[AgentId, int] = {}
    end = 0
    for e in events:
        end = max(end, e.step)
        if isinstance(e, AgentDisabled):
            failed.setdefault(e.agent_id, e.step)
        elif isinstance(e, EnergySpent):
            spent[e.agent_id] = spent.get(e.agent_id, 0) + e.amount
        elif isinstance(e, EpisodeEnd):
            end = e.step
    out: dict[str, Any] = {
        "failed_agent_steps_lost": sum(end - t for t in failed.values()),
        "failed_agent_energy_spent": sum(spent.get(a, 0) for a in failed),
    }
    log = list(decision_log)
    detector_names = [k for k in COMM_METRIC_DEFINITIONS if k not in out]
    detector_names = detector_names[detector_names.index("failure_pairs") :]
    if not any(entry.get("kind") == "detector" for entry in log):
        return out | dict.fromkeys(detector_names)

    def alive(agent: AgentId, step: int) -> bool:
        return agent not in failed or step < failed[agent]

    # Detection: first observer marks per (observer, peer).
    first_disabled: dict[tuple[AgentId, AgentId], int] = {}
    history: dict[tuple[AgentId, AgentId], list[tuple[int, str]]] = {}
    previous: dict[tuple[AgentId, AgentId], str] = {}
    false_suspicions = recoveries = 0
    holds: list[tuple[int, AgentId, int | None]] = []
    for entry in log:
        if entry.get("kind") == "peer_state":
            key = (entry["observer"], entry["peer"])
            step, state = int(entry["step"]), str(entry["state"])
            if state == "disabled":
                first_disabled.setdefault(key, step)
            peer = entry["peer"]
            history.setdefault(key, []).append((step, state))
            if state == "suspected" and alive(peer, step):
                false_suspicions += 1
            if previous.get(key) == "suspected" and state in ("active", "disconnected"):
                recoveries += 1
            previous[key] = state
        elif entry.get("kind") == "hold":
            holds.append((int(entry["step"]), entry["agent"], entry["sector"]))

    pairs = [(d, o) for d, t in sorted(failed.items()) for o in agents if o != d and alive(o, t)]
    det = [first_disabled[(o, d)] - failed[d] for d, o in pairs if (o, d) in first_disabled]
    aware = [
        delay
        for d, o in pairs
        if (delay := _awareness_delay(history.get((o, d), []), failed[d])) is not None
    ]

    # Holdings over time (as logged by agents).
    holds.sort(key=lambda h: (h[0], h[1]))
    held: dict[AgentId, int | None] = dict.fromkeys(agents)
    timeline: list[dict[AgentId, int | None]] = []
    i = 0
    for step in range(end + 1):
        while i < len(holds) and holds[i][0] <= step:
            held[holds[i][1]] = holds[i][2]
            i += 1
        timeline.append(dict(held))
    dup_steps = dup_sector_steps = 0
    for step, snapshot in enumerate(timeline):
        counts: dict[int, int] = {}
        for agent, sector in snapshot.items():
            if sector is not None and alive(agent, step):
                counts[sector] = counts.get(sector, 0) + 1
        extra = sum(n - 1 for n in counts.values() if n > 1)
        dup_steps += int(extra > 0)
        dup_sector_steps += extra

    # Abandoned sectors: held by an agent when it failed; reassigned when another active
    # agent holds it at or after the failure.
    delays: list[int] = []
    abandoned = unresolved = 0
    for d, t in sorted(failed.items()):
        sector = timeline[min(t, end)][d]
        if sector is None:
            continue
        abandoned += 1
        hit = next(
            (
                s
                for s in range(t, end + 1)
                if any(a != d and alive(a, s) and timeline[s][a] == sector for a in agents)
            ),
            None,
        )
        if hit is None:
            unresolved += 1
        else:
            delays.append(hit - t)
    return out | {
        "failure_pairs": len(pairs),
        "disabled_detection_delay_mean": sum(det) / len(det) if det else None,
        "disabled_detection_unresolved": len(pairs) - len(det),
        "suspicion_detection_delay_mean": sum(aware) / len(aware) if aware else None,
        "suspicion_detection_unresolved": len(pairs) - len(aware),
        "false_suspicions": false_suspicions,
        "recoveries": recoveries,
        "abandoned_sectors": abandoned,
        "reassignment_delay_mean": sum(delays) / len(delays) if delays else None,
        "unresolved_sectors": unresolved,
        "duplicate_ownership_steps": dup_steps,
        "duplicate_ownership_sector_steps": dup_sector_steps,
    }


def _awareness_delay(transitions: list[tuple[int, str]], failed_at: int) -> int | None:
    """Steps from a failure until the observer suspects or marks the agent: 0 if its logged state
    at the failure step is already ``suspected`` or ``disabled``; ``None`` if never."""
    state = "active"
    for step, new in sorted(transitions):
        if step > failed_at:
            break
        state = new
    if state in ("suspected", "disabled"):
        return 0
    for step, new in sorted(transitions):
        if step > failed_at and new in ("suspected", "disabled"):
            return step - failed_at
    return None

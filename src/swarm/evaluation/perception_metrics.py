"""Perception metrics (ADR-0025): evaluator-only counts of corrupted readings.

Merged into an episode's optional ``comm_metrics`` block when a perception model is configured,
so episodes without one keep their exact result layout.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Final

from swarm.core.events import Detection, Event, PerceptionError

PERCEPTION_METRIC_DEFINITIONS: Final[dict[str, str]] = {
    "perception_terrain_errors": "footprint cells whose terrain type was misread",
    "perception_hazards_missed": "hazard cells reported safe",
    "perception_hazards_false": "safe cells reported hazardous",
    "perception_true_reports": "survivor reports that were true (detection events)",
    "perception_false_reports": "survivor reports that were false alarms",
}


def compute_perception_metrics(events: Iterable[Event]) -> dict[str, Any]:
    counts = dict.fromkeys(PERCEPTION_METRIC_DEFINITIONS, 0)
    key = {
        "terrain": "perception_terrain_errors",
        "hazard_missed": "perception_hazards_missed",
        "hazard_false": "perception_hazards_false",
    }
    for e in events:
        if isinstance(e, PerceptionError):
            counts[key[e.error]] += 1
        elif isinstance(e, Detection):
            counts[
                "perception_true_reports" if e.true_positive else "perception_false_reports"
            ] += 1
    return counts

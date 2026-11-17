"""Event-based team reward for MARL training (ADR-0005, ADR-0022). Training-only.

The core emits events and no reward. :class:`TeamReward` turns each primitive step's events and
observations into separately logged components. All agents share their sum.

Components (default weight; the exploit each could invite; the metric that would reveal it):

- ``survivor`` (+1.0 per survivor first detected, true positive, by anyone). Duplicate or false
  detections earn nothing. Revealed by recall and precision.
- ``coverage`` (+0.01 per footprint cell the team senses for the first time). Could reward
  sweeping open area instead of finishing the search. Revealed by recall versus coverage_sensed.
- ``time`` (-0.05 per primitive step; at -0.01 the speed signal was negligible next to the
  detection terms in development runs on seeds 1000-1019). Could reward ending episodes early
  by losing agents. Revealed by agents_disabled.
- ``disabled`` (-1.0 per agent disabled by energy or a hazard; injected failures are excluded).
  The counterweight to ``time``.
- ``hazard`` (-0.1 per hazard exposure event).
- ``success`` (+2.0 on mission success).

``survivor`` reads the evaluator-only ``true_positive`` flag in detection events. This is
privileged information used **only as a training signal** (CTDE). Actors never see it: their
features come from :mod:`swarm.learning.sector`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Final

from swarm.core.events import AgentDisabled, Detection, EpisodeEnd, Event, HazardExposure
from swarm.core.observation import Observation
from swarm.core.types import AgentId, Position

COMPONENTS: Final[tuple[str, ...]] = (
    "survivor",
    "coverage",
    "time",
    "disabled",
    "hazard",
    "success",
)
DEFAULT_WEIGHTS: Final[dict[str, float]] = {
    "survivor": 1.0,
    "coverage": 0.01,
    "time": -0.05,
    "disabled": -1.0,
    "hazard": -0.1,
    "success": 2.0,
}


@dataclass(slots=True)
class TeamReward:
    """Stateful per-episode reward; call :meth:`reset` then :meth:`step` once per primitive step."""

    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    _found: set[int] = field(default_factory=set)
    _sensed: set[Position] = field(default_factory=set)
    totals: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = sorted(set(self.weights) - set(COMPONENTS))
        if unknown:
            raise ValueError(f"TeamReward: unknown components {unknown}; known: {list(COMPONENTS)}")
        self.reset(())

    def reset(self, observations: Iterable[Observation]) -> None:
        self._found = set()
        self._sensed = set()
        self.totals = dict.fromkeys(COMPONENTS, 0.0)
        self._sense(observations)

    def _sense(self, observations: Iterable[Observation]) -> int:
        new = 0
        for obs in observations:
            r0, c0 = obs.position
            rad = obs.sensor_radius
            h, w = obs.grid_shape
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    r, c = r0 + dr, c0 + dc
                    if obs.local_terrain[dr + rad, dc + rad] < 0 or not (0 <= r < h and 0 <= c < w):
                        continue
                    if (r, c) not in self._sensed:
                        self._sensed.add((r, c))
                        new += 1
        return new

    def step(
        self, events: Iterable[Event], observations: Mapping[AgentId, Observation]
    ) -> dict[str, float]:
        """Weighted components for one primitive step (added to :attr:`totals`)."""
        counts = dict.fromkeys(COMPONENTS, 0.0)
        counts["time"] = 1.0
        for e in events:
            if isinstance(e, Detection) and e.true_positive and e.survivor_index is not None:
                if e.survivor_index not in self._found:
                    self._found.add(e.survivor_index)
                    counts["survivor"] += 1
            elif isinstance(e, AgentDisabled) and e.reason != "injected":
                counts["disabled"] += 1
            elif isinstance(e, HazardExposure):
                counts["hazard"] += 1
            elif isinstance(e, EpisodeEnd) and e.reason == "mission_success":
                counts["success"] += 1
        counts["coverage"] = float(self._sense(observations.values()))
        out = {k: self.weights.get(k, 0.0) * v for k, v in counts.items()}
        for k, v in out.items():
            self.totals[k] += v
        return out

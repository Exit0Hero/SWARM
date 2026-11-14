"""Decentralized navigation baselines: ``astar_waypoint`` and ``frontier`` (ADR-0016).

Each agent keeps its own :class:`~swarm.navigation.known_map.KnownMap` built only from its own
observations, plans with A* on known cells (unknown cells impassable), treats visible same-layer
agents as obstacles for the next move, and follows the energy commit rule. No information is
shared between agents.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from swarm.core.observation import Observation
from swarm.core.oracle import OracleView
from swarm.core.types import ACTION_DELTAS, Action, AgentId, AgentStatus, Position
from swarm.navigation.astar import UNREACHABLE, Path, astar, cost_field
from swarm.navigation.energy import EnergyModel
from swarm.navigation.known_map import KnownMap
from swarm.policies.base import DECENTRALIZED, Diagnostics, MissionBrief, TeamFactory
from swarm.policies.planning import (
    BLACKLIST_STEPS,
    Counters,
    FieldCache,
    block_cells,
    feasible_mask,
    planning_costs,
)
from swarm.scenarios.config import AgentTypeSpec

_DELTA_TO_ACTION = {delta: action for action, delta in ACTION_DELTAS.items()}
BACKOFF_PROBABILITY = 0.5
"""Probability of waiting after a planned move did not happen (ADR-0016 rule 12)."""
_MAX_CANDIDATES = 12
"""Candidates whose actual A* path is checked against the commit rule per selection."""


def step_action(here: Position, nxt: Position) -> Action:
    return _DELTA_TO_ACTION[(nxt[0] - here[0], nxt[1] - here[1])]


class NavigatorAgent:
    """One agent: own known map, target, path and the energy commit rule."""

    def __init__(
        self,
        agent_id: AgentId,
        spec: AgentTypeSpec,
        brief: MissionBrief,
        *,
        target_rule: str,
        hazard_aware: bool,
        counters: Counters,
    ) -> None:
        if target_rule not in ("frontier", "waypoint"):
            raise ValueError(f"target_rule: {target_rule!r}")
        self.agent_id = agent_id
        self.layer = spec.layer
        self.brief = brief
        self.target_rule = target_rule
        self.hazard_aware = hazard_aware
        self.model = EnergyModel.for_agent(spec, brief.hazard_layers, brief.hazard_energy_cost)
        self.known = KnownMap(brief.grid_shape, brief.base)
        self.cache = FieldCache()
        self.counters = counters
        self.target: Position | None = None
        self.target_since = 0
        self.path: Path = []
        self.returning = False
        self.blacklist: dict[Position, int] = {}
        self.expected: Position | None = None

    # ----------------------------------------------------------------- helpers

    def _costs(self) -> np.ndarray[Any, np.dtype[np.int64]] | None:
        return planning_costs(self.known, self.layer, self.brief.hazard_layers, self.hazard_aware)

    def _plan(
        self, passable: np.ndarray[Any, np.dtype[np.bool_]], here: Position, goal: Position
    ) -> Path | None:
        return astar(passable, here, goal, self._costs())

    def _target_valid(self) -> bool:
        assert self.target is not None
        if self.target_rule == "frontier":
            return bool(self.cache.frontier_for(self.known, self.layer)[self.target])
        return True

    def _commit_ok(
        self, path: Path, energy: int, to_base: np.ndarray[Any, np.dtype[np.int64]]
    ) -> bool:
        enter = self.model.enter_energy(self.known)
        return self.model.can_commit(
            energy, self.model.route_energy(path, enter), int(to_base[path[-1]])
        )

    def _select_target(
        self,
        obs: Observation,
        rng: np.random.Generator,
        passable: np.ndarray[Any, np.dtype[np.bool_]],
        to_base: np.ndarray[Any, np.dtype[np.int64]],
    ) -> tuple[Position, Path] | None:
        here = obs.position
        feasible = feasible_mask(self.model, obs.energy, passable, here, self.known, to_base)
        if self.target_rule == "frontier":
            candidates = feasible & self.cache.frontier_for(self.known, self.layer)
        else:
            candidates = feasible & self.cache.passable_for(self.known, self.layer)
        candidates[here] = False
        for cell, until in list(self.blacklist.items()):
            if until <= obs.step:
                del self.blacklist[cell]
            else:
                candidates[cell] = False
        cells = [(int(r), int(c)) for r, c in np.argwhere(candidates)]
        if not cells:
            return None
        if self.target_rule == "frontier":
            dist = cost_field(passable, here, self._costs())
            cells = [c for c in cells if dist[c] != UNREACHABLE]
            cells.sort(key=lambda c: (int(dist[c]), c[0], c[1]))
        else:
            cells = [cells[int(rng.integers(len(cells)))]]
        for cell in cells[:_MAX_CANDIDATES]:
            path = self._plan(passable, here, cell)
            if path is not None and self._commit_ok(path, obs.energy, to_base):
                return cell, path
            self.blacklist[cell] = obs.step + BLACKLIST_STEPS
        return None

    # ----------------------------------------------------------------- decision

    def act(self, obs: Observation, rng: np.random.Generator) -> int:
        action = self._decide(obs, rng)
        if action is not Action.STAY:
            dr, dc = ACTION_DELTAS[action]
            self.expected = (obs.position[0] + dr, obs.position[1] + dc)
        else:
            self.expected = None
        return int(action)

    def _decide(self, obs: Observation, rng: np.random.Generator) -> Action:
        self.known.update(obs)
        here = obs.position
        base = self.brief.base
        # Randomized backoff (ADR-0016 rule 12): identical decentralized agents that collided
        # would otherwise repeat the same blocked move forever. One draw per detected failure.
        if self.expected is not None and here != self.expected:
            self.counters.move_failures += 1
            if rng.random() < BACKOFF_PROBABILITY:
                self.counters.backoff_waits += 1
                return Action.STAY
        static = self.cache.passable_for(self.known, self.layer)
        visible = [
            v.position for v in obs.visible_agents if v.layer == self.layer and v.position != here
        ]
        passable = block_cells(static, visible, base)
        to_base = self.cache.to_base_for(self.known, self.model)

        # Target bookkeeping: reached, exhausted, or no longer energetically safe.
        if self.target is not None and not self.returning:
            if here == self.target:
                self.counters.targets_reached += 1
                self.counters.completion_times.append(obs.step - self.target_since)
                self.target, self.path = None, []
            elif not self._target_valid():
                self.counters.target_invalidations += 1
                self.target, self.path = None, []
        if self.returning and here == base:
            self.returning = False

        if self.target is not None and not self.returning:
            path = self._plan(passable, here, self.target)
            if path is None:
                self.counters.failed_plans += 1
                self.counters.target_invalidations += 1
                self.blacklist[self.target] = obs.step + BLACKLIST_STEPS
                self.target, self.path = None, []
            elif not self._commit_ok(path, obs.energy, to_base):
                self.target, self.path = None, []
                self.returning = here != base
                self.counters.returns_to_base += int(self.returning)
            else:
                self.path = path

        if self.target is None and not self.returning:
            choice = self._select_target(obs, rng, passable, to_base)
            if choice is not None:
                self.target, self.path = choice
                self.target_since = obs.step
                self.counters.targets_selected += 1
            elif here != base:
                self.returning = True
                self.counters.returns_to_base += 1

        if self.returning:
            enter = self.model.enter_energy(self.known)
            path = astar(passable, here, base, enter) or astar(static, here, base, enter)
            if path is None:
                self.counters.failed_plans += 1
                return Action.STAY
            self.path = path

        if len(self.path) < 2:
            return Action.STAY
        nxt = self.path[1]
        if not passable[nxt]:
            self.counters.planned_waits += 1
            return Action.STAY
        return step_action(here, nxt)


class DecentralizedTeam:
    """Runs one :class:`NavigatorAgent` per agent with no information sharing."""

    information = DECENTRALIZED
    uses_oracle = False
    version = "1"

    def __init__(
        self, name: str, brief: MissionBrief, *, target_rule: str, hazard_aware: bool
    ) -> None:
        self.name = name
        self.counters = Counters()
        self.agents = {
            aid: NavigatorAgent(
                aid,
                spec,
                brief,
                target_rule=target_rule,
                hazard_aware=hazard_aware,
                counters=self.counters,
            )
            for aid, spec in brief.agents.items()
        }

    def act(
        self,
        observations: Mapping[AgentId, Observation],
        rngs: Mapping[AgentId, np.random.Generator],
    ) -> dict[AgentId, int]:
        return {
            aid: self.agents[aid].act(obs, rngs[aid])
            for aid, obs in sorted(observations.items())
            if obs.status is AgentStatus.ACTIVE
        }

    def attach_oracle(self, view: OracleView) -> None:
        raise RuntimeError(f"{self.name} is decentralized and never uses the oracle")

    def diagnostics(self) -> Diagnostics:
        return self.counters.as_dict()

    def decision_log(self) -> list[dict[str, Any]]:
        return []


def _team(name: str, target_rule: str, hazard_aware: bool) -> TeamFactory:
    return TeamFactory(
        name,
        lambda brief: DecentralizedTeam(
            name, brief, target_rule=target_rule, hazard_aware=hazard_aware
        ),
    )


ASTAR_WAYPOINT = _team("astar_waypoint", "waypoint", hazard_aware=False)
FRONTIER = _team("frontier", "frontier", hazard_aware=False)
FRONTIER_HAZARD_AWARE = _team("frontier_hazard_aware", "frontier", hazard_aware=True)

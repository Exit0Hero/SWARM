"""Centralized baselines: ``prioritized_frontier``, ``assignment`` and ``assignment_oracle``.

Energy feasibility (ADR-0016 rules 4 and 16) is checked on the route the planner actually executes:
allocation screens targets with the energy of a planning-cost-optimal route, and after every
planning round each exploring agent's planned timed path (moves, waits and known-hazard exposure)
plus the minimum energy back to base must satisfy the commit rule, otherwise the agent returns.
Returning agents plan on energy costs, matching the energy-to-base field.

One planner sees every agent's observations (``centralized_pooled``: perfect communication is
assumed, no hidden state), or additionally the ground truth through an ``OracleView``
(``oracle``). Targets are chosen greedily in priority order or by minimum-cost assignment, and all
paths are planned jointly with prioritized space-time planning (ADR-0014 to ADR-0016).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from swarm.coordination.assignment import min_cost_assignment
from swarm.core.observation import Observation, footprint_offsets
from swarm.core.oracle import OracleView
from swarm.core.types import AgentId, AgentStatus, Layer, Position
from swarm.navigation.astar import UNREACHABLE, cost_field_with_secondary
from swarm.navigation.energy import EnergyModel
from swarm.navigation.known_map import KnownMap
from swarm.navigation.prioritized import (
    PlanRequest,
    PlanRound,
    default_horizon,
    first_action,
    plan_prioritized,
    priority_order,
)
from swarm.policies.base import (
    CENTRALIZED_POOLED,
    ORACLE,
    Diagnostics,
    MissionBrief,
    TeamFactory,
)
from swarm.policies.planning import Counters, FieldCache, planning_costs


@dataclass
class _AgentState:
    model: EnergyModel
    position: Position
    energy: int
    target: Position | None = None
    task: Position | None = None  # frontier cell, or the survivor cell for the oracle
    target_since: int = 0
    returning: bool = False


class PooledPlannerTeam:
    """Centralized team planner. ``allocation`` is ``"greedy"`` or ``"hungarian"``."""

    version = "1"

    def __init__(
        self,
        name: str,
        brief: MissionBrief,
        *,
        allocation: str,
        hazard_aware: bool,
        oracle: bool,
    ) -> None:
        if allocation not in ("greedy", "hungarian"):
            raise ValueError(f"allocation: {allocation!r}")
        self.name = name
        self.brief = brief
        self.allocation = allocation
        self.hazard_aware = hazard_aware
        self.uses_oracle = oracle
        self.information = ORACLE if oracle else CENTRALIZED_POOLED
        self.known = KnownMap(brief.grid_shape, brief.base)
        self.cache = FieldCache()
        self.counters = Counters()
        self.log: list[dict[str, Any]] = []
        self.states = {
            aid: _AgentState(
                EnergyModel.for_agent(spec, brief.hazard_layers, brief.hazard_energy_cost),
                brief.base,
                spec.energy_capacity,
            )
            for aid, spec in brief.agents.items()
        }
        self.disabled: dict[AgentId, Position] = {}
        self.survivors: tuple[Position, ...] = ()
        self.reported: set[Position] = set()
        self.horizon = default_horizon(brief.grid_shape)
        self._active_last: frozenset[AgentId] = frozenset()
        self.last_round: PlanRound | None = None
        """The final planning round of the latest step (for diagnostics and tests)."""

    # ----------------------------------------------------------------- oracle

    def attach_oracle(self, view: OracleView) -> None:
        if not self.uses_oracle:
            raise RuntimeError(f"{self.name} is not an oracle policy")
        self.known = KnownMap.from_ground_truth(view.terrain, view.hazards, view.base)
        self.survivors = view.survivors

    # ----------------------------------------------------------------- helpers

    def _passable(self, layer: Layer) -> npt.NDArray[np.bool_]:
        """Known passable cells; disabled agents are blocked in the known map itself, so the
        energy-to-base field treats them as obstacles too (ADR-0016 rule 15)."""
        return self.cache.passable_for(self.known, layer)

    def _route_fields(self, aid: AgentId) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Planning cost from the agent to every cell, and the energy of that planning-optimal
        route (minimum energy among equally cheap routes)."""
        st = self.states[aid]
        layer = self.brief.agents[aid].layer
        return cost_field_with_secondary(
            self._passable(layer),
            st.position,
            self._costs(layer),
            st.model.enter_energy(self.known),
        )

    def _plan_feasible(self, aid: AgentId, path: list[Position]) -> bool:
        """Commit rule on the planned timed path itself (ADR-0016 rule 16)."""
        st = self.states[aid]
        route = st.model.route_energy(path, st.model.enter_energy(self.known))
        to_base = self.cache.to_base_for(self.known, st.model)
        return st.model.can_commit(st.energy, route, int(to_base[path[-1]]))

    def _costs(self, layer: Layer) -> npt.NDArray[np.int64] | None:
        return planning_costs(self.known, layer, self.brief.hazard_layers, self.hazard_aware)

    def _tasks(self) -> list[Position]:
        """Task cells: frontier cells of any layer, or unreported survivors (oracle)."""
        if self.uses_oracle:
            return [s for s in self.survivors if s not in self.reported]
        layers = {self.brief.agents[a].layer for a in self.states if a not in self.disabled}
        mask = np.zeros(self.brief.grid_shape, dtype=np.bool_)
        for layer in sorted(layers):
            mask |= self.cache.frontier_for(self.known, layer)
        return [(int(r), int(c)) for r, c in np.argwhere(mask)]

    def _task_valid(self, aid: AgentId, task: Position) -> bool:
        if self.uses_oracle:
            return task not in self.reported
        return bool(self.cache.frontier_for(self.known, self.brief.agents[aid].layer)[task])

    def _target_for(
        self, aid: AgentId, task: Position, dist: npt.NDArray[np.int64]
    ) -> tuple[Position, int] | None:
        """Where agent ``aid`` should go for ``task`` and the planning cost to get there."""
        if not self.uses_oracle:
            if dist[task] == UNREACHABLE:
                return None
            return task, int(dist[task])
        radius = self.brief.agents[aid].sensor_radius
        h, w = self.brief.grid_shape
        best: tuple[int, int, int] | None = None
        for dr, dc in footprint_offsets(radius):
            r, c = task[0] + dr, task[1] + dc
            if 0 <= r < h and 0 <= c < w and dist[r, c] != UNREACHABLE:
                key = (int(dist[r, c]), r, c)
                if best is None or key < best:
                    best = key
        return None if best is None else ((best[1], best[2]), best[0])

    def _feasible(
        self, aid: AgentId, target: Position, energy_field: npt.NDArray[np.int64]
    ) -> bool:
        st = self.states[aid]
        to_base = self.cache.to_base_for(self.known, st.model)
        if energy_field[target] == UNREACHABLE:
            return False
        return st.model.can_commit(st.energy, int(energy_field[target]), int(to_base[target]))

    def _log(
        self, step: int, aid: AgentId, target: Position | None, cost: int | None, trigger: str
    ) -> None:
        self.log.append(
            {
                "step": step,
                "agent": aid,
                "target": None if target is None else list(target),
                "cost": cost,
                "trigger": trigger,
            }
        )

    # ----------------------------------------------------------------- allocation

    def _allocate(self, step: int, agents: list[AgentId], trigger: str) -> None:
        tasks = self._tasks()
        options: dict[tuple[AgentId, int], tuple[Position, int]] = {}
        for aid in agents:
            dist, energy_field = self._route_fields(aid)
            for j, task in enumerate(tasks):
                choice = self._target_for(aid, task, dist)
                if choice is not None and self._feasible(aid, choice[0], energy_field):
                    options[(aid, j)] = choice

        chosen: dict[AgentId, tuple[int, Position, int]] = {}
        if self.allocation == "greedy":
            claimed: set[int] = set()
            for aid in priority_order(agents, step):
                best = min(
                    (
                        (cost, tasks[j], j)
                        for (a, j), (_, cost) in options.items()
                        if a == aid and j not in claimed
                    ),
                    default=None,
                )
                if best is not None:
                    claimed.add(best[2])
                    chosen[aid] = (best[2], *options[(aid, best[2])])
        else:
            ordered = sorted(agents)
            cost = np.zeros((len(ordered), len(tasks)), dtype=np.int64)
            feas = np.zeros((len(ordered), len(tasks)), dtype=np.bool_)
            for i, aid in enumerate(ordered):
                for j in range(len(tasks)):
                    if (aid, j) in options:
                        cost[i, j] = options[(aid, j)][1]
                        feas[i, j] = True
            for i, j in min_cost_assignment(cost, feas):
                chosen[ordered[i]] = (j, *options[(ordered[i], j)])

        for aid in agents:
            st = self.states[aid]
            if aid in chosen:
                j, target, cost_value = chosen[aid]
                if (st.task, st.target) != (tasks[j], target):
                    if st.task is not None:
                        self.counters.reassignments += 1
                    st.task, st.target, st.target_since = tasks[j], target, step
                    self.counters.targets_selected += 1
                    self._log(step, aid, target, cost_value, trigger)
            else:
                if st.task is not None or st.target is not None:
                    self._log(step, aid, None, None, f"{trigger}:no_feasible_task")
                st.task = st.target = None
                if st.position != self.brief.base and not st.returning:
                    st.returning = True
                    self.counters.returns_to_base += 1

    # ----------------------------------------------------------------- decision

    def act(
        self,
        observations: Mapping[AgentId, Observation],
        rngs: Mapping[AgentId, np.random.Generator],
    ) -> dict[AgentId, int]:
        step = next(iter(observations.values())).step if observations else 0
        active: list[AgentId] = []
        for aid, obs in sorted(observations.items()):
            st = self.states[aid]
            st.position, st.energy = obs.position, obs.energy
            if not self.uses_oracle:
                self.known.update(obs)
            self.reported.update(obs.detections)
            if obs.status is AgentStatus.ACTIVE:
                active.append(aid)
            elif aid not in self.disabled:
                self.disabled[aid] = obs.position
                self.known.mark_blocked(self.brief.agents[aid].layer, obs.position)
        base = self.brief.base

        triggers: list[str] = []
        if frozenset(active) != self._active_last:
            triggers.append("agent_set_changed")
            self._active_last = frozenset(active)
        for aid in active:
            st = self.states[aid]
            if st.returning and st.position == base:
                st.returning = False
            if st.target is None or st.task is None:
                continue
            if not self._task_valid(aid, st.task):
                self.counters.target_invalidations += 1
                if st.position == st.target:
                    self.counters.targets_reached += 1
                    self.counters.completion_times.append(step - st.target_since)
                st.task = st.target = None
                triggers.append("task_done_or_invalid")
            elif st.position == st.target and not self.uses_oracle:
                self.counters.targets_reached += 1
                self.counters.completion_times.append(step - st.target_since)
                st.task = st.target = None
                triggers.append("target_reached")
            else:
                _, energy_field = self._route_fields(aid)
                if not self._feasible(aid, st.target, energy_field):
                    st.task = st.target = None
                    st.returning = st.position != base
                    self.counters.returns_to_base += int(st.returning)
                    triggers.append("energy")
        explorers = [a for a in active if not self.states[a].returning]
        if any(self.states[a].target is None for a in explorers):
            triggers.append("unassigned_agent")
        if triggers and explorers:
            self._allocate(step, explorers, triggers[0])

        plans = self._plan_round(active, step)
        actions: dict[AgentId, int] = {}
        for aid in active:
            path = plans.paths[aid]
            st = self.states[aid]
            goal = base if st.returning else (st.target or st.position)
            if path is None:
                self.counters.failed_plans += 1
                if aid in plans.unreachable:
                    self.counters.failed_plans_unreachable += 1
                else:
                    self.counters.failed_plans_temporary += 1
                actions[aid] = 0
                continue
            action = first_action(path)
            if action == 0 and st.position != goal:
                self.counters.planned_waits += 1
            actions[aid] = int(action)
        return actions

    def _plan_round(self, active: list[AgentId], step: int) -> PlanRound:
        """Plan jointly; any explorer whose planned route breaks the commit rule gives up its
        target and returns (or waits on the base), then everyone is replanned. Terminates: each
        repeat removes at least one target."""
        base = self.brief.base
        while True:
            requests = []
            for aid in active:
                st = self.states[aid]
                layer = self.brief.agents[aid].layer
                if st.returning:
                    # Return on energy costs, as assumed by the energy-to-base field. Waits cost at
                    # least 1 so earlier arrival is preferred, plus known-hazard exposure.
                    hz = st.model.hazard_energy(self.known)
                    requests.append(
                        PlanRequest(
                            aid,
                            layer,
                            st.position,
                            base,
                            self._passable(layer),
                            st.model.enter_energy(self.known),
                            max(1, st.model.stay_cost) + hz,
                        )
                    )
                else:
                    costs = self._costs(layer)
                    requests.append(
                        PlanRequest(
                            aid,
                            layer,
                            st.position,
                            st.target or st.position,
                            self._passable(layer),
                            costs,
                            costs,  # hazard-aware variants also avoid *waiting* in hazards
                        )
                    )
            plans = plan_prioritized(requests, base, self.horizon, step)
            violators = [
                aid
                for aid in active
                if not self.states[aid].returning
                and self.states[aid].target is not None
                and (path := plans.paths[aid]) is not None
                and not self._plan_feasible(aid, path)
            ]
            if not violators:
                self.last_round = plans
                return plans
            for aid in violators:
                st = self.states[aid]
                self._log(step, aid, None, None, "energy_route_infeasible")
                st.task = st.target = None
                st.returning = st.position != base
                self.counters.energy_route_aborts += 1
                self.counters.returns_to_base += int(st.returning)

    def diagnostics(self) -> Diagnostics:
        return self.counters.as_dict()

    def decision_log(self) -> list[dict[str, Any]]:
        return list(self.log)


def _team(name: str, allocation: str, hazard_aware: bool, oracle: bool) -> TeamFactory:
    return TeamFactory(
        name,
        lambda brief: PooledPlannerTeam(
            name, brief, allocation=allocation, hazard_aware=hazard_aware, oracle=oracle
        ),
    )


PRIORITIZED_FRONTIER = _team("prioritized_frontier", "greedy", hazard_aware=False, oracle=False)
ASSIGNMENT = _team("assignment", "hungarian", hazard_aware=False, oracle=False)
ASSIGNMENT_HAZARD_AWARE = _team(
    "assignment_hazard_aware", "hungarian", hazard_aware=True, oracle=False
)
ASSIGNMENT_ORACLE = _team("assignment_oracle", "hungarian", hazard_aware=False, oracle=True)

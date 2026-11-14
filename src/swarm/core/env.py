"""The SWARM environment core (ADR-0001 to ADR-0006, ADR-0008, ADR-0010).

``SwarmEnv`` follows the *shape* of the PettingZoo Parallel API without depending on it:
``reset(seed)`` returns per-agent observations and infos; ``step(actions)`` takes a dict of
integer actions keyed by agent ID and returns a :class:`StepResult` with observations,
terminations, truncations, infos and the ordered event list. The core computes no reward.

Step order (docs/04): 1 validate actions, 2 resolve movement, 3 energy and recharge (agents at
zero energy away from base are disabled), 4 hazards (none in Phase 1 M1), 5 sensing,
6 mission state. Within a phase, agents are processed in sorted ID order.

State is held in NumPy arrays indexed by sorted agent order. Positions and energy are integers.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, SupportsIndex

import numpy as np
import numpy.typing as npt

from swarm.core.events import (
    AgentDisabled,
    Detection,
    EnergySpent,
    EpisodeEnd,
    Event,
    InvalidAction,
    MoveConflict,
    Moved,
    Recharged,
)
from swarm.core.movement import MoveRequest, resolve_moves
from swarm.core.observation import (
    OUT_OF_BOUNDS,
    UNKNOWN,
    Observation,
    VisibleAgent,
    footprint_offsets,
    in_footprint,
)
from swarm.core.rng import episode_streams
from swarm.core.types import (
    ACTION_DELTAS,
    NUM_ACTIONS,
    Action,
    AgentId,
    AgentStatus,
    Layer,
    Position,
    is_traversable,
)
from swarm.scenarios.config import AgentTypeSpec, ScenarioConfig
from swarm.scenarios.generator import Scenario, check_scenario_matches, generate_scenario

AgentInfo = dict[str, Any]


@dataclass(frozen=True, slots=True)
class StepResult:
    """Return value of :meth:`SwarmEnv.step` (ADR-0010). There is no reward (ADR-0005)."""

    observations: dict[AgentId, Observation]
    terminations: dict[AgentId, bool]
    truncations: dict[AgentId, bool]
    infos: dict[AgentId, AgentInfo]
    events: tuple[Event, ...]


@dataclass(frozen=True, slots=True)
class AgentState:
    agent_id: AgentId
    agent_type: str
    layer: Layer
    position: Position
    energy: int
    energy_capacity: int
    sensor_radius: int
    status: AgentStatus


@dataclass(frozen=True, slots=True)
class EnvSnapshot:
    """Read-only full state including ground truth. For evaluation and rendering only.

    Policies must never receive a snapshot (ADR-0008).
    """

    step: int
    scenario: Scenario
    agents: tuple[AgentState, ...]
    done: bool
    end_reason: str | None = field(default=None)


class SwarmEnv:
    """Deterministic 2D multi-agent search environment.

    Args:
        config: the scenario family. Agent IDs and type specs are fixed by it.
    """

    def __init__(self, config: ScenarioConfig) -> None:
        if not isinstance(config, ScenarioConfig):
            raise TypeError(f"config: must be a ScenarioConfig, got {type(config).__name__}")
        self._config = config
        roster = config.agents()
        self.possible_agents: tuple[AgentId, ...] = tuple(roster)
        self._specs: tuple[AgentTypeSpec, ...] = tuple(spec for spec, _ in roster.values())
        self._initial_energy = np.array([e for _, e in roster.values()], dtype=np.int64)
        self._index = {aid: i for i, aid in enumerate(self.possible_agents)}
        self._scenario: Scenario | None = None
        self._dynamics: np.random.Generator | None = None
        self._pos = np.zeros((len(self.possible_agents), 2), dtype=np.int64)
        self._energy = np.zeros(len(self.possible_agents), dtype=np.int64)
        self._active = np.zeros(len(self.possible_agents), dtype=np.bool_)
        self._step = 0
        self._done = False
        self._end_reason: str | None = None

    # ----------------------------------------------------------------- public API

    @property
    def config(self) -> ScenarioConfig:
        return self._config

    @property
    def agents(self) -> tuple[AgentId, ...]:
        """Agents still acting (active), sorted. Empty before reset and after the episode ends."""
        if self._scenario is None or self._done:
            return ()
        return tuple(aid for i, aid in enumerate(self.possible_agents) if self._active[i])

    @property
    def step_count(self) -> int:
        return self._step

    @property
    def done(self) -> bool:
        return self._done

    def reset(
        self, seed: int, *, scenario: Scenario | None = None
    ) -> tuple[dict[AgentId, Observation], dict[AgentId, AgentInfo]]:
        """Start an episode. All agents spawn on the base with their configured energy.

        Args:
            seed: root seed (ADR-0006). Determines the scenario and environment noise.
            scenario: optional explicit ground truth (for tests and fixed benchmark maps). It
                must match the config's grid shape and base. The ``scenario`` stream is still
                derived from ``seed`` but is not used.

        Returns:
            Observations (with no detections) and empty infos for every agent.
        """
        streams = episode_streams(seed, self.possible_agents)
        if scenario is None:
            scenario = generate_scenario(self._config, streams.scenario)
        else:
            check_scenario_matches(self._config, scenario)
        self._scenario = scenario
        self._dynamics = streams.dynamics
        self._pos[:] = scenario.base
        self._energy[:] = self._initial_energy
        self._active[:] = True
        self._step = 0
        self._done = False
        self._end_reason = None
        observations = {aid: self._observe(i, ()) for i, aid in enumerate(self.possible_agents)}
        return observations, {aid: {} for aid in self.possible_agents}

    def step(self, actions: Mapping[AgentId, SupportsIndex]) -> StepResult:
        """Advance one step with a joint action for exactly the agents in :attr:`agents`.

        Raises:
            RuntimeError: before :meth:`reset` or after the episode has ended.
            ValueError: on missing or unknown agent IDs or actions outside ``0..4``.
        """
        scenario, rng = self._require_running()
        acting = self.agents
        parsed = self._parse_actions(actions, acting)
        t = self._step + 1
        events: list[Event] = []

        # 1. Validate actions: masked actions become STAY plus an event (ADR-0004).
        effective: dict[AgentId, Action] = {}
        for aid in acting:
            i = self._index[aid]
            action = parsed[aid]
            reason = self._invalid_reason(i, action)
            if reason is not None:
                events.append(InvalidAction(t, aid, int(action), reason))
                action = Action.STAY
            effective[aid] = action

        # 2. Resolve movement simultaneously (ADR-0003). Disabled agents are stationary.
        requests = []
        for i, aid in enumerate(self.possible_agents):
            pos = self._position(i)
            target = pos
            if aid in effective:
                dr, dc = ACTION_DELTAS[effective[aid]]
                target = (pos[0] + dr, pos[1] + dc)
            requests.append(MoveRequest(aid, self._specs[i].layer, pos, target))
        resolution = resolve_moves(requests, scenario.base)
        by_id = {r.agent_id: r for r in requests}
        for aid in acting:
            r = by_id[aid]
            if aid in resolution.conflicts:
                events.append(
                    MoveConflict(t, aid, r.position, r.target, resolution.conflicts[aid].value)
                )
            elif aid in resolution.moved:
                events.append(Moved(t, aid, r.position, r.target))
        for i, aid in enumerate(self.possible_agents):
            self._pos[i] = resolution.positions[aid]
        moved = set(resolution.moved)

        # 3. Energy: pay for the executed action, recharge on base, disable at zero (ADR-0002).
        disabled_now: list[AgentId] = []
        for aid in acting:
            i = self._index[aid]
            spec = self._specs[i]
            cost = spec.move_cost if aid in moved else spec.stay_cost
            spent = min(cost, int(self._energy[i]))
            self._energy[i] -= spent
            if spent > 0:
                events.append(EnergySpent(t, aid, spent, int(self._energy[i])))
            on_base = self._position(i) == scenario.base
            if on_base:
                gain = min(spec.recharge_rate, spec.energy_capacity - int(self._energy[i]))
                if gain > 0:
                    self._energy[i] += gain
                    events.append(Recharged(t, aid, gain, int(self._energy[i])))
            elif self._energy[i] == 0:
                self._active[i] = False
                disabled_now.append(aid)
                events.append(AgentDisabled(t, aid, self._position(i), "energy_depleted"))

        # 4. Hazards: not modelled in Phase 1 M1 (ADR-0002 rule 4 arrives in M2).

        # 5. Sensing: each active agent reports each survivor in its footprint w.p. p_detect.
        reports: dict[AgentId, list[Position]] = {}
        for aid in acting:
            i = self._index[aid]
            if not self._active[i]:
                continue
            here = self._position(i)
            radius = self._specs[i].sensor_radius
            found: list[Position] = []
            for s_index, survivor in enumerate(scenario.survivors):
                # One draw per (agent, survivor in footprint), in sorted order (ADR-0006).
                in_range = in_footprint(here, survivor, radius)
                if in_range and rng.random() < self._config.p_detect:
                    found.append(survivor)
                    events.append(Detection(t, aid, survivor, True, s_index))
            reports[aid] = found

        # 6. Mission state (ADR-0009, M1 subset: time limit and all-disabled only).
        self._step = t
        all_disabled = not bool(self._active.any())
        time_up = t >= self._config.time_limit
        if all_disabled or time_up:
            self._done = True
            self._end_reason = "all_disabled" if all_disabled else "time_limit"
            events.append(EpisodeEnd(t, self._end_reason, all_disabled, not all_disabled))

        terminations = {aid: all_disabled or aid in disabled_now for aid in acting}
        truncations = {aid: time_up and not terminations[aid] for aid in acting}
        observations = {
            aid: self._observe(self._index[aid], tuple(reports.get(aid, ()))) for aid in acting
        }
        return StepResult(
            observations=observations,
            terminations=terminations,
            truncations=truncations,
            infos={aid: {} for aid in acting},
            events=tuple(events),
        )

    def snapshot(self) -> EnvSnapshot:
        """Full state including ground truth, for the evaluator and renderer (not policies)."""
        scenario = self._require_scenario()
        agents = tuple(
            AgentState(
                agent_id=aid,
                agent_type=self._specs[i].name,
                layer=self._specs[i].layer,
                position=self._position(i),
                energy=int(self._energy[i]),
                energy_capacity=self._specs[i].energy_capacity,
                sensor_radius=self._specs[i].sensor_radius,
                status=AgentStatus.ACTIVE if self._active[i] else AgentStatus.DISABLED,
            )
            for i, aid in enumerate(self.possible_agents)
        )
        return EnvSnapshot(self._step, scenario, agents, self._done, self._end_reason)

    def action_mask(self, agent_id: AgentId) -> tuple[bool, ...]:
        """The current action mask for one agent (also included in its observation)."""
        self._require_scenario()
        return self._mask(self._index[agent_id])

    # ----------------------------------------------------------------- internals

    def _require_scenario(self) -> Scenario:
        if self._scenario is None:
            raise RuntimeError("call reset(seed) before using the environment")
        return self._scenario

    def _require_running(self) -> tuple[Scenario, np.random.Generator]:
        scenario = self._require_scenario()
        if self._done:
            raise RuntimeError(
                f"episode ended at step {self._step} ({self._end_reason}); call reset(seed)"
            )
        assert self._dynamics is not None
        return scenario, self._dynamics

    @staticmethod
    def _parse_actions(
        actions: Mapping[AgentId, SupportsIndex], acting: tuple[AgentId, ...]
    ) -> dict[AgentId, Action]:
        if not isinstance(actions, Mapping):
            raise ValueError(f"actions: must be a mapping of agent ID to int, got {actions!r}")
        missing = sorted(set(acting) - set(actions))
        extra = sorted(set(actions) - set(acting))
        if missing or extra:
            raise ValueError(
                f"actions: must contain exactly the active agents {list(acting)}; "
                f"missing {missing}, unexpected {extra}"
            )
        parsed: dict[AgentId, Action] = {}
        for aid in acting:
            raw = actions[aid]
            if isinstance(raw, bool | np.bool_):
                raise ValueError(f"actions[{aid!r}]: must be an int in 0..4, got {raw!r}")
            try:
                value = operator.index(raw)
            except TypeError:
                raise ValueError(f"actions[{aid!r}]: must be an int in 0..4, got {raw!r}") from None
            if not 0 <= value < NUM_ACTIONS:
                raise ValueError(f"actions[{aid!r}]: must be in 0..4, got {value}")
            parsed[aid] = Action(value)
        return parsed

    def _position(self, i: int) -> Position:
        return (int(self._pos[i, 0]), int(self._pos[i, 1]))

    def _invalid_reason(self, i: int, action: Action) -> str | None:
        """Static validity (ADR-0004): bounds, terrain for the layer, energy. Not other agents."""
        if action is Action.STAY:
            return None
        scenario = self._require_scenario()
        r, c = self._position(i)
        dr, dc = ACTION_DELTAS[action]
        nr, nc = r + dr, c + dc
        h, w = scenario.shape
        if not (0 <= nr < h and 0 <= nc < w):
            return "out_of_bounds"
        if not is_traversable(self._specs[i].layer, int(scenario.terrain[nr, nc])):
            return "impassable_terrain"
        if self._energy[i] < self._specs[i].move_cost:
            return "insufficient_energy"
        return None

    def _mask(self, i: int) -> tuple[bool, ...]:
        if not self._active[i]:
            return (True,) + (False,) * (NUM_ACTIONS - 1)
        return tuple(self._invalid_reason(i, a) is None for a in Action)

    def _local_terrain(self, i: int) -> npt.NDArray[np.int8]:
        scenario = self._require_scenario()
        radius = self._specs[i].sensor_radius
        size = 2 * radius + 1
        view = np.full((size, size), UNKNOWN, dtype=np.int8)
        r, c = self._position(i)
        h, w = scenario.shape
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if not (0 <= r + dr < h and 0 <= c + dc < w):
                    view[dr + radius, dc + radius] = OUT_OF_BOUNDS
        for dr, dc in footprint_offsets(radius):
            if 0 <= r + dr < h and 0 <= c + dc < w:
                view[dr + radius, dc + radius] = scenario.terrain[r + dr, c + dc]
        view.flags.writeable = False
        return view

    def _observe(self, i: int, detections: tuple[Position, ...]) -> Observation:
        scenario = self._require_scenario()
        spec = self._specs[i]
        here = self._position(i)
        visible = tuple(
            VisibleAgent(
                agent_id=other,
                agent_type=self._specs[j].name,
                layer=self._specs[j].layer,
                position=self._position(j),
                status=AgentStatus.ACTIVE if self._active[j] else AgentStatus.DISABLED,
            )
            for j, other in enumerate(self.possible_agents)
            if j != i and in_footprint(here, self._position(j), spec.sensor_radius)
        )
        return Observation(
            agent_id=self.possible_agents[i],
            agent_type=spec.name,
            layer=spec.layer,
            step=self._step,
            position=here,
            energy=int(self._energy[i]),
            energy_capacity=spec.energy_capacity,
            status=AgentStatus.ACTIVE if self._active[i] else AgentStatus.DISABLED,
            sensor_radius=spec.sensor_radius,
            grid_shape=scenario.shape,
            base_position=scenario.base,
            local_terrain=self._local_terrain(i),
            visible_agents=visible,
            detections=tuple(sorted(set(detections))),
            action_mask=self._mask(i),
        )

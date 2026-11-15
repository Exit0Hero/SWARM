"""The SWARM environment core (ADR-0001 to ADR-0006, ADR-0008, ADR-0010).

``SwarmEnv`` follows the *shape* of the PettingZoo Parallel API without depending on it:
``reset(seed)`` returns per-agent observations and infos; ``step(actions)`` takes a dict of
integer actions keyed by agent ID and returns a :class:`StepResult` with observations,
terminations, truncations, infos and the ordered event list. The core computes no reward.

Step order (docs/04): 1 validate actions, 2 resolve movement, 3 energy and recharge (agents at
zero energy away from base are disabled), 4 static hazards (ADR-0011), 5 sensing: true reports
then false reports (ADR-0012), 5b communication channel when configured (ADR-0017), 6 mission
state: mission success > all disabled > time limit (ADR-0009). Within a phase, agents are
processed in sorted ID order.

State is held in NumPy arrays indexed by sorted agent order. Positions and energy are integers.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, SupportsIndex

import numpy as np
import numpy.typing as npt

from swarm.core.comms import Channel
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
from swarm.core.messages import Delivery, Message, outbox_size, validate_message
from swarm.core.movement import MoveRequest, resolve_moves
from swarm.core.observation import (
    OUT_OF_BOUNDS,
    UNKNOWN,
    Observation,
    VisibleAgent,
    footprint_offsets,
    in_footprint,
)
from swarm.core.oracle import OracleView
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
        self._hazard_rng: np.random.Generator | None = None
        self._false_positive_rng: np.random.Generator | None = None
        self._channel: Channel | None = None
        if config.comms is not None:
            ranges = {aid: config.comms.range_for(spec.name) for aid, (spec, _) in roster.items()}
            self._channel = Channel(config.comms, ranges)
        self._last_seq: dict[AgentId, int] = {}
        self._hazard_layers = (
            frozenset() if config.hazards is None else frozenset(config.hazards.affected_layers)
        )
        # Exact rational target: avoids float rounding at the boundary (ADR-0009).
        self._recall_target = Fraction(str(config.recall_target))
        self._found: set[int] = set()
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
        self._hazard_rng = streams.hazard
        self._false_positive_rng = streams.false_positive
        if self._channel is not None:
            self._channel.reset(streams.comms)
        self._last_seq = {}
        self._found = set()
        self._pos[:] = scenario.base
        self._energy[:] = self._initial_energy
        self._active[:] = True
        self._step = 0
        self._done = False
        self._end_reason = None
        observations = {aid: self._observe(i, ()) for i, aid in enumerate(self.possible_agents)}
        return observations, {aid: {} for aid in self.possible_agents}

    def step(
        self,
        actions: Mapping[AgentId, SupportsIndex],
        messages: Mapping[AgentId, Sequence[Message]] | None = None,
    ) -> StepResult:
        """Advance one step with a joint action for exactly the agents in :attr:`agents`.

        ``messages`` is the outbox: messages each active agent sends this step (ADR-0017). It
        requires a ``comms`` block in the config. ``None`` (the default) sends nothing; queued
        messages from earlier steps are still delivered. Without a channel, behaviour is exactly
        as in Phase 1 and 2.

        Raises:
            RuntimeError: before :meth:`reset` or after the episode has ended.
            ValueError: on missing or unknown agent IDs, actions outside ``0..4``, messages
                without a channel, messages from inactive agents, non-increasing ``seq`` or an
                outbox over the bandwidth.
            TypeError: on malformed messages.
        """
        scenario, rng = self._require_running()
        acting = self.agents
        parsed = self._parse_actions(actions, acting)
        outbox = self._parse_messages(messages, acting)
        for aid, batch in outbox.items():
            self._last_seq[aid] = batch[-1].seq
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

        # 4. Static hazards (ADR-0011): exposure when an affected agent ends the step in a hazard.
        hazard_rng = self._hazard_rng
        assert hazard_rng is not None
        hazards = scenario.hazard_grid
        cfg_hazards = self._config.hazards
        for aid in acting:
            i = self._index[aid]
            spec = self._specs[i]
            here = self._position(i)
            if (
                cfg_hazards is None
                or not self._active[i]
                or spec.layer not in self._hazard_layers
                or not hazards[here]
            ):
                continue
            lost = min(cfg_hazards.energy_cost, int(self._energy[i]))
            self._energy[i] -= lost
            events.append(HazardExposure(t, aid, here, lost, int(self._energy[i])))
            if self._energy[i] == 0:
                reason = "energy_depleted"
            elif hazard_rng.random() < cfg_hazards.disable_probability:
                reason = "hazard"
            else:
                continue
            self._active[i] = False
            disabled_now.append(aid)
            events.append(AgentDisabled(t, aid, here, reason))

        # 5. Sensing (ADR-0012). True reports: one dynamics draw per survivor in the footprint.
        #    False reports: one false_positive draw per other in-grid footprint cell, row-major.
        fp_rng = self._false_positive_rng
        assert fp_rng is not None
        survivor_index = {pos: k for k, pos in enumerate(scenario.survivors)}
        h, w = scenario.shape
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
                    self._found.add(s_index)
                    events.append(Detection(t, aid, survivor, True, s_index))
            empty_cells = [
                (here[0] + dr, here[1] + dc)
                for dr, dc in footprint_offsets(radius)
                if 0 <= here[0] + dr < h
                and 0 <= here[1] + dc < w
                and (here[0] + dr, here[1] + dc) not in survivor_index
            ]
            draws = fp_rng.random(len(empty_cells))
            for cell, u in zip(empty_cells, draws, strict=True):
                if u < self._config.p_false_positive:
                    found.append(cell)
                    events.append(Detection(t, aid, cell, False, None))
            reports[aid] = found

        # 5b. Communication channel (ADR-0017): transmit at post-move positions, then deliver.
        inboxes: dict[AgentId, tuple[Delivery, ...]] = {}
        if self._channel is not None:
            active_now = frozenset(
                aid for i, aid in enumerate(self.possible_agents) if self._active[i]
            )
            positions = {aid: self._position(i) for i, aid in enumerate(self.possible_agents)}
            events.extend(self._channel.transmit(t, outbox, positions, active_now))
            inboxes, delivery_events = self._channel.deliver(t, active_now)
            events.extend(delivery_events)

        # 6. Mission state (ADR-0009): mission_success > all_disabled > time_limit.
        self._step = t
        n_survivors = len(scenario.survivors)
        success = n_survivors > 0 and Fraction(len(self._found), n_survivors) >= self._recall_target
        all_disabled = not bool(self._active.any())
        time_up = t >= self._config.time_limit
        if success or all_disabled or time_up:
            self._done = True
            if success:
                self._end_reason = "mission_success"
            elif all_disabled:
                self._end_reason = "all_disabled"
            else:
                self._end_reason = "time_limit"
            ended_terminal = self._end_reason != "time_limit"
            events.append(EpisodeEnd(t, self._end_reason, ended_terminal, not ended_terminal))

        terminal = self._done and self._end_reason != "time_limit"
        terminations = {aid: terminal or aid in disabled_now for aid in acting}
        truncations = {
            aid: self._end_reason == "time_limit" and not terminations[aid] for aid in acting
        }
        observations = {
            aid: self._observe(self._index[aid], tuple(reports.get(aid, ())), inboxes.get(aid, ()))
            for aid in acting
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

    def oracle_view(self) -> OracleView:
        """Ground truth for oracle baselines only (ADR-0014). Never give this to other policies."""
        scenario = self._require_scenario()
        return OracleView(
            terrain=scenario.terrain,
            hazards=scenario.hazard_grid,
            survivors=scenario.survivors,
            base=scenario.base,
        )

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

    def _parse_messages(
        self,
        messages: Mapping[AgentId, Sequence[Message]] | None,
        acting: tuple[AgentId, ...],
    ) -> dict[AgentId, tuple[Message, ...]]:
        """Validate an outbox before any state changes (ADR-0017)."""
        if messages is None:
            return {}
        if self._channel is None:
            raise ValueError(
                "messages: this config has no comms block, so nothing can be sent (ADR-0017)"
            )
        if not isinstance(messages, Mapping):
            raise ValueError(f"messages: must map agent ID to messages, got {messages!r}")
        unexpected = sorted(set(messages) - set(acting))
        if unexpected:
            raise ValueError(f"messages: only active agents {list(acting)} may send; {unexpected}")
        bandwidth = self._config.comms.bandwidth if self._config.comms is not None else None
        parsed: dict[AgentId, tuple[Message, ...]] = {}
        for aid in sorted(messages):
            batch: object = messages[aid]
            if isinstance(batch, str | bytes) or not isinstance(batch, Sequence):
                raise TypeError(f"messages[{aid!r}]: must be a sequence of Message")
            msgs = tuple(validate_message(m, aid) for m in batch)
            last = self._last_seq.get(aid, -1)
            for m in msgs:
                if m.seq <= last:
                    raise ValueError(
                        f"messages[{aid!r}]: seq must strictly increase over the episode; "
                        f"got {m.seq} after {last}"
                    )
                last = m.seq
            size = outbox_size(msgs)
            if bandwidth is not None and size > bandwidth:
                raise ValueError(
                    f"messages[{aid!r}]: {size} units exceed the bandwidth of {bandwidth} per step"
                )
            if msgs:
                parsed[aid] = msgs
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

    def _local_hazard(self, i: int) -> npt.NDArray[np.bool_]:
        scenario = self._require_scenario()
        radius = self._specs[i].sensor_radius
        size = 2 * radius + 1
        view = np.zeros((size, size), dtype=np.bool_)
        r, c = self._position(i)
        h, w = scenario.shape
        hazards = scenario.hazard_grid
        for dr, dc in footprint_offsets(radius):
            if 0 <= r + dr < h and 0 <= c + dc < w:
                view[dr + radius, dc + radius] = hazards[r + dr, c + dc]
        view.flags.writeable = False
        return view

    def _observe(
        self, i: int, detections: tuple[Position, ...], inbox: tuple[Delivery, ...] = ()
    ) -> Observation:
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
            local_hazard=self._local_hazard(i),
            visible_agents=visible,
            detections=tuple(sorted(set(detections))),
            action_mask=self._mask(i),
            inbox=inbox,
        )

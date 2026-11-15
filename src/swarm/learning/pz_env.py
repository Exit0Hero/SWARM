"""PettingZoo ``ParallelEnv`` adapters (ADR-0010 Phase 4 adapter, ADR-0022). Needs ``[learning]``.

- :class:`SwarmParallelEnv`: the thin primitive-action adapter. One step is one simulator step,
  the actions are the five moves, and the action mask comes from the core (ADR-0004).
- :class:`SectorParallelEnv`: the macro-action adapter used for training. One step is one
  decision epoch of :class:`~swarm.learning.sector.SectorTeam`. Observations are the actor
  features. :meth:`SectorParallelEnv.state` is the centralized critic state.

Both use :class:`~swarm.learning.rewards.TeamReward`. The reward is shared: every acting agent
receives the team reward. Observation spaces are
``Dict(observation=Box, action_mask=MultiBinary)``.

Seeds: ``reset(seed=s)`` runs scenario seed ``s``. ``reset()`` without a seed takes the next seed
from ``seeds`` (the training range), so training and evaluation seed sets stay disjoint by
construction.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from pettingzoo import ParallelEnv

from swarm.core.env import SwarmEnv
from swarm.core.observation import Observation
from swarm.core.rng import episode_streams
from swarm.core.types import AgentId, AgentStatus
from swarm.learning.rewards import TeamReward
from swarm.learning.sector import DECISION_INTERVAL, FREE, N_ACTIONS, OBS_SIZE, SectorTeam
from swarm.policies.base import MissionBrief
from swarm.scenarios.config import ScenarioConfig

ObsDict = dict[str, npt.NDArray[Any]]


def _seed_iter(seeds: Sequence[int] | range | None) -> Iterator[int]:
    if seeds is None:
        return iter(())
    return iter(seeds)


def _primitive_vector(obs: Observation, size: int) -> npt.NDArray[np.float32]:
    terrain = (obs.local_terrain.astype(np.float32) + 1.0) / 4.0  # -1 (outside) .. 3 -> 0 .. 1
    hazard = obs.local_hazard.astype(np.float32)
    h, w = obs.grid_shape
    own = np.array(
        [
            obs.energy / max(1, obs.energy_capacity),
            obs.position[0] / max(1, h - 1),
            obs.position[1] / max(1, w - 1),
        ],
        dtype=np.float32,
    )
    vec = np.concatenate([terrain.ravel(), hazard.ravel(), own]).astype(np.float32)
    assert vec.size == size
    return vec


class SwarmParallelEnv(ParallelEnv[AgentId, ObsDict, int]):
    """Primitive-action adapter: local terrain and hazard window, energy and position."""

    metadata: ClassVar[dict[str, Any]] = {"name": "swarm_primitive_v0", "render_modes": []}

    def __init__(
        self,
        config: ScenarioConfig,
        *,
        seeds: Sequence[int] | range | None = None,
        reward: TeamReward | None = None,
    ) -> None:
        self.config = config
        self.env = SwarmEnv(config)
        self.possible_agents = list(self.env.possible_agents)
        self.agents: list[AgentId] = []
        self.reward_model = reward or TeamReward()
        self._seeds = _seed_iter(seeds)
        radii = {aid: spec.sensor_radius for aid, (spec, _) in config.agents().items()}
        self._sizes = {aid: 2 * (2 * r + 1) ** 2 + 3 for aid, r in radii.items()}
        self.observation_spaces = {
            aid: spaces.Dict(
                {
                    "observation": spaces.Box(0.0, 1.0, (self._sizes[aid],), np.float32),
                    "action_mask": spaces.MultiBinary(5),
                }
            )
            for aid in self.possible_agents
        }
        self.action_spaces = {aid: spaces.Discrete(5) for aid in self.possible_agents}

    def observation_space(self, agent: AgentId) -> spaces.Space[Any]:
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentId) -> spaces.Space[Any]:
        return self.action_spaces[agent]

    def _obs(
        self, observations: Mapping[AgentId, Observation], agents: Sequence[AgentId]
    ) -> dict[AgentId, ObsDict]:
        return {
            aid: {
                "observation": _primitive_vector(o, self._sizes[aid]),
                "action_mask": np.array(o.action_mask, dtype=np.int8),
            }
            for aid, o in observations.items()
            if aid in agents
        }

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[AgentId, ObsDict], dict[AgentId, dict[str, Any]]]:
        scenario_seed = seed if seed is not None else next(self._seeds)
        observations, _ = self.env.reset(scenario_seed)
        self.agents = [a for a in self.possible_agents if a in self.env.agents]
        self.reward_model.reset(observations.values())
        infos: dict[AgentId, dict[str, Any]] = {a: {} for a in self.agents}
        return self._obs(observations, self.agents), infos

    def step(
        self, actions: Mapping[AgentId, int]
    ) -> tuple[
        dict[AgentId, ObsDict],
        dict[AgentId, float],
        dict[AgentId, bool],
        dict[AgentId, bool],
        dict[AgentId, dict[str, Any]],
    ]:
        acting = list(self.agents)
        result = self.env.step({a: int(actions[a]) for a in acting})
        live = {a: o for a, o in result.observations.items() if o.status is AgentStatus.ACTIVE}
        team = sum(self.reward_model.step(result.events, live).values())
        rewards = dict.fromkeys(acting, float(team))
        terms = {a: bool(result.terminations.get(a, False)) for a in acting}
        truncs = {a: bool(result.truncations.get(a, False)) for a in acting}
        for a in acting:
            if a not in live:
                terms[a] = True
        self.agents = [a for a in acting if not (terms[a] or truncs[a])]
        return (
            self._obs(result.observations, acting),
            rewards,
            terms,
            truncs,
            {a: {} for a in acting},
        )


class SectorParallelEnv(ParallelEnv[AgentId, ObsDict, int]):
    """Macro-action adapter over :class:`SectorTeam` (the training environment).

    ``info[agent]`` holds the per-component team reward for the epoch and, at the end, the
    episode's ``end_reason``.
    """

    metadata: ClassVar[dict[str, Any]] = {"name": "swarm_sector_v0", "render_modes": []}

    def __init__(
        self,
        config: ScenarioConfig,
        *,
        seeds: Sequence[int] | range | None = None,
        reward: TeamReward | None = None,
        decision_interval: int = DECISION_INTERVAL,
    ) -> None:
        if config.comms is None:
            raise ValueError("SectorParallelEnv: the scenario needs a comms block (ADR-0017)")
        self.config = config
        self.brief = MissionBrief.from_config(config)
        self.env = SwarmEnv(config)
        self.possible_agents = list(self.env.possible_agents)
        self.agents: list[AgentId] = []
        self.reward_model = reward or TeamReward()
        self.decision_interval = decision_interval
        self._seeds = _seed_iter(seeds)
        self.observation_spaces = {
            aid: spaces.Dict(
                {
                    "observation": spaces.Box(0.0, 1.0, (OBS_SIZE,), np.float32),
                    "action_mask": spaces.MultiBinary(N_ACTIONS),
                }
            )
            for aid in self.possible_agents
        }
        self.action_spaces = {aid: spaces.Discrete(N_ACTIONS) for aid in self.possible_agents}
        n = len(self.possible_agents)
        self.state_space = spaces.Box(0.0, 1.0, (n * OBS_SIZE + n,), np.float32)
        self.team: SectorTeam | None = None
        self._latest: dict[AgentId, npt.NDArray[np.float32]] = {}
        self._prim: dict[AgentId, Observation] = {}
        self._rngs: dict[AgentId, np.random.Generator] = {}
        self.events: list[Any] = []
        self.end_reason: str | None = None

    def observation_space(self, agent: AgentId) -> spaces.Space[Any]:
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentId) -> spaces.Space[Any]:
        return self.action_spaces[agent]

    def state(self) -> npt.NDArray[np.float32]:
        """Centralized critic state: every agent's latest actor features (zeros once inactive)
        followed by alive flags, in roster order. Built only from agents' own information."""
        parts = [self._latest.get(a, np.zeros(OBS_SIZE, np.float32)) for a in self.possible_agents]
        alive = [float(a in self.agents) for a in self.possible_agents]
        return np.concatenate([*parts, np.array(alive, dtype=np.float32)]).astype(np.float32)

    def _pack(
        self, decide: Mapping[AgentId, tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]]
    ) -> dict[AgentId, ObsDict]:
        out: dict[AgentId, ObsDict] = {}
        for aid, (feats, mask) in decide.items():
            self._latest[aid] = feats
            out[aid] = {"observation": feats, "action_mask": mask.astype(np.int8)}
        return out

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[AgentId, ObsDict], dict[AgentId, dict[str, Any]]]:
        scenario_seed = seed if seed is not None else next(self._seeds)
        self.team = SectorTeam(
            "marl_training", self.brief, decision_interval=self.decision_interval
        )
        observations, _ = self.env.reset(scenario_seed)
        self._rngs = episode_streams(scenario_seed, self.env.possible_agents).policies
        self._prim = dict(observations)
        self._latest = {}
        self.events = []
        self.end_reason = None
        self.reward_model.reset(o for o in observations.values() if o.status is AgentStatus.ACTIVE)
        decide = self.team.prepare(observations)
        self.agents = [a for a in self.possible_agents if a in decide]
        return self._pack(decide), {a: {} for a in self.agents}

    def step(
        self, actions: Mapping[AgentId, int]
    ) -> tuple[
        dict[AgentId, ObsDict],
        dict[AgentId, float],
        dict[AgentId, bool],
        dict[AgentId, bool],
        dict[AgentId, dict[str, Any]],
    ]:
        assert self.team is not None, "call reset() first"
        acting = list(self.agents)
        choices = {a: int(actions[a]) for a in acting}
        components = dict.fromkeys(self.reward_model.weights, 0.0)
        decide: dict[AgentId, tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]] = {}
        while True:
            prim_actions = self.team.advance(self._prim, self._rngs, choices)
            choices = {}
            result = self.env.step(prim_actions, self.team.outbox())
            self.events.extend(result.events)
            self._prim = dict(result.observations)
            live = {a: o for a, o in self._prim.items() if o.status is AgentStatus.ACTIVE}
            for k, v in self.reward_model.step(result.events, live).items():
                components[k] = components.get(k, 0.0) + v
            if self.env.done:
                self.end_reason = self.env.snapshot().end_reason
                break
            decide = self.team.prepare(self._prim)
            if decide:
                break
        team_reward = float(sum(components.values()))
        rewards = dict.fromkeys(acting, team_reward)
        done = self.env.done
        timeout = done and self.end_reason == "time_limit"
        terms = {a: (done and not timeout) or a not in decide for a in acting}
        truncs = dict.fromkeys(acting, timeout)
        if timeout:
            terms = dict.fromkeys(acting, False)
        infos: dict[AgentId, dict[str, Any]] = {
            a: {"reward_components": dict(components), "end_reason": self.end_reason}
            for a in acting
        }
        obs = self._pack(decide) if not done else {}
        self.agents = [a for a in self.possible_agents if a in obs]
        for a in acting:
            if a not in obs:  # final observation of an agent that stopped acting
                final = self._latest.pop(a, np.zeros(OBS_SIZE, np.float32))
                mask = np.zeros(N_ACTIONS, dtype=np.int8)
                mask[FREE] = 1
                obs[a] = {"observation": final, "action_mask": mask}
        return obs, rewards, terms, truncs, infos

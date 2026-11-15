"""Policy interfaces: per-agent policies (ADR-0010) and team policies (ADR-0014).

A policy receives only an :class:`~swarm.core.observation.Observation` and its own random
generator (ADR-0006, ADR-0008), never the environment, a snapshot or the scenario. Policies are
built per agent and per episode by a :data:`PolicyFactory`, so no state leaks between episodes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from swarm.core.messages import Message
from swarm.core.observation import Observation
from swarm.core.oracle import OracleView
from swarm.core.types import AgentId, AgentStatus, Layer, Position
from swarm.scenarios.config import AgentTypeSpec, CommsConfig, ScenarioConfig


class Policy(Protocol):
    """Decision rule for one agent."""

    @property
    def name(self) -> str:
        """Stable identifier recorded in results, for example ``"random_valid"``."""
        ...

    @property
    def version(self) -> str:
        """Bumped whenever behaviour changes, so results stay attributable."""
        ...

    @property
    def uses_oracle(self) -> bool:
        """``True`` if the policy needs hidden ground truth (an ``OracleView``, Phase 2)."""
        ...

    def act(self, observation: Observation, rng: np.random.Generator) -> int:
        """Return an action in ``0..4`` for ``observation.agent_id``."""
        ...


PolicyFactory = Callable[[AgentId, AgentTypeSpec], Policy]
"""Builds a fresh policy for one agent at the start of an episode."""


# ----------------------------------------------------------------- team policies (ADR-0014)

DECENTRALIZED = "decentralized"
CENTRALIZED_POOLED = "centralized_pooled"
ORACLE = "oracle"
DECENTRALIZED_COMM = "decentralized_comm"
"""Each agent decides from its own observations plus messages received over the channel
(ADR-0017, ADR-0018)."""
INFORMATION_MODELS = frozenset({DECENTRALIZED, CENTRALIZED_POOLED, ORACLE, DECENTRALIZED_COMM})

Diagnostics = dict[str, int | float | None]


@dataclass(frozen=True, slots=True)
class HazardBrief:
    """Public hazard *model* (no locations): ADR-0014 rule 3."""

    affected_layers: tuple[Layer, ...]
    energy_cost: int
    disable_probability: float


@dataclass(frozen=True, slots=True)
class MissionBrief:
    """Public mission information given to policy factories (ADR-0014 rule 3).

    Deliberately excludes survivor count and locations, terrain, hazard locations and the
    terrain/hazard generation fractions.
    """

    grid_shape: tuple[int, int]
    base: Position
    time_limit: int
    agents: dict[AgentId, AgentTypeSpec]
    hazard: HazardBrief | None
    p_detect: float
    p_false_positive: float
    recall_target: float
    comms: CommsConfig | None = None
    """Public channel parameters (range, loss, delay, bandwidth); ``None`` without a channel."""

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> MissionBrief:
        hz = config.hazards
        return cls(
            grid_shape=(config.height, config.width),
            base=config.base,
            time_limit=config.time_limit,
            agents={aid: spec for aid, (spec, _) in config.agents().items()},
            hazard=None
            if hz is None
            else HazardBrief(hz.affected_layers, hz.energy_cost, hz.disable_probability),
            p_detect=config.p_detect,
            p_false_positive=config.p_false_positive,
            recall_target=config.recall_target,
            comms=config.comms,
        )

    @property
    def hazard_layers(self) -> tuple[Layer, ...]:
        return () if self.hazard is None else self.hazard.affected_layers

    @property
    def hazard_energy_cost(self) -> int:
        return 0 if self.hazard is None else self.hazard.energy_cost


class TeamPolicy(Protocol):
    """Decides for every active agent at once (ADR-0014 rule 1)."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def uses_oracle(self) -> bool: ...

    @property
    def information(self) -> str:
        """One of ``decentralized``, ``centralized_pooled``, ``oracle``."""
        ...

    def act(
        self,
        observations: Mapping[AgentId, Observation],
        rngs: Mapping[AgentId, np.random.Generator],
    ) -> dict[AgentId, int]:
        """Actions for every observed agent whose status is active."""
        ...

    def attach_oracle(self, view: OracleView) -> None:
        """Receive ground truth. Only called when ``uses_oracle`` is ``True``."""
        ...

    def diagnostics(self) -> Diagnostics:
        """Deterministic numeric diagnostics for the episode so far."""
        ...

    def decision_log(self) -> list[dict[str, Any]]:
        """Recorded decisions (for example assignments)."""
        ...


@dataclass(frozen=True, slots=True)
class TeamFactory:
    """Builds a fresh :class:`TeamPolicy` per episode from the public :class:`MissionBrief`."""

    name: str
    build: Callable[[MissionBrief], TeamPolicy]


@runtime_checkable
class CommunicatingTeam(Protocol):
    """A team whose agents send messages (ADR-0017).

    After each :meth:`TeamPolicy.act`, the runner calls :meth:`outbox` and passes the result to
    ``SwarmEnv.step``. Each agent's messages must be composed only from that agent's own
    observations and inbox (ADR-0008).
    """

    def outbox(self) -> Mapping[AgentId, Sequence[Message]]:
        """Messages each agent sends this step, keyed by sender."""
        ...


class IndependentTeam:
    """Adapts per-agent policies to the team interface (ADR-0014 rule 2).

    Each active agent's policy is called in sorted agent order with that agent's own stream,
    exactly as the Phase 1 runner did, so Phase 1 event streams are unchanged.
    """

    information = DECENTRALIZED

    def __init__(self, policies: dict[AgentId, Policy]) -> None:
        identities = {(p.name, p.version, p.uses_oracle) for p in policies.values()}
        if len(identities) != 1:
            raise ValueError(f"all agents must share one policy identity, got {identities}")
        first = next(iter(policies.values()))
        if first.uses_oracle:
            raise ValueError(
                "per-agent policies cannot use the oracle; use a TeamPolicy (ADR-0014)"
            )
        self._policies = policies
        self.name = first.name
        self.version = first.version
        self.uses_oracle = False

    def act(
        self,
        observations: Mapping[AgentId, Observation],
        rngs: Mapping[AgentId, np.random.Generator],
    ) -> dict[AgentId, int]:
        return {
            aid: self._policies[aid].act(observations[aid], rngs[aid])
            for aid in sorted(observations)
            if observations[aid].status is AgentStatus.ACTIVE
        }

    def attach_oracle(self, view: OracleView) -> None:
        raise RuntimeError("IndependentTeam never uses the oracle")

    def diagnostics(self) -> Diagnostics:
        return {}

    def decision_log(self) -> list[dict[str, Any]]:
        return []


def build_team(factory: PolicyFactory | TeamFactory, brief: MissionBrief) -> TeamPolicy:
    """Build the episode's team policy from either kind of factory."""
    if isinstance(factory, TeamFactory):
        team = factory.build(brief)
        if team.information not in INFORMATION_MODELS:
            raise ValueError(f"information: must be one of {sorted(INFORMATION_MODELS)}")
        if team.uses_oracle != (team.information == ORACLE):
            raise ValueError("uses_oracle must be True exactly when information == 'oracle'")
        return team
    return IndependentTeam({aid: factory(aid, spec) for aid, spec in brief.agents.items()})


def policy_identity(team: TeamPolicy) -> dict[str, Any]:
    """The policy record stored in every result."""
    return {
        "name": team.name,
        "version": team.version,
        "uses_oracle": team.uses_oracle,
        "information": team.information,
    }

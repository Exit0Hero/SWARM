"""Policy interface (ADR-0010).

A policy receives only an :class:`~swarm.core.observation.Observation` and its own random
generator (ADR-0006, ADR-0008), never the environment, a snapshot or the scenario. Policies are
built per agent and per episode by a :data:`PolicyFactory`, so no state leaks between episodes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

from swarm.core.observation import Observation
from swarm.core.types import AgentId
from swarm.scenarios.config import AgentTypeSpec


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

"""Random valid-action baseline: the sanity baseline from docs/05.

Each step it picks uniformly among the actions its observation's action mask allows. It has no
memory, no goal and no coordination. Any useful policy must beat it on the same seeds.
"""

from __future__ import annotations

import numpy as np

from swarm.core.observation import Observation
from swarm.core.types import AgentId
from swarm.scenarios.config import AgentTypeSpec


class RandomValidPolicy:
    """Uniform random choice over the valid actions in the observation's mask."""

    name = "random_valid"
    version = "1"
    uses_oracle = False

    def act(self, observation: Observation, rng: np.random.Generator) -> int:
        valid = [a for a, ok in enumerate(observation.action_mask) if ok]
        # STAY is always valid, so ``valid`` is never empty (ADR-0004).
        return valid[int(rng.integers(len(valid)))]


def random_valid_factory(agent_id: AgentId, spec: AgentTypeSpec) -> RandomValidPolicy:
    """:data:`~swarm.policies.base.PolicyFactory` for :class:`RandomValidPolicy`."""
    del agent_id, spec  # the policy is identical for every agent
    return RandomValidPolicy()

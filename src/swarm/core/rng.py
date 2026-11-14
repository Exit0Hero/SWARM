"""Per-episode random-number streams derived from one root seed (ADR-0006).

``SeedSequence(root_seed)`` is spawned, in this fixed order, into ``scenario``, ``dynamics`` and
``policies``; ``policies`` is spawned again into one child per agent in sorted agent-ID order.
Changing the policy therefore cannot change the scenario, and policy draws can never shift
environment draws.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from swarm.core.types import AgentId

SPAWN_ORDER: Final[tuple[str, str, str]] = ("scenario", "dynamics", "policies")


@dataclass(frozen=True, slots=True)
class EpisodeStreams:
    """Independent generators for one episode."""

    root_seed: int
    scenario: np.random.Generator
    dynamics: np.random.Generator
    policies: dict[AgentId, np.random.Generator]


def validate_seed(seed: object) -> int:
    """Return ``seed`` if it is a non-negative ``int`` (not ``bool``), else raise ``ValueError``."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed: must be a non-negative integer, got {seed!r}")
    return seed


def episode_streams(root_seed: int, agent_ids: Sequence[AgentId]) -> EpisodeStreams:
    """Derive the scenario, dynamics and per-agent policy generators from ``root_seed``."""
    validate_seed(root_seed)
    scenario_seq, dynamics_seq, policies_seq = np.random.SeedSequence(root_seed).spawn(3)
    ids = sorted(agent_ids)
    if len(set(ids)) != len(ids):
        raise ValueError(f"agent_ids: must be unique, got {list(agent_ids)}")
    children = policies_seq.spawn(len(ids))
    return EpisodeStreams(
        root_seed=root_seed,
        scenario=np.random.default_rng(scenario_seq),
        dynamics=np.random.default_rng(dynamics_seq),
        policies={aid: np.random.default_rng(seq) for aid, seq in zip(ids, children, strict=True)},
    )


def seed_derivation(root_seed: int, agent_ids: Sequence[AgentId]) -> dict[str, Any]:
    """Describe how every stream was derived, for run metadata (ADR-0006, docs/11)."""
    return {
        "root_seed": validate_seed(root_seed),
        "method": "numpy.random.SeedSequence(root_seed).spawn(3); policies child spawned per agent",
        "spawn_order": list(SPAWN_ORDER),
        "policy_stream_order": sorted(agent_ids),
        "bit_generator": "PCG64",
    }

"""Per-episode random-number streams derived from one root seed (ADR-0006).

``SeedSequence(root_seed)`` is spawned, in this fixed order, into ``scenario``, ``dynamics`` and
``policies``; ``policies`` is spawned again into one child per agent in sorted agent-ID order.
Changing the policy therefore cannot change the scenario, and policy draws can never shift
environment draws.

The ``dynamics`` sequence drives true-detection draws (as in M1) and is additionally spawned into
``hazard``, ``false_positive``, ``comms`` and ``failure`` children (ADR-0013, ADR-0017,
ADR-0021). Spawning children
does not change the parent's generator, and a child's spawn key depends only on its index, so
adding ``comms`` (third) and ``failure`` (fourth) leaves the earlier streams exactly as they were.
``comms`` and ``failure`` are drawn from only when configured.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from swarm.core.types import AgentId

SPAWN_ORDER: Final[tuple[str, str, str]] = ("scenario", "dynamics", "policies")
DYNAMICS_CHILDREN: Final[tuple[str, str, str, str]] = (
    "hazard",
    "false_positive",
    "comms",
    "failure",
)


@dataclass(frozen=True, slots=True)
class EpisodeStreams:
    """Independent generators for one episode."""

    root_seed: int
    scenario: np.random.Generator
    dynamics: np.random.Generator
    """True-detection draws (ADR-0012 rule 1)."""
    hazard: np.random.Generator
    """Hazard disablement draws (ADR-0011)."""
    false_positive: np.random.Generator
    """False-report draws (ADR-0012 rule 2)."""
    comms: np.random.Generator
    """Message drop and jitter draws (ADR-0017)."""
    failure: np.random.Generator
    """Scheduled-failure draws (ADR-0021); used only when failures are configured."""
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
    hazard_seq, false_positive_seq, comms_seq, failure_seq = dynamics_seq.spawn(
        len(DYNAMICS_CHILDREN)
    )
    return EpisodeStreams(
        root_seed=root_seed,
        scenario=np.random.default_rng(scenario_seq),
        dynamics=np.random.default_rng(dynamics_seq),
        hazard=np.random.default_rng(hazard_seq),
        false_positive=np.random.default_rng(false_positive_seq),
        comms=np.random.default_rng(comms_seq),
        failure=np.random.default_rng(failure_seq),
        policies={aid: np.random.default_rng(seq) for aid, seq in zip(ids, children, strict=True)},
    )


def seed_derivation(root_seed: int, agent_ids: Sequence[AgentId]) -> dict[str, Any]:
    """Describe how every stream was derived, for run metadata (ADR-0006, docs/11)."""
    return {
        "root_seed": validate_seed(root_seed),
        "method": (
            "numpy.random.SeedSequence(root_seed).spawn(3); dynamics child spawned into "
            "(hazard, false_positive, comms, failure); policies child spawned per agent"
        ),
        "spawn_order": list(SPAWN_ORDER),
        "dynamics_children": list(DYNAMICS_CHILDREN),
        "policy_stream_order": sorted(agent_ids),
        "bit_generator": "PCG64",
    }

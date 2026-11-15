"""Registered policies by name (ADR-0016, ADR-0018). Order is the order of the baseline ladder.

``PHASE2_POLICIES`` are the eight Phase 2 baselines, run by default by ``run_baselines``.
``frontier_shared``, ``auction_comm`` and ``auction_no_reassign`` need a scenario with a
communication channel (ADR-0018, ADR-0019).
"""

from __future__ import annotations

from typing import Final

from swarm.policies.auction import AUCTION_COMM, AUCTION_NO_REASSIGN
from swarm.policies.base import PolicyFactory, TeamFactory
from swarm.policies.centralized import (
    ASSIGNMENT,
    ASSIGNMENT_HAZARD_AWARE,
    ASSIGNMENT_ORACLE,
    PRIORITIZED_FRONTIER,
)
from swarm.policies.decentralized import ASTAR_WAYPOINT, FRONTIER, FRONTIER_HAZARD_AWARE
from swarm.policies.random_valid import random_valid_factory
from swarm.policies.shared import FRONTIER_SHARED

POLICIES: Final[dict[str, PolicyFactory | TeamFactory]] = {
    "random_valid": random_valid_factory,
    "astar_waypoint": ASTAR_WAYPOINT,
    "frontier": FRONTIER,
    "frontier_hazard_aware": FRONTIER_HAZARD_AWARE,
    "prioritized_frontier": PRIORITIZED_FRONTIER,
    "assignment": ASSIGNMENT,
    "assignment_hazard_aware": ASSIGNMENT_HAZARD_AWARE,
    "assignment_oracle": ASSIGNMENT_ORACLE,
    "frontier_shared": FRONTIER_SHARED,
    "auction_comm": AUCTION_COMM,
    "auction_no_reassign": AUCTION_NO_REASSIGN,
}

PHASE2_POLICIES: Final[tuple[str, ...]] = (
    "random_valid",
    "astar_waypoint",
    "frontier",
    "frontier_hazard_aware",
    "prioritized_frontier",
    "assignment",
    "assignment_hazard_aware",
    "assignment_oracle",
)

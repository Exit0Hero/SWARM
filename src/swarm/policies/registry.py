"""Registered baseline policies by name (ADR-0016). Order is the order of the baseline ladder."""

from __future__ import annotations

from typing import Final

from swarm.policies.base import PolicyFactory, TeamFactory
from swarm.policies.centralized import (
    ASSIGNMENT,
    ASSIGNMENT_HAZARD_AWARE,
    ASSIGNMENT_ORACLE,
    PRIORITIZED_FRONTIER,
)
from swarm.policies.decentralized import ASTAR_WAYPOINT, FRONTIER, FRONTIER_HAZARD_AWARE
from swarm.policies.random_valid import random_valid_factory

POLICIES: Final[dict[str, PolicyFactory | TeamFactory]] = {
    "random_valid": random_valid_factory,
    "astar_waypoint": ASTAR_WAYPOINT,
    "frontier": FRONTIER,
    "frontier_hazard_aware": FRONTIER_HAZARD_AWARE,
    "prioritized_frontier": PRIORITIZED_FRONTIER,
    "assignment": ASSIGNMENT,
    "assignment_hazard_aware": ASSIGNMENT_HAZARD_AWARE,
    "assignment_oracle": ASSIGNMENT_ORACLE,
}

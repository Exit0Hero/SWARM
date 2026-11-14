"""Test helpers: small configs and agent type specs."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from swarm.core.observation import Observation
from swarm.core.types import AgentId, Layer
from swarm.scenarios.config import (
    DRONE,
    GROUND,
    AgentSpec,
    AgentTypeSpec,
    ScenarioConfig,
    TerrainConfig,
)
from swarm.scenarios.generator import Scenario, parse_ascii_map


def make_config(
    *,
    height: int = 8,
    width: int = 8,
    debris: float = 0.15,
    structure: float = 0.15,
    survivors: int = 4,
    drones: int = 2,
    grounds: int = 2,
    time_limit: int = 30,
    p_detect: float = 0.9,
    base: tuple[int, int] | None = None,
    agent_types: tuple[AgentTypeSpec, ...] = (DRONE, GROUND),
    roster: tuple[AgentSpec, ...] | None = None,
) -> ScenarioConfig:
    """Build a small valid config; keyword arguments override individual fields."""
    if roster is None:
        roster = tuple([AgentSpec("drone")] * drones + [AgentSpec("ground")] * grounds)
    return ScenarioConfig(
        height=height,
        width=width,
        base_position=base,
        terrain=TerrainConfig(debris_fraction=debris, structure_fraction=structure),
        survivor_count=survivors,
        agent_types=agent_types,
        roster=roster,
        time_limit=time_limit,
        p_detect=p_detect,
    )


def spec(
    name: str,
    layer: Layer,
    *,
    radius: int = 1,
    capacity: int = 10,
    move: int = 1,
    stay: int = 0,
    recharge: int = 0,
) -> AgentTypeSpec:
    """Build an agent type spec with small, easy-to-reason-about defaults."""
    return AgentTypeSpec(
        name=name,
        layer=layer,
        sensor_radius=radius,
        energy_capacity=capacity,
        move_cost=move,
        stay_cost=stay,
        recharge_rate=recharge,
    )


def config_for(
    scenario_rows: list[str],
    agent_types: tuple[AgentTypeSpec, ...],
    roster: tuple[AgentSpec, ...],
    *,
    time_limit: int = 20,
    p_detect: float = 1.0,
) -> tuple[ScenarioConfig, Scenario]:
    """Config plus explicit scenario for a hand-drawn ASCII map (see parse_ascii_map)."""
    scenario = parse_ascii_map(scenario_rows)
    h, w = scenario.shape
    config = ScenarioConfig(
        height=h,
        width=w,
        base_position=scenario.base,
        terrain=TerrainConfig(debris_fraction=0.0, structure_fraction=0.0),
        survivor_count=len(scenario.survivors),
        agent_types=agent_types,
        roster=roster,
        time_limit=time_limit,
        p_detect=p_detect,
    )
    return config, scenario


def mask_random_actions(
    observations: Mapping[AgentId, Observation], rng: np.random.Generator
) -> dict[AgentId, int]:
    """Uniform random valid action per agent (test helper, independent of swarm.policies)."""
    actions: dict[AgentId, int] = {}
    for aid in sorted(observations):
        valid = [a for a, ok in enumerate(observations[aid].action_mask) if ok]
        actions[aid] = valid[int(rng.integers(len(valid)))]
    return actions

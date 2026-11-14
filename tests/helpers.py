"""Test helpers: small configs and agent type specs."""

from __future__ import annotations

from swarm.core.types import Layer
from swarm.scenarios.config import (
    DRONE,
    GROUND,
    AgentSpec,
    AgentTypeSpec,
    ScenarioConfig,
    TerrainConfig,
)


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

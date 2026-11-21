"""Phase 8: MuJoCo 3D replay prototype (skipped without the optional sim3d extra)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from swarm.core.rng import episode_streams
from swarm.core.types import Terrain
from swarm.experiments.runner import run_episode
from swarm.policies.registry import POLICIES
from swarm.scenarios.config import CommsConfig
from swarm.scenarios.generator import generate_scenario
from tests.helpers import make_config

pytest.importorskip("mujoco")

from swarm.sim3d.mujoco_replay import build_mjcf, replay


def _episode(policy: str) -> tuple[Any, Any, Any]:
    comms = CommsConfig(ranges=(("drone", 8), ("ground", 5)))
    config = replace(make_config(drones=1, grounds=1, time_limit=40), comms=comms)
    agents = tuple(config.agents())
    result = run_episode(config, POLICIES[policy], 3, designation="development")
    scenario = generate_scenario(config, episode_streams(3, agents).scenario)
    layers = {aid: spec.layer for aid, (spec, _) in config.agents().items()}
    return scenario, layers, result


def test_extruded_world_has_one_box_per_blocking_cell() -> None:
    scenario, layers, _ = _episode("frontier")
    xml = build_mjcf(scenario, layers)
    blocking = int(
        ((scenario.terrain == Terrain.STRUCTURE) | (scenario.terrain == Terrain.DEBRIS)).sum()
    )
    assert xml.count('type="box"') == blocking + 1  # plus the base plate
    assert all(f'name="{aid}"' in xml for aid in layers)


@pytest.mark.parametrize("policy", ["frontier", "auction_comm"])
def test_two_d_legal_paths_are_collision_free_in_three_d(policy: str) -> None:
    scenario, layers, result = _episode(policy)
    report = replay(scenario, layers, result.events)
    assert report.collisions == 0
    assert report.agents == 2
    assert report.steps >= 1

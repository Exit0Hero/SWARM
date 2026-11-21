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


# ----------------------------------------------------------------- 3D scenes (ADR-0027)

from swarm.sim3d.scene import FLIGHT_ALTITUDE, GROUND_Z, agent_z, build_scene  # noqa: E402


def _view(terrain: list[list[int]]) -> dict[str, Any]:
    h, w = len(terrain), len(terrain[0])
    return {
        "shape": [h, w],
        "terrain": terrain,
        "hazard": [[0] * w for _ in range(h)],
        "base": [0, 0],
        "survivors_ground_truth": [],
    }


def _frame(agents: list[tuple[str, str, list[int]]], detections: list[Any]) -> dict[str, Any]:
    return {
        "agents": [
            {"id": a, "layer": layer, "pos": pos, "active": True} for a, layer, pos in agents
        ],
        "detections": detections,
    }


def test_drones_fly_except_on_the_base_and_ground_robots_stay_low() -> None:
    assert agent_z("air", [0, 0], [0, 0]) < 1
    assert agent_z("air", [0, 1], [0, 0]) == FLIGHT_ALTITUDE
    assert agent_z("ground", [0, 1], [0, 0]) == GROUND_Z


def test_line_of_sight_is_blocked_by_a_structure_but_not_from_the_air() -> None:
    from swarm.sim3d.mujoco_replay import analyze_scene

    view = _view([[0, 2, 0], [0, 0, 0]])  # a structure at row 0, column 1
    target = {"agent": "g", "cell": [0, 2], "true": True}
    ground = [_frame([("g", "ground", [0, 0])], [target])]
    air = [_frame([("d", "air", [1, 2])], [{"agent": "d", "cell": [0, 2], "true": True}])]
    blocked = analyze_scene(build_scene(view, ground), ground)
    clear = analyze_scene(build_scene(view, air), air)
    assert (blocked["reports_checked"], blocked["reports_occluded"]) == (1, 1)
    assert (clear["reports_checked"], clear["reports_occluded"]) == (1, 0)


def test_a_report_on_a_debris_cell_is_not_occluded_by_its_own_box() -> None:
    from swarm.sim3d.mujoco_replay import analyze_scene

    view = _view([[0, 0, 1]])
    frames = [_frame([("g", "ground", [0, 0])], [{"agent": "g", "cell": [0, 2], "true": False}])]
    assert analyze_scene(build_scene(view, frames), frames)["reports_occluded"] == 0


def test_scene_tracks_follow_the_recorded_episode() -> None:
    from swarm.dashboard import service
    from swarm.sim3d.mujoco_replay import analyze_scene

    scenario, layers, _ = _episode("auction_comm")
    del scenario, layers
    comms = CommsConfig(ranges=(("drone", 8), ("ground", 5)))
    config = replace(make_config(drones=1, grounds=1, time_limit=40), comms=comms)
    record = service.record_episode(config, "auction_comm", 3)
    scene = build_scene(record["scenario"], record["frames"])
    for aid, track in scene["tracks"].items():
        cells = [[p[1], p[0]] for p in track["points"]]
        assert cells == [
            next(a["pos"] for a in f["agents"] if a["id"] == aid) for f in record["frames"]
        ]
    report = analyze_scene(scene, record["frames"])
    assert report["contacts"] == 0  # 2D-legal movement is contact-free in 3D
    assert report["reports_checked"] == sum(len(f["detections"]) for f in record["frames"])

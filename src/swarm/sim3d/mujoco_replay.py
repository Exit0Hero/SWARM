"""Kinematic 3D replay of a SWARM episode in MuJoCo (Phase 8 feasibility prototype, ADR-0024).

The 2D simulator stays the source of truth: the scenario is extruded into a MuJoCo world (structures
as tall boxes, debris as low boxes, the base as a flat plate) and each agent becomes a mocap body
that follows its event-stream path, drones at flight altitude and ground robots on the floor. The
prototype measures what a 3D back end would cost and checks that every 2D-legal path is
collision-free in 3D. It adds no dynamics, sensing or policy behaviour.

Example::

    python -m swarm.sim3d.mujoco_replay --config configs/phase3/nominal.json \\
        --policy auction_comm --seed 1000
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swarm.core.types import AgentId, Layer, Terrain
from swarm.scenarios.generator import Scenario
from swarm.sim3d.scene import (
    CELL,
    DEBRIS_HEIGHT,
    FLIGHT_ALTITUDE,
    GROUND_Z,
    STRUCTURE_HEIGHT,
)
from swarm.visualization.replay import replay_state


@dataclass(frozen=True, slots=True)
class ReplayReport:
    steps: int
    agents: int
    boxes: int
    build_seconds: float
    seconds_per_step: float
    collisions: int


def build_mjcf(scenario: Scenario, layers: dict[AgentId, Layer]) -> str:
    """MJCF XML for the extruded scenario with one mocap body per agent."""
    geoms = []
    h, w = scenario.terrain.shape
    for r in range(h):
        for c in range(w):
            t = Terrain(int(scenario.terrain[r, c]))
            height = {Terrain.STRUCTURE: STRUCTURE_HEIGHT, Terrain.DEBRIS: DEBRIS_HEIGHT}.get(t)
            if height is None:
                continue
            geoms.append(
                f'<geom name="{t.name.lower()}_{r}_{c}" type="box" '
                f'size="{CELL / 2} {CELL / 2} {height / 2}" pos="{c} {r} {height / 2}" '
                'contype="1" conaffinity="2"/>'
            )
    br, bc = scenario.base
    agents = []
    for aid, layer in sorted(layers.items()):
        z = FLIGHT_ALTITUDE if layer is Layer.AIR else GROUND_Z
        agents.append(
            f'<body name="{aid}" mocap="true" pos="{bc} {br} {z}">'
            f'<geom type="sphere" size="0.2" contype="2" conaffinity="1"/></body>'
        )
    return (
        '<mujoco model="swarm"><worldbody>'
        f'<geom name="floor" type="plane" size="{w} {h} 0.1" pos="{w / 2} {h / 2} 0" '
        'contype="0" conaffinity="0"/>'
        f'<geom name="base" type="box" size="0.5 0.5 0.01" pos="{bc} {br} 0.01" '
        'contype="0" conaffinity="0"/>' + "".join(geoms) + "".join(agents) + "</worldbody></mujoco>"
    )


def replay(scenario: Scenario, layers: dict[AgentId, Layer], events: Sequence[Any]) -> ReplayReport:
    """Replay every agent's path step by step and count agent-obstacle contacts."""
    import mujoco

    start = time.perf_counter()
    model = mujoco.MjModel.from_xml_string(build_mjcf(scenario, layers))
    data = mujoco.MjData(model)
    build = time.perf_counter() - start
    traces = replay_state(scenario, tuple(layers), events, until=None)
    steps = max(len(t.path) for t in traces.values())
    mocap = {aid: model.body_mocapid[model.body(aid).id] for aid in layers}
    collisions = 0
    start = time.perf_counter()
    for k in range(steps):
        for aid, trace in traces.items():
            r, c = trace.path[min(k, len(trace.path) - 1)]
            z = FLIGHT_ALTITUDE if layers[aid] is Layer.AIR else GROUND_Z
            data.mocap_pos[mocap[aid]] = (c, r, z)
        mujoco.mj_forward(model, data)
        collisions += int(data.ncon)
    per_step = (time.perf_counter() - start) / max(steps, 1)
    boxes = sum(1 for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX)
    return ReplayReport(steps, len(layers), boxes - 1, build, per_step, collisions)


def main(argv: Sequence[str] | None = None) -> int:
    from swarm.core.rng import episode_streams
    from swarm.experiments.runner import run_episode
    from swarm.policies.registry import POLICIES
    from swarm.scenarios.config import ScenarioConfig
    from swarm.scenarios.generator import generate_scenario

    parser = argparse.ArgumentParser(prog="python -m swarm.sim3d.mujoco_replay")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy", choices=sorted(POLICIES), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--scene-out", type=Path, help="also write the 3D scene and its analysis (JSON)"
    )
    args = parser.parse_args(argv)
    config = ScenarioConfig.from_json(args.config.read_text(encoding="utf-8"))
    result = run_episode(config, POLICIES[args.policy], args.seed, designation="development")
    agents = tuple(config.agents())
    scenario = generate_scenario(config, episode_streams(args.seed, agents).scenario)
    layers = {aid: spec.layer for aid, (spec, _) in config.agents().items()}
    t0 = time.perf_counter()
    run_episode(config, POLICIES[args.policy], args.seed, designation="development")
    sim2d = (time.perf_counter() - t0) / max(result.steps, 1)  # whole 2D step incl. policies
    report = replay(scenario, layers, result.events)
    print(report, f"2d_seconds_per_step={sim2d:.6f}")
    if args.scene_out is not None:
        import json

        from swarm.dashboard.service import record_episode, scene3d

        record = record_episode(config, args.policy, args.seed)
        out = scene3d(record)
        args.scene_out.write_text(json.dumps(out, indent=1), encoding="utf-8")
        a = out["analysis"]
        print(
            f"scene: {a['boxes']} boxes, {a['contacts']} contacts, "
            f"{a['reports_occluded']}/{a['reports_checked']} reports occluded -> {args.scene_out}"
        )
    return 0 if report.collisions == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


# ----------------------------------------------------------------- scene analysis (ADR-0027)


def scene_mjcf(scene: dict[str, Any]) -> str:
    """MJCF for a :func:`swarm.sim3d.scene.build_scene` scene, one mocap body per track."""
    h, w = scene["shape"]
    geoms = [
        f'<geom name="{b["kind"]}_{b["y"]}_{b["x"]}" type="box" '
        f'size="{CELL / 2} {CELL / 2} {b["h"] / 2}" pos="{b["x"]} {b["y"]} {b["h"] / 2}" '
        'contype="1" conaffinity="2"/>'
        for b in scene["boxes"]
    ]
    bodies = []
    for aid, track in sorted(scene["tracks"].items()):
        x, y, z = track["points"][0]
        bodies.append(
            f'<body name="{aid}" mocap="true" pos="{x} {y} {z}">'
            '<geom type="sphere" size="0.2" group="1" contype="2" conaffinity="1"/></body>'
        )
    return (
        '<mujoco model="swarm_scene"><worldbody>'
        f'<geom name="floor" type="plane" size="{w} {h} 0.1" pos="{w / 2} {h / 2} 0" '
        'contype="0" conaffinity="0"/>' + "".join(geoms) + "".join(bodies) + "</worldbody></mujoco>"
    )


def analyze_scene(scene: dict[str, Any], frames: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Step every track through MuJoCo: contacts per step and 3D line of sight per report.

    A detection report is *occluded* when the ray from the reporting robot's 3D position to the
    reported cell (0.05 m above the floor) first hits a terrain box other than the reported cell's
    own. The 2D sensor model has no occlusion (ADR-0012), so this measures how many 2D reports a
    line-of-sight sensor would not have made. It does not change the episode.
    """
    import mujoco
    import numpy as np

    start = time.perf_counter()
    model = mujoco.MjModel.from_xml_string(scene_mjcf(scene))
    data = mujoco.MjData(model)
    build = time.perf_counter() - start
    mocap = {aid: model.body_mocapid[model.body(aid).id] for aid in scene["tracks"]}
    terrain_only = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)  # robots are geom group 1
    contacts = checked = occluded = 0
    by_agent: dict[str, list[int]] = {aid: [0, 0] for aid in scene["tracks"]}
    geom_id = np.zeros(1, dtype=np.int32)
    start = time.perf_counter()
    for k, frame in enumerate(frames):
        for aid, track in scene["tracks"].items():
            data.mocap_pos[mocap[aid]] = track["points"][min(k, len(track["points"]) - 1)]
        mujoco.mj_forward(model, data)
        contacts += int(data.ncon)
        for det in frame["detections"]:
            src = np.array(scene["tracks"][det["agent"]]["points"][k], dtype=np.float64)
            r, c = det["cell"]
            dst = np.array([c, r, 0.05])
            vec = dst - src
            dist = float(np.linalg.norm(vec))
            checked += 1
            by_agent[det["agent"]][0] += 1
            if dist < 1e-9:
                continue
            hit = mujoco.mj_ray(model, data, src, vec / dist, terrain_only, 1, -1, geom_id)
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id[0])) or ""
            blocked = 0 <= hit < dist - 1e-6 and not name.endswith(f"_{r}_{c}")
            if blocked:
                occluded += 1
                by_agent[det["agent"]][1] += 1
    seconds = time.perf_counter() - start
    return {
        "engine": f"mujoco {mujoco.__version__}",
        "steps": len(frames),
        "boxes": len(scene["boxes"]),
        "build_seconds": build,
        "seconds_per_step": seconds / max(len(frames), 1),
        "contacts": contacts,
        "reports_checked": checked,
        "reports_occluded": occluded,
        "occluded_fraction": occluded / checked if checked else None,
        "by_agent": {aid: {"checked": a, "occluded": b} for aid, (a, b) in by_agent.items()},
        "note": "kinematic co-simulation; the 2D episode is unchanged (ADR-0027)",
    }

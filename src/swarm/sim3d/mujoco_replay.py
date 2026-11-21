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
from swarm.visualization.replay import replay_state

CELL = 1.0
STRUCTURE_HEIGHT = 2.0
DEBRIS_HEIGHT = 0.4
FLIGHT_ALTITUDE = 3.0
GROUND_Z = 0.25


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
    return 0 if report.collisions == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

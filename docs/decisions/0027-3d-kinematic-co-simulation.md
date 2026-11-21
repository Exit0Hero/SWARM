# ADR-0027: 3D kinematic co-simulation and visualization

- **Status:** Accepted. Supersedes part of [ADR-0024](0024-3d-feasibility-decision.md): 3D
  environment, playback and visualization are now implemented. Flight dynamics remain deferred as
  ADR-0024 decided.
- **Date:** 2026-11-21
- **Phase:** 8

## Context
The user asked for a functional 3D environment: positions and movement, obstacles and collisions,
sensor integration, hazards, coordination, playback and 3D visualization. They required two
things:
- the 2D simulator stays intact;
- no claims of physics accuracy.

## Decision
1. **One authority.** Episodes still run in the 2D simulator with the existing policies (classical,
   auction and learned). Movement, sensing and communication are unchanged.
2. **Scene.** `swarm.sim3d.scene.build_scene` (no dependencies) lifts a recorded episode into 3D:
   - structure boxes are 2 m tall, debris 0.4 m;
   - hazards are floor markers;
   - one track per robot with one point per step: ground robots at 0.25 m; drones at 0.15 m on the
     base and 3 m elsewhere.

   The altitude profile is a kinematic convention.
3. **MuJoCo co-simulation.** `swarm.sim3d.mujoco_replay.analyze_scene`, in the optional `sim3d`
   extra, steps every track through MuJoCo as mocap bodies and measures:
   - robot–obstacle contacts, as a check that 2D-legal motion is collision-free in 3D;
   - 3D line of sight for every detection report, cast with `mj_ray` against terrain geoms only.
     Robots are in geom group 1 and excluded. The reported cell's own box does not count as a
     blocker.

   The 2D sensor has no occlusion (ADR-0012), so the occluded fraction measures how many reports a
   line-of-sight sensor would not have made. It is reported, not fed back into the episode.
4. **Visualization.**
   - The dashboard's "3D view" (ADR-0026) renders the scene with an orthographic projection and
     yaw/pitch control, using the same replay controls as the 2D map. It draws robots, altitudes,
     trails, message links, hazards and survivors (ground truth).
   - `GET /api/runs/<id>/scene3d` returns the scene and the MuJoCo analysis, or a note when
     `mujoco` is not installed.
5. **CLI.** `python -m swarm.sim3d.mujoco_replay --config … --policy … --seed … --scene-out
   scene.json`.

## Not done (still deferred)
- Rigid-body flight dynamics, controllers, wind or rotor models.
- 3D sensing that changes what agents observe.
- Policies that plan in 3D.
- Rendering from MuJoCo (no GPU/EGL in this environment).

## Consequences
- **Tests:** `tests/test_sim3d.py`, which is skipped without `mujoco`, and
  `tests/test_dashboard.py`. They check:
  - a known occlusion case and a known clear one;
  - that tracks follow the recorded frames exactly;
  - zero contacts for real episodes.
- **The 2D path is unchanged:** golden hashes pass.

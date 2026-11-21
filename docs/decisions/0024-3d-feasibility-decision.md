# ADR-0024: 3D feasibility study (Phase 8): defer a full 3D simulator

- **Status:** Accepted (decision: **defer**)
- **Date:** 2026-11-21
- **Phase:** 8

## Context
Phase 8 asks for four things: an evidence-based simulator comparison, an integration prototype, a
performance and cost assessment, and a decision to proceed or defer.

## Simulator comparison (from published descriptions; not benchmarked here)
| Option | Strengths | Costs for SWARM |
|---|---|---|
| MuJoCo (`pip install mujoco`) | CPU, pure-Python install. Fast and accurate rigid-body physics. MJX GPU vectorization. A PettingZoo multi-drone wrapper exists (MuJoCo-Drones-Gym, 2026). | No photoreal rendering. No ground-vehicle or ROS stack out of the box. |
| PyBullet / gym-pybullet-drones | The most widely used multi-quadrotor Gym environment, with downwash and drag models. | The engine is older and is reported slower and less accurate than MuJoCo. |
| Gazebo / Webots | Mature robot models, sensors and ROS integration. | A separate process and stack. Heavy for seeded batch studies. |
| Isaac Sim / Aerial Gym | Photoreal RTX rendering and massive GPU parallelism. | Needs an NVIDIA GPU. Large install. Rendering only pays off with image perception (Phase 5, deferred). |
| AirSim (Unreal) | High-fidelity rendering. | Needs large computational resources. |

## Integration prototype
`swarm.sim3d.mujoco_replay`, available through the optional `sim3d` extra:
- The 2D scenario is extruded into a MuJoCo world: structures become 2 m boxes, debris 0.4 m boxes.
- Every agent becomes a mocap body that follows its event-stream path. Drones fly at 3 m; ground
  robots stay at floor level.
- The 2D simulator remains the source of truth. The prototype adds no dynamics.
- Tests (`tests/test_sim3d.py`):
  - one box per blocking cell;
  - paths that are legal in 2D have zero contacts in 3D, for `frontier` and `auction_comm`.

## Performance (development seed 1000, one shared 2-core CPU; wall clock, not deterministic)
| Scenario, policy | 2D step (environment + policies) | 3D kinematic replay step | World build |
|---|---|---|---|
| nominal 20×20, `frontier_shared` | 4.0 ms | 0.009 ms | 3 ms, 107 boxes |
| nominal 20×20, `auction_comm` | 6.4 ms | 0.007 ms | 3 ms |
| map_40, `auction_comm` | 13.0 ms | 0.010 ms | 8 ms, 477 boxes |

- **Kinematic replay is negligible** next to policy cost: below 0.1% of a step.
- **Not measured:** full flight dynamics, meaning quadrotor control at physics rates with several
  sub-steps per decision. Published environments use their own control rates. Our cost model would
  need its own benchmark before any 3D study.

## Decision: defer full 3D simulation, and keep the prototype
1. **The open research questions are about allocation and communication, not flight dynamics.**
   Phase 7 found the cost and the effects on the policy side: target selection, belief merging and
   the auction. A 3D engine would add cost without touching them.
2. **Photoreal options pay off only with image perception**, and that phase (Phase 5) is deferred
   by the user.
3. **If 3D is resumed**, the recommended path is MuJoCo:
   - CPU install, no new process, PettingZoo precedent.
   - The prototype already maps scenarios and agents.
   - The next steps are a dynamics benchmark and a waypoint-tracking controller beneath the
     existing macro actions.
   - Isaac Sim becomes relevant only if GPU resources and an image-perception study exist.

## Consequences
- The core simulator is unchanged. `mujoco` is an optional extra, and CI type-checks without it.
- Phase 8 is complete with a **defer** decision. A reversal needs a new ADR.

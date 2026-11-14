# ADR-0002: Two-layer world model

- **Status:** Accepted
- **Date:** 2026-11-15
- **Phase:** 1

## Context
SWARM has drones and ground robots, but on a single 2D grid their difference was never
defined. Without a real difference, "heterogeneous agents" would just be different labels
on the same thing. [docs/07](../07_AGENT_AND_ACTION_MODEL.md) requires that differences be
configuration, not scattered conditionals.

## Decision
1. **Two occupancy layers:** `AIR` and `GROUND`. Every agent type belongs to exactly one layer.
   Agents on different layers never conflict with each other.
2. **Terrain types** (one per cell), with traversability per layer:

   | Terrain | AIR | GROUND | Meaning |
   |---|---|---|---|
   | `OPEN` | yes | yes | Clear ground |
   | `DEBRIS` | yes | no | Collapsed material that a drone can fly over |
   | `STRUCTURE` | no | no | Standing building or tall obstacle |
   | `BASE` | yes | yes | Deployment and recharge point |

   Terrain is stored as a small integer array. The traversability table is data, not code
   branches.
3. **Base:** exactly one base cell per scenario in Phase 1. All agents spawn there. The base
   cell has unlimited capacity: it is exempt from vertex conflicts. An agent on the base
   recharges by a configured amount per step.
4. **Hazards** are an overlay grid, separate from terrain. A hazardous cell keeps its terrain's
   traversability and adds an extra energy cost on entry, plus a configurable per-step chance of
   disabling the agent. The core supports this from Phase 1, but the first Phase 1 milestone
   (M1) uses no hazards.
5. **Agent types are data:** an `AgentTypeSpec` gives `layer`, `sensor_radius`,
   `energy_capacity`, per-action `energy_cost`, `recharge_rate`, and later `comm_range`.
   Default roster:

   | Type | Layer | Sensor radius | Energy | Role |
   |---|---|---|---|---|
   | `drone` | AIR | larger (default 3) | smaller (default 200) | Wide-area search |
   | `ground` | GROUND | smaller (default 1) | larger (default 600) | Ground access |

6. **Survivors** are static points placed on `OPEN` or `DEBRIS` cells, never on `STRUCTURE` or
   `BASE`. A survivor under debris can be detected from the air but not reached on the ground.
   This is an intended asymmetry, and it becomes relevant when reaching or confirming survivors
   is added (see [ADR-0009](0009-mission-objectives-and-metrics.md)).

## Alternatives considered
- **Single layer, with differences only in parameters:** does not capture the main real difference
  (drones pass over rubble), and would cause unrealistic drone–robot "collisions".
- **Altitude as a continuous variable:** out of scope for 2D.
- **Subclasses per agent type:** rejected in favour of data specs, as docs/07 requires.

## Consequences
- Coverage needs a denominator that respects layers (ADR-0009).
- Scenario generation must guarantee the base is reachable and must report cells that no
  agent type can reach.
- A third agent type (for example, a large UAV) is a new spec, not new code.

## Amendment (Phase 1 M2)
Rule 4 (hazards) is made precise by [ADR-0011](0011-static-hazards.md), which governs placement, affected layers,
per-step exposure, damage, disablement draws and observability.

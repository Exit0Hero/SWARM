# 2D Simulation Specification

The binding rules are in the ADRs. This document gives the overview and links to them.

## World
The world is a 2D, 4-connected grid with discrete time ([ADR-0001](decisions/0001-grid-and-discrete-time.md)).
Each cell has one terrain type (`OPEN`, `DEBRIS`, `STRUCTURE`, `BASE`), and its traversability depends on
the agent's layer (`AIR` or `GROUND`). A hazard overlay is optional
([ADR-0002](decisions/0002-two-layer-world-model.md)). The cell model is extensible through new terrain types
and traversability-table entries.

## Agents
Each agent has:
- stable ID and type (type parameters are an `AgentTypeSpec`, not subclasses);
- layer (`AIR` or `GROUND`) and position;
- sensor radius (≥ 1) and detection model;
- communication range (Phase 3);
- integer energy capacity, current level, per-action costs and recharge rate;
- status: `active` or `disabled`. Returning to base is a policy behaviour, not an environment status
  ([ADR-0004](decisions/0004-action-masks-and-invalid-actions.md)).

## Hidden state and observations
Survivor locations are hidden from agents. A local observation contains only what lies inside the agent's
sensor footprint, its own state, and its action mask
([ADR-0008](decisions/0008-observation-and-hidden-state-boundaries.md)). Do not leak hidden state into
decentralized policies.

## Actions
`Discrete(5)`: stay, north, east, south, west. Sensing is passive every step. Communication becomes a separate
channel in Phase 3. Return-to-base is composed of moves by a policy. The environment validates actions
centrally and exposes an action mask ([ADR-0004](decisions/0004-action-masks-and-invalid-actions.md)).

## Step order
1. Validate proposed actions (contract violations raise; masked actions become stay plus an event).
2. Resolve movement simultaneously per layer: vertex, swap and blocked conflicts make the movers stay; cycles of three
   or more succeed; the base is exempt ([ADR-0003](decisions/0003-simultaneous-movement-and-collision-resolution.md)).
3. Apply energy costs and base recharge. Disable agents at zero energy away from base.
4. Update hazards and environment dynamics.
5. Generate detections. 5b. When a channel is configured, transmit this step's messages at post-move positions
   and deliver those due ([ADR-0017](decisions/0017-communication-channel.md)).
6. Update mission state, termination and truncation ([ADR-0009](decisions/0009-mission-objectives-and-metrics.md)).
7. Emit the ordered event list ([ADR-0005](decisions/0005-event-driven-simulation.md)).

Tie-breaking never depends on agent ID order or container iteration order, so seeded runs are reproducible
([ADR-0006](decisions/0006-random-number-generation-and-reproducibility.md)).

## Implementation details (Phase 1 M1 and M2)
- **Spawn:** every agent starts on the base with its configured `initial_energy` (default: full capacity).
- **Energy accounting:** a successful move costs `move_cost`; staying, a failed move and an invalid action cost
  `stay_cost`. An agent never pays more than it has, so energy never goes negative. After paying, an agent on the
  base gains up to `recharge_rate`, capped at capacity. An agent that ends a step at zero energy away from the base
  is disabled. On the base it simply recharges.
- **Disabled agents** stay where they are, keep occupying the cell (ADR-0003 clarifications), stop sensing, and
  leave `env.agents`.
- **Detection draws:** one draw from the `dynamics` stream for each (active agent, survivor inside its footprint)
  pair, in sorted agent order and then survivor order. Reported with probability `p_detect`.
- **False reports (M2):** after an agent's true reports, one draw from the `false_positive` stream per in-grid
  footprint cell without a survivor, in row-major order, reported with probability `p_false_positive`. Policies see
  only the merged set of reported cells ([ADR-0012](decisions/0012-false-positive-detections-and-precision.md)).
- **Reset** observations include terrain and visible agents but no detection reports. Sensing happens during steps.
- **Episode end:** after sensing, `mission_success` when global recall reaches `recall_target` (exact rational
  comparison), otherwise `all_disabled` when no agent is active, otherwise `time_limit` at the last step, in that
  precedence. The first two set `terminated`; `time_limit` sets `truncated`. Zero-survivor episodes never succeed
  and run until all are disabled or time runs out ([ADR-0009](decisions/0009-mission-objectives-and-metrics.md)).
  Success and all-disabled cannot co-occur in practice, because disabled agents do not sense.
- **Hazards (M2):** phase 4. An active agent of an affected layer that ends the step on a hazard loses
  `energy_cost` (capped), is disabled on depletion, and otherwise makes one `hazard`-stream draw for disablement
  with probability `disable_probability` ([ADR-0011](decisions/0011-static-hazards.md)).
- **Generation:** terrain is drawn independently per cell. The base is the configured cell or the grid centre.
  Reachability is not enforced, so a survivor may be unreachable. Coverage uses eligible (reachable) cells only.
  Hazards are drawn last (one uniform per cell, `OPEN`/`DEBRIS` only), so terrain and survivors do not depend on
  the hazard settings.

## Scenario configuration
Map dimensions, terrain generation parameters, hazard parameters, base location, survivor count and placement
constraints, agent roster with type specs, time limit, sensor settings, communication settings, recall target,
and seed. The format is versioned JSON ([ADR-0007](decisions/0007-configuration-and-serialization.md)).

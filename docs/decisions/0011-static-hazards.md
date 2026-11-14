# ADR-0011: Static hazards

- **Status:** Accepted
- **Date:** 2026-11-15
- **Phase:** 1 (M2)

## Context
[ADR-0002](0002-two-layer-world-model.md) rule 4 sketched a hazard overlay with an entry energy cost and a chance of
disabling the agent, but left placement, layer effects, timing and event semantics open. M2 needs static hazards that
are deterministic given the configuration and RNG streams.

## Decision
1. **Representation.** Hazards are a boolean overlay, `Scenario.hazards`, with the same shape as the terrain. Terrain
   and traversability do not change: a hazardous cell can be entered like any other cell of its terrain type. Hazards
   are static for the whole episode.
2. **Placement.** Only `OPEN` and `DEBRIS` cells can be hazardous. `STRUCTURE` and `BASE` never are. Each eligible cell is
   hazardous independently with probability `hazards.fraction`. The draw uses the `scenario` stream, *after* terrain
   and survivor placement, so enabling hazards never changes terrain or survivor positions for the same seed.
   Survivors may lie in hazardous cells.
3. **Affected layers.** `hazards.affected_layers` is a non-empty subset of `{air, ground}`. Agents on other layers
   ignore hazards entirely.
4. **Exposure.** An *active* agent of an affected layer that **ends a step** (after movement) in a hazardous cell is
   exposed once for that step. This applies whether it moved in or stayed. Entering and leaving within one step is
   impossible, because agents move at most one cell.
5. **Effect, in order:**
   1. it loses `hazards.energy_cost` energy units, capped at its remaining energy, so energy never goes negative;
   2. if its energy is now zero, it is disabled with reason `energy_depleted`;
   3. otherwise one draw from the `hazard` stream ([ADR-0013](0013-environment-rng-substreams.md)) disables it with
      probability `hazards.disable_probability`, reason `hazard`.

   An exposed agent that is not disabled keeps acting normally next step: hazards do not slow it, block it or change
   its mask.
6. **Step order.** Hazards are phase 4 of the step order: after energy and recharge (phase 3), before sensing
   (phase 5). An agent disabled in phase 3 is not exposed. An agent disabled in phase 4 does not sense this step. It
   stays in its cell as an obstacle, as with any disabled agent
   ([ADR-0003](0003-simultaneous-movement-and-collision-resolution.md) clarifications).
7. **Events.** Per exposed agent, in sorted agent order: `hazard_exposure(step, agent_id, cell, energy_lost,
   remaining)`, always emitted even when `energy_lost` is 0; then, if applicable,
   `agent_disabled(step, agent_id, position, reason)` with reason `energy_depleted` or `hazard`.
8. **Draw rule.** Exactly one `hazard`-stream draw per exposure that did not already end in `energy_depleted`, in
   sorted agent order. The draw is taken even when `disable_probability` is 0 or 1, so the stream position depends
   only on exposures.
9. **Observation.** Hazards are observable inside the sensor footprint: `Observation.local_hazard` is a boolean view
   aligned with `local_terrain`. It is `False` outside the footprint, which there means "unknown", not "safe". Hazards
   outside the footprint are never visible ([ADR-0008](0008-observation-and-hidden-state-boundaries.md)).
10. **Configuration.** `hazards` is optional (`null` = no hazard layer, and no hazard draws at all).
    `fraction ∈ [0, 1]`, `energy_cost ≥ 0`, `disable_probability ∈ [0, 1]`.

## Alternatives considered
- **A separate health variable:** adds a second resource with no current research use. Energy damage plus a disablement
  chance captures "hazards cost resources and can destroy agents".
- **Hazards that block movement:** that is a terrain type, not a hazard. It would also change masks and reachability.
- **Damage on entry only:** lets agents loiter in hazards for free. Per-step exposure is simpler and more conservative.
- **Hidden hazards:** plausible for some hazards (gas), but invisible hazards make avoidance impossible for any policy.
  This can be added later as a scenario option.

## Consequences
- New metrics: hazard exposures, hazard energy lost, and disablements by cause ([ADR-0009](0009-mission-objectives-and-metrics.md)).
- Hazard-free scenarios behave and hash exactly as in M1.
- Required tests: placement on allowed terrain only, layer filtering, energy boundary at zero, disablement probability
  0 and 1, exposure when staying, no exposure for disabled agents, event order, and determinism.

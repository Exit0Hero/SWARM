# ADR-0008: Observation and hidden-state boundaries

- **Status:** Accepted
- **Date:** 2026-11-15
- **Phase:** 1

## Context
Decentralized policies must not see hidden global state unless an experiment declares an oracle
baseline ([docs/08](../08_PERCEPTION_AND_COMMUNICATION.md), [docs/15](../15_RISK_REGISTER_AND_LIMITATIONS.md)).
A boundary enforced only by convention is easy to break by accident and hard to detect in results.

## Decision
1. **What an agent knows at the start:** grid dimensions, base position, its own type spec, and the
   roster of agent IDs and types. Terrain beyond its sensor footprint is **unknown** (no prior map in
   Phase 1).
2. **Per-step observation** (a typed, immutable object) contains only:
   - own position, energy, status;
   - terrain inside the sensor footprint (cells within Euclidean `sensor_radius`);
   - other agents inside the footprint (ID, type, position);
   - this step's detection reports, as cell coordinates, possibly noisy;
   - `action_mask` ([ADR-0004](0004-action-masks-and-invalid-actions.md));
   - in Phase 3, messages received this step.
3. **Never in an observation:** survivor ground truth, hazard parameters outside the footprint, other
   agents' energy (unless communicated), RNG state, or the environment object itself.
4. **Detection model (Phase 1):** each step, each true survivor inside an active agent's footprint is
   reported with probability `p_detect` (default 0.9). False positives (a per-cell rate inside the
   footprint) are added in the second Phase 1 milestone. No occlusion in Phase 1.
5. **Policies receive only `(observation, rng)`.** A policy that needs global information (for example,
   centralized assignment with known survivor locations) must request an explicit `OracleView`
   object. The runner then records `oracle: true` in its results, and every report must label it.
6. **Memory is the policy's job.** Coverage maps, beliefs and visited sets live in the policy (or, later,
   the communication layer), not in the observation.

## Alternatives considered
- **Full map known, only survivors hidden:** plausible (pre-disaster maps exist), but it makes exploration
  trivial. It can be added later as a scenario option ("prior map available").
- **A global observation with a "don't look" convention:** unenforceable.

## Consequences
- Required test: two scenarios identical except for survivor positions *outside every footprint* must
  produce identical observations. This is the hidden-state leakage test.
- Requires `sensor_radius ≥ 1` (also needed by ADR-0004).
- The Phase 4 MAPPO critic may use a centralized state. That is allowed in training only, and it is documented
  under centralized training with decentralized execution in [docs/06](../06_MARL_RESEARCH_PLAN.md).

## Amendment (Phase 1 M2)
Observations gain `local_hazard`: hazards inside the footprint, aligned with `local_terrain`
([ADR-0011](0011-static-hazards.md)). `detections` merges true and false reports with no labels
([ADR-0012](0012-false-positive-detections-and-precision.md)). Required tests now also cover hazards outside every
footprint, which must not change observations.

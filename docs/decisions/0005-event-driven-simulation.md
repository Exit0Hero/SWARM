# ADR-0005: Event-driven simulation architecture

- **Status:** Accepted
- **Date:** 2026-11-15
- **Phase:** 1

## Context
The docs require several separations: simulation from visualization, metrics from policies, and
reward from mission success ([docs/02](../02_SYSTEM_ARCHITECTURE.md),
[docs/06](../06_MARL_RESEARCH_PLAN.md), [docs/12](../12_VISUALIZATION_DASHBOARD.md)). Without a
shared mechanism, each consumer would reach into environment internals.

## Decision
1. Each `step()` produces an ordered list of **typed, immutable events**: frozen dataclasses with
   `step`, `kind`, `agent_id` (where applicable), and fields specific to the kind. The Phase 1
   kinds are `moved`, `move_conflict`, `invalid_action`, `energy_spent`, `recharged`,
   `agent_disabled`, `detection`, `episode_end`. Phase 3 adds `message_sent`, `message_delivered` and
   `message_dropped`. False-positive detections carry a ground-truth flag that is used only by evaluation.
2. **Event order** follows the step order in [docs/04](../04_SIMULATION_SPECIFICATION.md) (actions,
   movement, energy, hazards, sensing, mission state). Within a phase, events are sorted by agent ID.
   Equal inputs therefore always produce identical event lists.
3. **The core computes no reward.** Rewards are pluggable functions in `swarm.learning` (Phase 4)
   that read events and snapshots, and they log each reward component separately.
4. **Consumers are read-only:** metrics, reward functions, run loggers, text and graphical
   renderers. They receive events and immutable state snapshots, never a mutable environment
   reference.
5. Events carry a `schema_version`. The episode event stream serializes to JSON Lines and is
   the basis of the determinism test: an identical seed and config must give a byte-identical stream.

## Alternatives considered
- **Metrics computed inside `step()`:** couples evaluation to dynamics. Adding a metric would mean
  touching the core.
- **Reward computed in the core:** ties the simulator to one reward design and invites reward
  hacking to leak into baselines.
- **Callbacks or observers:** the same information, but harder to replay and to test.

## Consequences
- Metrics can be recomputed after the fact from stored event logs.
- Event streams can be large. Logging granularity (all events or summaries) is a runner option.
- The PettingZoo adapter (Phase 4) supplies the reward dictionary that the core leaves out
  ([ADR-0010](0010-api-contract.md)).

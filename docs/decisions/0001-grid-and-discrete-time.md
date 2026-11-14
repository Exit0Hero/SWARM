# ADR-0001: 2D grid and discrete time

- **Status:** Accepted
- **Date:** 2026-11-15
- **Phase:** 1

## Context
The earlier briefs left the world model open ("grid- or continuous 2D environment"). The research
questions in [docs/06](../06_MARL_RESEARCH_PLAN.md) are about coordination, communication and
partial observability, not flight dynamics. Classical multi-agent pathfinding (Phase 2)
and discrete-action MARL (Phase 4) are both most mature on grids.

## Decision
1. The world is a rectangular grid of `height × width` cells, stored as NumPy arrays of shape
   `(height, width)` and indexed `(row, col)` with `(0, 0)` at the top-left.
2. Movement is 4-connected (north, east, south, west). No diagonal moves.
3. Time is a discrete step counter `t = 0, 1, …`. Each `step()` call advances `t` by exactly
   one. An episode is truncated when `t` reaches the scenario's `time_limit`.
4. In Phase 1 every agent moves at most one cell per step. Agent speed differences are **not**
   modelled in Phase 1. Heterogeneity comes from layer, sensing, energy and costs
   ([ADR-0002](0002-two-layer-world-model.md)). Multi-cell moves per step need a new ADR because
   they change conflict resolution ([ADR-0003](0003-simultaneous-movement-and-collision-resolution.md)).
5. Distances for sensing and communication are Euclidean between cell centres.

## Alternatives considered
- **Continuous 2D:** more realistic kinematics, but it complicates collision checking, pathfinding
  and action spaces, and adds no value to the current research questions.
- **8-connected grid:** shorter paths, but diagonal corner-cutting rules and √2 costs add
  edge cases to collision resolution with little scientific benefit at this stage.
- **Per-type speed via cooldowns** (a ground robot moves every *k* steps): deferred. It is a
  reasonable later extension, but it complicates baselines before they exist.

## Consequences
- Simple, exact state, so it's easy to test and to replay deterministically.
- Drones cannot yet be "faster" than ground robots. This is a stated limitation, recorded in
  [docs/15](../15_RISK_REGISTER_AND_LIMITATIONS.md).
- Grid resolution is a modelling choice. Results must state the grid size.

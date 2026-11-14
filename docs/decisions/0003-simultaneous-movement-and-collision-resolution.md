# ADR-0003: Simultaneous movement and collision resolution

- **Status:** Accepted
- **Date:** 2026-11-15
- **Phase:** 1

## Context
[docs/04](../04_SIMULATION_SPECIFICATION.md) asks for consistent movement resolution with
documented tie-breaking. [docs/05](../05_EXPERIMENTS_AND_METRICS.md) asks for a collision count. If
the simulator prevents collisions, a raw "collision count" is always zero, so the metric needs
a definition that measures something real.

## Decision
All agents act simultaneously. On each layer independently:

1. **Static check.** Actions are first validated against static rules (bounds, terrain, agent
   status, energy; [ADR-0004](0004-action-masks-and-invalid-actions.md)). Invalid actions become
   `STAY`.
2. **Vertex conflict.** If two or more agents move to the same non-base cell, *all* of them stay.
   No agent wins by ID, so the rule has no ordering bias.
3. **Swap conflict.** If two agents would exchange cells, both stay.
4. **Blocked by a stationary agent.** A move into a cell whose occupant is staying fails, and the
   mover stays. This is repeated to a fixed point, because a failed move can block another move
   (chains).
5. **Cycles.** Rotations of length three or more, where every agent moves into the cell being
   vacated by the next, succeed. No agent in the cycle is stationary, and no pair swaps.
6. **Base exemption.** The base cell is never a conflict target (unlimited capacity).
7. Each failed move emits a `move_conflict` event naming the agent, its intended target and the
   conflict kind (`vertex`, `swap`, `blocked`). A failed move is charged the `STAY` energy cost.

**Invariant:** after every step, no two agents on the same layer share a non-base cell. Physical
collisions cannot happen by construction. The collision metric is therefore **move conflicts**:
the number of `move_conflict` events, which counts coordination failures.

**Determinism:** the outcome depends only on the set of (agent, action) pairs, not on dictionary
order or agent ID order. This permutation invariance is a required property test in Phase 1.

## Alternatives considered
- **Lower ID wins:** simple, but it biases outcomes toward particular agents and makes results
  depend on naming.
- **Random winner:** requires consuming randomness in movement and makes conflicts harder to analyse.
- **Sequential moves in a fixed order:** not simultaneous, and it gives early movers an advantage.
- **Allow collisions and count them:** physically meaningless for agents that would crash or deadlock.

## Consequences
- Deadlocks are possible (two agents repeatedly blocking each other). Policies, not the
  environment, must resolve them. Phase 2 planners use reservation tables.
- Required tests: vertex, swap, chain, three-cycle, base co-occupancy, cross-layer
  non-interaction, and permutation invariance.

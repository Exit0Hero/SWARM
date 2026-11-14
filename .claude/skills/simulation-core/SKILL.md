---
name: simulation-core
description: Implement or modify the SWARM 2D environment core - scenario generation, grid and terrain, agent state, movement and conflict resolution, energy, hazards, sensing, events, termination. Use for any change to environment dynamics or the reset and step contract, and for diagnosing nondeterminism.
---

# Simulation core

## Read first
- `docs/04_SIMULATION_SPECIFICATION.md` for the step order
- ADRs 0001–0008 and 0010 in `docs/decisions/`, especially:
  - 0002: layers, terrain and traversability, base, agent type specs
  - 0003: simultaneous movement resolution and the no-shared-cell invariant
  - 0004: action space, masks, invalid-action handling
  - 0005: event kinds and ordering
  - 0006: RNG streams and integer state

## Rules
- State is NumPy arrays plus frozen config objects. Positions and energy are integers.
- `step()` follows the documented phase order. Events within a phase are sorted by agent ID.
- Never read global RNG state. Each stochastic process has its own stream (ADR-0013): true detections use
  `dynamics`, hazard draws `hazard`, false reports `false_positive`. Draw in a fixed iteration order. A new
  process gets a new child stream, never a share of an existing one.
- The core computes no reward and imports nothing from policies, learning, evaluation or visualization.
- Contract violations (bad agent IDs, out-of-space actions, stepping after the episode ends) raise
  exceptions. Masked-but-valid actions become `STAY` plus an event.

## Procedure for a change
1. State which step phase(s) and which ADR rules the change affects.
2. Write failing tests first (use the testing-and-review skill). At minimum, a hand-built tiny grid with the
   exact expected events.
3. Implement the smallest change. Keep functions pure where possible: (state, actions) → (new state, events).
4. Run the determinism check: the same seed and config, run twice, must give an identical event stream
   (compare a hash of the JSON Lines).
5. Run the property tests (in bounds, terrain respected, no shared non-base cell per layer, energy ≥ 0,
   permutation invariance of joint actions).
6. If a rule changed, update the ADR (new superseding ADR) and `docs/04_SIMULATION_SPECIFICATION.md`.

## Diagnosing nondeterminism
Diff the two event streams and find the first differing event. Check, in order: set or dict iteration feeding a
draw, a draw made conditionally (which shifts later draws), float accumulation, and shared generators
between components.

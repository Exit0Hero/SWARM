# ADR-0004: Action masks and invalid actions

- **Status:** Accepted
- **Date:** 2026-11-15
- **Phase:** 1

## Context
[docs/07](../07_AGENT_AND_ACTION_MODEL.md) requires one consistent invalid-action strategy.
[docs/04](../04_SIMULATION_SPECIFICATION.md) originally listed eight actions: stay, four moves,
`scan`, `communicate`, `return-to-base`. `scan` and `communicate` would make sensing and messaging
compete with movement for the same action slot. `return-to-base` would require the environment to
plan a path, which pulls navigation into the core.

## Decision
1. **Action space:** `Discrete(5)`, with `0 STAY`, `1 NORTH`, `2 EAST`, `3 SOUTH`, `4 WEST`. North is
   decreasing row.
2. **Sensing is passive.** Every active agent senses at the end of every step
   ([ADR-0008](0008-observation-and-hidden-state-boundaries.md)). There is no `scan` action in Phase 1.
   An active, energy-costed scan is a possible later extension and would need a new ADR.
3. **Communication is a separate channel** introduced in Phase 3. It will not be a movement-exclusive
   action.
4. **Return-to-base is a policy behaviour,** made of ordinary moves using a planner. Agents recharge
   whenever they are on the base cell.
5. **Action mask.** Each observation includes `action_mask: bool[5]`. A move is masked out if it
   leaves the grid, enters terrain impassable for the agent's layer, or costs more energy than the
   agent has. `STAY` is always valid. Disabled agents may only `STAY`. The mask uses only static
   information the agent is entitled to see (its own position, energy and adjacent terrain). It does
   not predict other agents' moves.
6. **Two kinds of invalid input:**
   - *Inside the action space but masked out* (for example, a move into a structure): the action
     becomes `STAY`, an `invalid_action` event is recorded, and the agent is charged the `STAY` cost.
     This is a policy error that the simulation tolerates and measures.
   - *Outside the contract* (an integer not in 0–4, a missing or extra agent ID): the environment
     raises `ValueError`. This is a programming error and must not be hidden.

## Alternatives considered
- **Reject with an exception for every masked action:** makes the random baseline and exploring RL
  policies crash.
- **Silently map to `STAY` with no event:** hides policy bugs and makes the "invalid action" rate
  unmeasurable.
- **Composite actions (move + scan + message):** deferred until communication exists. The Phase 4
  MARL interface can then choose between a multi-discrete or a factored action space.

## Consequences
- MARL can use masking directly (MAPPO with invalid-action masking).
- Requires `sensor_radius ≥ 1` so that the mask does not reveal terrain the agent cannot see.
- [docs/04](../04_SIMULATION_SPECIFICATION.md) and [docs/07](../07_AGENT_AND_ACTION_MODEL.md) are
  updated to reference this ADR.

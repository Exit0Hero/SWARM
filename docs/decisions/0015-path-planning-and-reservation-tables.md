# ADR-0015: Path planning and prioritized planning with reservation tables

- **Status:** Accepted
- **Date:** 2026-11-15
- **Phase:** 2

## Context
Phase 2 adds single-agent A* and prioritized multi-agent planning. Planners must respect per-layer traversability
([ADR-0002](0002-two-layer-world-model.md)), the movement rules of [ADR-0003](0003-simultaneous-movement-and-collision-resolution.md)
(vertex and swap conflicts, base exemption, disabled agents as obstacles) and the information boundary
([ADR-0008](0008-observation-and-hidden-state-boundaries.md)).

## Decision

### Maps and unknown cells
1. Non-oracle planners plan on a **known map**: the union of `local_terrain` and `local_hazard` observations seen
   so far (own observations for decentralized policies, all agents' for pooled ones). It is a deterministic
   record of what was seen, not a probabilistic belief map (Phase 3).
2. **Unknown cells are impassable** (pessimistic). Every planned move enters a known traversable cell, so a planned
   action is never masked out for terrain reasons. The oracle plans on the full terrain.

### Single-agent A*
3. 4-connected grid; entering a cell costs `enter_cost[cell] ≥ 1` (all 1 for shortest path). Heuristic: Manhattan
   distance, admissible because every step costs at least 1.
4. **Deterministic tie-breaking:** the open list is ordered by `(f, h, row, col)`. Neighbours are expanded N, E, S, W.
   A parent is replaced only on strictly lower `g`.
5. Start equal to goal returns `[start]`. An impassable or unreachable goal returns `None`. The start is always
   allowed, even on an otherwise impassable cell.
6. **Cost models:** `shortest` (all costs 1) is the reference baseline. `hazard_aware` adds `hazard_penalty`
   (default 10) to entering a *known* hazard cell when the agent's layer is affected. It is a separately named variant,
   never applied silently.

### Prioritized planning (centralized pooled and oracle teams)
7. **Space-time A*** over `(cell, t)` with actions N, E, S, W and wait. Waiting costs 1. Moves cost `enter_cost`.
   The horizon is `H = 2 × (height + width)` steps. If no plan is found within `H`, the plan fails: the agent waits
   that step and a failed plan is counted.
8. **Reservation table** per layer, after the agent's plan is accepted:
   - vertex reservations `(cell, t)` for each step of its plan;
   - edge reservations `(from, to, t)` to forbid swaps;
   - a **goal hold**: the goal cell is reserved for every `t` after arrival up to `H`.

   A move from `a` at `t` to `b` at `t + 1` is forbidden if `(b, t+1)` is reserved, `b` is held from a time
   `≤ t + 1`, or the edge `(b, a, t)` is reserved. A goal is accepted at time `T` only if no reservation by another
   agent uses that cell after `T`.
9. **Base exemption:** the base cell is never reserved, held or conflict-checked, matching its unlimited capacity.
   Swaps through the base are allowed.
10. **Layers** are planned in separate reservation tables. Air and ground agents never constrain each other.
11. **Obstacles:** disabled agents' cells (last known positions) are impassable for their layer, except the base.
    Agents not yet planned in the current round hold a provisional reservation of their current non-base cell at
    `t = 1`. Higher-priority agents therefore do not move into a cell whose occupant might stay.
12. **Receding horizon:** all plans are recomputed every step from current positions, and only the first action is
    executed. The environment remains the authority on conflicts ([ADR-0003](0003-simultaneous-movement-and-collision-resolution.md)).
13. **Priority order:** in planning round `k` (the step index), agents are ordered by sorted agent ID rotated left by
    `k mod n` (`n` = agents planning that round). Over any `n` consecutive rounds each agent holds each rank exactly
    once. There is no permanent ID bias, and the order is deterministic.

### Amendment (PR #5 review, 2026-11-15)
- The space-time search accepts a per-cell **wait cost** (default 1, the original behaviour). Pooled planners use
  the hazard-aware penalty for waits in hazard-aware variants, and `max(1, stay_cost)` plus known-hazard exposure
  for returning agents, which also plan moves on energy costs ([ADR-0016](0016-energy-feasibility-and-classical-baselines.md) rule 16).
- A failed plan is classified as **unreachable** (no route on the planner's map, ignoring reservations) or
  **temporary** (reservations or the horizon).

## Alternatives considered
- **Optimistic unknown cells:** shorter plans, but planned moves can hit masked terrain, which counts as invalid
  actions. That is also a form of guessing.
- **Conflict-based search:** optimal, but much more complex. The path-planning skill requires measured conflict and
  deadlock rates before escalating. These are reported from the Phase 2 runs.
- **Fixed ID priority:** simple, but biased. The longest-remaining-path-first heuristic was also rejected, because a
  stuck agent could keep top priority forever.

## Consequences
- Complexity: A* is `O(V log V)` per call on `V = height × width` cells. Space-time A* is `O(V H log(V H))` in the
  worst case per agent per round, and much less in practice because the heuristic is informative. A static
  reachability check runs before every space-time search, so an unreachable goal fails fast.
- Priority fairness, reservation correctness, swap prevention, base exemption and conflict-free execution in the
  environment are tested.

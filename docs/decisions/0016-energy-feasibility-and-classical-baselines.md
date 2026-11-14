# ADR-0016: Energy feasibility and the classical baseline policies

- **Status:** Accepted
- **Date:** 2026-11-15
- **Phase:** 2

## Context
Phase 2 baselines must be scientifically comparable with the random valid-action policy. Agents must not commit to
routes they cannot return from. The task set, cost model and reassignment triggers of every allocator must be
explicit (task-allocation skill).

## Decision

### Energy model and feasibility rule
1. **Energy of a route** under planner assumptions: `move_cost` per move, plus `stay_cost` per planned wait, plus the
   hazard `energy_cost` for each move into a *known* hazard cell if the agent's layer is affected (from the mission
   brief, [ADR-0014](0014-team-policies-and-information-models.md)). Hazard *disablement* risk is not converted into
   energy. The hazard-aware variants reduce exposure instead.
2. **Energy to base** from any known cell is the minimum route energy to the base over the known map. It is computed
   for all cells at once by a reverse Dijkstra from the base. Recharging happens only on the base and is never
   assumed en route.
3. **Reserve** = `ceil(reserve_fraction × energy_capacity)`, with `reserve_fraction = 0.1` by default.
4. **Commit rule:** an agent may commit to target `T` along route `P` only if
   `energy − reserve ≥ E(P) + E_to_base(T)`.
   The rule is re-checked every step against the remaining route. If it fails, the agent switches to returning to
   base along a minimum-energy route.
5. **At the base:** an agent with no feasible target stays and recharges. It departs as soon as a target becomes
   feasible.
6. **Geometric versus energetic feasibility:** a target can be reachable on the map yet infeasible under the rule.
   Such targets are excluded, not attempted.

### Frontier
7. A **frontier cell** for layer `L` is a known cell, traversable for `L`, with at least one in-grid 4-neighbour that
   is unknown. Standing on or near it reveals unknown cells.
8. Targets are selected among frontier cells **reachable on the known map and feasible under the commit rule**. The
   nearest is chosen by route cost, with ties broken by `(row, col)`. The target is kept while it is still a
   frontier cell and still feasible (hysteresis). The agent replans when the target is exhausted (no longer a
   frontier), unreachable, infeasible or reached. Unreachable and exhausted targets are never re-selected while
   that remains true, because candidates are recomputed from the current map.

### Baseline policies
| Name | Information | Target choice | Paths |
|---|---|---|---|
| `random_valid` | decentralized | none (uniform valid action) | none |
| `astar_waypoint` | decentralized | uniform random known, reachable, feasible cell (policy stream) | A*, own map |
| `frontier` | decentralized | nearest feasible frontier | A*, own map, visible same-layer agents avoided |
| `frontier_hazard_aware` | decentralized | as `frontier`, with hazard-aware costs | hazard-aware A* |
| `prioritized_frontier` | centralized_pooled | greedy: agents in priority order take the nearest unclaimed feasible frontier | prioritized planning |
| `assignment` | centralized_pooled | min-cost assignment of agents to distinct frontier cells | prioritized planning |
| `assignment_hazard_aware` | centralized_pooled | as `assignment`, with hazard-aware costs | prioritized planning, hazard-aware |
| `assignment_oracle` | **oracle** | min-cost assignment to unreported survivors; target = nearest cell within sensor range | prioritized planning on full terrain |

9. **Assignment** uses the Hungarian method (Kuhn–Munkres with potentials, `O(n² m)` for `n ≤ m`), implemented in
   pure Python. It is tested against brute force. Infeasible pairs are excluded. Ties are broken by task order
   (row, col), then agent ID. **SciPy is not added.**
10. **Reassignment triggers:** an assigned target stops being valid (exhausted, or for the oracle, reported); an
    agent is disabled; an agent reaches its target; an agent becomes energy-infeasible. Every assignment change is
    logged with step, agent, target, cost and trigger.
11. **Policy diagnostics:** failed plans, planned waits, targets assigned and reached, mean completion time (steps
    from assignment to arrival), reassignments, returns to base, detected move failures and backoff waits.
12. **Randomized backoff (decentralized only):** when an agent's previous planned move did not happen (its
    position is not the expected cell), it waits this step with probability 0.5, using one draw from its own
    policy stream. Without this, identical deterministic agents leaving the base towards the same frontier
    collide every step and deadlock forever. A development-seed smoke test showed 800 conflicts in 200 steps
    with zero targets reached. Centralized planners do not need backoff, because reservations prevent
    within-team conflicts.

## Alternatives considered
- **Pure return-when-low thresholds:** ignore the target's distance from the base and strand agents far away.
- **Optimistic unknown cells for frontier paths:** rejected in [ADR-0015](0015-path-planning-and-reservation-tables.md).
- **SciPy `linear_sum_assignment`:** correct and fast, but a heavy dependency for problems of size ≤ 10 × 300.

## Consequences
- Agents following the commit rule should rarely run out of energy away from the base. Remaining depletions come
  from waits that the energy estimate did not account for, and are counted in `agents_disabled_energy`.
- The baselines form a ladder of ablations: random → planned motion (`astar_waypoint`) → informed targets
  (`frontier`) → pooled coordination (`prioritized_frontier`, `assignment`) → oracle reference.

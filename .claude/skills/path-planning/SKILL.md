---
name: path-planning
description: Implement or evaluate SWARM navigation - single-agent grid search (BFS, A*), multi-agent pathfinding (prioritized planning with reservation tables, later conflict-based search), return-to-base feasibility and deadlock handling. Use for any work in the navigation package or any policy that plans routes.
---

# Path planning

## Read first
- ADR-0001 (4-connected grid, unit-cost moves) and ADR-0002 (per-layer traversability) in `docs/decisions/`
- ADR-0003 for how the environment resolves conflicting moves. Planners should avoid conflicts, but the
  environment remains the authority.
- ADR-0008: a decentralized planner may only plan over terrain the agent has observed or been told about.

## Rules
- Planners are pure functions of a map *as known to the agent*, a start, a goal and a layer. They never take the
  environment or ground-truth terrain, except in a clearly labelled oracle baseline.
- Unknown cells need an explicit policy (optimistic: assume traversable; pessimistic: assume blocked). State which
  one is used.
- Heuristic: Manhattan distance (admissible on a 4-connected unit-cost grid).
- Tie-breaking must be deterministic. Order the heap by (f, h, row, col), never by object identity.
- Use the standard library `heapq`. Do not add a graph library without an architecture review.

## Procedure
1. Implement single-agent A* with tests: straight line, detour, unreachable goal, start equals goal, goal blocked,
   layer-specific terrain (a drone crosses debris, a ground robot cannot).
2. Return-to-base feasibility: an agent must start returning when
   `energy ≤ path_cost_to_base × move_cost + safety_margin`. Test the boundary exactly.
3. Multi-agent: prioritized planning with a space-time reservation table (vertex and swap reservations,
   matching ADR-0003). Record the priority order rule. Test vertex, swap and corridor cases.
4. Escalate to conflict-based search only if measured conflict or deadlock rates justify it. Show the numbers
   from real runs.
5. Report: path optimality checks against BFS on small maps, planning time per call, and conflicts and
   deadlocks per episode from actual evaluation runs.

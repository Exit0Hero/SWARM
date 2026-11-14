---
name: task-allocation
description: Design, implement or evaluate SWARM task allocation and coordination - search-region partitioning, centralized assignment, decentralized auction or market methods, reassignment after agent failure, and communication-limited coordination. Use for work in the coordination package or any multi-agent baseline policy.
---

# Task allocation and coordination

## Read first
- `docs/05_EXPERIMENTS_AND_METRICS.md` (initial baselines) and ADR-0009 (metrics) in `docs/decisions/`
- `docs/08_PERCEPTION_AND_COMMUNICATION.md` (communication model, information boundaries)
- ADR-0008: observation boundary and the `OracleView` rule

## Rules
- State the **information model** of every allocator: centralized with full information (oracle),
  centralized with pooled observations, or decentralized with communication only. Put it in the policy
  name and in the results.
- Tasks are explicit objects (for example, a frontier cell or region to search, or a survivor to revisit) with a
  cost model that is documented: path length from the path-planning skill, energy, and travel-then-return feasibility.
- Allocation must be deterministic given the inputs and RNG stream (ADR-0006). Break cost ties by task ID, then
  agent ID.
- Never let a decentralized allocator read another agent's state except through received messages.

## Procedure
1. Define the task set and cost function, and write unit tests on hand-built instances with known optimal
   assignments.
2. Implement in order, measuring each against the previous: greedy nearest-task → centralized optimal
   assignment (Hungarian method; SciPy may be added in Phase 2 after an architecture review) → decentralized
   auction with communication limits.
3. Reassignment: define the trigger (agent disabled, task completed, new information) and test it with an agent
   failure scenario.
4. Evaluate on the same evaluation seeds as all other baselines (experiment-analysis skill). Report recall, time to
   recall target, coverage, energy, move conflicts and message delivery rate.
5. Never describe an allocator as better without a seed-matched comparison that includes confidence intervals.

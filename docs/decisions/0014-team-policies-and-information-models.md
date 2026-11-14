# ADR-0014: Team policies, mission brief, oracle view and information labels

- **Status:** Accepted
- **Date:** 2026-11-15
- **Phase:** 2

## Context
Phase 2 needs centralized baselines (prioritized multi-agent planning, centralized assignment) that decide for all
agents at once, plus an oracle reference. The ADR-0010 `Policy` acts for one agent. Planning policies also need
public mission parameters, such as the hazard energy cost, that are not in an observation. [ADR-0008](0008-observation-and-hidden-state-boundaries.md)
requires oracle access to go through an explicit `OracleView` and to be labelled in results.

## Decision
1. **Team policy.** A `TeamPolicy` maps the dict of current observations (one per agent that acted last step,
   including agents disabled in that step) and the per-agent policy RNG streams to one action per *active* agent.
   It exposes `name`, `version`, `uses_oracle`, `information`, `diagnostics()` and `decision_log()`.
2. **Factories.** A `TeamFactory(name, build)` builds a team policy from a `MissionBrief` at the start of every
   episode. Existing per-agent `PolicyFactory`s still work: the runner wraps them in `IndependentTeam`, which calls
   each agent's policy in sorted agent order with its own stream, exactly as in Phase 1. Random-baseline event streams
   and golden hashes are therefore unchanged.
3. **Mission brief (public information).** Grid shape, base, time limit, each agent's ID and type spec, the hazard
   *model* (affected layers, energy cost, disable probability, but not hazard locations), `p_detect`,
   `p_false_positive` and `recall_target`. It **excludes** survivor count and locations, hazard locations, terrain,
   and the hazard and terrain generation fractions.
4. **Information labels**, recorded in every result as `policy.information`:
   - `decentralized`: each agent decides from its own observations only. Other agents are visible only inside its
     footprint, and there is no information sharing.
   - `centralized_pooled`: one planner sees *all agents' observations* (perfect, instantaneous communication is
     assumed) but no hidden state. This is an idealized upper bound on communication, not a communication model.
   - `oracle`: additionally reads an `OracleView` (terrain, hazards, survivor positions, base). It is a reference
     upper bound and must never be reported as decentralized or pooled performance.
5. **Oracle access.** Only a team policy with `uses_oracle = True` receives an `OracleView`, through
   `attach_oracle(view)`, which the runner calls after `reset`. `SwarmEnv.oracle_view()` builds it as read-only
   copies. A non-oracle policy never receives one. The runner raises an error if an oracle policy does not accept the view.
6. **Diagnostics.** Policies may report numeric `diagnostics()` (for example failed plans, waits, reassignments) and
   a `decision_log()` (for example assignment decisions with step, agent, target, cost and trigger). They go into
   episode results; experiments summarize the numeric diagnostics like metrics. Diagnostics must be deterministic,
   so no wall-clock times.

## Alternatives considered
- **Coordinator shared behind per-agent policies:** per-agent `act` calls arrive one at a time, so a coordinator
  cannot see the whole team's observations before the first agent acts.
- **Passing the config to policies:** it contains survivor count and generation parameters that are not public
  knowledge. The brief is an explicit allow-list.
- **SciPy for assignment:** see [ADR-0016](0016-energy-feasibility-and-classical-baselines.md). Not needed.

## Consequences
- The episode and experiment schemas gain `policy.information`, `policy_diagnostics` and `decision_log`.
- Leakage tests cover team policies: observations and brief only, unless the policy is labelled oracle.
- Communication-limited coordination (Phase 3) will be a fourth information model. It must not reuse the
  `centralized_pooled` label.

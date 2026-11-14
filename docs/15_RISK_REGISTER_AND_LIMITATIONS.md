# Risk Register and Limitations

| Risk | Impact | Mitigation |
|---|---|---|
| Premature 3D scope | High schedule cost | Prove 2D dynamics and metrics first |
| Reward hacking | Misleading learned behavior | Evaluate task metrics and inspect trajectories |
| Hidden-state leakage | Invalid decentralized results | Enforce observation boundaries and tests |
| Non-reproducible runs | Weak research claims | Seed control and complete run metadata |
| Simulator simplification | Overstated real-world relevance | Document assumptions and avoid field-readiness claims |
| Dependency sprawl | Fragile setup | Add dependencies only with a clear need |
| Dashboard coupled to core | Slow experiments and brittle tests | Event/snapshot read-only interface |
| Slow Python step loop limits RL training | MARL phase stalls | NumPy array state from Phase 1; profile before Phase 4 |
| Silent bugs in an in-repo MAPPO implementation | Wrong conclusions | Validate on a known small task; compare against independent PPO and heuristic baselines |
| Scope creep across nine phases | Nothing finished well | A phase closes only when its acceptance criteria pass (`docs/14_ROADMAP.md`) |
| Instruction drift between briefs and docs | Inconsistent work | Single roadmap; ADRs override numbered docs; documentation skill |
| Oracle results mistaken for achievable performance | Overstated claims | `information` label in every result; oracle rows marked as references only |
| Decentralized livelocks and deadlocks | Stranded agents, misleading baselines | Traffic conventions (ADR-0016 rules 12–14), regression tests, per-episode diagnostics |

## Limitations
Modelling limitations accepted for Phase 1 (see the ADRs): a 4-connected grid; all agents move at most one cell per step,
so drones are not faster than ground robots; static survivors; no occlusion; no prior map; and a single base.
Hazards are static, have a single type per scenario, and act through energy loss and a disablement probability.
They are not physical fire, gas or flood models. Detection is an independent Bernoulli draw per step and survivor
in range. False reports are independent per footprint cell and step, with no localization error and no
confirmation rule, so location-level precision falls in long episodes under any non-zero false-positive rate.
There is still no communication. Centralized baselines assume perfect, instantaneous communication. Decentralized
baselines see others only within their footprint, use simple traffic conventions, and can still occasionally strand
an agent, because the energy rule does not budget for waiting. Prioritized planning is incomplete: it can fail where a
joint plan exists. Assignment uses frontier cells as tasks and does not model information gain. Agent-type parameters
(costs, capacities, sensor radii) are illustrative defaults, not calibrated to any real platform. Results from the
random baseline describe that baseline on these synthetic maps only, and are not evidence about drones or
ground robots in general.

SWARM is initially a research simulator. Simulated performance does not establish safety, reliability, or effectiveness in real disaster response. Real deployment would require domain experts, validated hardware, operational procedures, regulatory approvals, and extensive field testing.

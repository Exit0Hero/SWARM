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

## Limitations
Modelling limitations accepted for Phase 1 (see the ADRs): a 4-connected grid; all agents move at most one cell per step,
so drones are not faster than ground robots; static survivors; no occlusion; no prior map; and a single base.

SWARM is initially a research simulator. Simulated performance does not establish safety, reliability, or effectiveness in real disaster response. Real deployment would require domain experts, validated hardware, operational procedures, regulatory approvals, and extensive field testing.

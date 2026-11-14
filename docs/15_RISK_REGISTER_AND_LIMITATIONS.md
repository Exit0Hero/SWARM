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

## Limitations
SWARM is initially a research simulator. Simulated performance does not establish safety, reliability, or effectiveness in real disaster response. Real deployment would require domain experts, validated hardware, operational procedures, regulatory approvals, and extensive field testing.

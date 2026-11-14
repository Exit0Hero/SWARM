# System Architecture

## Design
Use a layered Python architecture with explicit interfaces and minimal coupling.

```text
Scenario Config
     |
Scenario Generator ---> Map / Hazards / Survivors
     |
Environment Core <---- Agent State
     |                      |
Observation Builder     Agent Policies
     |                      |
Communication Layer <--- Coordination / Task Allocation
     |
Step / Reward / Termination
     |
Experiment Runner ---> Metrics / Logs / Artifacts
     |
Visualization / Analysis
```

## Suggested package boundaries
- `swarm.core`: environment state, clock, transitions, events.
- `swarm.scenarios`: scenario schema, generators, validation.
- `swarm.agents`: agent models, capabilities, resource state.
- `swarm.observations`: local/global observations and sensor noise.
- `swarm.navigation`: grid path planning and collision handling.
- `swarm.coordination`: assignment, messaging, reassignment.
- `swarm.policies`: random, greedy, heuristic, learned policy adapters.
- `swarm.learning`: training adapters and algorithms.
- `swarm.evaluation`: metrics, benchmark suites, statistical summaries.
- `swarm.visualization`: optional rendering and dashboard.
- `swarm.experiments`: configs, runners, result storage.

Subpackages are created only when they contain working, tested code (see
[repository conventions](decisions/README.md#repository-conventions)). The list above is the target layout, not
the current one.

## Implemented modules (Phases 1 and 2)
For M1, agent models and observations live in `swarm.core` and `swarm.scenarios` rather than in separate
`swarm.agents` and `swarm.observations` packages. Agent types are configuration data (ADR-0002), so a separate
package would be empty. Dependencies point downward only: `experiments` → `policies`, `evaluation`,
`visualization` → `core` → `scenarios.config`.

| Module | Contents |
|---|---|
| `src/swarm/core/types.py` | `Layer`, `Terrain`, `Action`, `AgentStatus`, traversability table |
| `src/swarm/core/rng.py` | Per-episode RNG streams and sub-streams, seed-derivation record (ADR-0006, ADR-0013) |
| `src/swarm/core/movement.py` | Pure simultaneous move resolution (ADR-0003) |
| `src/swarm/core/events.py` | Typed events, canonical JSON Lines, stream hash (ADR-0005) |
| `src/swarm/core/observation.py` | `Observation`, footprint geometry (ADR-0008) |
| `src/swarm/core/env.py` | `SwarmEnv`, `StepResult`, `EnvSnapshot` (ADR-0010) |
| `src/swarm/scenarios/config.py` | Frozen configs, strict JSON, config hash, default agent types (ADR-0007) |
| `src/swarm/scenarios/generator.py` | `Scenario` ground truth with hazard overlay, seeded generator, ASCII maps, reachability |
| `src/swarm/core/oracle.py` | `OracleView`: the only channel to ground truth for policies (ADR-0014) |
| `src/swarm/navigation/` | Known map, A* and cost fields, energy model, reservation tables and prioritized planning (ADR-0015, ADR-0016) |
| `src/swarm/coordination/assignment.py` | Pure-Python Hungarian assignment (ADR-0016) |
| `src/swarm/policies/` | `Policy` and `TeamPolicy` protocols, `MissionBrief`, decentralized and centralized baselines, registry (ADR-0014, ADR-0016) |
| `src/swarm/evaluation/metrics.py` | Event-replay metric registry with ground-truth cross-checks (ADR-0009) |
| `src/swarm/evaluation/statistics.py` | Summaries, percentile bootstrap, paired differences (ADR-0009) |
| `src/swarm/experiments/runner.py` | `run_episode`, result record, metadata |
| `src/swarm/experiments/run_episode.py` | Command-line entry point (one episode) |
| `src/swarm/experiments/experiment.py` | `run_experiment`, `compare_experiments`, result records |
| `src/swarm/experiments/run_experiment.py`, `src/swarm/experiments/compare.py` | Command-line entry points (multi-seed, comparison) |
| `src/swarm/experiments/run_baselines.py` | All registered policies on identical seeds, plus pre-registered comparisons |
| `src/swarm/visualization/text.py` | Read-only text renderer |

## Event stream
`Step / Reward / Termination` in the diagram is implemented as an ordered stream of typed events. Metrics, rewards,
logging and visualization consume events and read-only snapshots, and never mutate the environment. The core
computes no reward ([ADR-0005](decisions/0005-event-driven-simulation.md)).

## Architectural constraints
- Environment transitions must not depend on rendering.
- Policies consume observations and return valid actions; they should not mutate the environment directly.
- Scenario generation must accept an explicit seed.
- Keep algorithm-specific dependencies optional where practical.
- Define interfaces before adding parallel implementations.

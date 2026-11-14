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

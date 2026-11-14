# Product Requirements Document (PRD)

## 1. Vision
SWARM is an extensible research platform for studying cooperative autonomous agents in simulated disaster-response missions.

## 2. Users
- Researchers studying multi-agent systems and MARL.
- Students learning robotics simulation and cooperative AI.
- Developers building reproducible search-and-rescue experiments.

## 3. Core use cases
1. Configure a disaster scenario and random seed.
2. Run multiple heterogeneous simulated agents.
3. Observe exploration, survivor discovery, communication, and resource use.
4. Compare classical and learned policies.
5. Export metrics and experiment artifacts.

## 4. MVP scope
- 2D map and discrete simulation clock.
- Obstacles and configurable hazard zones.
- Drone and ground-robot agent types with distinct movement/resource parameters.
- Hidden survivor locations and noisy observations.
- Battery/energy budget.
- Basic communication range and dropout.
- Baseline search and task-allocation policies.
- Reproducible evaluation and machine-readable results.

## 5. Out of scope for MVP
- Physical drone control or flight certification.
- Operational deployment in active disaster zones.
- Photorealistic 3D graphics.
- Real victim identification from real-world imagery.
- Claims of field readiness.

## 6. Success criteria
- Same seed and configuration produce the same scenario and deterministic baseline trace.
- Agents cannot move through blocked cells or exceed configured resources.
- Metrics can be exported and compared across policies.
- Baselines run without a GPU.
- Tests cover core transition, observation, reward, and termination behavior.

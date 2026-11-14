# SWARM — Master Project Prompt

You are the principal software architect, multi-agent systems engineer, reinforcement-learning researcher, simulation engineer, and technical writer for SWARM.

## Project
**SWARM: Autonomous Multi-Agent Disaster Response and Search & Rescue Intelligence**

Build a research-oriented platform in which simulated aerial drones and ground robots coordinate to:
- explore a disaster area;
- search for simulated survivors;
- share uncertain observations over imperfect communication links;
- allocate and reallocate search tasks;
- navigate around obstacles and dynamic hazards;
- manage battery, payload, and time constraints;
- return useful, measurable mission outcomes.

## What makes this more than a simulator
The platform must support comparative experiments across:
- multi-agent pathfinding and collision avoidance;
- centralized and decentralized coordination;
- task allocation;
- partial observability and communication constraints;
- classical planning and heuristic baselines;
- multi-agent reinforcement learning, introduced only after reliable baselines;
- perception noise and false-positive/false-negative detections.

## Development strategy
### Stage 0 — Repository and research foundation
Inspect the repository, document its current state, define interfaces, pin dependencies, and establish tests and reproducibility conventions.

### Stage 1 — 2D simulator
Implement a grid- or continuous 2D environment with terrain, obstacles, hazards, survivor locations, agent state, battery, time steps, and deterministic seeded scenarios.

### Stage 2 — Classical baselines
Implement random/greedy search, nearest-target assignment, centralized assignment, and a collision-aware path planner. Establish baseline metrics.

### Stage 3 — Multi-agent coordination
Add decentralized local observations, message passing, communication range/dropout/latency, task reassignment, and robust handling of agent failure.

### Stage 4 — MARL
Add a PettingZoo-compatible multi-agent interface where appropriate. Implement a simple training baseline before MAPPO. Keep algorithm code modular and document observation, action, reward, and termination definitions. Do not claim MAPPO is implemented until it is implemented and tested.

### Stage 5 — Perception
Introduce synthetic sensor observations and a replaceable perception interface. Start with controlled noise models; add computer vision only when there is a clear dataset/simulation need.

### Stage 6 — Visualization and analysis
Provide a lightweight dashboard or renderer for scenarios, trajectories, detections, communication, agent status, and experiment comparisons. Visualization must not be coupled to simulation correctness.

### Stage 7 — 3D extension
Only after 2D benchmarks are stable, evaluate a 3D simulator such as a robotics simulator. Document integration cost and scientific value before adoption.

## Required engineering outcomes
- Reproducible scenario generation using explicit seeds.
- Unit and integration tests.
- Baseline algorithms.
- Experiment configuration and result logging.
- Metrics with precise definitions.
- Ablation and comparison support.
- Documentation of limitations and assumptions.
- A clear separation between simulated capabilities and real-world claims.

## Rules for Claude Code
1. Inspect before editing. Preserve existing work.
2. Do not create placeholder implementations disguised as completed features.
3. Do not fabricate benchmark results, citations, repository findings, or test outcomes.
4. Prefer the simplest architecture that supports the research questions.
5. Ask before destructive operations, major dependency changes, or broad refactors.
6. Deliver work in small milestones with acceptance criteria.
7. After each milestone, report changed files, tests run, results, known limitations, and next step.

## First task
Inspect the current repository and report:
- current file tree and stack;
- existing implementation and tests;
- gaps against the staged architecture;
- recommended minimal next milestone;
- risks and dependency choices.

Do not start a large implementation until the inspection report is complete.

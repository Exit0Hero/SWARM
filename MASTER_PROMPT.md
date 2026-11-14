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
Phases are numbered 0–8. Acceptance criteria and current status are in `docs/14_ROADMAP.md`, which is
the single source of truth. Design rules are in the ADRs in `docs/decisions/`.

### Phase 0 — Foundation
Repository structure, tooling, CI, project skills, and ADRs for the core design decisions.

### Phase 1 — Deterministic 2D core
A 4-connected grid with air and ground layers, terrain, survivors, heterogeneous agents, integer energy,
simultaneous movement resolution, action masks, passive noisy sensing, a typed event stream and seeded
scenarios (ADR-0001 to ADR-0008). Delivered as milestones M1 ("one honest episode") and M2.

### Phase 2 — Navigation and classical baselines
A* and multi-agent path planning, return-to-base feasibility, greedy exploration, centralized assignment,
and baseline metrics on fixed evaluation seeds.

### Phase 3 — Coordination under constraints
Local beliefs, message passing with range, dropout and latency, decentralized and auction-based task
allocation, reassignment, and robust handling of agent failure.

### Phase 4 — Multi-agent reinforcement learning
A PettingZoo-compatible adapter, event-based rewards, an independent-learner baseline, then MAPPO. Document
observation, action, reward and termination definitions. Do not claim MAPPO is implemented until it is
implemented and tested.

### Phase 5 — Perception
A replaceable perception interface and synthetic sensor benchmarks. Add computer vision only when there is a
clear dataset and evaluation need.

### Phase 6 — Visualization dashboard and analytics
Live and replay views of scenarios, trajectories, detections, communication, agent status and experiment
comparisons. Visualization must not be coupled to simulation correctness.

### Phase 7 — Research experiments and reporting
Pre-registered experiments on the research questions, with confidence intervals and documented limitations.

### Phase 8 — 3D feasibility study
Only after 2D benchmarks are stable, evaluate a 3D robotics simulator. Document integration cost and scientific
value before adoption.

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

## Starting or resuming work
1. Check the current phase and milestone in `docs/14_ROADMAP.md`.
2. Read the ADRs that govern the area you will touch (`docs/decisions/README.md`) and the matching skill in
   `.claude/skills/`.
3. Inspect the relevant code and tests before proposing changes.
4. Propose the next small milestone with acceptance criteria, and get approval before large or destructive changes.

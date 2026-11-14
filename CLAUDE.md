# Claude Code Instructions — SWARM

## Mission
Build SWARM: a research-grade, reproducible multi-agent simulation platform for disaster response and search & rescue. The initial system uses simulated aerial and ground agents in a 2D environment. It is not a chatbot and is not initially a physical-drone control system.

## Read first
Before making substantial changes, read:
- `MASTER_PROMPT.md`
- `docs/14_ROADMAP.md` for the current phase. Phases are numbered 0–8, and this file is the source of truth.
- `docs/decisions/README.md` (ADR index). ADRs override the numbered docs when they conflict.
- `docs/01_PRODUCT_REQUIREMENTS.md`
- `docs/02_SYSTEM_ARCHITECTURE.md`
- `docs/03_SOFTWARE_REQUIREMENTS.md`
- `docs/04_SIMULATION_SPECIFICATION.md`
- `docs/05_EXPERIMENTS_AND_METRICS.md`

Read other documents and skills when relevant. Do not start work from a later phase until the current
phase's acceptance criteria are met, unless the user approves.

## Engineering principles
1. Inspect the existing repository before proposing or changing its structure.
2. Prefer small, reviewable, test-driven changes over large rewrites.
3. Keep the simulator deterministic when a seed is supplied.
4. Separate environment dynamics, agent policies, coordination, perception, training, evaluation, and visualization.
5. Establish classical baselines before reinforcement learning.
6. Keep training code separate from inference/evaluation code.
7. Log configuration, random seeds, software versions, metrics, and artifacts for every experiment.
8. Never report a result that has not been produced by an actual run.
9. Avoid unnecessary dependencies and heavyweight simulation frameworks in the first milestone.
10. Use type hints, clear interfaces, docstrings for public APIs, and meaningful error messages.

## Safety and scope
- Simulated agents only in the initial release.
- Do not provide code that directly controls real aircraft or bypasses geofencing, safety systems, or aviation rules.
- Treat simulated survivor detections as uncertain observations, not guaranteed ground truth.
- Make communication failures, sensor noise, battery limits, obstacles, and changing hazards explicit modelled conditions.
- Do not claim the simulator is validated for real disaster operations.

## Workflow
For each task:
1. Summarize the requested change and inspect related files.
2. State assumptions and identify risks.
3. Write or update tests with the implementation.
4. Implement the smallest complete change.
5. Run the quality gate from the repository root: `ruff check .`, `ruff format --check .`, `mypy`, `pytest`.
6. Report exact commands and actual outcomes. If a command was not run, say so.
7. Update documentation when behavior, configuration, or interfaces change.

## Code quality
- Use Python as the initial core language unless the existing repository has a justified alternative.
- Prefer a `src/` layout and a `pyproject.toml` for packaging.
- Keep configuration explicit; avoid hidden global state.
- Use structured data classes or validated config objects for scenario and agent settings.
- Keep random number generation local and seedable.
- Add tests for edge cases and failure modes, not only happy paths.

## Skills
Use the project skills in `.claude/skills/` when their descriptions match the task: architecture-review, simulation-core, path-planning, task-allocation, marl-research, perception, experiment-analysis, testing-and-review, documentation. Skills are workflow instructions; do not treat them as permission to skip repository inspection, testing, or user approval for destructive actions.

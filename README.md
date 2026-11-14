# SWARM

**Autonomous Multi-Agent Disaster Response and Search & Rescue Intelligence.** SWARM is a research
platform for studying how simulated drones and ground robots search a disaster area, share
uncertain information, divide work and learn to cooperate. It is a simulator for research. It does
not control real aircraft or robots, and it is not validated for real disaster operations.

## Current status: Phase 1, milestone M1 ("one honest episode")

| What | Status |
|---|---|
| Docs, ADR-0001 to ADR-0010, 9 Claude Code skills, tooling and CI (Phase 0) | Complete |
| Deterministic 2D grid with air and ground layers, terrain, survivors, heterogeneous agents | Implemented (M1) |
| Simultaneous movement with conflict resolution, action masks, integer energy and disabling | Implemented (M1) |
| Probabilistic survivor detection in sensor range, typed event stream | Implemented (M1) |
| Random valid-action baseline, episode runner with JSON results, text renderer | Implemented (M1) |
| Hazards, false positives, mission-success termination, multi-seed aggregation | Planned (M2) |
| Path planning, informed baselines, coordination, communication | Planned (Phases 2–3) |
| MARL, perception models, dashboard, 3D | Planned (Phases 4–8) |

The only policy is a **random** baseline. It is a sanity check, not a search strategy. In M1 all agents move
one cell per step, so nothing here shows that drones outperform ground robots, or the reverse. The simulator
is a simplified abstraction, not a physically realistic model (see `docs/15_RISK_REGISTER_AND_LIMITATIONS.md`).
Phases and acceptance criteria are in `docs/14_ROADMAP.md`.

## Run an episode

```bash
python -m swarm.experiments.run_episode --config configs/m1_default.json --seed 0 \
    --out runs/m1_seed0.json --events runs/m1_seed0.events.jsonl --render
```

This writes a versioned JSON result (config, seed derivation, software versions, git commit, metric values with
definitions and units, and event-stream hash) and prints a summary. Re-running with the same config and seed
reproduces the same `event_stream_sha256` and metrics. The format is described in
`docs/11_REPRODUCIBILITY_AND_EXPERIMENT_LOGGING.md`.

## Install (development)

Requires Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

## Quality gate

The same commands run in CI on Python 3.12, 3.13 and 3.14:

```bash
ruff check .
ruff format --check .
mypy
pytest
```

## Python support

`requires-python >=3.12`. NumPy 2.5 dropped Python 3.11, and the planned later dependencies (PyTorch,
PettingZoo, Gymnasium, SciPy, matplotlib) were checked on 2026-11-15 to resolve with binary wheels on
Python 3.12–3.15. The runtime dependency is NumPy only. Heavier libraries will be optional extras
added in the phase that needs them (`docs/decisions/README.md`).

## Repository layout

| Path | Purpose |
|---|---|
| `CLAUDE.md` | Standing instructions for Claude Code |
| `MASTER_PROMPT.md` | Project brief |
| `docs/` | Requirements, specifications, roadmap, risks (numbered 01–15) |
| `docs/decisions/` | Architecture Decision Records |
| `.claude/skills/` | Project-specific Claude Code skills |
| `src/swarm/` | Python package: `core`, `scenarios`, `policies`, `evaluation`, `experiments`, `visualization` |
| `configs/` | Example scenario configs |
| `tests/` | Test suite |
| `.github/workflows/ci.yml` | Continuous integration |

## Claude Code skills

Claude Code discovers project skills from `.claude/skills/<name>/SKILL.md`. They are instruction
packages, not executable plugins.

| Skill | Use for |
|---|---|
| `architecture-review` | Checking changes against package boundaries, ADRs and dependency policy |
| `simulation-core` | Environment dynamics, the reset/step contract, determinism |
| `path-planning` | A*, multi-agent pathfinding, return-to-base feasibility |
| `task-allocation` | Assignment, auctions, reassignment, coordination baselines |
| `marl-research` | PettingZoo adapter, rewards, PPO/MAPPO (Phase 4 and later) |
| `perception` | Sensor models and any later vision-based detection |
| `experiment-analysis` | Seed protocols, aggregation, comparisons, reporting |
| `testing-and-review` | Tests, the quality gate, code review |
| `documentation` | ADRs, doc consistency, status honesty |

## License

MIT. See `LICENSE`.

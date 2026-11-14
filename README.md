# SWARM

**Autonomous Multi-Agent Disaster Response and Search & Rescue Intelligence.** SWARM is a research
platform for studying how simulated drones and ground robots search a disaster area, share
uncertain information, divide work and learn to cooperate. It is a simulator for research. It does
not control real aircraft or robots, and it is not validated for real disaster operations.

## Current status: Phase 0 (foundation) — no simulator yet

| What | Status |
|---|---|
| Requirements, architecture and specification documents (`docs/`) | Present |
| Architecture Decision Records ADR-0001 to ADR-0010 (`docs/decisions/`) | Accepted |
| Claude Code project skills (9, in `.claude/skills/`) | Present |
| Python packaging, lint, type-check and test tooling, CI | Present |
| 2D environment, agents, policies, metrics, experiment runner | **Not implemented** (Phase 1–2) |
| Coordination, communication, MARL, perception, dashboard, 3D | **Not implemented** (Phase 3–8) |

The `swarm` package currently exposes only `swarm.__version__`. The tests check packaging, the
dependency policy and documentation integrity. They do not test simulation behaviour, because
none exists yet. Phases and acceptance criteria are in `docs/14_ROADMAP.md`.

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
| `src/swarm/` | Python package (Phase 0: version only) |
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

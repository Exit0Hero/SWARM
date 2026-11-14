# SWARM

**Autonomous Multi-Agent Disaster Response and Search & Rescue Intelligence.** SWARM is a research
platform for studying how simulated drones and ground robots search a disaster area, share
uncertain information, divide work and learn to cooperate. It is a simulator for research. It does
not control real aircraft or robots, and it is not validated for real disaster operations.

## Current status: Phase 1, milestone M2 (in review)

| What | Status |
|---|---|
| Docs, 13 ADRs, 9 Claude Code skills, tooling and CI (Phase 0) | Complete |
| M1: deterministic 2D grid, two layers, heterogeneous agents, simultaneous moves, masks, energy, detection, events, random baseline, episode runner, text renderer | Complete |
| M2: static hazards, false-positive reports and precision, mission-success termination, full metric registry | Implemented, in review |
| M2: multi-seed experiment runner with bootstrap confidence intervals and paired comparison | Implemented, in review |
| Path planning, informed baselines, coordination, communication | Planned (Phases 2–3) |
| MARL, perception models, dashboard, 3D | Planned (Phases 4–8) |

The only policy is a **random** baseline. It is a sanity check, not a search strategy, and any numbers it produces
describe that baseline on these synthetic maps only. All agents move one cell per step, so nothing here shows that
drones outperform ground robots, or the reverse. Hazards and false reports are simple probabilistic models, not
physical ones. The simulator is a simplified abstraction, not a realistic model
(see `docs/15_RISK_REGISTER_AND_LIMITATIONS.md`). Phases and acceptance criteria are in `docs/14_ROADMAP.md`.

## Run an episode

```bash
python -m swarm.experiments.run_episode --config configs/m2_default.json --seed 0 \
    --out runs/m2_seed0.json --events runs/m2_seed0.events.jsonl --render
```

This writes a versioned JSON result (config, seed derivation, software versions, git commit, metric values with
definitions and units, and event-stream hash) and prints a summary. Re-running with the same config and seed
reproduces the same `event_stream_sha256` and metrics. `configs/m1_default.json` is the M1 scenario: a schema-v1
file that still loads and reproduces its M1 results exactly.

## Run a multi-seed experiment

```bash
python -m swarm.experiments.run_experiment --config configs/m2_default.json \
    --policy random_valid --seeds 0-29 --out runs/m2_random.json
```

This reports n, mean, sample standard deviation and a 95% percentile-bootstrap interval per metric, plus counts of
failed episodes and undefined values. Formats are in `docs/11_REPRODUCIBILITY_AND_EXPERIMENT_LOGGING.md`; metric
definitions are in `docs/decisions/0009-mission-objectives-and-metrics.md`.

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
| `configs/` | Example scenario configs (`m1_default.json` schema v1, `m2_default.json` schema v2) |
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

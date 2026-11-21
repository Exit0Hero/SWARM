# SWARM

**Autonomous Multi-Agent Disaster Response and Search & Rescue Intelligence.** SWARM is a research
platform for studying how simulated drones and ground robots search a disaster area, share
uncertain information, divide work and learn to cooperate. It is a simulator for research. It does
not control real aircraft or robots, and it is not validated for real disaster operations.

## Current status: all roadmap phases implemented; Phases 5, 6 and the 3D co-simulation in review

| What | Status |
|---|---|
| Docs, 27 ADRs, 9 Claude Code skills, tooling and CI (Phase 0) | Complete |
| Phase 1: deterministic 2D core, hazards, noisy detection, mission success, metric registry, multi-seed statistics | Complete |
| Phase 2: A*, prioritized multi-agent planning, energy feasibility, frontier exploration, assignment, oracle reference | Complete (merged) |
| Phase 2: pre-registered baseline study (8 policies × 2 configs × 50 seeds) | Run, then re-run after the PR #5 review corrections (original results superseded). See `docs/16_PHASE2_BASELINE_STUDY.md` |
| Phase 2: pre-registered validation on fresh seeds 100–149 | Run: all 800 episodes complete; partially successful (13 of 17 headline effects replicated; planned-vs-random not established). See `docs/17_PHASE2_VALIDATION_STUDY.md` |
| Phase 3 M3.1: communication channel (range, drops, latency, bandwidth, message events; opt-in, ADR-0017) | Complete (merged) |
| Phase 3 M3.2: per-agent belief maps, information sharing (`frontier_shared`), survivor-information delay metric (ADR-0018) | Complete (merged) |
| Phase 3 M3.3: lease-based auction over 4×4 sectors (`auction_comm`, `auction_no_reassign` control; ADR-0019) | Complete (merged) |
| Phase 3 M3.4: failure detection (active / disconnected / suspected / disabled) and failure, reassignment and duplicate-ownership metrics (ADR-0020) | Complete (merged) |
| Phase 3 M3.5: scheduled failures and partitions (ADR-0021), study runner, pre-registered coordination study (4 450 episodes) and replay viewer | Complete (merged). Sharing and auction coordination speed up search; reassignment reclaims sectors but did not speed up the mission. See `docs/18_PHASE3_COORDINATION_STUDY.md` |
| Phase 4: learned sector selection with IPPO/MAPPO (optional `learning` extra; ADR-0022), nine trained models, pre-registered evaluation | In review. MAPPO beats belief sharing alone (−7 steps) but is slower than the lease auction (+6 steps) at this small training budget. See `docs/19_PHASE4_MARL_STUDY.md` |
| Phase 5: replaceable perception model (range falloff, report confidence, terrain and hazard misreads; ADR-0025), exploratory sensor benchmark | In review. See `docs/22_PERCEPTION.md` |
| Phase 6: web dashboard (`python -m swarm.dashboard`; replay, robots, comms, failures, experiments, history, exports; ADR-0026) | In review. See `docs/23_DASHBOARD.md` |
| Phase 8: 3D feasibility (ADR-0024), then 3D kinematic co-simulation with line-of-sight analysis and a dashboard 3D view (ADR-0027) | Feasibility merged; co-simulation in review. Flight dynamics deferred. See `docs/24_3D_SIMULATION.md` |
| Phase 7 final validation (directive "Phase 5"): 17 conditions × 8 policies, 6 850 episodes, scalability; paper draft | Run, in review. See `docs/20_PHASE7_FINAL_VALIDATION.md` and `docs/21_RESEARCH_PAPER_DRAFT.md`. Phases 5 and 6 deferred |
| 3D feasibility (Phase 8) | Planned |

**Policies** (`python -m swarm.experiments.run_episode --policy <name>`):

| Policy | Information model |
|---|---|
| `random_valid` | decentralized (sanity baseline) |
| `astar_waypoint`, `frontier`, `frontier_hazard_aware` | decentralized (own observations only) |
| `prioritized_frontier`, `assignment`, `assignment_hazard_aware` | centralized_pooled (all agents' observations; perfect communication assumed) |
| `assignment_oracle` | **oracle** (reads hidden survivor locations; reference only, never reported as non-oracle performance) |

All results are simulation results on synthetic maps with illustrative parameters. All agents move one cell per
step, so nothing here shows that drones outperform ground robots, or the reverse. Hazards and false reports are simple
probabilistic models. The simulator is a simplified abstraction, not a realistic model
(see `docs/15_RISK_REGISTER_AND_LIMITATIONS.md`). Phases and acceptance criteria are in `docs/14_ROADMAP.md`.

## Dashboard, perception and 3D
```bash
python -m swarm.dashboard          # web UI at http://127.0.0.1:8765/ (docs/23_DASHBOARD.md)
pip install -e ".[sim3d]"          # optional MuJoCo for the 3D analysis (docs/24_3D_SIMULATION.md)
```
Add a `perception` block to any config for noisy sensing (docs/22_PERCEPTION.md).

## Run an episode

```bash
python -m swarm.experiments.run_episode --config configs/m2_default.json --seed 0 \
    --out runs/m2_seed0.json --events runs/m2_seed0.events.jsonl --render
```

This writes a versioned JSON result (config, seed derivation, software versions, git commit, metric values with
definitions and units, and event-stream hash) and prints a summary. Re-running with the same config and seed
reproduces the same `event_stream_sha256` and metrics. `configs/m1_default.json` is the M1 scenario: a schema-v1
file that still loads and reproduces its M1 results exactly.

## Run the baseline study

```bash
python -m swarm.experiments.run_baselines --config configs/m2_default.json --seeds 0-49 \
    --out-dir runs/phase2/m2
```

This runs every registered policy on identical seeds, writes one experiment JSON per policy, the pre-registered paired
comparisons, and generated summary tables.

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

Optional extras:
- `.[viz]`: matplotlib, for the replay viewer and study figures.
- `.[learning]`: PyTorch, PettingZoo and Gymnasium, for MARL training (Phase 4). For CPU-only
  PyTorch, first run `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
  Evaluating a trained policy needs only NumPy.
- `.[sim3d]`: MuJoCo, for the 3D co-simulation analysis (Phase 8). The dashboard's 3D view works
  without it.
- The dashboard needs no extra.

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
| `src/swarm/` | Python package: `core`, `scenarios`, `navigation`, `coordination`, `policies`, `evaluation`, `experiments`, `visualization` |
| `docs/results/` | Archived generated results of pre-registered studies |
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

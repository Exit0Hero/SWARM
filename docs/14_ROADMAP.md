# SWARM Development Roadmap

This is the **single source of truth** for phase numbering and status. `MASTER_PROMPT.md`,
`CLAUDE.md` and `README.md` refer here and must not restate a different scheme.

A phase is complete only when every acceptance criterion is met *and* verified by tests or
recorded runs. Work from a later phase does not start before the current phase is complete,
unless the user explicitly approves it.

| Phase | Name | Status |
|---|---|---|
| 0 | Foundation | Complete (PR #1, merged 2026-11-15) |
| 1 | Deterministic 2D core | In progress: M1 complete (PR #2); M2 implemented, in review |
| 2 | Navigation and classical baselines | Planned |
| 3 | Coordination under constraints | Planned |
| 4 | Multi-agent reinforcement learning | Planned |
| 5 | Perception | Planned |
| 6 | Visualization dashboard and analytics | Planned |
| 7 | Research experiments and reporting | Planned |
| 8 | 3D feasibility study | Planned |

## Phase 0 — Foundation
Acceptance:
- Repository content at the root. Nine project skills in `.claude/skills/`.
- `pyproject.toml` (NumPy-only runtime), `.gitignore`, `LICENSE`, `src/` layout, `tests/`.
- Quality gate (ruff, ruff format, strict mypy, pytest) passing locally and in CI on Python 3.12–3.14.
- ADR-0001 to ADR-0010 recorded in `docs/decisions/`.
- Phase numbering unified across README, CLAUDE.md, MASTER_PROMPT.md and this file.

## Phase 1 — Deterministic 2D core
Implements ADR-0001 to ADR-0008, ADR-0011 to ADR-0013, ADR-0009 revision 2 and the core part of ADR-0010.
Phase 1 is complete when M2 is reviewed and merged.

**Milestone M1 — "one honest episode"** (smallest end-to-end slice). Status: complete (PR #2, merged 2026-11-15).
The file-level map of what was built is in `docs/02_SYSTEM_ARCHITECTURE.md` ("Implemented modules").
- Frozen, validated `ScenarioConfig` with JSON round-trip and config hash. Seeded generator for terrain
  (`OPEN`, `DEBRIS`, `STRUCTURE`, `BASE`), static survivors and a drone/ground roster.
- `reset(seed)` / `step(actions)` with action masks, simultaneous two-layer movement resolution,
  integer energy with recharge and disablement, passive detection with `p_detect`, and a typed event stream.
- Random valid-action policy. Episode runner writing versioned JSON with run metadata.
- Metrics: recall, time to first detection, coverage (sensed and visited), energy used, move conflicts,
  invalid actions (plus an auxiliary count of agents disabled).
- Text renderer for debugging (no dependencies).
- Tests: byte-identical event stream for the same seed. Property tests (bounds, terrain, no shared cell,
  energy ≥ 0, permutation invariance, policy swap does not change the scenario). Hidden-state leakage test.
  docs/10 edge cases. Golden event-stream hashes for fixed seeds (regression).
- Not in M1: hazards, false positives, communication, pathfinding, rewards, PettingZoo, plotting.

**Milestone M2 — hazards, false detections, mission success, multi-seed statistics.** Status: implemented,
in review.
- Static hazard overlay with layer-specific effects, energy damage and disablement draws (ADR-0011).
- False-positive reports, exact-cell association by ground truth, location-level and report-level precision
  (ADR-0012).
- Per-process environment RNG sub-streams that keep M1 streams byte-identical (ADR-0013).
- Mission-success termination after sensing with exact recall-target comparison; precedence
  mission_success > all_disabled > time_limit; zero-survivor success undefined (ADR-0009 revision 2).
- Full metric registry with formula, unit, undefined case and role (ADR-0009 revision 2).
- Multi-seed experiment runner with failure accounting and percentile-bootstrap intervals; paired
  comparison that refuses different seed lists or configs; CLIs for both.
- Config schema v2 (v1 files migrate). M1 golden hashes unchanged; M2 golden hashes added.
- Not in M2: dynamic hazards, localization noise, confirmation rules, communication, pathfinding, rewards.

## Phase 2 — Navigation and classical baselines
Acceptance: A* per layer. Prioritized multi-agent planning with reservation tables. Return-to-base
feasibility rule. Greedy frontier exploration. Centralized assignment (labelled oracle where it uses
hidden state). Evaluation on a fixed seed list with the full metric suite from ADR-0009.

## Phase 3 — Coordination under constraints
Acceptance: per-agent belief and coverage maps. Communication channel with range, drop probability,
latency and timestamps. Decentralized heuristic. Auction or market task allocation with reassignment.
Agent-failure scenarios. Message delivery rate metric. Lightweight replay viewer (matplotlib, optional
extra).

## Phase 4 — Multi-agent reinforcement learning
Acceptance: PettingZoo `ParallelEnv` adapter passing the API test. Event-based reward modules.
Separate training and evaluation entry points with disjoint seed sets. Independent PPO diagnostic,
then MAPPO with action masking and a centralized critic. Checkpointing. Evaluation on the same
seeds as the baselines.

## Phase 5 — Perception
Acceptance: replaceable perception interface. Synthetic sensor benchmark (`p_detect` and
false-positive sweeps). An image-based detector only with a justified dataset, held-out evaluation
and a baseline to beat.

## Phase 6 — Visualization dashboard and analytics
Acceptance: live and replay views, policy comparison views, and run summaries. These are all read-only
consumers of events and snapshots. A headless mode is preserved.

## Phase 7 — Research experiments and reporting
Acceptance: the research questions in [docs/06](06_MARL_RESEARCH_PLAN.md) run as pre-registered
experiment configs. A reproducible report with methodology, confidence intervals and limitations.

## Phase 8 — 3D feasibility study
Acceptance: evidence-based simulator comparison, an integration prototype, a performance and cost
assessment, and a recorded decision to proceed or defer.

## Mapping from earlier numbering
Earlier briefs used other schemes. They map to the phases above as follows:

| Earlier brief | Maps to |
|---|---|
| Repo `MASTER_PROMPT.md` Stages 0–7 | 0, 1, 2, 3, 4, 5, 6, 8 (Stage 6 analysis is split into Phases 6 and 7) |
| Original roadmap Phases 0–7 | 0, 1, 2, 3, 4, 5, 6/7, 8 |
| Master project initialization Phases 1–11 | 1→0, 2–3→1, 4→2, 5→3, 6→2 (baselines) and 3 (coordination), 7→4, 8→5, 9→6, 10→7, 11→8 |

In the 11-phase brief, measurable baselines were placed after communication. They are now
in Phase 2, before coordination, so that every later feature is measured against a baseline.

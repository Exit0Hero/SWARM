# SWARM: complete platform and implementation status

Status as of 2026-11-22, `main` at the merge of PR #20. This document summarizes and links. The
numbered documents, ADRs and archived result files remain the authoritative sources, and every
number below is quoted from them.

**Status labels used throughout**

| Label | Meaning |
|---|---|
| **Implemented and tested** | Code is merged and covered by the automated test suite. |
| **Implemented, experimentally evaluated** | Implemented and tested, and measured in an exploratory run (development seeds or a single example, no pre-registration). The numbers are not validated results. |
| **Validated result** | A finding from a pre-registered study on reserved evaluation seeds, with multiple-comparison correction (docs/16–20). |
| **Partially implemented** | Works within stated limits. Part of the intended capability is missing. |
| **Deferred** | Not built, by recorded decision. |

SWARM is a simulation research platform. Nothing in it is validated for real disaster operations,
and it controls no real vehicle.

---

## 1. Final architecture

```
configs/*.json ──► ScenarioConfig (frozen, validated, hashed; ADR-0007)
                        │
                        ▼
              swarm.core.env.SwarmEnv  ◄── seeded RNG streams (ADR-0006, 0013)
   2D two-layer grid · simultaneous moves · energy · hazards · sensing (+ perception model)
   · communication channel · scheduled failures · typed events (ADR-0005)
        │ observations (no ground truth, ADR-0008)          │ events + snapshots (evaluation only)
        ▼                                                    ▼
  policies / coordination / learning                swarm.evaluation (metrics, statistics)
  classical · belief sharing · lease auction        swarm.experiments (runner, studies, analysis)
  · failure detection · learned actors                   │
                                                         ├─► swarm.dashboard (web UI, read-only)
                                                         ├─► swarm.visualization (replay figures)
                                                         └─► swarm.sim3d (3D scene, MuJoCo analysis)
```

The packages under `src/swarm/`:

| Package | Role |
|---|---|
| `core` | Environment, events, observations, RNG, channel and perception model. Imports no consumer. |
| `scenarios` | Config schema and seeded scenario generation. |
| `navigation` | A*, reservation tables, energy feasibility. |
| `coordination` | Belief maps, lease auction, failure detector. |
| `policies` | Registry of baseline and coordinated team policies. |
| `learning` | Sector-selection agent, rewards, PettingZoo adapter, PPO/MAPPO training. Training needs the `learning` extra; inference is NumPy only. |
| `evaluation` | Metrics: core, communication, failure and perception. Statistics: bootstrap, sign-flip, Holm, Benjamini–Hochberg. |
| `experiments` | Episode runner (with a read-only `on_step` hook), multi-seed experiments, pre-registered studies and their analysis, scalability. |
| `dashboard` | Standard-library HTTP server and a single-page user interface. |
| `visualization` | Replay viewer (matplotlib, `viz` extra) and text renderer. |
| `sim3d` | 3D scene builder (no dependencies) and MuJoCo co-simulation (`sim3d` extra). |

**Design rules that hold throughout**
- **The 2D environment is the only simulator.** Every other component consumes its events and
  snapshots.
- **No ground truth reaches agents.** Policies see only their observations and the messages they
  receive. Oracle access goes through a labelled `OracleView`.
- **Determinism:** a seed and a config hash fully determine an episode.

## 2. Phase status (canonical numbering 0–8, docs/14)

| Phase | Content | Status | PRs |
|---|---|---|---|
| 0 Foundation | Repository, tooling, CI, ADRs, skills | Implemented and tested | #1, #4 |
| 1 Deterministic 2D core | Two-layer grid, heterogeneous agents, conflict resolution, energy, hazards, noisy detection, events, runner | Implemented and tested | #2, #3 |
| 2 Navigation and classical baselines | A*, prioritized planning, frontier, centralized assignment, oracle reference, multi-seed statistics | Implemented and tested. Baseline study and independent validation in docs/16–17 | #5, #6 |
| 3 Coordination under constraints | Channel, belief maps, lease auction, failure detection and reassignment, fault injection, replay viewer, pre-registered study | Implemented and tested. Validated results in docs/18 | #7–#11 |
| 4 MARL | Sector-selection agent, PettingZoo adapter, IPPO/MAPPO, nine trained models, pre-registered evaluation | Implemented and tested. Validated results in docs/19 (small CPU training budget) | #12–#14, #17 |
| 5 Perception | Replaceable sensor model, noisy synthetic sensor, sensor benchmark | Implemented and tested. Benchmark is exploratory (docs/22). Image detector deferred | #18 |
| 6 Dashboard | Web dashboard over the real runner | Implemented and tested (docs/23) | #20 |
| 7 Research experiments | Pre-registered final validation (17 conditions), scalability, paper draft | Implemented. Validated results in docs/20; paper draft in docs/21 | #15 (via #17) |
| 8 3D | Feasibility study, then kinematic co-simulation and 3D view | Feasibility: decision recorded (ADR-0024). Co-simulation implemented and tested, but **partial**: no flight dynamics (ADR-0027) | #16 (via #17), #19 |

The 2026-11-15 development directive's "Phase 5" is roadmap Phase 7.

## 3. Perception (Phase 5; ADR-0025, docs/22)

**Implemented and tested**
- **Optional config block.** `perception` is an optional `ScenarioConfig` block. Without it,
  sensing is exactly the original synthetic sensor (ADR-0012), and every existing event stream and
  golden hash is unchanged.
- **Replaceable sensor interface.** The `SensorModel` protocol and `SENSOR_MODELS` registry
  (`swarm.core.perception`) let a new model be registered without changing the environment. The
  built-in `noisy` model provides:
  - **Range limitation:** detection probability is `p_detect × (1 − range_falloff × d / (r + 1))`.
    False positives per scanned cell are unchanged.
  - **Obstacle perception uncertainty:** terrain is misread with probability `terrain_error`
    (open ↔ debris, structure → debris). The robot's own cell and the base are never corrupted.
  - **Hazard perception uncertainty:** hazards are missed (`hazard_miss`) or reported where there
    are none (`hazard_false`).
  - **Report confidence:** a posterior computed from public parameters only, with
    `report_confidence(...)`. It is deliberately **not** an observation field. The existing
    leakage test, which forbids any truth-revealing or "confidence" field, is unchanged.
- **Random streams.** Draws come from a fifth `dynamics` stream (`perception`), two per footprint
  cell, so sweeps keep common random numbers.
- **Error logging.** Every corrupted reading is logged as an evaluator-only `perception_error`
  event. New metrics count terrain errors, missed and false hazards, and true and false reports.
- **Integration.** Noisy views flow unchanged into the existing known maps, shared belief maps
  (last-writer-wins conflict resolution, ADR-0018), the auction, every classical policy and the
  MARL observation encoder. The environment still enforces true terrain, so a move planned from a
  misread becomes an invalid action.

**Implemented, experimentally evaluated:** the sensor benchmark. See section 11.

**Partially implemented:** no existing policy uses report confidence, and none was retuned for
noisy sensing.

**Deferred:** an image-based detector. The roadmap requires a dataset, a held-out evaluation and
a baseline to beat, and none exists.

## 4. Dashboard (Phase 6; ADR-0026, docs/23)

**Implemented and tested.** Start it with `python -m swarm.dashboard`. It needs no extra
packages, binds to 127.0.0.1 and stores data in `.swarm-dashboard/`.

**Back end** (`swarm.dashboard.service`, `server`)
- **Episodes:** run through `run_episode`. Each frame is recorded from `env.snapshot()` through the
  read-only `on_step` hook. A test shows the hooked and unhooked event hashes are identical, so
  the dashboard never simulates anything itself.
- **Comparisons:** run through `run_experiment` on matched seeds, as a background job, capped at
  600 episodes.
- **Storage:** run and experiment records, history and saved configs are stored as JSON.
- **Exports:** run JSON, event JSON Lines, metrics CSV, experiment JSON and seed-level CSV.
- **Policies:** every registered policy, plus the trained Phase 4 actors as `learned:<name>`.
  Policies that cannot run on a configuration are shown as unavailable, with the reason.

**Interface** (`static/index.html`)
- **Configuration:** a base config plus settings for policy, seed, time limit, `p_detect`, the
  communication channel, scheduled failures and noisy perception. The effective JSON is
  editable. Configs can be validated and saved.
- **Map:** terrain, hazards, base, robots (drone, ground, disabled), sensor ranges, trails,
  message links, true and false detection reports, and ground-truth survivors. Ground truth is
  labelled as the evaluation view.
- **Replay controls:** play, pause, step, jump to start or end, scrub and speed.
- **Panels:** mission progress, per-robot energy and status, per-step communication and perception
  counts, failure log and final metrics.
- **Experiments tab:** summaries with 95% intervals, a bar chart, paired differences against a
  baseline, and exports. Results are labelled exploratory.
- **History tab:** open, delete and export stored runs and experiments.

**Testing:** service and HTTP end-to-end tests are in `tests/test_dashboard.py`. During
development every control was also exercised in headless Chrome with no console errors. That
browser check is not part of CI.

**Partially implemented:** "live" viewing replays completed runs (episodes finish in under a
second). There is no server-side session that steps an episode on demand.

## 5. 3D co-simulation (Phase 8; ADR-0024, ADR-0027, docs/24)

**Implemented and tested.** This is a **kinematic co-simulation**, not a flight simulation.
- **Source of every episode:** the 2D simulator, with any policy, classical or learned.
- **`build_scene`** (no dependencies) lifts the recorded episode into 3D:
  - structures become 2 m boxes and debris 0.4 m boxes; hazards are floor markers; 1 cell = 1 m;
  - each robot has a track with one point per step;
  - ground robots stay at 0.25 m; drones are at 0.15 m on the base and at 3 m elsewhere. This is a
    fixed altitude convention, not dynamics.
- **`analyze_scene`** (optional `sim3d` extra, MuJoCo) moves the tracks as mocap bodies and
  measures:
  - **Robot–obstacle contacts:** a check that motion legal in 2D is collision-free in 3D.
  - **Line of sight:** whether a 3D ray from each reporting robot to the reported cell is blocked
    by terrain. Robots are excluded from the ray, and so is the reported cell's own box. The
    result is reported and does not change the episode.
- **Visualization:** the dashboard's rotatable 3D view (`GET /api/runs/<id>/scene3d`), and the
  `python -m swarm.sim3d.mujoco_replay --scene-out` command.
- **Tests:** a known blocked line of sight and a known clear one; tracks equal the recorded frames
  exactly; zero contacts for real episodes.

**Limitations: deferred, not implemented**
- Rigid-body flight dynamics, controllers, wind or rotor models.
- 3D sensing that changes what agents observe.
- Policies that plan in 3D.
- Rendering from MuJoCo.

No claim of physical flight accuracy is made.

## 6. Integration with classical policies and MARL

**Implemented and tested.** Every policy runs through the same `run_episode` path and therefore
works with:
- perception, communication limits and scheduled failures;
- the dashboard and the 3D scene.

The policies are:
- **Classical:** `random_valid`, `astar_waypoint`, `frontier`, `prioritized_frontier`, hazard-aware
  variants, and centralized `assignment`. `assignment_oracle` is labelled as an oracle reference.
- **Communication-limited:** `frontier_shared` (belief sharing), `auction_comm` (lease auction with
  failure detection and reassignment) and `auction_no_reassign`.
- **Learned:** the Phase 4 IPPO and MAPPO actors (nine models in `docs/results/phase4/models`).
  They run with NumPy-only inference, through the study tools (`learned_policies`) and the
  dashboard (`learned:<name>`).

## 7. Communication and failure handling

**Implemented and tested** (ADR-0017 to ADR-0021)
- **Channel:** range per agent type, Bernoulli loss, latency plus jitter, and a bandwidth cap.
  Scheduled partitions are hidden from agents.
- **Belief sharing:** commutative merges, so the order messages arrive in does not change the
  result.
- **Lease auction:** 4×4 sectors, 15-step leases and heartbeats.
- **Failure detector:** marks a teammate active, then disconnected, then suspected after more than
  10 silent steps (suspicion is reversible). Disablement is permanent and is learned from
  tombstone records.
- **Reassignment:** the sector of a suspected or disabled holder is released and reassigned.
- **Scheduled agent failures** for experiments.
- **Metrics:** message overhead, delivery rate, information-propagation delay, detection and
  suspicion delays, false suspicions, reassignment delay, unresolved sectors and duplicate
  ownership.

## 8. Testing and supported Python versions

Supported Python versions: 3.12, 3.13 and 3.14 (`requires-python >=3.12`). The CI matrix runs
all three, plus Python 3.13 with the `learning` extra. Each job runs:
- `ruff check .`
- `ruff format --check .`
- `mypy` (strict)
- `pytest`

| Run | Result |
|---|---|
| Local, Python 3.13, with optional extras, on the tree merged in #18–#20 | ruff and format clean, strict mypy clean, **789 passed** |
| Local, without optional extras | **772 passed, 6 skipped**: the MuJoCo tests skip without `mujoco` |
| CI on `main` after the merges | all four jobs passed |

Test layers: unit, property-based (Hypothesis, derandomized), determinism, golden-hash
regression, leakage, API and HTTP end to end, plus study and analysis tests.

## 9. Preservation of golden hashes and archived experiments

- **Golden hashes:** `tests/test_regression.py` was not modified by Phases 5, 6 or 8, and passes
  (6 tests).
- **No change in behaviour:** every new capability is optional and additive.
  - New config blocks are written only when set, so existing config hashes are unchanged.
  - New RNG children are spawned after the existing ones, so earlier streams are unchanged.
  - The runner hook defaults to off.
- **Existing tests:** only two were edited, and only to add the new field to lists that must match
  exactly: the RNG children list and the mission-brief allow-list. The brief test also gained an
  assertion that `perception` is `None` by default. Nothing was removed or weakened.
- **Archived results:** `docs/results/phase2_validation`, `phase3`, `phase4` and `phase7`, and
  their configs, are unchanged.

## 10. Installation and reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # core + tests (NumPy only at runtime)
pip install -e ".[viz]"                 # optional: figures, replay viewer
pip install -e ".[sim3d]"               # optional: MuJoCo 3D analysis
pip install -e ".[learning]"            # optional: MARL training (PyTorch, PettingZoo, Gymnasium)

ruff check . && ruff format --check . && mypy && pytest      # quality gate
python -m swarm.dashboard                                     # http://127.0.0.1:8765/
python -m swarm.sim3d.mujoco_replay --config configs/phase3/nominal.json \
    --policy auction_comm --seed 1000 --scene-out scene.json
```

**Reproducing a study:**
1. Unpack the archive: `tar xzf runs.tar.gz` in `docs/results/<phase>/`.
2. Rerun the analysis: `python -m swarm.experiments.study_analysis --spec configs/<phase>/<spec>.json --runs … --out …`.
3. Or regenerate every episode with `python -m swarm.experiments.study --spec …`.

Episodes are deterministic. Every result file records the commit, the config hash and the seeds.
Commands for each study are in docs/18–20 and `docs/results/*/README.md`.

## 11. Findings

### Validated results (pre-registered, reserved seeds, Holm-corrected; simulation only)
- **docs/18 (Phase 3; 4 450 episodes).** The metric is steps to reach the recall target.
  - Belief sharing: 81.5 → 44.3 steps. The auction adds a further −10.4 steps.
  - The auction is not distinguishable from centralized pooled assignment at nominal.
  - With two early failures, reassignment cut unresolved sectors from 1.52 to 1.06 but did not
    change the time to target.
  - An early partition cost the auction about 7 steps.
  - The first fault study had little exposure to faults. This is reported in docs/18 as a design
    flaw and was corrected by a second study on new seeds.
- **docs/19 (Phase 4; 3 200 episodes).**
  - MAPPO was 7 steps faster than belief sharing and 6 steps slower than the auction at nominal.
  - IPPO and MAPPO were not distinguishable.
  - The coverage reward term had no detectable effect.
  - Training budget: about 2 400 episodes per model on CPU.
- **docs/20 (Phase 7; 6 850 episodes, 54 tests).**
  - Belief sharing was supported in 17 of 17 conditions.
  - The auction was supported in 7 of 17 and never contradicted.
  - MAPPO differed from belief sharing in 1 of 17 conditions.
  - Reassignment and mission success at a recall target of 1.0 were inconclusive.
  - On 40 × 40 maps, success was about 0.45 for the decentralized methods and 0.78 for
    centralized assignment.

### Experimental (exploratory: not validated)
- **Sensor benchmark** (docs/22; development seeds 1000–1019, n = 20, Benjamini–Hochberg).
  - A 10% terrain misread rate slowed `frontier_shared` by +11.8 steps and `auction_comm` by
    +11.2 steps.
  - `p_detect` 0.5 cost +2 to +4 steps.
  - Hazard misreads changed no decision. The current planners use known hazards only in the
    energy reserve.
- **3D line of sight** (docs/24): a single example episode (development seed 1000, nominal,
  `auction_comm`) had 6 of 62 reports occluded and 0 contacts. This is one episode, not a study.
- **Scalability** (docs/20; development seeds, one machine): auction policy cost per step grows
  superlinearly with team size. Runtimes are machine-dependent.
- **Dashboard comparisons:** always exploratory (development seeds, unadjusted p-values).

## 12. Partially completed and deferred

**Partially implemented**
- 3D is kinematic co-simulation. The 2D simulator decides all movement, and altitude is a
  convention.
- Report confidence is available, but no policy uses it.
- Dashboard "live" mode replays completed runs.
- The MARL results come from a small CPU budget on one training scenario.
- Mission success saturates at the default time limits, so most validated findings concern search
  speed.

**Deferred**
- Flight dynamics and controllers; 3D-aware sensing and policies; MuJoCo rendering.
- An image-based survivor detector, pending a dataset and a held-out baseline.
- Primitive-action MARL training, and learned communication content (ADR-0022).
- Factorial experiments measuring interactions between faults.
- Larger MARL training budgets.
- Any validation against real systems.

## 13. References

- **Roadmap and architecture:** [14_ROADMAP](14_ROADMAP.md), [02_SYSTEM_ARCHITECTURE](02_SYSTEM_ARCHITECTURE.md),
  [15_RISK_REGISTER_AND_LIMITATIONS](15_RISK_REGISTER_AND_LIMITATIONS.md).
- **Studies:** [16](16_PHASE2_BASELINE_STUDY.md), [17](17_PHASE2_VALIDATION_STUDY.md),
  [18](18_PHASE3_COORDINATION_STUDY.md), [19](19_PHASE4_MARL_STUDY.md), [20](20_PHASE7_FINAL_VALIDATION.md).
  Paper draft: [21](21_RESEARCH_PAPER_DRAFT.md); its citations must be verified before submission.
- **New-capability guides:** [22 Perception](22_PERCEPTION.md), [23 Dashboard](23_DASHBOARD.md),
  [24 3D simulation](24_3D_SIMULATION.md).
- **ADRs** ([index](decisions/README.md)):
  - 0001–0016: core and baselines;
  - 0017 channel; 0018 belief maps; 0019 lease auction; 0020 failure detection;
  - 0021 fault injection; 0022 MARL; 0023 final-validation factors;
  - 0024 3D feasibility (superseded in part); 0025 perception; 0026 dashboard;
  - 0027 3D co-simulation.
- **Merged PRs:** #1–#20 on `Exit0Hero/SWARM`:
  - Phase 5: #18; Phase 6: #20; 3D co-simulation: #19.
  - Phase 4 and Phase 7 reached `main` through #17, which included #14, #15 and #16.

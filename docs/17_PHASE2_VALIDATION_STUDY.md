# Phase 2 Validation Study (fresh seeds 100–149)

This document is the **pre-registration** of an independent validation of the corrected Phase 2 baselines. It is
committed before any validation episode is run. Results are appended in the [Results](#results) section, and nothing
above that section changes after the runs.

## Purpose
The Phase 2 baseline study (`docs/16_PHASE2_BASELINE_STUDY.md`) is exploratory. During it, the evaluation seeds 0–49 were
partly inspected while debugging, and the implementation was then corrected after code review. This study tests whether
its findings hold on **fresh seeds that no one has looked at**, with the implementation frozen.

## Implementation under test
- Commit `226bf9da7f4e29396daff54d26f9cfabc8f5ecac` on `main`: the merge of PR #5. Its tree is identical to the
  reviewed and corrected PR head `31f289c`, which contains the review corrections from `4895575`.
- **Frozen:** no code, configuration or parameter changes from pre-registration until the results are archived. The
  runner, policies, energy reserve (10%), hazard penalty (10), horizon `2(h+w)` and traffic rules are used as they are.
- Seeds 100–149 have not been used for development, debugging or any reported run.

## Design
- **Seeds:** 100 through 149 (n = 50), identical for every policy and both configurations.
- **Configurations:** `configs/m1_default.json` (config hash `fec0dca1c6f801fc`) and `configs/m2_default.json`
  (config hash `8db09395238eb122`), unchanged.
- **Policies:** all eight: `random_valid`, `astar_waypoint`, `frontier`, `frontier_hazard_aware`,
  `prioritized_frontier`, `assignment`, `assignment_hazard_aware`, `assignment_oracle` (the oracle is a labelled
  reference only).
- **Paired comparisons:** the seven defined in `swarm.experiments.run_baselines.PREREGISTERED_PAIRS`. Questions Q1–Q5 are
  those of `docs/16_PHASE2_BASELINE_STUDY.md`:

  | Pair | Baseline → candidate | Question |
  |---|---|---|
  | P1 | `random_valid` → `astar_waypoint` | Q1 |
  | P2 | `astar_waypoint` → `frontier` | Q2 |
  | P3 | `frontier` → `prioritized_frontier` | Q3 |
  | P4 | `prioritized_frontier` → `assignment` | Q3 |
  | P5 | `frontier` → `frontier_hazard_aware` | Q4 |
  | P6 | `assignment` → `assignment_hazard_aware` | Q4 |
  | P7 | `assignment` → `assignment_oracle` | Q5 (reference gap) |

- **Commands** (unchanged runner, default bootstrap settings):
  ```
  python -m swarm.experiments.run_baselines --config configs/m1_default.json --seeds 100-149 --out-dir runs/phase2_validation/m1
  python -m swarm.experiments.run_baselines --config configs/m2_default.json --seeds 100-149 --out-dir runs/phase2_validation/m2
  ```

## Metrics and statistics
- **Metrics:** the ADR-0009 registry as in the original study. Primary: `recall`, `mission_success`.
  Secondary: `time_to_recall_target`, `energy_used`, `agents_disabled_energy`, `agents_disabled_hazard`,
  `hazard_exposures`, `coverage_sensed`, `move_conflicts`, `invalid_actions`, `precision`. Policy diagnostics:
  `failed_plans` (`_unreachable` / `_temporary`), `energy_route_aborts`, planned waits, reassignments.
- **Per policy:** n, mean, sample standard deviation, 95% percentile-bootstrap interval (10 000 resamples, bootstrap
  seed 0), with counts of failed episodes and undefined values.
- **Comparisons:** paired per-seed differences (candidate − baseline), each with a 95% bootstrap interval. `time_to_recall_target` is
  compared only on seeds where both policies reached the target; the number of pairs is always reported.
- **Multiplicity:** no correction. The headline effects below are fixed in advance to limit the number of claims.
  Every other comparison is reported as descriptive.

## Headline effects (from the corrected exploratory run, `docs/results/phase2_corrected/`)
Effects whose 95% paired interval excluded 0 in the corrected exploratory run. Direction: `+` means the candidate is higher.

| ID | Config | Pair | Metric | Expected direction |
|---|---|---|---|---|
| H1 | M1 | P1 | recall | + |
| H2 | M2 | P1 | recall | + |
| H3 | M2 | P1 | mission_success | + |
| H4 | M1 | P2 | recall | + |
| H5 | M1 | P2 | mission_success | + |
| H6 | M2 | P2 | recall | + |
| H7 | M2 | P2 | mission_success | + |
| H8 | M1 | P3 | recall | + |
| H9 | M1 | P3 | mission_success | + |
| H10 | M1 | P3 | time_to_recall_target | − |
| H11 | M2 | P3 | time_to_recall_target | − |
| H12 | M2 | P5 | hazard_exposures | − |
| H13 | M2 | P5 | agents_disabled_hazard | − |
| H14 | M2 | P6 | hazard_exposures | − |
| H15 | M2 | P6 | agents_disabled_hazard | − |
| H16 | M1 | P7 | time_to_recall_target | − (reference gap) |
| H17 | M2 | P7 | time_to_recall_target | − (reference gap) |

**Expected nulls** (no detectable difference in the corrected exploratory run):
- N1: P4 (`assignment` vs `prioritized_frontier`): `recall`, `mission_success`, `time_to_recall_target`, both configs.
- N2: P5 and P6 on M2 (cost of hazard awareness): `recall`, `mission_success`.

## Decision rules (fixed before running)
- **Replicated:** the validation 95% paired interval excludes 0, with the expected sign.
- **Direction consistent, not replicated:** same sign as expected, but the interval includes 0.
- **Contradicted:** the interval excludes 0 with the opposite sign.
- **Expected null:**
  - *Consistent* if the interval includes 0, or if it excludes 0 with |mean difference| below the practical threshold.
  - *Not consistent* if the interval excludes 0 and |mean difference| is at least the threshold.
- **Practical-significance thresholds**, used to separate a detectable difference from a meaningful one:
  - recall 0.02;
  - mission success 0.05;
  - time to recall target 5 steps;
  - hazard exposures 1.0 per episode;
  - disablements 0.05 agents per episode;
  - energy used 50 units.

  An interval that excludes 0 while the whole interval lies below the threshold is reported as "detectable but small".
- **Overall verdict:**
  - **Successful:** every run completes without a crash, all 17 headline effects replicate, and every expected null is consistent.
  - **Partially successful:** no crash and no contradiction, but some headline effects are only direction-consistent,
    or some nulls are not consistent. The list is reported.
  - **Failed:** any crash, or any contradicted headline effect.

## Failures, crashes and livelocks
- **Crash:** an exception in an episode. It is recorded by the runner (episode marked failed, never dropped), the runner
  exits with status 3, and the affected seeds and policies are reported. Any crash makes the verdict "failed".
- **Livelock or stranding:** an episode in which agents stop making progress or run out of energy is a valid outcome, not a failure.
  It shows up in `agents_disabled_energy`, `time_to_recall_target` (undefined) and the diagnostics, and is reported.
- **No selective reruns, no exclusions, no code changes during the evaluation.** If a defect is found, the
  validation is reported as it stands. Any fix is made as a separate, disclosed change, and needs a new pre-registration
  on new seeds (not 0–49 or 100–149).
- Development seeds 1000–1019 may still be used for debugging. They are never reported.

## Archive
`docs/results/phase2_validation/{m1,m2}/`:
- `summary.md` and `comparisons.json`;
- the per-policy result JSON files, with per-episode records, metadata and fingerprints;
- the seed list, config hashes, implementation and pre-registration commits;
- a methodology and limitations note.

The exploratory results in `docs/results/phase2/` and `docs/results/phase2_corrected/` are not modified.

## Results
*(Appended after the runs.)*

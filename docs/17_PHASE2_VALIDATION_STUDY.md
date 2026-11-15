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

### Provenance
- **Implementation evaluated:** `226bf9d` (merge of PR #5). **Pre-registration:** `613c5f3`, which has code identical to
  `226bf9d` and from which the runs were executed with a clean tree.
- **Commands:** as in [Design](#design). Both exited with status 0. **All 800 episodes completed** (8 policies × 2
  configurations × 50 seeds). No crashes, no failed seeds, no reruns, no exclusions.
- **Archive:** `docs/results/phase2_validation/`, containing per-episode results, summaries, comparisons, seeds, the
  replication analysis, methodology and limitations. The tables below are copied from the generated
  `replication.md`, not typed by hand.

### Verdict (pre-registered rule): partially successful
- **Headline effects:** 13 of 17 replicated. The other 4 (H1, H2, H3, H15) had the expected direction, but their
  intervals included 0. None was contradicted.
- **Expected nulls:** all 10 were consistent.
- **All 98 pair × metric rows** (7 pairs × 7 reported metrics × 2 configurations, in `replication.md`): 89 reached the
  same interval conclusion as the corrected exploratory run, and 82 had the same point-estimate direction. All 16
  direction changes have validation intervals that include 0. The 9 conclusion changes are:
  - H1, H2, H3, H15 above;
  - M2 P1 hazard disablements (interval now includes 0);
  - M1 P2 time to target (only 2 pairs; uninformative);
  - three differences detectable only in validation, none pre-registered:
    - M2 P2 time to target: −35.0 [−55.2, −16.3] (22 pairs);
    - M2 P5 time to target: −6.8 [−12.5, −2.0];
    - M2 P5 energy used: −38.9 [−67.1, −13.2].

  These are exploratory observations, not established effects.

#### Headline effects

| ID | Config | Pair | Metric | Exploratory | Validation | Verdict | Practical |
|---|---|---|---|---|---|---|---|
| H1 | M1 | P1 | recall | +0.085 [+0.035, +0.135] (n=50) | +0.0325 [-0.025, +0.0875] (n=50) | direction consistent, not replicated | not detectable |
| H2 | M2 | P1 | recall | +0.0725 [+0.0275, +0.117] (n=50) | +0.02 [-0.0275, +0.065] (n=50) | direction consistent, not replicated | not detectable |
| H3 | M2 | P1 | mission_success | +0.16 [+0.02, +0.3] (n=50) | +0.12 [-0.06, +0.3] (n=50) | direction consistent, not replicated | not detectable |
| H4 | M1 | P2 | recall | +0.32 [+0.263, +0.378] (n=50) | +0.295 [+0.242, +0.35] (n=50) | replicated | practically significant (whole interval beyond threshold) |
| H5 | M1 | P2 | mission_success | +0.6 [+0.46, +0.74] (n=50) | +0.64 [+0.48, +0.78] (n=50) | replicated | practically significant (whole interval beyond threshold) |
| H6 | M2 | P2 | recall | +0.158 [+0.115, +0.203] (n=50) | +0.13 [+0.09, +0.175] (n=50) | replicated | practically significant (whole interval beyond threshold) |
| H7 | M2 | P2 | mission_success | +0.62 [+0.48, +0.76] (n=50) | +0.54 [+0.4, +0.68] (n=50) | replicated | practically significant (whole interval beyond threshold) |
| H8 | M1 | P3 | recall | +0.0575 [+0.0325, +0.0825] (n=50) | +0.0575 [+0.0325, +0.085] (n=50) | replicated | practically significant (whole interval beyond threshold) |
| H9 | M1 | P3 | mission_success | +0.34 [+0.22, +0.48] (n=50) | +0.3 [+0.18, +0.42] (n=50) | replicated | practically significant (whole interval beyond threshold) |
| H10 | M1 | P3 | time_to_recall_target | -81.1 [-91.8, -70.2] (n=33) | -83.8 [-94.3, -72.4] (n=35) | replicated | practically significant (whole interval beyond threshold) |
| H11 | M2 | P3 | time_to_recall_target | -50.7 [-59.6, -42.5] (n=49) | -48 [-58.4, -38.1] (n=49) | replicated | practically significant (whole interval beyond threshold) |
| H12 | M2 | P5 | hazard_exposures | -10.3 [-12.3, -8.42] (n=50) | -12.5 [-15.3, -9.98] (n=50) | replicated | practically significant (whole interval beyond threshold) |
| H13 | M2 | P5 | agents_disabled_hazard | -0.26 [-0.4, -0.14] (n=50) | -0.12 [-0.22, -0.04] (n=50) | replicated | point estimate beyond threshold, interval reaches below it |
| H14 | M2 | P6 | hazard_exposures | -5 [-5.9, -4.16] (n=50) | -5.06 [-5.98, -4.16] (n=50) | replicated | practically significant (whole interval beyond threshold) |
| H15 | M2 | P6 | agents_disabled_hazard | -0.08 [-0.16, -0.02] (n=50) | -0.06 [-0.14, +0] (n=50) | direction consistent, not replicated | not detectable |
| H16 | M1 | P7 | time_to_recall_target | -34.4 [-37.4, -31.3] (n=50) | -35.3 [-38, -32.4] (n=49) | replicated | practically significant (whole interval beyond threshold) |
| H17 | M2 | P7 | time_to_recall_target | -24 [-27, -20.8] (n=50) | -24.3 [-27.2, -21.2] (n=50) | replicated | practically significant (whole interval beyond threshold) |

#### Expected nulls

| ID | Config | Pair | Metric | Exploratory | Validation | Verdict |
|---|---|---|---|---|---|---|
| N1 | M1 | P4 | recall | +0 [+0, +0] (n=50) | -0.0025 [-0.0075, +0] (n=50) | consistent |
| N1 | M1 | P4 | mission_success | +0 [+0, +0] (n=50) | -0.02 [-0.06, +0] (n=50) | consistent |
| N1 | M1 | P4 | time_to_recall_target | -1.98 [-4.48, +0.48] (n=50) | -0.469 [-2.51, +1.71] (n=49) | consistent |
| N1 | M2 | P4 | recall | +0 [-0.0075, +0.0075] (n=50) | +0.0025 [-0.005, +0.0125] (n=50) | consistent |
| N1 | M2 | P4 | mission_success | +0 [+0, +0] (n=50) | +0 [+0, +0] (n=50) | consistent |
| N1 | M2 | P4 | time_to_recall_target | -0.98 [-2.9, +0.94] (n=50) | -1 [-2.64, +0.68] (n=50) | consistent |
| N2 | M2 | P5 | recall | +0.0025 [+0, +0.0075] (n=50) | +0.0025 [+0, +0.0075] (n=50) | consistent |
| N2 | M2 | P5 | mission_success | +0.02 [+0, +0.06] (n=50) | +0.02 [+0, +0.06] (n=50) | consistent |
| N2 | M2 | P6 | recall | +0.0025 [-0.0075, +0.0126] (n=50) | +0 [-0.01, +0.01] (n=50) | consistent |
| N2 | M2 | P6 | mission_success | +0 [+0, +0] (n=50) | +0 [+0, +0] (n=50) | consistent |

### Interpretation (on these scenarios and in simulation only)
- **Replicated, and practically meaningful.** The whole interval lies beyond the pre-registered threshold for each of these:
  - Informed frontier targets beat random waypoints (Q2): recall +0.30 (M1) and +0.13 (M2); mission success +0.64 (M1)
    and +0.54 (M2).
  - Pooled coordination beats independent frontier exploration (Q3): on M1, recall +0.06, mission success +0.30 and
    time to target −84 steps; on M2, time to target −48 steps.
  - Hazard-aware planning cuts hazard exposures: −12.5 per episode for `frontier`, −5.1 for `assignment`.
  - The oracle reference gap: 35 steps (M1) and 24 steps (M2).
- **Replicated, with a smaller margin.** Hazard-aware `frontier` reduces hazard disablements by 0.12 agents per
  episode [−0.22, −0.04]. The point estimate exceeds the 0.05 threshold, but the interval reaches below it.
- **Not established on fresh seeds:**
  - **Q1 (planned vs random motion).** Recall +0.03 [−0.03, +0.09] (M1) and +0.02 [−0.03, +0.07] (M2); M2
    mission success +0.12 [−0.06, +0.30]. The directions match, but the intervals include 0. `random_valid` did
    better on these seeds than on 0–49 (M1 recall 0.615 vs 0.537), while `astar_waypoint` changed little (0.647 vs
    0.623). So the exploratory recall advantage of planned waypoints is small and depends on the seed set.
    Planned motion still removes energy disablements: on M1, 1.86 agents per episode for random vs 0 for
    `astar_waypoint`.
  - **H15.** Hazard-aware `assignment` reduces hazard disablements by 0.06 [−0.14, 0.00]. The interval's upper bound
    sits exactly at 0, so the effect is not established.
- **Consistent nulls:**
  - Hungarian assignment shows no detectable difference from greedy allocation (`prioritized_frontier`) on recall,
    mission success or time to target, on either configuration.
  - Hazard awareness has no detectable cost in recall or mission success.
- **Robustness observations (descriptive):**
  - No planned policy ran out of energy on any validation seed. The decentralized strandings seen on seeds 0–49 did
    not recur.
  - All pooled-planner plan failures were temporary (reservations or horizon): 0.36–0.72 per episode, none
    unreachable. All decentralized plan failures were static (no known route).
  - The corrected executed-route energy check (ADR-0016 rule 16) never triggered.
  - `assignment` missed full recall once on M1 (seed 128: 7 of 8 survivors in 200 steps, with every cell sensed): a
    missed detection (`p_detect = 0.9`) that frontier-based policies never revisit. This is a known design limit of
    these baselines, not a fault.
- **Limitations:** see `docs/results/phase2_validation/README.md`. These include one map family, 50 seeds, no
  multiplicity correction, perfect communication for pooled policies, the M2 recall ceiling, and conditional time to
  target.

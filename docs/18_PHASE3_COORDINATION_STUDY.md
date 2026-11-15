# Phase 3 Coordination Study (M3.5) — pre-registration

Status: **pre-registered**. This document and `configs/phase3/study.json` were committed before any run on
evaluation seeds 200–249. Results are appended below the "Results" heading after the run and never edited
into the design sections.

## Purpose
Test whether communication-limited distributed coordination improves search and rescue in this simulator.
Also measure how much coordination degrades under realistic channel faults and agent failures.

The study covers:
- belief sharing;
- the lease-based auction with failure detection and reassignment (ADR-0018 to ADR-0020);
- the same auction without reassignment.

They are compared against independent and centralized baselines. The Phase 3 design step stated
the questions but did not commit hypotheses to the repository. This document is the binding version.

## Implementation under test
- Branch `phase-3-m3.5-evaluation` at the commit that adds this document.
- No policy, channel or detector parameter is changed by this study.
- The fault-injection mechanisms are new (ADR-0021). They add scheduled failures and partitions,
  and change no default behaviour.

## Questions
- **Q1:** Does sharing local belief maps (`frontier_shared`) speed up the search compared with independent
  `frontier`?
- **Q2:** Does auction coordination (`auction_comm`) add to belief sharing, and does it beat independent
  search?
- **Q3:** How far is communication-limited coordination from centralized assignment with perfect, pooled information
  (`assignment`)? This is a difference test, not an equivalence claim.
- **Q4:** When agents fail, does failure detection with lease expiry and reassignment beat the same auction
  without reassignment (`auction_no_reassign`)?
- **Q5:** How does the auction degrade under heavy packet loss, short range and a temporary partition?

## Scenario
- **Base (`nominal`):** `configs/phase3/nominal.json` (config hash `3412a5225c75db89…`).
  - `configs/m2_default.json` unchanged: 20×20, 2 drones and 2 ground robots, 8 survivors, hazards,
    false positives, recall target 0.75, time limit 200.
  - Plus the approved channel: range 8 for drones and 5 for ground robots, no loss, no delay, no
    bandwidth limit.
- Every other condition changes one factor from `nominal`, as listed in `configs/phase3/study.json`.

| Condition | Change from nominal | Policies |
|---|---|---|
| `nominal` | — | all 8 |
| `range_short` / `range_long` / `range_unlimited` | ranges (4, 3) / (16, 10) / unlimited | S |
| `drop_10` / `drop_30` / `drop_60` | `p_drop` 0.1 / 0.3 / 0.6 | S |
| `delay_2_1` / `delay_5_3` | (latency, jitter) (2, 1) / (5, 3) | S |
| `bw_32` / `bw_16` | bandwidth 32 / 16 units per agent per step | S |
| `partition_split` | steps 40–119: {drone_0, ground_0} cut from {drone_1, ground_1} | S |
| `isolate_drone` | steps 40–119: drone_0 cut from everyone | S |
| `fail_1` / `fail_2` | 1 / 2 agents disabled at steps uniform in 30–90 (ADR-0021) | S + `assignment` |

S = `frontier`, `frontier_shared`, `auction_comm`, `auction_no_reassign`.

How the levels were chosen:
- **Fixed in advance.** The ranges bracket the approved values. Loss and delay are spread across
  the configuration's range.
- **Bandwidth.** Levels sit below the median message size, which was measured on development
  seeds 1000–1019 under `nominal`:
  - `frontier_shared`: median 48 units, interquartile range 33–57;
  - `auction_comm`: median 56 units, interquartile range 39–68.

  This step measured traffic only, not performance.
- **No level was tuned toward an outcome.**
- **Hazard disablement and energy depletion stay active in every condition.** Injected failures
  come on top of them.

### Policies and information access

| Policy | Information model |
|---|---|
| `random_valid` | own observation only |
| `astar_waypoint` | own observation, fixed waypoint tour |
| `frontier` | own belief map only (ignores the channel) |
| `frontier_shared` | own belief + belief records received over the channel |
| `auction_comm` | own belief + received beliefs, claims and heartbeats; failure detection and reassignment |
| `auction_no_reassign` | as `auction_comm` without suspicion, lease expiry or reassignment |
| `assignment` | **centralized, pooled observations, perfect instantaneous communication** (ignores the channel) |
| `assignment_oracle` | **ORACLE: hidden ground truth.** Upper-bound reference only, never a competitor |

`assignment` and the oracle have different information access from the communication-limited policies. Every
comparison with them is labelled as such.

## Seeds
| Purpose | Seeds |
|---|---|
| Evaluation (this study, used once) | 200–249 (n = 50 per cell, identical for every cell) |
| Development and functional checks (never reported) | 1000–1019 |

The development checks used before pre-registration:
- traffic calibration on 1000–1019;
- a runner and analysis smoke test on 1000–1003.

The smoke test showed that on this scenario `recall` and `mission_success` saturate for every
coordinated policy: every episode reached the 0.75 target. So the primary outcome is search speed,
not success. This was decided before the evaluation run and is disclosed here.

## Metrics
- **Primary: `time_censored`.**
  - For a successful episode, its `time_to_recall_target`.
  - Otherwise `time_limit + 1` (201), a conservative penalty.
  - Defined for every completed episode.
- **Secondary (descriptive):**
  - outcome: `mission_success`, `recall`, `coverage_sensed`, `energy_used`, `agents_disabled`,
    `precision`;
  - communication: `messages_sent`, `message_units_per_step` (communication overhead),
    `message_delivery_rate`, `survivor_info_delay_mean` and `survivor_info_informed_fraction`
    (information propagation);
  - failure detection and reassignment (auction policies only): detection delays and unresolved
    counts, `false_suspicions`, `abandoned_sectors`, `reassignment_delay_mean`,
    `unresolved_sectors`, `duplicate_ownership_steps`;
  - runtime per episode (wall clock, not deterministic).
- Metric definitions are in ADR-0009 and ADR-0017 to ADR-0021, and in the result files.

## Statistics
- **Per cell:**
  - n, mean, sample SD and 95% percentile-bootstrap interval (10 000 resamples, seed 0);
  - counts of failed episodes and undefined values.
- **Comparisons:**
  - Paired by seed. Pairing is valid because the scenario stream does not depend on the channel or
    the failure stream. The analysis checks that both cells generated the same scenario for every
    seed (`scenario_sha256`) and refuses to compare otherwise.
  - Each comparison reports: the mean difference (B − A) with a 95% bootstrap interval, Cohen's
    d_z, and a two-sided paired sign-flip randomization test (20 000 resamples, seed 0; exact
    when at most 16 differences are non-zero).
  - Pairs with an undefined value on either side are excluded and counted.
- **Confirmatory family:** H1–H9, Holm-adjusted, α = 0.05.
- **Exploratory family:** Benjamini–Hochberg adjusted and reported as descriptive only. It
  contains:
  - every non-nominal condition against `nominal`, per S policy, on `time_censored`,
    `mission_success`, `recall`, `message_delivery_rate`, `message_units_per_step` and
    `survivor_info_delay_mean`;
  - the pairs `frontier`→`frontier_shared`, `frontier_shared`→`auction_comm` and
    `auction_no_reassign`→`auction_comm` in every condition, on `time_censored`, `mission_success`
    and `recall`.
- **Practical thresholds:** `time_censored` 5 steps, `mission_success` 0.05, `recall` 0.02,
  `unresolved_sectors` 0.25.
- **Code:** `python -m swarm.experiments.study_analysis` (tested in `tests/test_study.py`).

### Confirmatory hypotheses (difference = B − A)

| ID | Q | A → B | Metric | Expected |
|---|---|---|---|---|
| H1 | Q1 | nominal: `frontier` → `frontier_shared` | time_censored | − |
| H2 | Q2 | nominal: `frontier_shared` → `auction_comm` | time_censored | − |
| H3 | Q2 | nominal: `frontier` → `auction_comm` | time_censored | − |
| H4 | Q3 | nominal: `assignment` (pooled, perfect comms) → `auction_comm` | time_censored | two-sided |
| H5 | Q4 | fail_2: `auction_no_reassign` → `auction_comm` | time_censored | − |
| H6 | Q4 | fail_2: `auction_no_reassign` → `auction_comm` | unresolved_sectors | − |
| H7 | Q5 | `auction_comm`: nominal → drop_60 | time_censored | + (slower) |
| H8 | Q5 | `auction_comm`: nominal → range_short | time_censored | + (slower) |
| H9 | Q5 | `auction_comm`: nominal → partition_split | time_censored | + (slower) |

### Decision rules (fixed before running)
- **Supported:** Holm-adjusted p < 0.05 and the mean difference has the expected sign. For H4, any
  sign counts as "difference detected".
- **Contradicted:** Holm-adjusted p < 0.05 with the opposite sign.
- **Inconclusive:** otherwise. Reported as such, never as evidence of no effect.
- A supported effect whose whole interval lies within the practical threshold is reported as "detectable but
  small".

## Integrity rules
- Every episode is recorded. A crash marks the episode failed, the runner exits with status 3, and the crash
  is reported. No episode is removed, rerun selectively or replaced.
- No code or parameter change during the run. If a defect is found, the results are reported as they stand, and
  any fix needs a new pre-registration on new seeds.
- Commands:
  ```
  python -m swarm.experiments.study --spec configs/phase3/study.json --out-dir docs/results/phase3/runs --jobs 2
  python -m swarm.experiments.study_analysis --spec configs/phase3/study.json --runs docs/results/phase3/runs --out docs/results/phase3/analysis.json
  ```

## Known limitations (stated in advance)
- One map size and team composition (20×20, 2 + 2). Larger scales belong to Phase 5.
- Success saturates on this scenario, so conclusions concern search speed, not mission success.
- One-factor-at-a-time sweeps do not measure interactions between channel faults.
- Simulated agents, sensors and channel only. No claim about real operations.

## Results (study A)

### Provenance
- Run from commit `8dfa7af` (clean tree, recorded in every result file); Python 3.13.18, NumPy 2.5.4.
- 66 cells × 50 seeds = 3 300 episodes. All completed: no crash, no failed episode, runner exit status 0.
- Files in `docs/results/phase3/`:
  - `runs.tar.gz`: per-cell result files and `seed_level.csv`;
  - `analysis.json`: every test;
  - `results.md`: all tables;
  - `figures/`.
- `analysis.json` was produced by the pre-registered analysis. After the run, one purely descriptive
  derived metric was added: `agents_disabled_injected`, the scheduled failures that happened before
  the episode ended. No test or decision rule changed.

### Confirmatory verdicts
Means in steps (lower is faster). Intervals are 95% bootstrap; p is Holm-adjusted.

| ID | Comparison | Mean A → B | Diff [95% CI] | d_z | p (Holm) | Verdict |
|---|---|---|---|---|---|---|
| H1 | `frontier` → `frontier_shared` | 81.5 → 44.3 | −37.1 [−47.5, −27.6] | −1.01 | 0.0004 | **supported** |
| H2 | `frontier_shared` → `auction_comm` | 44.3 → 34.0 | −10.4 [−14.5, −6.5] | −0.70 | 0.0004 | **supported** |
| H3 | `frontier` → `auction_comm` | 81.5 → 34.0 | −47.5 [−58.9, −36.7] | −1.18 | 0.0004 | **supported** |
| H4 | `assignment` (pooled, perfect comms) → `auction_comm` | 36.3 → 34.0 | −2.3 [−6.2, +1.3] | −0.17 | 1.0 | inconclusive |
| H5 | fail_2: `auction_no_reassign` → `auction_comm` | 36.8 → 34.7 | −2.1 [−8.3, +1.4] | −0.10 | 1.0 | inconclusive |
| H6 | fail_2: same, `unresolved_sectors` | 0.26 → 0.22 | −0.04 [−0.12, 0.00] | −0.14 | 1.0 | inconclusive |
| H7 | `auction_comm`: nominal → drop_60 | 34.0 → 36.3 | +2.4 [−0.9, +5.7] | 0.19 | 1.0 | inconclusive |
| H8 | `auction_comm`: nominal → range_short | 34.0 → 36.5 | +2.5 [−1.0, +7.6] | 0.15 | 1.0 | inconclusive |
| H9 | `auction_comm`: nominal → partition_split | 34.0 → 35.4 | +1.4 [−0.1, +4.3] | 0.14 | 1.0 | inconclusive |

H1–H3 are also practically significant: the whole interval lies beyond 5 steps.

### What the inconclusive results do and do not show
- **H5, H6 and H9 had little exposure to the fault. This is a design flaw, reported as such.**
  - Under `auction_comm` the median episode lasts 30 steps. Only 26 of 50 episodes were still
    running at step 30, and 13 at step 40.
  - So most scheduled failures (steps 30–90) and the partitions (from step 40) came after the
    mission had already ended. Only 0.26 of the 2 scheduled failures per episode actually happened
    under `auction_comm` in `fail_2`.
  - Across the 50 seeds, the partition changed the outcome of 2 `auction_comm` episodes, and
    `fail_2` changed only 7 in the H5 comparison.
  - These tests therefore say almost nothing about failure handling. This finding motivated study
    B below.
- **H7 and H8** (loss and range applied for the whole episode) were not detected. On this map, the
  auction slowed by 2–3 steps on average under 60% loss or ranges (4, 3). The intervals include 0.
- **H4:** the communication-limited auction was not distinguishable from centralized pooled
  assignment. This is not an equivalence claim: the interval is −6.2 to +1.3 steps.

### Exploratory findings (BH-adjusted, descriptive)
- **Latency and bandwidth slow coordination much more than loss or range.**
  - `auction_comm` against nominal:
    - delay (2, 1): +5.3 steps [2.6, 8.0], q = 0.001;
    - delay (5, 3): +17.6 [12.6, 22.8], q < 0.001;
    - bandwidth 32: +9.1 [3.4, 15.7], q = 0.009;
    - bandwidth 16: +17.1 [9.8, 25.2], q < 0.001.
  - `frontier_shared` is the most delay-sensitive: delay (2, 1) gives +23.5 steps [17.3, 30.3].
- **The auction's advantage over belief sharing grows under channel faults.**
  - drop_60: −13.7 steps [−18.8, −8.5];
  - delay (2, 1): −28.6 [−36.4, −21.2];
  - both q < 0.001.
- **Reassignment versus no reassignment:** no difference was detected anywhere.
- **False suspicions are frequent:** about 8.8 per `auction_comm` episode at nominal range. A
  teammate that is merely out of range for more than 10 steps is suspected. The suspicion clears
  when the agent is heard again (ADR-0020), so this is how the detector is designed to work, not a
  defect, but it is a cost.
- **Metric caveat:** under `auction_no_reassign` a failed agent's sector can still show a
  "reassignment". This happens when another agent already held it concurrently (duplicate
  ownership). It is not a lease takeover.

Full tables: `docs/results/phase3/results.md`. Figures: `figures/nominal_policies.png` and
`figures/channel_sweeps.png`.

## Study B: fault-exposure follow-up (pre-registered after study A, before any run on its seeds)

### Why
Study A's fault windows mostly fell after episodes had ended, so they could not test failure
handling. Study B moves the same faults earlier.
- **New windows:** failures at steps 5–25; partitions during steps 5–59.
- **New seeds:** 300–349, never used before, n = 50.
- **Basis for the windows:** the episode lengths observed in study A. This is a structural
  property, not the direction of any effect.
- **Unchanged:** hypotheses, policies, statistics and decision rules are the same as study A,
  except that the family is new.
- **Study A stands as reported.** Study B does not replace it.

### Design
- **Spec:** `configs/phase3/study_b.json`, with the same base config as study A.
- **Conditions:**
  - `nominal`: S + `assignment`;
  - `fail_1_early`: count 1, steps 5–25; S + `assignment`;
  - `fail_2_early`: count 2, steps 5–25; S + `assignment`;
  - `partition_split_early`: {drone_0, ground_0} cut during steps 5–59; S;
  - `isolate_drone_early`: drone_0 isolated during steps 5–59; S.
- **Confirmatory family (Holm, α = 0.05):**

| ID | A → B | Metric | Expected |
|---|---|---|---|
| B1 | fail_2_early: `auction_no_reassign` → `auction_comm` | time_censored | − |
| B2 | fail_2_early: `auction_no_reassign` → `auction_comm` | unresolved_sectors | − |
| B3 | fail_1_early: `auction_no_reassign` → `auction_comm` | time_censored | − |
| B4 | fail_2_early: `assignment` (pooled, perfect comms) → `auction_comm` | time_censored | two-sided |
| B5 | `auction_comm`: nominal → partition_split_early | time_censored | + (slower) |
| B6 | `auction_comm`: nominal → isolate_drone_early | time_censored | + (slower) |

- **Exploratory family (BH):** the same as study A, applied to these conditions.

### Results (study B)
Not yet run.

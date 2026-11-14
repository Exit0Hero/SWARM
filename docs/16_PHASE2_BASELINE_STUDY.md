# Phase 2 Baseline Study

This document was pre-registered before any Phase 2 baseline was run: the questions, configurations, seeds,
metrics and comparisons below were committed first. Results are appended in the
[Results](#results) section, produced by the command shown there, and nothing above that section changes after
the runs.

## Questions
1. Does planned motion beat random motion? (`astar_waypoint` vs `random_valid`)
2. Do informed targets beat uninformed ones with the same planner? (`frontier` vs `astar_waypoint`)
3. Does pooled centralized coordination beat independent frontier exploration?
   (`prioritized_frontier` vs `frontier`; `assignment` vs `prioritized_frontier`)
4. Does hazard-aware planning reduce hazard exposure and disablement, and at what cost in recall?
   (`frontier_hazard_aware` vs `frontier`; `assignment_hazard_aware` vs `assignment`)
5. How far are the non-oracle baselines from an oracle reference? (`assignment_oracle` vs `assignment`). This is
   reported as a reference gap, never as decentralized or pooled performance.

## Configurations
- **Primary:** `configs/m2_default.json`. 20 × 20 grid, 2 drones and 2 ground robots, 8 survivors, ground-only
  hazards, `p_false_positive = 0.002`, `recall_target = 0.75`, 200 steps.
- **Secondary:** `configs/m1_default.json`. The same map family without hazards or false positives,
  `recall_target = 1.0`.

## Seeds
- **Evaluation:** 0–49 for every policy and both configurations, with identical lists and config hashes.
- **Development:** 1000–1009, used only while implementing and debugging. They are never reported.

## Metrics (ADR-0009 registry)
- **Primary:** `recall`, `mission_success`.
- **Secondary:** `time_to_recall_target`, `coverage_sensed`, `energy_used`, `move_conflicts`, `invalid_actions`,
  `agents_disabled_energy`, `agents_disabled_hazard`, `hazard_exposures`, `precision`.
- **Policy diagnostics** (ADR-0016): failed plans, planned waits, reassignments.

## Analysis
- Per policy: n, mean, sample standard deviation and a 95% percentile-bootstrap interval (10 000 resamples,
  bootstrap seed 0), with counts of failed episodes and undefined values (ADR-0009).
- Comparisons: paired per-seed differences (candidate − baseline) with 95% bootstrap intervals, for the pairs
  in the Questions section.
- A difference is described as evidence only if its 95% paired interval excludes 0. With several comparisons and no
  multiplicity correction, the results are **exploratory** and are labelled that way.
- No claims beyond these scenarios. Results are simulation results for these synthetic maps and parameters only.

## Results
*Pending. Appended after the runs.*

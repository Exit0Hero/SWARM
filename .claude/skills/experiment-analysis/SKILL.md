---
name: experiment-analysis
description: Design, run and report SWARM experiments - seed protocols, multi-seed evaluation, metric aggregation, confidence intervals, policy comparisons, ablations, and result tables or charts. Use whenever results are produced, compared or written up, and before making any claim that one method performs better.
---

# Experiment analysis

## Read first
- `docs/05_EXPERIMENTS_AND_METRICS.md` (protocol)
- `docs/11_REPRODUCIBILITY_AND_EXPERIMENT_LOGGING.md` (run metadata)
- ADR-0006 (seeds) and ADR-0009 (metric definitions, aggregation) in `docs/decisions/`

## Non-negotiable
- **Only report numbers produced by a run in this repository.** Quote the command, config hash and commit.
  If something was not run, say so. Never estimate, extrapolate or "expect" results in place of measuring them.
- Every compared policy uses the **same evaluation seed list** and the same config hash.
- Training, validation and evaluation seed sets are disjoint.
- Oracle policies are labelled as oracle in every table and chart.

## Procedure
1. **Pre-register.** Before running, write the question, the compared policies, the config, the seed list, the primary
   metric, and what result would answer the question. Commit it with the experiment config.
2. **Run** through the experiment runner only. Results are versioned JSON with full metadata.
3. **Aggregate** per ADR-0009: n, mean, sample standard deviation, 95% bootstrap confidence interval with a recorded seed,
   and the count of null values reported separately.
4. **Compare** using paired differences per seed (the seeds are matched), with a bootstrap confidence interval on the mean
   difference. Overlapping marginal intervals are not a test.
5. **Report.** Each table or chart states the metric definition, n, seed list or range, training or evaluation, and
   commit. Use the dataviz skill (if available) for charts.
6. **Limitations.** List what the experiment cannot show (map sizes, agent counts, simulator assumptions from
   `docs/15_RISK_REGISTER_AND_LIMITATIONS.md`).

## Language
Use "on these scenarios" and "in simulation". Never write "proves", "guarantees" or "real-world ready".

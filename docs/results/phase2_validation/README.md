# Phase 2 validation results (fresh seeds 100–149)

Pre-registration and interpretation: `docs/17_PHASE2_VALIDATION_STUDY.md`.

## Provenance
- **Implementation evaluated:** `226bf9da7f4e29396daff54d26f9cfabc8f5ecac` (`main`, merge of PR #5; tree identical to the
  reviewed and corrected PR head `31f289c`).
- **Pre-registration commit:** `613c5f36bb88a1eead11e04d5577e7fc12c25c1c`. It adds only
  `docs/17_PHASE2_VALIDATION_STUDY.md`; `src/`, `configs/` and `pyproject.toml` are identical to `226bf9d`. The runs
  were executed from this commit with a clean tree, so every result file records `commit 613c5f36…, dirty: False`.
- **Seeds:** 100–149 (`seeds.txt`), identical for every policy and both configurations.
- **Config hashes:** M1 `fec0dca1c6f801fc…` (`configs/m1_default.json`), M2 `8db09395238eb122…`
  (`configs/m2_default.json`). The full hashes are in every result file.
- **Software:** Python 3.13.18, NumPy 2.5.4, Linux x86-64.
- **Commands:**
  ```
  python -m swarm.experiments.run_baselines --config configs/m1_default.json --seeds 100-149 --out-dir runs/phase2_validation/m1
  python -m swarm.experiments.run_baselines --config configs/m2_default.json --seeds 100-149 --out-dir runs/phase2_validation/m2
  python docs/results/phase2_validation/replication_analysis.py
  ```
  Both runs exited with status 0: 8 policies × 2 configurations × 50 seeds = 800 episodes, all completed, none failed.

## Files
- `m1/`, `m2/`: per-policy result files (`<policy>.json`: per-episode records with seed, status, end reason, steps,
  every metric, policy diagnostics, scenario and event-stream SHA-256; summaries; metadata), `summary.md` and
  `comparisons.json`, copied unchanged from the run outputs.
- `replication_analysis.py`: applies the pre-registered decision rules. It only reads files.
- `replication.json`, `replication.md`: its output (headline effects, expected nulls, all pre-registered pairs
  compared with the corrected exploratory run in `docs/results/phase2_corrected/`).

## Methodology
Per policy: n, mean, sample standard deviation and a 95% percentile-bootstrap interval (10 000 resamples, bootstrap
seed 0). Comparisons: paired per-seed differences (candidate − baseline) with 95% bootstrap intervals;
`time_to_recall_target` is paired only on seeds where both policies reached the target. The replication rules,
practical-significance thresholds and verdict rule are those pre-registered in
`docs/17_PHASE2_VALIDATION_STUDY.md`.

## Limitations
- One procedural map family (20 × 20), four agents, fixed parameters, synthetic sensing. These are results on these
  scenarios, in simulation only.
- 50 seeds per configuration; no multiplicity correction. Bootstrap interval bounds on discrete metrics can sit
  exactly at 0, which makes some decisions sensitive to single episodes (for example H15).
- Pooled policies assume perfect, instantaneous communication. The oracle is a reference, not an achievable policy.
- On the M2 configuration (`recall_target = 0.75`), recall saturates near 0.75 for successful episodes. Time to target
  is conditional on both policies succeeding, and is uninformative where few pairs exist (M1 P2: 2 pairs).
- The corrected energy rule (ADR-0016 rule 16) never triggered in these runs (`energy_route_aborts` = 0). It is
  exercised by the regression tests, not by these configurations.

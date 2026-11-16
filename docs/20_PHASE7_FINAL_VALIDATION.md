# Phase 7 Final Validation: pre-registration

Status: **complete** (run and analysed). This document and `configs/phase7/study.json` were committed before any run on
evaluation seeds 500–549, and before the Phase 4 evaluation results (docs/19) were known. Results are appended
under "Results".

This is roadmap Phase 7, which the 2026-11-15 directive calls "Phase 5". Roadmap Phases 5 (perception) and 6
(dashboard) are deferred by the user.

## Questions
1. **Generality:** do the Phase 3 coordination findings hold across map sizes, layouts, team
   compositions, survivor distributions, obstacle densities, hazards, communication and failure
   rates?
2. **Ablations:** what does each component contribute?
   - communication with shared beliefs (`frontier` → `frontier_shared`);
   - auction allocation (`frontier_shared` → `auction_comm`);
   - lease expiry, failure detection and reassignment (`auction_no_reassign` → `auction_comm`);
   - learned allocation (`frontier_shared` → `mappo`).
   - Reward components were ablated in Phase 4 (docs/19 P4).
   - Energy-aware navigation has no variant without it (ADR-0023), so it is not ablated.
3. **Mission success:** when success is not saturated (recall target 1.0), does coordination change
   it?

## Design
- **Seeds:** 500–549 (n = 50, never used before), identical for every cell.
- **Base:** `configs/phase3/nominal.json`.
- **Conditions:** 17, each changing one factor from nominal (`configs/phase7/study.json`). The
  factors and their levels:
  - map 15 / 30 / 40, with the same survivor density and time limit 200 / 300 / 400;
  - obstacles low (5% + 5%) or high (25% + 25%);
  - hazards none or high (20%, disable probability 0.05);
  - clustered survivors (2 clusters, spread 2; ADR-0023);
  - base in the corner;
  - teams of 1+1, 3+3 and 4+4;
  - a combined poor channel (range 4/3, loss 0.3, latency 2, jitter 1);
  - 1 or 2 early failures (steps 5–25);
  - recall target 1.0.
- **Policies in every condition:**
  - `frontier`, `frontier_shared`, `auction_comm`, `auction_no_reassign`;
  - `assignment` (centralized, pooled, perfect communication; reference with different
    information access);
  - `mappo`: the three Phase 4 MAPPO models, pooled per seed, with the validation-selected mode.
    They were trained only on the nominal 20 × 20, 2 + 2 scenario, so every other condition is
    out of distribution for them.
- **Oracle:** `assignment_oracle` in nominal only (reference).
- **Size:** 137 cells × 50 seeds = 6 850 episodes.

### Confirmatory family
54 tests, Holm-adjusted together, α = 0.05. The metric is `time_censored` unless stated. Difference
= B − A.

| Family | Comparison (A → B) | Per | Expected |
|---|---|---|---|
| C (17 tests) | `frontier` → `frontier_shared` | each condition | − |
| A (17 tests) | `frontier_shared` → `auction_comm` | each condition | − |
| L (17 tests) | `frontier_shared` → `mappo` | each condition | two-sided |
| R (2 tests) | `auction_no_reassign` → `auction_comm`, `unresolved_sectors` | fail_1_early, fail_2_early | − |
| S (1 test) | `frontier_shared` → `auction_comm`, `mission_success` | recall_full | + |

- **Statistics, decision rules and integrity rules:** as in docs/18.
- **Exploratory family:** Benjamini–Hochberg, descriptive only. It contains:
  - each condition against nominal for four policies;
  - `assignment` → `auction_comm`, `auction_no_reassign` → `auction_comm` and `auction_comm` →
    `mappo` in every condition.

### Scalability (descriptive, development seeds)
```
python -m swarm.experiments.scalability --config configs/phase3/nominal.json \
    --out docs/results/phase7/scalability.json --policies frontier_shared auction_comm --profile
```
- Varies team (1–8 per type), map side (20–60) and survivors (4–32).
- Five development seeds per point.
- Runtimes are machine-dependent and reported as such.

## Known limitations (stated in advance)
- **One factor at a time:** interactions are not measured, except in the combined `comms_poor`
  condition.
- **Results with zero exposure:** a component can only matter where its trigger occurs. For
  example, reassignment needs failures that happen before the mission ends.
- **MARL is out of distribution:** the learned models are tested outside their training
  distribution in every non-nominal condition, after a small training budget (docs/19).
- **Simulation only:** no claim about real operations.

## Results

### Provenance
- Every cell ran at commit `75f5cd5` (clean tree). That commit is the pre-registration plus the
  merged Phase 4 models.
- The run was interrupted once by an environment shutdown after 31 of 137 cells. It was resumed
  with `--resume`, which skips only finished cells with the same config hash and seeds. Episodes
  are deterministic, so the interruption cannot change any result.
- 137 cells × 50 seeds = 6 850 episodes. All completed, with no crash.
- Files: `docs/results/phase7/runs.tar.gz`, `analysis.json`, `results.md`, `figures/` and
  `scalability.json`.
- Not tested: 144 exploratory "condition vs nominal" comparisons that cannot be paired, because the
  map or team differs (listed in `analysis.json`). No confirmatory test was affected.

### Confirmatory verdicts (54 tests, Holm)
| Family | Supported / detected | Inconclusive | Contradicted |
|---|---|---|---|
| C: `frontier` → `frontier_shared` (belief sharing) | 17 / 17 | 0 | 0 |
| A: `frontier_shared` → `auction_comm` (auction allocation) | 7 / 17 | 10 | 0 |
| L: `frontier_shared` → `mappo` (learned allocation, two-sided) | 1 / 17 | 16 | 0 |
| R: reassignment, `unresolved_sectors` (fail_1_early, fail_2_early) | 0 / 2 | 2 | 0 |
| S: auction, `mission_success` at recall target 1.0 | 0 / 1 | 1 | 0 |

Details in steps (lower is faster); intervals are 95% bootstrap:
- **C: sharing helps everywhere.** Reductions range from −12.9 steps (`comms_poor`) to −125.7 to
  −91.7 (`map_30`, interval). At nominal: 92.0 → 48.0.
- **A: the auction's advantage is clear where communication or team size makes allocation
  matter.**
  - Supported: nominal −11.9 [−16.5, −7.3]; `obstacles_low` −27.7; `comms_poor` −30.4; `team_3x3`
    −11.6; `team_4x4` −13.9; `hazards_none` −12.0; `hazards_high` −10.0.
  - Inconclusive on larger maps (`map_30` −2.6 [−24.4, +21.1]), with high obstacle density
    (−1.0), with early failures, and with a 1 + 1 team.
  - All A point estimates favour the auction. None is contradicted.
- **L: no detectable difference** between learned allocation and belief sharing in 16 of 17
  conditions. The one exception is `obstacles_low`, where MAPPO was faster: −13.6 [−19.4, −7.9].
  MAPPO was trained only at nominal, so 16 conditions are out of distribution.
- **R: reassignment did not survive the correction.** Unresolved sectors fell from 0.80 to 0.66
  and from 1.46 to 1.28, with intervals excluding 0 (unadjusted). After Holm across 54 tests both
  are inconclusive. Study B of docs/18 (B2) had detected this effect with a 6-test family.
- **S: with full recall required, success stays saturated.** 0.98 vs 0.96 [−0.08, +0.04] at time
  limit 200. Question 3 cannot be answered on this scenario, because the time limit is generous
  enough for every coordinated policy.

### Scale and generalization (descriptive)
- **Success.** At map 40 (time limit 400), success falls to:
  - 0.02 for `frontier`;
  - about 0.45 for `frontier_shared`, `auction_comm` and `mappo`;
  - 0.78 for centralized `assignment`.

  This is the clearest gap between communication-limited and centralized coordination in the
  project.
- **Team size.** Larger teams help the auction most: 36 → 26 → 22 steps for 2 + 2, 3 + 3 and
  4 + 4 agents.
- **Runtime (`scalability.json`, development seeds, one machine).**
  - `auction_comm` costs 1.9 ms per step for a team of 1 + 1 and 38 ms for 8 + 8. Growth with team
    size is superlinear: claim tables and record sharing are pairwise.
  - Map size matters less: 5 ms per step at 20 × 20 and 13 ms at 60 × 60.
  - Profiling hotspots are policy-side (target selection and belief merging), not the
    environment.

### Conclusions (simulation only, these scenarios)
1. **Belief sharing is the single most robust improvement.** It was supported in all 17
   conditions.
2. **The lease auction adds a further, frequent speedup.** It was supported in 7 of 17
   conditions, and the point estimate favours it in all 17. The benefit is largest under poor
   communication and with larger teams. It was not shown on larger maps or under early failures.
3. **The learned allocator (MAPPO, small CPU budget) matches belief sharing but not the auction.**
   - It generalizes without collapse to unseen maps, teams and faults.
   - It did not outperform the classical methods, apart from one condition.
4. **Failure reassignment reclaims sectors**, but the effect is small and only detectable in a
   focused test.
5. **Centralized pooled assignment remains the upper reference among non-oracle policies.** The
   gap grows with map size.

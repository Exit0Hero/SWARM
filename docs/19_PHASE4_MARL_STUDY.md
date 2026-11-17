# Phase 4 MARL Study: pre-registration

Status: **complete** (trained at `0e0c609`, evaluated at `9bb1042`). This document, `configs/phase4/study.json` and the three training configs were
committed before any training run of this study and before any run on evaluation seeds 400–449. Results are
appended under "Results". The design sections are not edited afterwards.

## Question
Does learned sector selection improve communication-limited search? The learned policy is MAPPO
with a centralized critic (ADR-0022).

It is compared with the classical policies that use the same information, navigation and channel:
- `frontier_shared`: same belief sharing, no allocation;
- `auction_comm`: hand-designed allocation.

The study also measures:
- what the centralized critic adds (IPPO);
- what the coverage reward term adds;
- whether the learned policy generalizes beyond its training distribution.

## Training (fixed budget, disclosed)

| Model set | Config | Training seeds | Budget per model |
|---|---|---|---|
| `mappo_s0..2` | `configs/phase4/train_mappo.json` | 0, 1, 2 | 2 400 episodes (150 updates of 16 episodes) |
| `ippo_s0..2` | `configs/phase4/train_ippo.json` | 0, 1, 2 | same |
| `mappo_nocov_s0..2` | `configs/phase4/train_mappo_no_coverage.json` (coverage weight 0) | 0, 1, 2 | same |

- **Scenario:** `configs/phase3/nominal.json` (20 × 20, 2 drones and 2 ground robots, nominal
  channel).
- **Scenario seeds:** training seed *s* uses scenario seeds 10000 + 5000·*s* onward.
- **Checkpoint and decision mode:** validation every 25 updates on seeds 3000–3019. The greedy or
  sampled mode is selected there too (ADR-0022). Evaluation seeds are never used for selection.
- **Hardware:** CPU only, two cores, single-threaded PyTorch.
- **This is a small budget.** About 2 400 episodes is a few percent of what MARL papers typically
  use. A negative or null result here says only that this budget was insufficient. It does not
  say that the approach cannot work.
- **Hyperparameters** were fixed from two development runs on seeds 1000–1019 (ADR-0022). Nothing
  was tuned on validation or evaluation seeds.

## Evaluation
- **Seeds:** 400–449 (n = 50, never used before). The same seeds are used for every cell.
- **Policies, all through the normal runner:**
  - classical: `frontier`, `frontier_shared`, `auction_comm`;
  - `assignment` (centralized, pooled observations, perfect communication: a reference with
    different information access);
  - `assignment_oracle` (**oracle**, nominal only);
  - the nine learned models.
- **Learned policies are reported two ways:**
  - **pooled**: for each scenario seed, the mean over the three training seeds. This pooled value
    is used in the hypotheses;
  - **per training seed**: descriptive, to show the variance between training seeds.

| Condition | Change from nominal | Distribution |
|---|---|---|
| `nominal` | none | in-distribution (training scenario, new seeds) |
| `drop_30` | `p_drop` 0.3 | out of distribution: channel |
| `delay_2_1` | latency 2, jitter 1 | out of distribution: channel |
| `fail_2_early` | 2 agents disabled at steps 5–25 | out of distribution: failures |
| `map_30` | 30 × 30, 18 survivors (same density), time limit 300 | out of distribution: map size |
| `team_3x3` | 3 drones + 3 ground robots | out of distribution: team composition |

### Confirmatory hypotheses
Holm-corrected as one family, α = 0.05. The metric is `time_censored`, as in docs/18. Difference =
B − A, paired by scenario seed.

| ID | A → B | Expected |
|---|---|---|
| P1 | nominal: `frontier_shared` → `mappo` | − (learned allocation is faster than none) |
| P2 | nominal: `auction_comm` → `mappo` | two-sided |
| P3 | nominal: `ippo` → `mappo` | − (the centralized critic helps) |
| P4 | nominal: `mappo_nocov` → `mappo` | two-sided (reward ablation) |
| P5 | fail_2_early: `auction_comm` → `mappo` | two-sided |
| P6 | map_30: `frontier_shared` → `mappo` | − (generalizes to a larger map) |
| P7 | team_3x3: `frontier_shared` → `mappo` | − (generalizes to a larger team) |

- **Statistics, decision rules and integrity rules:** identical to docs/18.
- **Exploratory family:** Benjamini–Hochberg, descriptive only. It contains:
  - each condition against nominal;
  - `frontier_shared`/`auction_comm`/`ippo` → `mappo` and `frontier_shared` → `ippo` in every
    condition;
  - also on mission success, recall and energy.

## Known limitations (stated in advance)
- The training budget is small and covers one scenario family. MARL results depend strongly on
  budget and hyperparameters.
- Three training seeds give only a rough view of training variance. Pooling them answers "this
  training recipe", not "the best model".
- The learned policy chooses sectors, not moves or messages (ADR-0022).
- In development, the greedy policy behaved much like `FREE` search (similar to `frontier_shared`)
  on seeds 1000–1019. This is a warning sign, recorded here before the evaluation.
- Simulation only.

## Results

### Provenance
- **Training:** the nine models were trained exactly as pre-registered, at commit `0e0c609` on a
  clean tree.
  - Each model ran 2 400 episodes, about 57 000–64 000 agent decisions.
  - Wall clock: about 15–30 minutes per model on a shared 2-core CPU.
  - Models, logs and version records are in `docs/results/phase4/models/`.
- **Evaluation:** run at commit `9bb1042`, a clean tree that adds only the models.
  - 64 cells × 50 seeds = 3 200 episodes. All completed, with no crash.
  - Files in `docs/results/phase4/`: `runs.tar.gz`, `analysis.json`, `results.md` and
    `figures/` (training curves and nominal policies).
- **Deviation, after the run (analysis code only).** The exploratory family included "each
  condition against nominal". For `map_30` this comparison cannot be paired by seed, because the
  map itself differs, and the analysis stopped with a pairing error. Those 12 comparisons are now
  listed as not pairable (`exploratory_not_pairable` in `analysis.json`) and are not tested. No
  confirmatory test was affected.

### Confirmatory verdicts
`time_censored` in steps, lower is faster. `mappo` and `ippo` are the per-seed means over three
training seeds.

| ID | Comparison | Mean A → B | Diff [95% CI] | d_z | p (Holm) | Verdict |
|---|---|---|---|---|---|---|
| P1 | nominal: `frontier_shared` → `mappo` | 47.4 → 40.2 | −7.2 [−12.2, −2.5] | −0.41 | 0.037 | **supported** |
| P2 | nominal: `auction_comm` → `mappo` | 34.2 → 40.2 | +6.1 [+1.8, +10.4] | 0.39 | 0.046 | **difference detected: MAPPO slower** |
| P3 | nominal: `ippo` → `mappo` | 40.5 → 40.2 | −0.3 [−3.5, +2.8] | −0.02 | 1.0 | inconclusive (within ±5) |
| P4 | nominal: `mappo_nocov` → `mappo` | 39.7 → 40.2 | +0.5 [−1.6, +2.5] | 0.07 | 1.0 | inconclusive (within ±5) |
| P5 | fail_2_early: `auction_comm` → `mappo` | 69.7 → 77.0 | +7.3 [−2.9, +17.2] | 0.20 | 0.88 | inconclusive |
| P6 | map_30: `frontier_shared` → `mappo` | 164.6 → 167.5 | +2.9 [−10.0, +15.5] | 0.06 | 1.0 | inconclusive |
| P7 | team_3x3: `frontier_shared` → `mappo` | 33.5 → 33.6 | +0.1 [−3.0, +3.4] | 0.01 | 1.0 | inconclusive (within ±5) |

### Interpretation (this budget, these scenarios, simulation only)
- **Learned allocation beats having no allocation, in distribution, but not the hand-designed
  auction.**
  - MAPPO was 7 steps faster than belief sharing alone (P1).
  - It was 6 steps slower than the lease auction (P2), which needs no training.
  - Centralized pooled assignment (32.4) and the auction (34.2) remain the best non-oracle
    policies.
- **The centralized critic added nothing measurable here** (P3). IPPO and MAPPO are
  indistinguishable, with the interval within ±3.5 steps.
  - This matches reports that independent PPO can be competitive (de Witt et al., 2020).
  - A plausible reason, not tested: the actor already receives teammates' intents, and the critic
    state holds no information the actors lack.
- **The coverage reward term made no detectable difference** (P4).
- **No sign of a generalization advantage** (P5–P7).
  - Out of distribution, the learned policies did no better than `frontier_shared` on a larger
    map or a larger team.
  - Under early failures, the gap to the auction was inconclusive.
  - They did not collapse either: success rates stayed within a few points of the classical
    policies (`results.md`).
- **Exploratory:** under channel faults, both learned policies stayed clearly ahead of
  `frontier_shared` (delay (2, 1): −16 steps; loss 0.3: −8 to −10 steps) and clearly behind the
  auction (+10 to +12 steps). All BH q < 0.05.
- **Training variance:**
  - Mean time to target at nominal ranged from 39.1 to 41.7 across MAPPO training seeds, and from
    37.5 to 43.3 across IPPO seeds.
  - Validation scores, based on 20 seeds, were noisy (`figures/training_curves.png`). Selection on
    them is a source of optimism that the fresh evaluation seeds remove.
  - Training curves flattened after about 1 000 episodes, at about 40 steps per episode with the
    sampled policy.
- **Bottom line:** with this budget and action design, MARL learned a useful allocation heuristic
  but did not match a simple communication-limited auction. Whether more training, recurrence or
  learned messages would close the gap is untested.

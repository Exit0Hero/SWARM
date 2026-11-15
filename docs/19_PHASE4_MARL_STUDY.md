# Phase 4 MARL Study: pre-registration

Status: **pre-registered.** This document, `configs/phase4/study.json` and the three training configs were
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
Not yet run.

# Phase 7 Final Validation: pre-registration

Status: **pre-registered**. This document and `configs/phase7/study.json` were committed before any run on
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
Not yet run.

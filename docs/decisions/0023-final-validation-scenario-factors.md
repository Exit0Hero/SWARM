# ADR-0023: Scenario factors for final validation: clustered survivors and scalability benchmark

- **Status:** Accepted
- **Date:** 2026-11-15
- **Phase:** 7

## Context
Final validation (roadmap Phase 7; the user directive's "Phase 5") has to vary the following
factors:
- map size;
- layout;
- team composition;
- survivor distribution;
- obstacle density;
- hazards;
- communication;
- failure rates.

It also has to measure scalability. Every factor except survivor distribution can already be set
through existing config fields, because the study runner applies one-factor overrides. Survivors
were always uniform.

## Decision
1. **Clustered survivors.**
   - `ScenarioConfig.survivor_layout` is optional and is written only when set, so existing configs
     keep their bytes and hashes.
   - It takes `{"kind": "clustered", "clusters", "spread"}`.
   - Placement, using the same `scenario` stream after terrain:
     - draw `clusters` centres uniformly from the eligible cells;
     - draw survivors without replacement with weight `exp(-d²/2σ²)` to the nearest centre, where
       σ = `spread`;
     - every eligible cell keeps a small non-zero weight.
   - Uniform placement (no layout) is unchanged, so golden hashes are preserved.
2. **Scalability benchmark** (`swarm.experiments.scalability`).
   - One-factor sweeps from a base config: team size, map side (survivor density and time limit
     scaled with the map) and survivor count.
   - Records per point:
     - wall-clock milliseconds per step (descriptive, machine dependent);
     - deterministic traffic and outcomes;
     - with `--profile`, the five `swarm` functions with the most cumulative time.
   - Runs on development seeds only.
3. **Ablations without a variant are reported as such.**
   - Energy-aware navigation has no energy-unaware variant: removing it strands agents by design
     (ADR-0016).
   - Lease expiry, failure detection and reassignment are switched off together by
     `auction_no_reassign` (ADR-0019) and cannot be separated.
   - No new policy variants are added to manufacture these ablations.

## Consequences
- Phase 7 conditions are ordinary configs with recorded hashes (docs/20).
- Survivor clustering is a stylized model, not an empirical casualty distribution.

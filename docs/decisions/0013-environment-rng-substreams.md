# ADR-0013: Environment RNG sub-streams per stochastic process

- **Status:** Accepted (amends [ADR-0006](0006-random-number-generation-and-reproducibility.md) rule 1)
- **Date:** 2026-11-15
- **Phase:** 1 (M2)

## Context
M1 used one `dynamics` stream for all environment randomness, and its only consumer was true detection. M2 adds two
more stochastic processes: hazard disablement and false-positive reports. If they shared the `dynamics` stream:
- every M1 event stream would change, even with hazards and false positives switched off; and
- changing `p_false_positive` would also change which true detections happen. That ruins controlled sensor sweeps,
  which need common random numbers.

## Decision
1. The `dynamics` SeedSequence child keeps producing the `dynamics` generator, used for **true detections only**,
   exactly as in M1.
2. The same `dynamics` SeedSequence is additionally spawned into two children, in this order:
   `hazard` (hazard disablement draws, [ADR-0011](0011-static-hazards.md)) and `false_positive` (false-report draws,
   [ADR-0012](0012-false-positive-detections-and-precision.md)). Spawning children does not change the parent
   generator's output. A test checks this.
3. Full derivation, recorded in run metadata:
   `SeedSequence(root).spawn(3) → scenario, dynamics, policies`;
   `dynamics.spawn(2) → hazard, false_positive`;
   `policies.spawn(n_agents)` in sorted agent-ID order.
4. Any future stochastic process gets a new child, spawned *after* the existing ones, never a share of an existing stream.

## Alternatives considered
- **Everything on one `dynamics` stream:** breaks M1 reproducibility and common random numbers (see Context).
- **Unrelated root seeds per process:** harder to audit, and loses the single-root-seed guarantee of ADR-0006.

## Consequences
- M1 scenarios and event streams are byte-identical under M2 when hazards are absent and `p_false_positive = 0`.
  The M1 golden hashes stay valid without edits.
- For a fixed seed, sweeping `p_false_positive` leaves true-detection draws unchanged. Trajectories still diverge if
  policies react to false reports.

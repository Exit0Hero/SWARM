# ADR-0006: Random number generation and reproducibility

- **Status:** Accepted
- **Date:** 2026-11-15
- **Phase:** 1

## Context
[docs/11](../11_REPRODUCIBILITY_AND_EXPERIMENT_LOGGING.md) requires independent RNG streams and
recorded seed derivation. Comparing algorithms on "identical scenario seeds"
([docs/05](../05_EXPERIMENTS_AND_METRICS.md)) only works if changing the policy cannot change the
scenario or the environment noise.

## Decision
1. **One root seed per episode.** `numpy.random.SeedSequence(root_seed)` is spawned in a fixed
   order into child streams:
   - `scenario`: map, hazards, survivor placement;
   - `dynamics`: environment stochasticity (detection noise, hazard disablement, packet loss);
   - `policies`: further spawned into one stream per agent, in sorted agent-ID order.

   Each stream drives its own `numpy.random.Generator` (default bit generator, PCG64).
2. **Isolation property.** Swapping the policy must not change the scenario or the dynamics
   stream. This holds by construction, and a Phase 1 test checks it.
3. **Forbidden:** the global `numpy.random.*` functions, the `random` module, time-based seeds, and
   iterating over a `set` or other unordered container where the order affects the outcome (string
   hashing varies between processes). Ruff rule `NPY002` enforces the first. Code review enforces
   the rest.
4. **Integer state where exactness matters.** Positions and energy are integers (energy in
   integer units), so trajectories are exactly reproducible across platforms. Floats are used
   only for probabilities.
5. **Seed sets.** Training and evaluation use disjoint, recorded seed sets. Evaluation seed lists
   are part of the experiment config.
6. **Recorded with every run:** root seed, spawn order, NumPy version, Python version, code commit,
   and config hash ([ADR-0007](0007-configuration-and-serialization.md)).

## Alternatives considered
- **A single shared generator:** any extra policy draw would shift every later environment draw,
  so policies would face different noise on the "same" seed.
- **Separate user-supplied seeds per stream:** more flexible, but error-prone. Derivation from one root
  is simpler to record and audit.

## Consequences
- Determinism is testable: same root seed and config gives a byte-identical event stream
  ([ADR-0005](0005-event-driven-simulation.md)).
- Results remain tied to the NumPy bit generator. A NumPy upgrade that changes a stream (rare) must
  be noted in the run metadata.
- PyTorch seeding for Phase 4 will get its own recorded seed derived from the same root.

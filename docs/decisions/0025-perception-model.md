# ADR-0025: Replaceable perception model

- **Status:** Accepted
- **Date:** 2026-11-21
- **Phase:** 5

## Context
Before this ADR, sensing (ADR-0012) worked as follows:
- detection probability `p_detect` was the same at every distance;
- false positives occurred per scanned cell;
- terrain and hazard readings were always exact.

Phase 5 asks for a replaceable perception interface with range limits, uncertainty, obstacle and
hazard perception errors, and report confidence, without leaking ground truth.

## Decision
1. **Configuration.** `ScenarioConfig.perception: PerceptionConfig | None` holds:
   - `model`, a name in `swarm.core.perception.SENSOR_MODELS`; the default and only model is
     `noisy`;
   - `range_falloff`, `terrain_error`, `hazard_miss`, `hazard_false` and `prior`.

   With no block, the environment runs exactly as before, so streams and golden hashes are
   unchanged.
2. **`SensorModel` protocol.** A model provides:
   - `detect_probability(p_detect, distance, radius)`;
   - `confidence(...)`;
   - `corrupt_terrain(code, u)`;
   - `corrupt_hazard(flag, u)`.

   A new model is one class plus one registry entry.
3. **Survivor detection** uses the distance-dependent probability. The number of `dynamics` draws
   is unchanged, so common random numbers still hold across sweeps.
4. **Obstacle and hazard perception.**
   - Each in-grid footprint cell takes exactly two draws from a new fifth `dynamics` child,
     `perception`, row-major.
   - A misread terrain cell reports open ↔ debris; a structure reads as debris.
   - A hazard can be missed, or flagged where there is none.
   - An agent's own cell and the base are never corrupted.
   - Each error is logged as an evaluator-only `perception_error` event:
     - errors from reset readings go with the first step;
     - errors from a step's readings go just before `episode_end`.
5. **Confidence is not an observation field.** It is a function of public parameters only, so the
   mission brief carries the public `perception` block and policies call `report_confidence`.
   This keeps the existing leakage test unchanged: no observation field mentions "true" or
   "confidence".
6. **Integration.** Corrupted readings flow into the existing `KnownMap` and `BeliefMap`.
   - Conflicting readings are resolved by the existing last-writer-wins records, so a later
     correct reading or a teammate's record overrides an error (ADR-0018).
   - Every policy, including the MARL observation encoder, consumes the noisy views unchanged.
7. **Metrics.** The optional `comm_metrics` block gains `perception_terrain_errors`,
   `perception_hazards_missed`, `perception_hazards_false`, `perception_true_reports` and
   `perception_false_reports`.

## Consequences
- **Not image-based.** The roadmap keeps an image detector conditional on a dataset and a held-out
  baseline. That is not done.
- **Confidence is unused so far.** Existing policies do not use report confidence, and none was
  retuned for noisy perception.
- **Hazard misreads currently change no decision.** Known hazards only affect the energy reserve
  calculation (`navigation.energy`), not route choice. This is measured in docs/22.

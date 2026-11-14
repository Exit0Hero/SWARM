# ADR-0007: Configuration and serialization

- **Status:** Accepted
- **Date:** 2026-11-15
- **Phase:** 1

## Context
[docs/09](../09_API_AND_DATA_CONTRACTS.md) asks for validated, immutable scenario configs and
versioned JSON. The runtime dependency policy allows NumPy only
(see [repository conventions](README.md#repository-conventions)).

## Decision
1. Configs (`ScenarioConfig`, `AgentTypeSpec`, `SensorConfig`, later `CommConfig` and
   `ExperimentConfig`) are `@dataclass(frozen=True, slots=True)`. They use tuples instead of
   lists, so they are immutable and hashable.
2. Validation runs in `__post_init__`. Failures raise `ConfigError` with the field path and the
   offending value (for example, `agents[1].sensor_radius: must be >= 1, got 0`).
3. The on-disk format is **JSON**, with a top-level `"schema_version"` integer. Unknown keys are
   rejected, not ignored, so typos are caught. Serialization uses explicit `to_dict` and
   `from_dict` functions, never `pickle`.
4. **Config hash:** SHA-256 of the canonical JSON (`sort_keys=True`, compact separators, UTF-8).
   It is recorded in every run's metadata and identifies a scenario family independent of the
   seed.
5. Example configs are stored in a top-level `configs` directory, created in Phase 1 with the
   first real config.

## Alternatives considered
- **Pydantic:** excellent validation, but a heavy runtime dependency that isn't yet needed. Revisit if
  schemas become large or nested.
- **YAML:** friendlier to edit, but it adds a dependency and has well-known parsing pitfalls.
- **Plain dictionaries:** no validation and no immutability.

## Consequences
- Hand-written validation code needs tests for every rule, including boundary values.
- Schema changes need a version bump and a migration note.

## Amendment (Phase 1 M2)
Schema version 2 adds `hazards` (object or `null`), `p_false_positive` and `recall_target`, all required in v2
files. Version-1 files are still read and migrated with `hazards = null`, `p_false_positive = 0.0` and
`recall_target = 1.0`, which reproduces M1 behaviour exactly. The serializer always writes the current version.

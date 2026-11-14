# ADR-0009: Mission objectives and metrics

- **Status:** Accepted
- **Date:** 2026-11-15
- **Phase:** 1 (definitions), 2 (full suite)

## Context
"Mission completion rate" was a required metric, but the mission was never defined. Several metric
definitions in [docs/05](../05_EXPERIMENTS_AND_METRICS.md) deferred their denominators. Detection
precision was missing, even though the sensor model produces false positives.

## Decision

### Mission objective (Phase 1–2)
A survivor is **found** the first time any agent produces a true detection report of it.
The mission **succeeds** when `recall ≥ recall_target` (config, default 1.0) at any step before
`time_limit`.

Episode end:
- `terminated`: the mission succeeded, or every agent is disabled;
- `truncated`: `t == time_limit` without success.

A scenario with zero survivors is valid for tests. Its mission succeeds trivially, and the episode
terminates after the first step. Recall is reported as `null` and excluded from recall averages.

Later extension (needs its own ADR): survivors must be *reached* by a ground agent, or *confirmed* by a
second observation.

### Metric definitions
All metrics are computed from the event stream ([ADR-0005](0005-event-driven-simulation.md)) by the
evaluator, which is the only component allowed to read ground truth.

| Metric | Definition | Undefined case |
|---|---|---|
| `recall` | unique survivors found ÷ total survivors | `null` if 0 survivors |
| `precision` (from false-positive milestone) | true detection reports ÷ all detection reports | `null` if 0 reports |
| `mission_success` | boolean, as above | — |
| `time_to_first_detection` | step index of the first true detection | `null` if none |
| `time_to_recall_target` | step index at which success occurred | `null` if not reached |
| `coverage_sensed` | eligible cells ever inside any active agent's footprint ÷ eligible cells | — |
| `coverage_visited` | eligible cells ever occupied ÷ eligible cells | — |
| `energy_used` | total energy units spent by all agents (recharge not subtracted) | — |
| `energy_efficiency` | unique survivors found per 1000 energy units | `null` if 0 energy |
| `move_conflicts` | count of `move_conflict` events ([ADR-0003](0003-simultaneous-movement-and-collision-resolution.md)) | — |
| `invalid_actions` | count of `invalid_action` events | — |
| `agent_utilization` | active agent-steps ÷ (agents × steps run). An agent-step is active if the agent is not disabled and not idle on base at full energy | — |
| `message_delivery_rate` (Phase 3) | delivered ÷ attempted messages | `null` if 0 attempted |

**Eligible cells:** cells that are not `STRUCTURE` and are reachable from the base by at least one agent
type in the roster ([ADR-0002](0002-two-layer-world-model.md)).

### Aggregation across seeds
Report `n`, mean, sample standard deviation (`ddof=1`), and a 95% percentile bootstrap confidence
interval (10,000 resamples, with a recorded bootstrap seed). Report the count of `null` values separately.
Never fold them into averages. Comparisons must use the same evaluation seed list for every policy.

## Alternatives considered
- **Success = all survivors found:** this is the default (`recall_target = 1.0`), but it is configurable
  because some scenarios have survivors that are unreachable within the time budget.
- **Coverage over all cells:** inflated by unreachable cells and structures.

## Consequences
- Metric implementations need unit tests on hand-built event streams with known answers.
- [docs/05](../05_EXPERIMENTS_AND_METRICS.md) now refers to this ADR for definitions.

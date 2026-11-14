# ADR-0009: Mission objectives and metrics

- **Status:** Accepted, revision 2
- **Date:** 2026-11-15 (revision 1), 2026-11-15 (revision 2, Phase 1 M2)
- **Phase:** 1

## Context
"Mission completion rate" was a required metric, but the mission was never defined. Several metric definitions in
[docs/05](../05_EXPERIMENTS_AND_METRICS.md) deferred their denominators. M2 adds hazards
([ADR-0011](0011-static-hazards.md)), false positives ([ADR-0012](0012-false-positive-detections-and-precision.md)),
mission-success termination and multi-seed statistics. Each needs an exact definition.

## Decision

### Mission objective and episode end
- A survivor is **found** the first time any agent makes a *true* report of it (ADR-0012). False reports never find a
  survivor.
- **Global recall** = found survivors ÷ *all* survivors in the scenario. Unreachable survivors stay in the
  denominator. They are never excluded silently.
- `recall_target` is a config value with `0 < recall_target ≤ 1` (default 1.0). Success is tested **after sensing**
  (phase 6) of every step, using exact rational arithmetic: `Fraction(found, N) ≥ Fraction(str(recall_target))`. This
  avoids float rounding at the boundary (for example `0.3 × 10`).
- If the target cannot be reached (for example, a survivor that no agent can ever sense), the episode never succeeds.
  It ends by time limit or all-disabled and reports `mission_success = false`.
- **Zero survivors** *(changed in revision 2)*: recall is undefined, so success is undefined. There is no success
  termination; the episode ends by all-disabled or time limit and reports `mission_success = null`. Revision 1 said
  "succeeds trivially". That would let empty maps inflate success rates, so it is withdrawn.
- **End reasons and precedence** when several conditions hold on the same step:
  `mission_success` > `all_disabled` > `time_limit`.
  `mission_success` and `all_disabled` set `terminated`; `time_limit` sets `truncated`. The result records
  `end_reason`, `terminated` and `truncated`.

### Metric registry
Computed by the evaluator from the event stream plus scenario ground truth
([ADR-0005](0005-event-driven-simulation.md)). No metric uses information that policies receive. Notation: `N` total
survivors; `F` found survivors; `R` survivors on eligible cells; `T` episode steps; `n` agents.
**Eligible cells** are cells that are not `STRUCTURE` and are reachable from the base by at least one agent type in the
roster ([ADR-0002](0002-two-layer-world-model.md)).

| Metric | Formula | Unit | Undefined (`null`) when |
|---|---|---|---|
| `recall` | `|F| / N` | fraction | `N = 0` |
| `reachable_recall` (diagnostic) | `|F ∩ R| / |R|` | fraction | `|R| = 0` |
| `precision` | distinct reported cells with a survivor ÷ distinct reported cells | fraction | no reports |
| `report_precision` (diagnostic) | true reports ÷ all reports | fraction | no reports |
| `false_positive_reports` | number of false reports | count | — |
| `mission_success` | `end_reason == "mission_success"` | boolean | `N = 0` |
| `time_to_first_detection` | step of the first true report | steps | no true report |
| `time_to_recall_target` | step at which the episode succeeded | steps | not successful |
| `coverage_sensed` | eligible cells ever inside an active agent's footprint (including at reset) ÷ eligible cells | fraction | — (denominator ≥ 1: the base) |
| `coverage_visited` | eligible cells ever occupied ÷ eligible cells | fraction | — |
| `energy_used` | Σ action energy spent (`energy_spent` events); hazard damage excluded | energy units | — |
| `energy_efficiency` | `|F| × 1000 / energy_used` | survivors per 1000 energy units | `energy_used = 0` |
| `hazard_exposures` | number of `hazard_exposure` events (agent-steps ending in a hazard) | count | — |
| `hazard_energy_lost` | Σ `energy_lost` over hazard exposures | energy units | — |
| `move_conflicts` | number of `move_conflict` events ([ADR-0003](0003-simultaneous-movement-and-collision-resolution.md)) | count | — |
| `invalid_actions` | number of `invalid_action` events | count | — |
| `agent_utilization` | active agent-steps ÷ `n × T`. An agent-step is active if the agent is active at the end of the step and not idle on base (on the base, did not move, energy at capacity) | fraction | — (`T ≥ 1`) |
| `agents_disabled` (diagnostic) | distinct agents disabled during the episode | count | — |
| `agents_disabled_energy` | … of which reason `energy_depleted` | count | — |
| `agents_disabled_hazard` | … of which reason `hazard` | count | — |

An agent can be disabled at most once, so the number of disablement events always equals `agents_disabled`. No separate
event-count metric is reported. Reachable-survivor recall is a diagnostic. The primary recall metric is always global.

### Multi-seed aggregation
- An experiment runs one config and one policy over an explicit, duplicate-free list of evaluation seeds, all recorded.
- **Failed episodes** (any exception during an episode) are recorded with seed, exception type and message, and never
  dropped. Statistics are computed over completed episodes. Every summary reports `n_seeds`, `n_failed`,
  `n_completed`, `n_defined` and `n_undefined`, and the experiment is flagged `complete: false` if any episode failed.
- Per metric, over the `n_defined` non-null values (booleans as 0/1): **mean**; **sample standard deviation**
  (`ddof = 1`, `null` if `n_defined < 2`); and a **95% percentile-bootstrap confidence interval of the mean**:
  - resample the `n_defined` values with replacement `B` times (default `B = 10 000`) and take the mean of each resample;
  - the interval is the `(1 − c)/2` and `(1 + c)/2` quantiles of those means (`c = 0.95`, NumPy `linear` quantile method);
  - the generator for each metric is `default_rng(SeedSequence(bootstrap_seed, spawn_key=(crc32(metric_name),)))`,
    which is independent of episode seeds and stable when metrics are added;
  - `null` if `n_defined < 2`. Identical values give a zero-width interval.
- **Comparisons** require identical seed lists and identical config hashes, and raise an error otherwise. They use
  paired per-seed differences (B − A) over seeds where both episodes completed and the metric is defined in both, and
  report the mean difference with the same bootstrap procedure and the number of pairs. Overlapping marginal intervals
  are not a comparison.

## Alternatives considered
- **Success = all survivors found:** this is the default (`recall_target = 1.0`), but it is configurable because some
  survivors can be undetectable within the budget.
- **Excluding unreachable survivors from recall:** hides scenario difficulty. They are kept in the denominator and
  reported separately through `reachable_recall`.
- **Coverage over all cells:** inflated by unreachable cells and structures.
- **Normal-approximation or t intervals:** poor for bounded and binary metrics at small `n`. The percentile bootstrap is
  simple and needs no extra dependency. BCa intervals can be added later if needed.

## Consequences
- Metric implementations need unit tests on hand-built event streams with known answers.
- [docs/05](../05_EXPERIMENTS_AND_METRICS.md) refers to this ADR for definitions.

## Revision history
- **Revision 2 (Phase 1 M2):** zero-survivor success becomes undefined; added end-reason precedence, exact target
  comparison, global versus reachable recall, location-level and report-level precision, hazard metrics, disablement
  breakdown, the bootstrap generator derivation, failure accounting and paired comparisons.

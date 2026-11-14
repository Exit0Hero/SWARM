# ADR-0012: False-positive detections, report association and precision

- **Status:** Accepted
- **Date:** 2026-11-15
- **Phase:** 1 (M2)

## Context
[ADR-0008](0008-observation-and-hidden-state-boundaries.md) rule 4 deferred false positives to M2. Precision needs exact
definitions: what counts as a report, how reports are matched to survivors, how duplicates are counted, and what happens
when there are no reports. Policies must not be able to tell true reports from false ones.

## Decision
1. **True detections (unchanged from M1).** For each active agent and each survivor inside its footprint, one draw from
   the `dynamics` stream reports the survivor's exact cell with probability `p_detect`.
2. **False positives.** For each active agent and each in-grid footprint cell that does *not* contain a survivor, one
   draw from the `false_positive` stream ([ADR-0013](0013-environment-rng-substreams.md)) reports that cell with
   probability `p_false_positive`. Draws go in sorted agent order, then row-major cell order. Expected false reports per
   agent-step are `p_false_positive × (footprint cells in the grid − survivors in the footprint)`. A drone with radius 3
   has 29 footprint cells; a ground robot with radius 1 has 5.
3. **No localization noise.** A report always names one cell. A true report names the survivor's own cell, and a false
   report names a cell with no survivor. A cell can therefore never get both a true and a false report.
4. **Association (evaluator only).** A report is **true** if and only if its cell contains a survivor in the scenario
   ground truth, and it is then associated with that survivor. The evaluator decides this from the scenario, not from
   the event's `true_positive` flag. It also checks that the flag agrees and raises an error if not.
5. **What policies see.** `Observation.detections` is the sorted set of cells reported by *that agent* in the current
   step, with true and false reports merged and deduplicated. No flag, confidence or survivor index is exposed.
   Detection events, which carry the flags, never reach policies.
6. **Duplicates.** Reports are counted at two levels:
   - *report level:* every (step, agent, cell) report counts once. Repeated reports of the same cell by the same agent
     on later steps, or by several agents, all count.
   - *location level:* the set of distinct cells reported by any agent at any step in the episode.
7. **Precision (primary)** = distinct reported cells containing a survivor ÷ distinct reported cells. It answers: "of
   the locations the team flagged, what fraction were real?" It is `null` when nothing was reported.
   **Report precision (diagnostic)** = true reports ÷ all reports, also `null` with no reports. It is affected by
   repeated reports, so it is diagnostic only.
8. **Recall uses true reports only.** False positives never increase recall.

## Alternatives considered
- **Report-level precision as the primary metric:** dominated by how often agents loiter near one survivor.
- **False positives only on non-structure cells:** the model is a generic sensor false alarm, so every footprint cell
  is eligible.
- **Confirmation rules** (for example, "k reports before a location counts"): these are policy or coordination
  behaviour (Phase 3), not sensor behaviour.
- **Localization noise** (reporting a neighbouring cell): needs a distance-based association rule. Deferred until a
  sensor-model study needs it.

## Consequences
- With `p_false_positive = 0` and no `false_positive` draws affecting other streams, M1 detection behaviour and event
  streams are unchanged.
- Location-level precision falls as episodes get longer under persistent false-alarm rates. This is expected for
  unconfirmed reports and is stated in limitations.
- Required tests: false-positive rate statistics, false positives never on survivor cells, policies cannot distinguish
  true from false reports, duplicate counting at both levels, precision undefined when there are no reports, and the
  evaluator rejecting a mislabelled event.

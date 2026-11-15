# ADR-0021: Scheduled failures and communication partitions for controlled experiments

- **Status:** Accepted
- **Date:** 2026-11-15
- **Phase:** 3 (M3.5)

## Context
The Phase 3 study (docs/18) must vary agent failures and network partitions independently of other
factors. Before this ADR the only failures were hazard disablement and energy depletion, which depend
on the policy's route. As a result, failure rate could not be set as an experimental factor. The
channel (ADR-0017) had no way to cut links on a schedule.

## Decision
1. **Scheduled failures.** `ScenarioConfig.failures: FailureConfig | None` with `count`, `start`
   and `end`.
   - At reset, the episode's new `failure` stream draws `count` distinct agents with one
     `permutation` of the sorted agent IDs, then `count` steps uniform on `start..end`.
   - In a new step phase **4b** (after hazards, before sensing), each scheduled agent still active
     is disabled with `agent_disabled(reason="injected")`.
   - Policies are told nothing. They see only the agent's silence, exactly as with any other
     failure.
2. **RNG.** `failure` is the fourth child of the `dynamics` sequence. Existing children keep their
   spawn keys, so every earlier stream, scenario and golden hash is unchanged. The stream is drawn
   from only when `failures` is set.
3. **Partitions.** `CommsConfig.partitions` is a list of `(start, end, group)` entries.
   - During steps `start ≤ t < end`, a link between an agent in `group` and one outside it is
     treated exactly as out of range: it gets no recipient and no `comms` draws.
   - A group of one is a single-agent isolation. Links return at `end`, which is the recovery.
   - Messages already queued are still delivered (the cut is at transmission).
4. **Serialization.** `failures` and `partitions` are written only when set. Configs without them
   keep their bytes and hashes. A schema-3 config may hold `failures` with `comms: null`.
5. **Traffic metrics.** The communication metrics gain the following (overhead and delivery rate):
   - `messages_sent`, `message_units_sent` and `message_units_per_step`;
   - `messages_unaddressed`;
   - `links_addressed`, `links_delivered`, `links_dropped_random` and
     `links_dropped_receiver_disabled`;
   - `message_delivery_rate` (delivered ÷ addressed).

## Alternatives considered
- *Raise the hazard disable probability.* This confounds failure with route choice and hazard
  awareness.
- *Fixed (agent, step) lists in configs.* These are deterministic but give every seed the same
  victim, so results would depend on one agent's role.
- *Partitions as random link failures.* Random link failures are already modelled by `p_drop`. A
  partition is a structured, temporary cut.

## Consequences
- Failure count and partition schedule can be swept like any other config field.
- `agents_disabled` now includes injected failures. The injected count is `agents_disabled` minus
  `agents_disabled_energy` minus `agents_disabled_hazard`.
- Tests: `tests/test_fault_injection.py`.

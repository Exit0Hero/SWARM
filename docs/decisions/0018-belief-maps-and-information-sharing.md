# ADR-0018: Belief maps and information sharing

- **Status:** Accepted
- **Date:** 2026-11-15
- **Phase:** 3 (milestone M3.2)

## Context
With the M3.1 channel ([ADR-0017](0017-communication-channel.md)), agents can exchange messages. They now need beliefs
they can merge despite drops, delays, duplicates and reordering. Beliefs must come only from their own observations and
received messages, and a metric is needed for how fast survivor information spreads.

## Decision
1. **Belief map.** `swarm.coordination.belief.BeliefMap` subclasses the Phase 2 `KnownMap`, so the existing planners
   use it unchanged. It holds three kinds of provenance-stamped records:
   - **cell** `(row, col, terrain, hazard, step, observer)`;
   - **det** `(row, col, observer, first_step, last_step, count)`: survivor evidence;
   - **dis** `(agent, layer, row, col, step, observer)`: a disabled-agent sighting.
2. **Merge rules.** These make every merge commutative, associative and idempotent:
   - **cell:** per cell, the greatest `(step, observer, terrain, hazard)` wins. Older information never overwrites
     newer; ties resolve the same way on every agent. A disagreement in value is counted as a conflict.
   - **det:** per (cell, observer), the fields merge as `min` / `max` / `max`. Only the observer increments its own
     count, so relayed copies are never double-counted.
   - **dis:** per agent, the smallest record wins. Blocked cells are derived from the winning records only.

   Hypothesis property tests check order and grouping independence.
3. **Uncertainty.** A cell becomes known only through a cell record, and only an observation creates one. Survivor
   evidence never makes terrain known.
4. **Sharing.**
   - Each step an agent broadcasts one `"belief"` message. It holds the records changed since its last message, own or
     relayed, in key order, then a rotating anti-entropy slice of 16 records from the whole store, so information lost
     to drops is eventually resent.
   - Relaying gives multi-hop spread through stored records, with no routing.
   - A bandwidth limit caps the records sent; fresh records that do not fit wait for the next step.
5. **Policy.** `frontier_shared` is the Phase 2 frontier navigator on a `BeliefMap`, labelled `decentralized_comm`.
   - Each step, an agent merges its own inbox, decides, then composes its message.
   - There are no claims, no auction and no reassignment.
   - The mission brief adds only the public channel parameters (`MissionBrief.comms`, needed to respect the bandwidth).
6. **Survivor-information delay.**
   - **Tagging.** Records tagged `"det"` with an integer cell are the only message content the environment reads. On
     delivery, the channel logs `survivor_report_received(receiver, cell, sender)` once per (receiver, cell). This is
     evaluation only and never changes delivery.
   - **Pairs.** For each detected survivor, the origin is its earliest true detection, at step t0. Each other agent
     still active at t0 forms a pair.
   - **Delay.** A pair is informed at the earlier of the agent's own detection and its first received report; the
     delay is that step minus t0.
   - **Unresolved.** Pairs never informed are counted, and never given a zero delay.
   - **Reporting.** `comm_metrics` are recorded only for configs with a channel. The ADR-0009 registry and Phase 2 result
     layouts are unchanged.

## Alternatives considered
- **Version-vector anti-entropy:** fewer redundant records, but more state and protocol, and loss handling is
  harder to test. It can replace the sweep later without changing the merge rules.
- **Counting received cells as learned without provenance:** this cannot resolve conflicts, and loses which
  observation is newest.
- **Putting communication metrics in the ADR-0009 registry:** this would add fields to every Phase 1 and 2 result.

## Consequences
- With every message dropped, `frontier_shared` behaves exactly like `frontier`; the only difference is message
  events. This is tested.
- Phase 2 baselines run by default in `run_baselines` (`PHASE2_POLICIES`). `frontier_shared` needs a config with a
  channel.
- The sweep resends records whether or not peers need them, so bandwidth use is not minimal. Belief size grows with
  the observed area (about one record per cell, plus survivor evidence).

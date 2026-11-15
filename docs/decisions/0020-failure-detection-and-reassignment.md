# ADR-0020: Failure detection and reassignment metrics

- **Status:** Accepted
- **Date:** 2026-11-15
- **Phase:** 3 (milestone M3.4)

## Context
The auction ([ADR-0019](0019-lease-based-auction.md)) already releases claims on tombstones and expired leases. M3.4
asks for explicit failure detection from communication evidence, together with metrics that separate what agents
observe from what actually happened in the simulator.

## Decision
1. **Detector.** `swarm.coordination.failure.FailureDetector`, one per agent. It classifies each teammate from relayed
   heartbeats and tombstones only:
   - `active`: heard within the channel's normal heartbeat age, `1 + latency + jitter`;
   - `disconnected`: not heard for longer than that, but for at most `suspicion_steps`; a temporary gap;
   - `suspected`: not heard for more than `suspicion_steps` (default 10, configurable). Hearing the peer again
     recovers it;
   - `disabled`: a tombstone, from the agent's own observation or a relayed sighting. It is the only certain state, and
     it is absorbing.

   Missing messages never produce `disabled`. Transitions are a deterministic function of the step, the last-heard
   times and the tombstones. All teammates count as heard at step 0, since they start together at the base.
2. **Reassignment is unchanged** (ADR-0019):
   - A tombstoned holder is released at once, and an expired lease (15 steps) allows reassignment, so a suspected
     holder keeps a sector for at most the lease.
   - The same `suspicion_steps` removes a holder's incumbency allowance.
   - A recovered agent's renewed claims become valid again, and the claim table settles a single owner.
   - `auction_no_reassign` keeps infinite leases and no tombstone release.
3. **Logging.** Auction teams log only observable facts in the decision log: a detector marker, detector transitions
   `(step, observer, peer, state)`, and holding changes `(step, agent, sector)`. No agent reads the log.
4. **Metrics** (`comm_metrics`; evaluation only). These join the log with the true `agent_disabled` events:
   - **Pairs:** `failure_pairs` counts (failed agent, observer still active at the failure).
   - **Detection delays:** `disabled_detection_delay_mean` and `suspicion_detection_delay_mean`, each reported with an
     `_unresolved` count.
   - **Detector errors:** `false_suspicions` (suspicions of agents that were actually active) and `recoveries`.
   - **Sectors:** `abandoned_sectors` (held by an agent when it failed), `reassignment_delay_mean` (until another
     active agent holds the sector), and `unresolved_sectors` (never reassigned by the end).
   - **Duplicate ownership:** `duplicate_ownership_steps` and `duplicate_ownership_sector_steps`.
   - **Effort lost:** `failed_agent_steps_lost` and `failed_agent_energy_spent`.

   Anything unresolved is counted, never treated as a success. Detection and reassignment metrics are `None` for
   policies without a detector.

## Consequences
- `auction_comm` and `auction_no_reassign` take the same actions as before. Their event streams on development seeds
  are identical to M3.3, and the detector replaced only the ad hoc suspicion counter.
- Detection quality depends on heartbeats reaching observers. Tombstones need a sighting, so many failures are only
  suspected, never confirmed; the metrics report this rather than hiding it.

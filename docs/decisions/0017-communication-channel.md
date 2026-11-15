# ADR-0017: Communication channel in the environment

- **Status:** Accepted
- **Date:** 2026-11-15
- **Phase:** 3 (milestone M3.1)

## Context
Phase 3 replaces the perfect-communication assumption of the Phase 2 pooled planners. Message delivery depends on
true positions, needs seeded randomness, and must be auditable from the event stream for metrics (ADR-0005,
ADR-0009). Phase 1 and 2 behaviour must not change when no channel is configured.

## Decision
1. **Location.** The channel is part of the environment step, as step 5b after sensing (`swarm.core.comms`).
   Policies only compose messages; the environment decides delivery.
2. **Configuration.** `ScenarioConfig.comms: CommsConfig | None`, default `None`. A `CommsConfig` holds:
   - a range per agent type (cells; `None` = unlimited);
   - `p_drop`;
   - `latency`;
   - `jitter`;
   - `bandwidth` (units per agent per step; `None` = unlimited).

   A config with a channel is written as schema version 3. A config without one is written exactly as version 2,
   so its JSON and `config_hash` are unchanged (amends ADR-0007).
3. **Link.** A link from i to j exists when:
   - both agents are active after steps 1–4;
   - their Euclidean distance at post-move positions is at most min(range_i, range_j).

   The link is symmetric and checked at send time only. A sender disabled during the step reaches no one.
4. **Loss and delay.**
   - For every (message, receiver) pair in range, in (sender, seq, receiver) order, the channel always draws
     exactly two uniforms from the new `comms` child of the dynamics stream: `u` for the drop test (dropped when
     `u < p_drop`) and `v` for the jitter, `floor(v × (jitter + 1))`. The draw count never depends on `p_drop`,
     `jitter` or drop outcomes, so the stream stays aligned across channel settings (amends ADR-0006/0013).
   - A pair that survives is delivered in the observation returned by step `t + latency + jitter`.
   - If the receiver is disabled by then, the pair is dropped with reason `receiver_disabled`.
5. **Messages.** `Message(sender, seq, kind, records)`:
   - `records` are flat tuples of exact `int`, `str` or `bool` values, so no arrays, floats or object references.
     The message, its record container and each record must be exactly those types, never subclasses.
   - Size = 4 header units + 1 per record.
   - `seq` strictly increases per sender over the episode.
   - The environment validates every outbox before changing any state. A wrong sender, a malformed record, an
     inactive sender, a non-increasing `seq` or exceeding the bandwidth raises.
6. **Interface.**
   - `SwarmEnv.step(actions, messages=None)`.
   - `Observation.inbox: tuple[Delivery, ...] = ()`.
   - Teams implementing `CommunicatingTeam.outbox()` have their outbox passed by the runner. A communicating
     team on a config without a channel is an error.
   - The mission brief is unchanged.
7. **Events** (amends ADR-0005):
   - `message_sent(sender, seq, message_kind, size, recipients)`;
   - `message_dropped(receiver, sender, seq, reason)`;
   - `message_delivered(receiver, sender, seq, sent_step, latency)`.

   Events carry no message contents.

## Alternatives considered
- **A policy-side channel:** simpler, but delivery could not be audited from events, and the randomness would
  sit outside the environment's streams.
- **Re-checking range at delivery:** more realistic for long delays, but it couples delivery to later movement.
  It can be added as an option.
- **Bumping every config to version 3:** this would change every existing config hash, and so break comparisons
  with archived Phase 2 results.

## Consequences
- With no channel, step 5b does nothing, and no `comms` draws are made. Tests pin the M1/M2 config hashes, the
  unchanged hazard and false-positive streams, and exact reproduction of archived Phase 2 validation episodes.
- With a channel but no messages, the event stream equals the no-channel stream.
- Every (message, recipient) pair ends in exactly one `message_delivered` or `message_dropped` event, except
  pairs still in flight when the episode ends, which get none. Delivery metrics must count those as unresolved,
  not as delivered or dropped.
- Range is checked only at send time, and jitter can reorder one sender's messages. Receivers must not assume
  in-order arrival.
- Communication metrics (delivery rate, information age, survivor-information delay), typed payloads, belief
  maps, auctions and failure handling are later milestones (M3.2–M3.6). They are not part of this decision.

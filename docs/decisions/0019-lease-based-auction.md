# ADR-0019: Lease-based auction over search sectors

- **Status:** Accepted
- **Date:** 2026-11-15
- **Phase:** 3 (milestone M3.3)

## Context
The approved Phase 3 design (decisions D5–D7) calls for decentralized task allocation over the M3.1 channel
([ADR-0017](0017-communication-channel.md)) and M3.2 beliefs ([ADR-0018](0018-belief-maps-and-information-sharing.md)).
Allocation must work with message loss, delay and agent failure, and needs a control without failure handling (H5).

## Decision
1. **Tasks.** Fixed 4 × 4-cell sectors, numbered row-major. A sector is done for an agent when every cell in it is
   known in that agent's belief.
2. **Bids.** An agent's bid on a sector is its integer planning cost, on its own belief, to the nearest frontier cell
   in the sector that is reachable and energy-feasible (ADR-0016 commit rule). A sector with no such cell gets no bid.
3. **Claims.**
   - Format: `("claim", sector, holder, bid, stamp, active)`. Only the holder publishes its own claims, and claims are
     relayed like belief records.
   - Merging: per (sector, holder), the greatest `(stamp, active, bid)` is kept, which makes merging order-independent
     and idempotent. A release is a newer record with `active = False`.
   - Validity: a claim is valid while active, while `stamp + 15 ≥ now` (lease), and while its holder is not known to
     be disabled.
   - Winner: the smallest `(bid, holder)` among valid claims.
4. **Each step**, after merging its inbox:
   - **Holding a sector:** renew it (new stamp and current bid) while it is undone, feasible and still won.
     "Won" is judged from the claim table alone, including the agent's own last published claim, so all agents
     with the same table agree. Otherwise release it.
   - **Holding none:** claim the sector with the lowest own bid among those it would win. A holder heard within
     the last 10 steps keeps an incumbency allowance of `now - stamp`, so a claimant must beat its published bid
     by more than the steps since that bid was stamped. Otherwise stale bids would make two approaching agents
     take the sector from each other every step. A suspected holder gets no allowance.
   - **No such sector:** fall back to the Phase 2 frontier rule.
   - **Search scope:** the agent targets frontier cells only inside the sector it holds.
5. **Failures.** A holder seen disabled (a belief tombstone) loses its claims at once. A silent holder loses them when
   its lease expires; an unreachable agent is treated the same way. Heartbeats `("beat", agent, step)` are relayed.
   A peer unheard for more than 10 steps is *suspected*. Suspicion is local: it removes the incumbency allowance
   (rule 4) and is counted. It never releases a claim; only leases and tombstones do. This replaces the design
   draft's "lower traffic priority", for which the decentralized traffic rules have no mechanism.
6. **`auction_no_reassign`.** The same code with `honor_failures = False`:
   - leases never expire;
   - tombstones do not release claims;
   - its fallback skips sectors held by others.

   A failed agent's sector can then be taken only by a live agent that bids lower.

## Consequences
- Conflicts resolve once agents exchange claims. On M1 development seeds 1000–1004, every per-sector conflict lasted
  1 step under an ideal channel, and 1–3 steps with latency 1. Under a network partition, two agents may hold the same sector
  (duplicate effort, counted as `auction_duplicate_claim_steps`).
- The Phase 2 navigator gains one hook, `_scope()`, which returns `None` by default. Event streams for `frontier`,
  `frontier_shared` and the other Phase 2 policies are unchanged.

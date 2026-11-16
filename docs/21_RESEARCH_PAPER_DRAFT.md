# SWARM: Communication-Limited Coordination for Simulated Multi-Agent Search and Rescue

*Draft. Every number below comes from the pre-registered studies in docs/16–20 and their result files. Citations
are to well-known works and must be checked against the original sources before any submission.*

## Abstract
We present SWARM, a deterministic 2D simulator for heterogeneous teams of drones and ground robots searching
for survivors under hazards, noisy detection, energy limits, communication limits and agent failures. Using
pre-registered, seed-matched studies, we compare four approaches:
- independent frontier exploration;
- belief sharing;
- a lease-based sector auction with failure detection and reassignment;
- a learned sector allocator trained with MAPPO.

Belief sharing reduced time to the recall target in all 17 scenario conditions tested. The auction reduced it
further in 7 of 17, most strongly under poor communication and with larger teams. On a small CPU training
budget, the learned allocator was indistinguishable from belief sharing in 16 of 17 conditions and slower than
the auction at nominal. Centralized pooled assignment remained best among non-oracle methods, and its
advantage grew with map size. All results are from simulation. None is a claim about real operations.

## 1. Research problem
How much does communication-limited decentralized coordination help a heterogeneous search-and-rescue team?
How does that help degrade under channel faults and agent failures? Can multi-agent reinforcement learning
(MARL) improve on hand-designed allocation?

## 2. Motivation
Disaster sites degrade communication and destroy agents. Coordination methods are often evaluated with ideal
channels, a single scenario, or without statistical controls. We wanted a reproducible testbed in which
channel faults and failures are explicit, controllable factors, and in which claims are tested before they are
made.

## 3. Related work
- **Frontier-based exploration:** Yamauchi, 1997.
- **Market and auction task allocation:** for example, consensus-based bundle algorithms (Choi, Brunet and
  How, 2009).
- **PPO:** Schulman et al., 2017.
- **Multi-agent PPO with centralized critics (MAPPO):** Yu et al., 2022.
- **Independent PPO as a strong baseline:** de Witt et al., 2020.
- **Environment API:** PettingZoo (Terry et al., 2021).
- **Multiple-comparison control:** Holm, 1979; Benjamini and Hochberg, 1995.

A full literature review is future work.

## 4. System architecture
- **Event spine:** the environment emits typed events and computes no reward (ADR-0005). Metrics, rewards,
  logging and visualization all consume the events.
- **Packages:** the core, navigation, coordination, policies, learning (optional extra), evaluation,
  experiments and visualization are separate packages.
- **Information boundary:** policies receive only their own observation and the messages delivered to them.
  Oracle access must go through a labelled `OracleView` (ADR-0008).
- **Reproducibility:** each episode has one seed, spawned into independent random streams (ADR-0006). Golden
  event-stream hashes guard against regressions.

## 5. Simulation environment
- **World:** a two-layer grid. Drones fly over debris; ground robots cannot cross it, and structures block
  both.
- **Agents:** integer energy with recharge at the base. Agents are disabled when their energy runs out or by
  hazards.
- **Sensing:** survivors are static and detected with probability `p_detect`. False positives occur.
- **Mission:** success means reaching the recall target within the time limit.
- **Failures:** scheduled agent failures (ADR-0021).

## 6. Communication model
Messages travel between agents within range, with these impairments (ADR-0017):
- Bernoulli loss;
- latency plus uniform jitter;
- a per-agent bandwidth cap.

Scheduled partitions cut the links across a group, and agents do not know about them in advance (ADR-0021).
Agents share belief records (ADR-0018).

## 7. Coordination algorithm
The map is divided into 4×4 sectors, which agents claim through a lease auction (ADR-0019):
- **Claims and heartbeats:** agents bid with claim records and send heartbeats.
- **Lease:** 15 steps; the winner is the lowest bid.
- **Failure detection:** a teammate is marked active, disconnected, then suspected after more than 10 silent
  steps. Suspicion is reversible. Disablement is permanent and is learned from tombstone records.
- **Reassignment:** the sector of a suspected or disabled holder is released and reassigned (ADR-0020).

## 8. MARL methodology
- **Learned decision:** only the sector choice is learned (ADR-0022). Each agent picks one of 18 macro actions
  per decision epoch from a 73-value observation built from its own beliefs and received intents. Invalid
  choices are masked.
- **Reward:** a shared, event-based team reward.
- **Algorithms:** PPO with parameter sharing.
  - IPPO uses a local critic.
  - MAPPO uses a centralized critic, during training only.
- **Training:** nine models, each trained for 2 400 episodes at about 15–30 CPU-minutes, all on the nominal
  scenario.
- **Model selection:** checkpoints were chosen on validation seeds kept separate from the evaluation seeds.

## 9. Experimental methodology
- **Pre-registration:** each study was pre-registered in the repository before its evaluation seeds were used.
- **Seed sets:** disjoint sets for development, training, validation and evaluation.
- **Design:** one factor varied at a time. Every policy ran on the same 50 seeds per condition.
- **Primary metric:** time to the recall target, with failed episodes censored at the time limit + 1.
- **Tests:** comparisons are paired by seed and use a sign-flip randomization test. Results report bootstrap
  confidence intervals and Cohen's d_z.
- **Corrections:** Holm for confirmatory families, Benjamini–Hochberg for exploratory ones.
- **Integrity:** no episode was removed or rerun selectively.

## 10. Results
**Phase 3** (docs/18; 4 450 episodes)
- Sharing: 81 → 44 steps.
- Auction: a further −10 steps.
- The auction was indistinguishable from centralized pooled assignment at nominal.
- Latency and bandwidth limits hurt much more than loss or range.
- With early failures, reassignment reduced unresolved sectors by 30% but did not change search time.
- An early partition cost 7 steps.

**Phase 4** (docs/19; 3 200 episodes)
- MAPPO was 7 steps faster than sharing alone and 6 steps slower than the auction.
- IPPO and MAPPO were indistinguishable.
- The coverage reward term had no detectable effect.

**Phase 7** (docs/20; 6 850 episodes, 54 Holm-corrected tests)
- Sharing was supported in 17 of 17 conditions.
- The auction was supported in 7 of 17. It was never contradicted, but was inconclusive on larger maps and
  under early failures.
- MAPPO was indistinguishable from sharing in 16 of 17 conditions.
- On a 40 × 40 map, success was about 0.45 for decentralized methods and 0.78 for centralized assignment.
- Policy runtime grows superlinearly with team size: 38 ms per step for the auction with 8 + 8 agents.

## 11. Ablation studies
Measured contributions:
- **Communication:** frontier vs belief sharing, and the channel sweeps.
- **Belief maps:** frontier vs belief sharing.
- **Auction allocation:** belief sharing vs the auction.
- **Reassignment and lease expiry:** `auction_no_reassign`.
- **Failure handling under faults:** early-failure conditions.
- **Centralized critic:** IPPO vs MAPPO.
- **Coverage reward term:** `mappo_nocov`.

Energy-aware navigation has no variant without it, so it was not ablated (ADR-0023).

## 12. Limitations
- **One simulator:** 2D grids up to 40 × 40 and up to 8 agents in the main studies.
- **Saturated success:** at the default time limits, mission success saturates, so most conclusions concern
  speed.
- **Interactions:** one-factor designs measure only one combined channel condition.
- **MARL budget:** training was small and CPU-only, with one training scenario. Larger budgets might change
  the MARL conclusions.
- **Sensing:** synthetic detection only, with perception deferred. No validation against real systems.

## 13. Future work
- Harder time limits, to test mission success.
- Factorial designs, to measure interactions.
- Larger MARL budgets and curriculum training over scenario families.
- Learned communication content.
- The deferred perception and dashboard phases.
- 3D feasibility (Phase 8).

## 14. Conclusion
In this simulator, sharing beliefs is the most dependable improvement. A simple lease auction adds speed where
allocation matters. A small-budget learned allocator does not yet beat hand-designed coordination.
Pre-registration exposed a design flaw in our own first fault study, and the follow-up corrected it. Every
claim here is reproducible from the repository and its archived results.

## References (to verify)
- Benjamini, Y., Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS B*.
- Choi, H.-L., Brunet, L., How, J. P. (2009). Consensus-based decentralized auctions for robust task
  allocation. *IEEE Transactions on Robotics*.
- de Witt, C. S., et al. (2020). Is independent learning all you need in the StarCraft multi-agent challenge?
  arXiv.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of
  Statistics*.
- Schulman, J., et al. (2017). Proximal policy optimization algorithms. arXiv.
- Terry, J. K., et al. (2021). PettingZoo: Gym for multi-agent reinforcement learning. *NeurIPS*.
- Yamauchi, B. (1997). A frontier-based approach for autonomous robot exploration. *IEEE CIRA*.
- Yu, C., et al. (2022). The surprising effectiveness of PPO in cooperative multi-agent games. *NeurIPS
  Datasets and Benchmarks*.

# ADR-0022: MARL design: learned sector selection with CTDE

- **Status:** Accepted
- **Date:** 2026-11-15
- **Phase:** 4

## Context
Phase 4 adds learning-based coordination. It must be compared fairly with the Phase 3 classical
policies, use only information an agent has (ADR-0008), keep the core reward-free (ADR-0005) and
the simulator free of new dependencies. Compute is small: CPU training runs of about 10 minutes.

Learning primitive moves from scratch on a 20 × 20 map with a 200-step horizon would spend most of
that budget relearning A* and energy-safe return. It would also not isolate the research question.
That question is coordination: who searches where.

## Decision
1. **Macro actions (options).**
   - The learned policy chooses, for each agent, every `decision_interval = 8` steps (from step 0),
     one of 18 actions:
     - one of 16 sectors of a fixed 4 × 4 sector grid;
     - `FREE` (nearest frontier anywhere);
     - `RETURN` (go to base).
   - The Phase 2 energy-aware navigator executes the choice. It searches only frontier cells in
     the chosen sector, falls back to `FREE` when that sector is exhausted, and keeps its automatic
     energy-safe return.
   - The learned policy replaces exactly what the auction does (sector allocation). Navigation,
     belief sharing (ADR-0018) and the channel stay identical.
   - The sector count is fixed, unlike the auction's 4 × 4-cell sectors, so one network fits any
     map size.
2. **Communication.**
   - Agents share beliefs as in `frontier_shared`.
   - They broadcast one `intent` record (their latest macro action), which costs one record of
     bandwidth.
   - What to communicate is not learned. This is a stated limitation.
3. **Actor observation:** 73 values in [0, 1], all from the agent's own observation, belief and
   received records.
   - Per sector:
     - unknown fraction;
     - normalized path cost to its nearest reachable, energy-feasible frontier cell;
     - survivor evidence;
     - the share of teammates intending it.
   - Own state: energy, position, layer, time, teammates recently heard, teammates known to be
     disabled, and whether the agent is at the base.
   - **Mask:** sectors with a targetable frontier cell; `FREE` if any cell is targetable; `RETURN`
     away from the base.
   - Tested:
     - moving unseen survivors does not change the features;
     - the training adapter and the runner produce identical event streams for identical choices.
4. **CTDE.**
   - One actor is shared by all agents (parameter sharing). Heterogeneity enters through the layer
     flags.
   - **MAPPO:** the centralized critic reads `SectorParallelEnv.state()`, which is every agent's
     actor features plus alive flags. It holds no hidden ground truth, so a critic advantage
     cannot come from privileged map knowledge.
   - **IPPO:** the same learner with a critic on the agent's own observation. It is the
     diagnostic.
   - No recurrence: the belief map is the memory.
   - At a time-limit truncation the learner bootstraps the value; at a termination it does not.
5. **Reward** (`swarm.learning.rewards`).
   - The team reward is shared by all agents.
   - Components are logged separately: first true survivor detection +1, new sensed cell +0.01,
     step −0.01, agent lost to energy or hazard −1, hazard exposure −0.1, mission success +2.
   - The module docstring lists the exploit each component could invite and the metric that would
     reveal it.
   - The survivor term uses the evaluator's true-positive flag as a training signal only.
6. **Inference without PyTorch.**
   - The actor is exported to `.npz` and evaluated by `NumpyActor`, greedily (argmax over valid
     actions), through the normal runner and study tools.
   - Evaluation results are therefore deterministic and need only NumPy.
7. **Dependencies.**
   - `torch`, `pettingzoo` and `gymnasium` form the optional `learning` extra.
   - Only `swarm.learning.pz_env`, `ppo` and `train` import them.
   - CI's main jobs run without them. One CI job installs the extra (CPU wheels) and runs the full
     suite, including the PettingZoo `parallel_api_test` for both adapters.
8. **Seeds.**
   - Training scenario seeds are drawn from 10 000–59 999.
   - Checkpoint selection uses validation seeds 3000–3019.
   - Evaluation seeds are fixed in each study's pre-registration and are disjoint from both.

## Alternatives considered
- *Primitive-action MARL.* The adapter exists (`SwarmParallelEnv`, passes the API test) for future
  work. It is not trainable to a meaningful level in this compute budget.
- *Learned communication content.* It doubles the action space and is not needed to test
  allocation.
- *A critic on hidden ground truth.* This is legal under CTDE but confounds what the critic adds.
  It can be added as an ablation later.

## Consequences
- The learned policy is evaluated as a peer of `auction_comm`: same channel, navigation, seeds and
  metrics.
- Results speak to learned allocation on top of classical navigation, not to end-to-end control.

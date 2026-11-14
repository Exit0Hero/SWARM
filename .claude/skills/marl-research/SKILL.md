---
name: marl-research
description: Plan, implement or evaluate SWARM multi-agent reinforcement learning - the PettingZoo adapter, reward functions, independent PPO and MAPPO, action masking, centralized critics, training and evaluation separation, checkpoints. Use only for Phase 4 work or later, after classical baselines and metrics exist.
---

# MARL research

## Preconditions (check first and stop if unmet)
- Phases 1–3 acceptance criteria are met (`docs/14_ROADMAP.md`).
- Classical baselines have recorded results on a fixed evaluation seed list.
- The user has approved adding `torch`, `pettingzoo` and `gymnasium` as an optional extra.

## Read first
- `docs/06_MARL_RESEARCH_PLAN.md` (research questions, MAPPO design decisions)
- ADR-0004 (action masks), ADR-0005 (no reward in core), ADR-0008 (actor observations vs critic state),
  ADR-0010 (adapter contract) in `docs/decisions/`

## Procedure
1. **Adapter.** Wrap the core in a PettingZoo `ParallelEnv` in the learning package. Pass PettingZoo's
   `parallel_api_test` in a test. Rewards come from a pluggable reward function over events.
2. **Reward.** Write each reward component as a separate, logged term. Before training, write down how each term
   could be exploited (duplicate detections, clustering, idling on base) and the metric that would reveal it.
3. **Sanity checks before SWARM.** Confirm the training code learns on a tiny known task, and that a learned policy
   beats random on a tiny SWARM map. If not, debug the learner, not the environment.
4. **Independent PPO first** (diagnostic), **then MAPPO.** Document every design decision from docs/06: critic
   input, parameter sharing, masking, normalization, recurrence, termination versus truncation handling.
5. **Separation.** Training scripts, evaluation scripts and inference policies are separate modules. Training and
   evaluation seed sets are disjoint and recorded. Checkpoint selection uses a validation seed set, never the
   evaluation set.
6. **Evaluation.** Learned policies run through the same runner, seeds and metrics as the baselines
   (experiment-analysis skill). Report multiple training seeds (at least 3, more if variance is high).

## Rules
- Never write "MAPPO works" or "MAPPO is implemented" without a passing test suite and real evaluation output.
- Reward curves are not results. Mission metrics on evaluation seeds are.
- A centralized critic may read global state during training only. The actor gets only the agent's observation.

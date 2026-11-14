# Experiments and Metrics

Exact metric definitions, undefined-value handling, the mission objective and the aggregation method are in
[ADR-0009](decisions/0009-mission-objectives-and-metrics.md). This document summarizes them and sets the protocol.

## Required metrics
- **Survivor recall:** unique survivors found ÷ total survivors.
- **Detection precision:** true detection reports ÷ all detection reports (once false positives exist).
- **Mission success rate:** fraction of episodes meeting the recall target before the time limit.
- **Time to first detection** and **time to recall target**, in simulation steps.
- **Coverage:** sensed and visited, over *eligible* cells (not `STRUCTURE` and reachable from base by some agent type).
- **Energy used** and **energy efficiency** (survivors found per 1000 energy units).
- **Move conflicts:** blocked simultaneous moves. This replaces "collision count": physical collisions are impossible by
  construction ([ADR-0003](decisions/0003-simultaneous-movement-and-collision-resolution.md)).
- **Invalid actions:** masked actions submitted by a policy.
- **Agent utilization:** active agent-steps ÷ available agent-steps.
- **Message delivery rate:** delivered ÷ attempted messages (Phase 3).

### Implemented in Phase 1 M1
`recall`, `time_to_first_detection`, `coverage_sensed`, `coverage_visited`, `energy_used`, `move_conflicts`,
`invalid_actions`, and the auxiliary count `agents_disabled`. Each result file embeds the definition, unit and
undefined case of every metric (`src/swarm/evaluation/metrics.py`). Precision, mission success, time to recall
target, energy efficiency, agent utilization and multi-seed aggregation are planned for M2.

## Experiment protocol
- Compare policies on identical evaluation seed lists and config hashes.
- Use multiple seeds and report n, mean, sample standard deviation and a 95% bootstrap confidence interval. Report
  undefined (`null`) values separately.
- Use paired, per-seed differences when comparing two policies.
- Keep training, validation and evaluation seeds disjoint.
- Record config, config hash, code revision, dependency versions, policy name and version, checkpoint and timestamp
  ([docs/11](11_REPRODUCIBILITY_AND_EXPERIMENT_LOGGING.md)).
- Do not compare methods across different scenario difficulties without explaining the protocol.

## Initial baselines
1. Random valid-action policy (sanity baseline, Phase 1).
2. Greedy nearest-unsearched-area policy (Phase 2).
3. Centralized assignment. Oracle information is clearly labelled where used (Phase 2).
4. Communication-limited decentralized heuristic (Phase 3).

## Reporting
Every chart and table states the metric definition, sample count, seeds, whether results are from training or
evaluation, and whether any policy used oracle information.

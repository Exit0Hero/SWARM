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

### Implemented (Phase 1 M2)
All metrics in the ADR-0009 registry: `recall`, `reachable_recall`, `precision`, `report_precision`,
`false_positive_reports`, `mission_success`, `time_to_first_detection`, `time_to_recall_target`,
`coverage_sensed`, `coverage_visited`, `energy_used`, `energy_efficiency`, `hazard_exposures`,
`hazard_energy_lost`, `move_conflicts`, `invalid_actions`, `agent_utilization`, and the diagnostics
`agents_disabled`, `agents_disabled_energy`, `agents_disabled_hazard`. Message delivery rate arrives with
communication in Phase 3. Every episode and experiment result embeds each metric's definition, unit, undefined
case and role (`src/swarm/evaluation/metrics.py`).

### Running and comparing experiments
```bash
python -m swarm.experiments.run_experiment --config configs/m2_default.json \
    --policy random_valid --seeds 0-29 --out runs/random.json
python -m swarm.experiments.compare --baseline runs/a.json --candidate runs/b.json --out runs/a_vs_b.json
```
The experiment runner records every seed. Failed episodes are listed with their error, counted in every summary,
and flagged with `complete: false` and exit code 3. The comparison CLI refuses experiments whose config hash,
seed list or designation differ.

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

# Experiments and Metrics

## Required metrics
- Survivor recall: unique survivors detected / total survivors.
- Time to first detection: elapsed simulation steps until the first valid detection.
- Mission completion rate: fraction of scenarios meeting the defined mission objective.
- Coverage: proportion of eligible map area visited or scanned, with the exact denominator documented.
- Energy efficiency: detections or coverage per energy unit; define zero-denominator behavior.
- Collision count: number of collision events under the simulator's collision definition.
- Communication delivery rate: delivered messages / attempted messages.
- Agent utilization: active agent-steps / available agent-steps.

## Experiment protocol
- Compare policies on identical scenario seeds.
- Use multiple seeds and report mean, standard deviation, and sample count.
- Keep training and evaluation seeds separate.
- Record config, code revision, dependency versions, policy checkpoint, and timestamp.
- Report confidence intervals when appropriate.
- Do not compare methods using different scenario difficulty without explaining the protocol.

## Initial baselines
1. Random valid-action policy (sanity baseline).
2. Greedy nearest-unsearched-area policy.
3. Centralized nearest-survivor/task assignment (oracle information clearly labelled if used).
4. Communication-limited decentralized heuristic.

## Reporting
Every chart/table must state metric definition, sample count, seeds, and whether results are training or evaluation results.

# MARL Research Plan

## Research questions
- How does communication availability affect cooperative search performance?
- When does decentralized assignment outperform a centralized planner under partial observability?
- How robust are learned policies to agent dropout and dynamic hazards?
- What is the trade-off between mission success, time, and energy?

## Algorithm progression
1. Heuristic and centralized planning baselines.
2. Independent learners only as a diagnostic comparison.
3. Value-based or policy-gradient multi-agent baseline suited to the action space.
4. MAPPO after environment/API and metrics are stable.

## MAPPO design decisions to document
- Centralized training / decentralized execution assumptions.
- Actor observation and critic state.
- Parameter sharing or per-agent networks.
- Action masking for invalid actions.
- Reward decomposition and normalization.
- Episode termination/truncation behavior.
- Recurrent policy need under partial observability.
- Training seeds, evaluation protocol, and checkpoint selection.

## Reward design caution
Avoid rewarding only raw detections if that creates duplicate scanning, unsafe clustering, or exploitation. Track reward components separately and evaluate against task-level metrics. Reward is not a substitute for mission success metrics.

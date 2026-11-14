---
name: perception
description: Implement or evaluate SWARM perception - the synthetic sensor model (detection probability, false positives, range, occlusion), the replaceable perception interface, and any later image-based survivor detector. Use for sensing changes, noise-model experiments, or proposals to add computer vision.
---

# Perception

## Read first
- `docs/08_PERCEPTION_AND_COMMUNICATION.md`
- ADR-0008 (observation contents, detection model, hidden-state boundary) and ADR-0009 (precision and recall
  definitions) in `docs/decisions/`

## Rules
- Perception is a replaceable component with one interface: given the agent's footprint and the true world state,
  return detection reports (cell, optional confidence). Only the environment calls it. Policies see only its output.
- Every detection report records, for evaluation only, whether it was true or false. That flag never reaches a policy.
- All stochastic draws use the `dynamics` RNG stream, in sorted (agent ID, cell) order (ADR-0006).
- Detections are uncertain observations, never ground truth (CLAUDE.md safety rules).

## Procedure
1. **Synthetic model first.** Parameters: `p_detect`, false-positive rate per footprint cell, `sensor_radius`,
   optional occlusion. Test each against its expectation with a statistical test on many fixed-seed draws, using
   bounds wide enough to be non-flaky.
2. **Sensor benchmark.** Sweep `p_detect` and false-positive rate on fixed seeds with a fixed baseline policy. Report
   recall, precision and time to recall target, with confidence intervals (experiment-analysis skill).
3. **Image-based perception is gated.** Propose it only with: what images are rendered from, how labels are produced,
   the train/validation/test split by scenario seed, a baseline to beat (the synthetic model with matched
   precision and recall), and the dependency cost. Get user approval before adding OpenCV or a model.
4. If a learned detector is adopted, report its precision and recall on held-out scenario seeds, then measure the mission-level
   effect through the same runner.

## Rules for claims
Do not describe synthetic detections as "AI perception" or "computer vision". Name the model that produced them.

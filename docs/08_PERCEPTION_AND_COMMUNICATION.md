# Perception and Communication

## Perception stages
MVP: synthetic sensor model with range limits, detection probability, false-positive probability, and optional occlusion.
Later: image-based perception module behind the same observation interface.

## Communication model
Support:
- range-limited connectivity;
- packet drop probability;
- optional latency;
- message size/type constraints;
- stale information and timestamping.

## Information boundaries
A decentralized policy must not access global survivor coordinates or hidden map state unless the experiment explicitly defines an oracle baseline. Label oracle access clearly in results.

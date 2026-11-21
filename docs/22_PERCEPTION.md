# Perception (Phase 5)

The design is in [ADR-0025](decisions/0025-perception-model.md).

## Using it
Add a block to any scenario config (schema 3):
```json
"perception": {"model": "noisy", "range_falloff": 0.5, "terrain_error": 0.1,
               "hazard_miss": 0.3, "hazard_false": 0.05, "prior": 0.02}
```

| Field | Effect |
|---|---|
| `range_falloff` | Detection probability is `p_detect × (1 − falloff × d / (r + 1))`. |
| `terrain_error` | Per-cell probability of misreading terrain (open ↔ debris, structure → debris). |
| `hazard_miss` / `hazard_false` | Per-cell probability of missing a hazard, or flagging a safe cell. |
| `prior` | Survivor prior used by `report_confidence`. |

- **Corrupted views:** agents receive the corrupted footprint views. The environment still enforces
  the true terrain, so a planned move into a real obstacle becomes an invalid action.
- **Confidence:** policies compute report confidence with
  `swarm.core.perception.report_confidence(make_sensor(...), p_detect, p_false, own_cell, cell,
  radius)`, using the public parameters in `MissionBrief.perception`.
- **New sensor model:** implement `SensorModel` and add it to `SENSOR_MODELS`.

## Sensor benchmark (exploratory; development seeds 1000–1019; not a validated result)
Spec: `configs/phase5/sensor_benchmark.json`. Files: `docs/results/phase5/` (`runs.tar.gz`,
`analysis.json`, `results.md`).

Method:
- one factor at a time from `configs/phase3/nominal.json`;
- paired by seed;
- Benjamini–Hochberg adjusted;
- n = 20.

The effect on `time_censored` (steps) against nominal: nominal is 46.5 for `frontier_shared` and
34.4 for `auction_comm`.

| Condition | `frontier_shared` | `auction_comm` |
|---|---|---|
| `p_detect` 0.5 (nominal 0.9) | +4.0 [+2.1, +5.9] | +2.4 [+0.9, +4.0] |
| `p_detect` 0.95 | −0.5 [−1.1, −0.1] | −0.1 [−0.4, 0.0] |
| false positives 0 (nominal 0.002) | +0.3 [−0.9, +1.9] | +0.7 [0.0, +2.1] |
| false positives 0.02 | +3.9 [−1.0, +12.2] | +0.4 [−0.1, +1.1] |
| range falloff 0.5 | +2.1 [+0.8, +3.7] | +1.9 [+0.7, +3.5] |
| terrain misread 10% | +11.8 [+3.9, +21.1] | +11.2 [+5.5, +17.2] |
| hazard miss 30% and false 5% | 0.0 (identical episodes) | 0.0 |
| all noise combined | +14.2 [+5.5, +23.9] | +17.9 [+8.8, +29.4] |

Reading it:
- **Terrain misreads hurt most.** A 10% misread rate (about 266 wrong cells per episode) slows both
  policies by about 11 steps.
- **Precision falls as false positives rise:** 0.65 at nominal and 0.16 at 2% for `auction_comm`.
  Search time barely changes, because the policies do not chase reports.
- **Hazard misreads change nothing**, even though about 100 false alarms and 40 misses per episode
  were logged. The current planners use known hazards only for the energy reserve, which never
  bound in these runs.
- **Mission success stayed at 1.0 throughout.** n = 20 development seeds: treat every number as
  exploratory.

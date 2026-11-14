# Reproducibility and Experiment Logging

Each run should record:
- run ID;
- scenario/configuration hash;
- random seed(s);
- policy name and version;
- code commit;
- Python and dependency versions;
- training/evaluation designation;
- start time and duration;
- metric values and metric schema version;
- artifact/checkpoint references.

## Episode result format (`result_schema_version` 1)
`python -m swarm.experiments.run_episode` writes one JSON file per episode, and optionally the event stream
as JSON Lines:

| Field | Content |
|---|---|
| `config`, `config_hash` | Full scenario config and its SHA-256 (ADR-0007) |
| `seed`, `metadata.seed_derivation` | Root seed, spawn order, per-agent policy stream order, bit generator |
| `policy` | `name`, `version`, `uses_oracle` |
| `designation` | `evaluation` or `development` (training arrives in Phase 4) |
| `episode` | Steps, end reason, terminated/truncated, event count |
| `scenario_sha256`, `event_stream_sha256` | Ground-truth and event-stream fingerprints |
| `metrics`, `metric_definitions` | Values, plus the definition, unit and undefined case of each |
| `metadata.schema_versions` | Config, event, metric and result schema versions |
| `metadata.software`, `metadata.code` | swarm, Python and NumPy versions; git commit and dirty flag |
| `metadata.run` | Run ID, start time, duration, platform. These are the only fields that differ between reruns |

**Reproducibility check:** the same config, seed, policy and commit must give the same
`event_stream_sha256` and identical metrics. `tests/test_regression.py` pins golden hashes for three seeds
of `configs/m1_default.json` (unchanged since M1) and for three seeds of `configs/m2_default.json`, and CI checks them on
Python 3.12–3.14. Environment randomness is split into `dynamics` (true detections), `hazard` and
`false_positive` sub-streams (ADR-0013).

## Experiment result format (`experiment_schema_version` 1, Phase 1 M2)
`python -m swarm.experiments.run_experiment` writes one JSON file per experiment:

| Field | Content |
|---|---|
| `config`, `config_hash` | Full config and its SHA-256. The hash is verified again when the file is loaded |
| `policy`, `designation` | Policy name, version and oracle flag; `evaluation` or `development` |
| `seeds` | The exact evaluation seed list, in run order |
| `episodes` | One record per seed: status, end reason, steps, scenario and event-stream hashes, metrics; or the error (type, message, traceback) for a failed episode |
| `complete`, `failed_seeds`, `end_reasons` | Whether every seed completed; which failed; counts per end reason |
| `summary` | Per metric: `n_seeds`, `n_failed`, `n_completed`, `n_defined`, `n_undefined`, mean, std, CI |
| `bootstrap` | Method, resamples, confidence, seed, quantile method, generator derivation |
| `metric_definitions` | Definition, unit, undefined case and role of each metric |
| `metadata` | Schema versions, software versions, git commit and dirty flag, run ID and timing |

`python -m swarm.experiments.compare` writes paired per-seed differences (candidate − baseline) with bootstrap
intervals. It refuses inputs whose config hash, seed list or designation differ.

## Random-number generators
Use independent random-number generators for scenario generation, environment stochasticity, and policy stochasticity where possible. Record seed derivation. Avoid relying on process-global random state.

# Reproducibility guide

## What makes an episode reproducible
- **Seeded randomness.** One root seed per episode is spawned into independent streams: scenario,
  dynamics (hazard, false positive, comms, failure, perception) and one stream per policy
  (ADR-0006, ADR-0013). There is no global random state.
- **Configs are frozen, validated and hashed.** The SHA-256 of the canonical JSON is the config
  hash (ADR-0007). An optional block (`comms`, `failures`, `perception`) is written only when set,
  so older configs keep their hash.
- **Every result records:**
  - the config and its hash, the seed and the seed derivation;
  - the policy identity and information model;
  - the git commit and dirty flag;
  - Python, NumPy and OS versions;
  - metric definitions;
  - the event-stream SHA-256.
- **Golden hashes.** `tests/test_regression.py` pins event-stream hashes of reference episodes. CI
  fails if behaviour changes. No release so far has changed them.

## Reproduce one episode
```bash
swarm run --config basic --policy auction_comm --seed 1000 --out a.json
swarm run --config basic --policy auction_comm --seed 1000 --out b.json
python -c "import json;a,b=(json.load(open(f)) for f in ('a.json','b.json'));print(a['event_stream_sha256']==b['event_stream_sha256'])"
```
The smoke test (`scripts/smoke_test.py`) performs exactly this check on every CI platform.

## Reproduce a published study
Archived studies are kept and never regenerated in place:

| Study | Spec | Archive |
|---|---|---|
| Phase 2 baseline and validation | `python -m swarm.experiments.run_baselines` (see docs/16, docs/17) | `docs/results/phase2_validation/` |
| Phase 3 coordination (A, B) | `configs/phase3/study.json`, `study_b.json` | `docs/results/phase3/` |
| Phase 4 MARL | `configs/phase4/study.json` | `docs/results/phase4/` |
| Phase 7 final validation | `configs/phase7/study.json` | `docs/results/phase7/` |
| Phase 5 sensor benchmark (exploratory) | `configs/phase5/sensor_benchmark.json` | `docs/results/phase5/` |

**Re-run only the analysis** (fast, from the archived episodes):
```bash
cd docs/results/phase3 && tar xzf runs.tar.gz && cd -
swarm analyze --spec configs/phase3/study.json --runs docs/results/phase3/runs --out /tmp/analysis.json
```

**Regenerate every episode:**
```bash
swarm study --spec configs/phase3/study.json --out-dir /tmp/p3 --jobs 2
```
The study runner refuses mismatched config hashes or seeds, and the analysis refuses to pair
cells whose scenarios differ.

Every episode is deterministic, so a regenerated run reproduces the archived event-stream hashes,
given the same code commit and NumPy version. Each result file records both. Wall-clock runtimes
are never reproducible and are labelled as such.

## Rules for new experiments
- **Pre-register first.** Commit the question, configs, seeds, metrics, tests and decision rules
  before running on evaluation seeds (see docs/18 and the `experiment-analysis` skill).
- **Seed sets** used so far:
  - evaluation: 200–249, 300–349, 500–549;
  - Phase 2: 0–49 and 100–149;
  - development: 1000–1019.

  New evaluation studies need new seed ranges.
- **Label everything else exploratory:** dashboard comparisons, `swarm compare`, `swarm demo`
  output and benchmarks.

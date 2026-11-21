# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow
[Semantic Versioning](https://semver.org/). Before 0.1.0 the project had no releases or tags. Its
history is recorded per phase in `docs/14_ROADMAP.md` and in merged PRs #1–#21.

## [0.1.0] — first release

The first tagged release of the research platform built in Phases 0–8.

### Stable (tested; validated results where noted)
- **Deterministic 2D simulator:**
  - two-layer grid, heterogeneous drones and ground robots;
  - simultaneous moves, energy and recharge, hazards, noisy detection;
  - typed events, seeded RNG streams, golden-hash regression tests.
- **Classical baselines:** A*, prioritized planning, frontier exploration, centralized
  assignment, and an oracle reference (validated: docs/16–17).
- **Communication-limited coordination:**
  - a channel with range, loss, latency, jitter, bandwidth and partitions;
  - shared belief maps; a lease-based sector auction;
  - failure detection and reassignment; scheduled failures.

  Validated: docs/18, docs/20.
- **MARL sector selection:** IPPO and MAPPO, nine trained models, NumPy inference (validated at a
  small CPU budget: docs/19).
- **Experiment tooling:** a pre-registered study runner and analysis (paired sign-flip tests,
  Holm and Benjamini–Hochberg correction, bootstrap intervals), archived results for Phases 2–7,
  and a research paper draft (docs/21).

### Added in this release
- **`swarm` command** (`swarm demo`, `run`, `validate`, `policies`, `compare`, `dashboard`,
  `sim3d`, `study`, `analyze`, `benchmark`):
  - consistent exit codes and one-line errors;
  - the older `python -m` commands still work.
- **Packaged example scenarios:**
  - `basic` has the same hash as the validated `configs/phase3/nominal.json`;
  - `advanced` adds loss, delay, failures, hazards and noisy perception.

  Both work from a wheel install.
- **`swarm demo`:** an end-to-end run of both examples with exports, a 3D scene and a comparison,
  filling a dashboard data directory.
- **`swarm benchmark`** and exploratory benchmark results (docs/28).
- **Perception model** (ADR-0025), **web dashboard** (ADR-0026) and **3D kinematic co-simulation**
  (ADR-0027), merged in #18–#20 and released here for the first time.
- **Dashboard hardening:**
  - a request size limit, a localhost Host check (DNS rebinding) and JSON-only bodies;
  - security headers and a warning when bound to a non-local address;
  - always-listed packaged examples, empty and loading states, and a responsive layout.
- **CI:**
  - a MuJoCo job;
  - wheel build plus a clean-install end-to-end smoke test on Linux (3.12, 3.14 with sim3d), macOS
    and Windows;
  - a Markdown link check.
- **Documentation:**
  - installation and CLI guide (docs/26), reproducibility guide (docs/27), benchmarks (docs/28);
  - `CONTRIBUTING.md`, `SECURITY.md`;
  - issue and PR templates;
  - README quick start with real screenshots.

### Experimental and partial
- **3D:** kinematic co-simulation only. Movement comes from the 2D simulator, altitude is a
  convention, and there are **no flight dynamics**.
- **Exploratory, not validated:** the sensor benchmark, the 3D line-of-sight figures, benchmarks,
  and dashboard and `swarm compare` results.
- **Report confidence** is available, but no policy uses it.

### Unchanged
- Golden hashes.
- Archived results for Phases 2, 3, 4 and 7.
- Validated configs.
- Simulation behaviour.

# Architecture Decision Records

An Architecture Decision Record (ADR) captures one significant decision: the context, the
decision itself, the alternatives considered, and the consequences. ADRs are the source of
truth when a numbered document in `docs/` and an ADR disagree; fix the document.

## Index

| ADR | Title | Status |
|---|---|---|
| [0001](0001-grid-and-discrete-time.md) | 2D grid and discrete time | Accepted |
| [0002](0002-two-layer-world-model.md) | Two-layer world model | Accepted |
| [0003](0003-simultaneous-movement-and-collision-resolution.md) | Simultaneous movement and collision resolution | Accepted |
| [0004](0004-action-masks-and-invalid-actions.md) | Action masks and invalid actions | Accepted |
| [0005](0005-event-driven-simulation.md) | Event-driven simulation architecture | Accepted, amended by 0017 |
| [0006](0006-random-number-generation-and-reproducibility.md) | Random number generation and reproducibility | Accepted, amended by 0013 and 0017 |
| [0007](0007-configuration-and-serialization.md) | Configuration and serialization | Accepted, amended by 0017 |
| [0008](0008-observation-and-hidden-state-boundaries.md) | Observation and hidden-state boundaries | Accepted |
| [0009](0009-mission-objectives-and-metrics.md) | Mission objectives and metrics | Accepted, revision 2 |
| [0010](0010-api-contract.md) | API contract | Accepted |
| [0011](0011-static-hazards.md) | Static hazards | Accepted |
| [0012](0012-false-positive-detections-and-precision.md) | False-positive detections, report association and precision | Accepted |
| [0013](0013-environment-rng-substreams.md) | Environment RNG sub-streams per stochastic process | Accepted |
| [0014](0014-team-policies-and-information-models.md) | Team policies, mission brief, oracle view and information labels | Accepted |
| [0015](0015-path-planning-and-reservation-tables.md) | Path planning and prioritized planning with reservation tables | Accepted |
| [0016](0016-energy-feasibility-and-classical-baselines.md) | Energy feasibility and the classical baseline policies | Accepted |
| [0017](0017-communication-channel.md) | Communication channel in the environment | Accepted |
| [0018](0018-belief-maps-and-information-sharing.md) | Belief maps and information sharing | Accepted |
| [0019](0019-lease-based-auction.md) | Lease-based auction over search sectors | Accepted |
| [0020](0020-failure-detection-and-reassignment.md) | Failure detection and reassignment metrics | Accepted |
| [0021](0021-fault-injection-for-experiments.md) | Scheduled failures and communication partitions | Accepted |
| [0022](0022-marl-sector-selection.md) | MARL design: learned sector selection with CTDE | Accepted |
| [0023](0023-final-validation-scenario-factors.md) | Final-validation scenario factors: clustered survivors, scalability benchmark | Accepted |
| [0024](0024-3d-feasibility-decision.md) | 3D feasibility study: defer full 3D (MuJoCo prototype) | Accepted |
| [0025](0025-perception-model.md) | Replaceable perception model | Accepted |
| [0026](0026-dashboard.md) | Dashboard: standard-library server over the real runner | Accepted |

ADR-0001 to ADR-0010 were accepted on 2026-11-15 from the Phase 0 repository audit; ADR-0011 to ADR-0013 and
ADR-0009 revision 2 with Phase 1 M2; ADR-0014 to ADR-0016 with Phase 2; ADR-0017 with Phase 3 M3.1 (it amends ADR-0005, 0006 and 0007); ADR-0018 with M3.2; ADR-0019 with M3.3; ADR-0020 with M3.4; ADR-0021 with M3.5; ADR-0022 with Phase 4; ADR-0023 with Phase 7. Parameter values
marked *default* are starting points that Phase 1 may tune without a new ADR. Changing a
rule (not a default) requires a new ADR that supersedes the old one.

## Repository conventions

These conventions came out of the same audit. They are recorded here instead of in a separate ADR:

- **Runtime dependencies:** NumPy only. A new runtime dependency needs a stated reason
  in the pull request and, if it is heavy (PyTorch, PettingZoo, OpenCV, web stack), an
  optional extra so the core simulator still installs with NumPy alone.
- **No empty packages:** a `swarm.*` subpackage is created only when it contains working,
  tested code, so the source tree never implies features that do not exist.
- **Python:** `>=3.12`, tested in CI on 3.12, 3.13 and 3.14.

## Writing a new ADR

Copy [the template](0000-template.md), take the next number, keep it to one decision, and add
it to the index above. `tests/test_repository_docs.py` fails if an ADR file is not indexed.

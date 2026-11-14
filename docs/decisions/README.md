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
| [0005](0005-event-driven-simulation.md) | Event-driven simulation architecture | Accepted |
| [0006](0006-random-number-generation-and-reproducibility.md) | Random number generation and reproducibility | Accepted |
| [0007](0007-configuration-and-serialization.md) | Configuration and serialization | Accepted |
| [0008](0008-observation-and-hidden-state-boundaries.md) | Observation and hidden-state boundaries | Accepted |
| [0009](0009-mission-objectives-and-metrics.md) | Mission objectives and metrics | Accepted |
| [0010](0010-api-contract.md) | API contract | Accepted |

All ten were accepted on 2026-11-15 from the Phase 0 repository audit. Parameter values
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

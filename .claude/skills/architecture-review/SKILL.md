---
name: architecture-review
description: Review a proposed or implemented SWARM change against the package boundaries, dependency policy and Architecture Decision Records. Use before starting any new module or phase, before adding a dependency, when a change crosses package boundaries, or when a design question is not answered by an existing ADR.
---

# Architecture review

## Sources of truth
- Layering and package boundaries: `docs/02_SYSTEM_ARCHITECTURE.md`
- Decisions: `docs/decisions/README.md` (ADR index and repository conventions)
- Phase scope and acceptance criteria: `docs/14_ROADMAP.md`
- Risks: `docs/15_RISK_REGISTER_AND_LIMITATIONS.md`

An ADR overrides a numbered document. If they disagree, flag it and fix the document.

## Procedure
1. **Locate the phase.** Confirm the change belongs to the current roadmap phase. Work from a later
   phase (MARL, perception models, dashboards, 3D) needs explicit user approval.
2. **Map to packages.** Name the `swarm.*` subpackage(s) touched. Check against the direction of
   dependencies in docs/02. The core never imports policies, learning, evaluation or visualization.
   Policies never import the environment's internals.
3. **Check every ADR the change touches.** For each one, state "complies" or "deviates". A deviation
   needs a new ADR (copy `docs/decisions/0000-template.md`) before the code is merged.
4. **Dependency check.** Runtime dependencies stay NumPy only unless the phase requires more. For a
   new dependency, state: why, which phase, a lighter alternative and why it was rejected, and whether it
   belongs in an optional extra. Verify the dependency supports Python 3.12–3.14.
5. **Hidden-state check.** Confirm no policy-facing path can reach ground truth (ADR-0008). Oracle access
   must go through an explicit `OracleView` and be labelled in results.
6. **Determinism check.** New randomness must use a spawned stream (ADR-0006). No set-iteration-order
   dependence and no global RNG.
7. **No empty scaffolding.** Do not create packages, classes or CLI commands that have no working,
   tested behaviour.

## Output
Write a short review: scope, affected packages, ADR compliance table, dependency verdict, risks, and
either "approve" or a list of required changes. Do not approve your own deviation from an ADR. Ask the
user.

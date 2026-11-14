---
name: testing-and-review
description: Write tests for SWARM code and review changes before they are committed - unit, integration, property-based (Hypothesis), determinism, regression and API tests, plus the pre-commit quality gate (ruff, mypy, pytest) and a code-review checklist. Use for every code change and every pull request review.
---

# Testing and review

## Read first
- `docs/10_TESTING_AND_QUALITY.md` (test layers, edge cases)
- The ADR that governs the code under test (`docs/decisions/README.md`)

## Quality gate (run all of these, from the repository root, and report the real output)
```bash
ruff check .
ruff format --check .
mypy
pytest
```
CI (`.github/workflows/ci.yml`) runs the same commands on Python 3.12, 3.13 and 3.14. If a command was not run,
say so. Never summarize as "tests pass" without having run them in this change.

## Writing tests
1. **Unit tests on tiny hand-built worlds** with the exact expected state and events. Prefer 3×3 to 5×5 grids
   with literal layouts.
2. **Property tests (Hypothesis)** for invariants: in bounds, terrain respected per layer, no shared non-base cell
   per layer, energy ≥ 0, permutation invariance of joint actions, policy swap does not change the scenario.
   Use `@settings(derandomize=True)` or fixed seeds in CI to avoid flaky runs.
3. **Determinism tests:** run twice with the same seed and compare the event-stream hash. A different seed must give a
   different scenario.
4. **Edge cases from docs/10:** zero survivors, fully blocked map, no communication peers, depleted battery, time
   limit, all agents disabled, false detections, simultaneous conflicts.
5. **Leakage test (ADR-0008):** moving survivors outside every footprint must not change any observation.
6. **Regression tests:** every fixed bug gets a test that failed before the fix.
7. **Contract tests:** a bad agent ID, an out-of-space action, or stepping after the end must raise.

## Review checklist
- [ ] Matches an ADR, or a new ADR is included
- [ ] No hidden-state path to policies; oracle access labelled
- [ ] Randomness from spawned streams only; no ordering on sets
- [ ] Public functions typed and documented; errors have actionable messages
- [ ] Tests cover failure modes, not only the happy path
- [ ] No placeholder code presented as complete; no fabricated numbers in docs or commits
- [ ] Docs updated when behaviour, config or interfaces changed (documentation skill)
- [ ] All four quality-gate commands were run, and their real output is reported

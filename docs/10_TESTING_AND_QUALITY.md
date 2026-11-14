# Testing and Quality Strategy

## Test layers
- Unit: movement, collision, energy, detection, communication, metrics.
- Integration: complete episode, reset reproducibility, policy-environment interaction.
- Property-style: positions remain in bounds; blocked cells remain impassable; energy never becomes negative unless explicitly modelled.
- Regression: fixed-seed scenarios and previously corrected bugs.
- API: environment interface compliance where applicable.

## Quality gates
Run from the repository root. CI (`.github/workflows/ci.yml`) runs the same commands on Python 3.12, 3.13 and 3.14.

```bash
ruff check .          # linting, including NPY rules against global NumPy RNG
ruff format --check . # formatting
mypy                  # strict static typing of src/ and tests/
pytest                # tests, with warnings treated as errors
```

- Formatting and linting.
- Static typing for public interfaces.
- Test suite passes.
- Documentation integrity: `tests/test_repository_docs.py` checks internal links, repository paths, skill
  frontmatter and the ADR index.
- No unexplained nondeterminism in deterministic mode.
- Documentation updated for changed contracts.

## Property-test determinism
The Hypothesis profile (`tests/conftest.py`) uses `derandomize=True`, so a given commit always runs the same
examples in the full suite. Hypothesis is pinned exactly in `pyproject.toml` (`hypothesis==6.170.15`), because
example generation changes between Hypothesis releases. Upgrade it deliberately, in its own commit, and expect
different examples. Hypothesis (6.170 here) also mines literal constants from loaded project modules into
generation, so running a *subset* of test files can produce different examples than the full suite. CI runs the
full suite. A counterexample found once is pinned with `@example` so it runs regardless of the generated set.

## Test design
Test edge cases: simultaneous movement conflicts, agent failure, zero survivors, all cells blocked, no communication peers, depleted battery, episode time limit, and false detections.

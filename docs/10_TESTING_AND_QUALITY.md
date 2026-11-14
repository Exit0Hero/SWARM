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

## Test design
Test edge cases: simultaneous movement conflicts, agent failure, zero survivors, all cells blocked, no communication peers, depleted battery, episode time limit, and false detections.

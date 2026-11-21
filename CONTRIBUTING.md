# Contributing to SWARM

Thank you for helping. SWARM is a research simulator, so correctness, determinism and honest
reporting matter more than features.

## Set up
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"                      # add ,sim3d / ,viz / ,learning as needed
```

## Before you open a pull request
Run the quality gate from the repository root. CI runs the same commands:
```bash
ruff check .
ruff format --check .
mypy            # strict
pytest
```
If you touch packaging, the CLI or the dashboard, also build and smoke-test a clean install:
```bash
python -m build && python -m venv /tmp/clean && /tmp/clean/bin/pip install dist/*.whl
cd /tmp && PATH=/tmp/clean/bin:$PATH /tmp/clean/bin/python <repo>/scripts/smoke_test.py
```

## Rules
- **Read first:** `docs/14_ROADMAP.md`, the [ADR index](docs/decisions/README.md) and
  `docs/02_SYSTEM_ARCHITECTURE.md`. ADRs override the numbered documents. A change of design
  needs a new ADR (copy `docs/decisions/0000-template.md`).
- **Determinism:**
  - Use only the spawned RNG streams (ADR-0006).
  - Never let set or dict iteration order drive random draws.
  - If a change alters event streams, the golden hashes in `tests/test_regression.py` fail. That
    is intended. Explain why the change is correct and update them deliberately in a separate
    commit.
- **Information boundary:** policies see only their observations and the messages they receive
  (ADR-0008). Oracle access must go through `OracleView` and be labelled.
- **Tests:**
  - Add tests with every change, including failure modes.
  - Never weaken or delete a test to make CI pass.
  - A bug fix includes a test that failed before the fix.
- **Results:**
  - Never report a number that no run produced.
  - Pre-register studies before using evaluation seeds (docs/27).
  - Never overwrite archived results in `docs/results/`.
- **Dependencies:** the runtime stays NumPy only. Anything heavier is an optional extra, justified
  in an ADR.
- **Commits:** keep them small and focused, written in the imperative ("Add …", "Fix …"). One
  topic per pull request.

## Reporting bugs and ideas
Use the issue templates. For security problems, see [SECURITY.md](SECURITY.md).

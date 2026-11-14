---
name: documentation
description: Create or update SWARM documentation - ADRs, the numbered documents in docs, the roadmap and README status, and docstrings - and keep them consistent with the code and with each other. Use whenever behaviour, configuration, interfaces, dependencies or phase status change, or when writing a new decision record.
---

# Documentation

## Map of the docs
- `README.md`: what exists *now*, how to install and test. Status must match reality.
- `CLAUDE.md`: standing instructions for Claude Code.
- `MASTER_PROMPT.md`: the project brief and phase list.
- `docs/14_ROADMAP.md`: the single source of truth for phases and their status.
- `docs/decisions/`: ADRs. These override numbered docs when they conflict.
- `docs/01_PRODUCT_REQUIREMENTS.md` to `docs/15_RISK_REGISTER_AND_LIMITATIONS.md`: requirements and specifications.

## Procedure
1. **Find every place the change touches.** Search the docs for the affected concept (for example, an action
   name, metric name or phase number) and update all occurrences, not just one.
2. **Decisions go in ADRs.** A new or changed rule gets an ADR, copied from `docs/decisions/0000-template.md`
   and added to the index in `docs/decisions/README.md`. Numbered docs link to ADRs instead of restating them.
3. **Status honesty.** The README and roadmap say "planned" for anything not implemented and tested. A phase is
   "complete" only when every acceptance criterion in `docs/14_ROADMAP.md` is met.
4. **Phase numbering** is 0–8 everywhere. If a document uses a different scheme, fix it.
5. **Links:** use relative Markdown links between docs, and backtick repository paths (for example
   `docs/14_ROADMAP.md`) for files. Both are checked by `tests/test_repository_docs.py`. Do not backtick a path
   that does not exist yet. Describe planned paths in words or as module names.
6. **Docstrings** for public APIs: purpose, parameters, return value, raised exceptions, and the ADR they implement.
7. Run `pytest tests/test_repository_docs.py` after any docs change.

## Style
Short sentences, precise terms, defined units. State limitations plainly. No marketing language, and no claims of
real-world validity.

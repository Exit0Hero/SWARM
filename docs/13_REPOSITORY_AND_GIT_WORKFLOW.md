# Repository and Git Workflow

## Target structure
Directories under `src/swarm/` and the `configs`, `scripts` and `artifacts` directories are created when the
first real content for them exists. The tree below is the target, not the current state (see `README.md`).

```text
swarm/
  .claude/
    skills/
  .github/
    workflows/
  docs/
    decisions/
  src/swarm/
    core/
    scenarios/
    agents/
    observations/
    navigation/
    coordination/
    policies/
    learning/
    evaluation/
    experiments/
    visualization/
  tests/
  configs/
  scripts/
  artifacts/        # normally ignored; store metadata or selected outputs intentionally
  pyproject.toml
  README.md
  CLAUDE.md
  MASTER_PROMPT.md
```

## Git hygiene
- Use focused branches and descriptive commits.
- Keep generated datasets, large checkpoints, and transient logs out of Git unless intentionally versioned.
- Include `.gitignore`, license, contribution guidance, and environment setup instructions.
- Never commit secrets or local environment files.
- Do not rewrite history or force-push without explicit approval.

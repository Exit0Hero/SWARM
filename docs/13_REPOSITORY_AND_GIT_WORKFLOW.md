# Repository and Git Workflow

## Suggested structure
```text
swarm/
  .claude/
    skills/
  docs/
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

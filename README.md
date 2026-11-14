# SWARM — Claude Code Project Starter

SWARM is a research-oriented simulation platform for autonomous multi-agent disaster response and search & rescue.

## Package layout

- `CLAUDE.md` — project-wide instructions automatically read by Claude Code.
- `MASTER_PROMPT.md` — full project brief to use when starting or resuming work.
- `docs/` — project documentation, with the Markdown documents grouped together.
- `.claude/skills/` — eight project-specific Claude Code skills. Each skill has its own `SKILL.md` and can be invoked by Claude Code when relevant.

## Install

1. Extract this folder into the root of your SWARM Git repository.
2. Keep the `.claude/skills/` directory at the repository root. Do not move it into `docs/`.
3. Open the repository in Claude Code.
4. Ask Claude Code to read `CLAUDE.md` and `MASTER_PROMPT.md`, inspect the existing repository, and propose a staged implementation plan before changing files.
5. Commit the package so collaborators receive the same instructions and skills.

Project-local skills are discovered from `.claude/skills/<skill-name>/SKILL.md`. These are instruction packages, not executable plugins or separately installed software. Claude Code uses them as needed.

## Important implementation rule

The first milestone is a deterministic, testable 2D simulator. Do not begin with 3D rendering, physical drone control, or a large deep-learning stack. Establish baselines and evaluation first.

## Suggested first Claude Code prompt

> Read `CLAUDE.md`, `MASTER_PROMPT.md`, and all relevant documents in `docs/`. Inspect the repository before editing. Report the current state, identify gaps against the architecture, and propose a small, testable Phase 0/Phase 1 implementation plan. Do not overwrite existing work or introduce dependencies until you explain why they are needed.

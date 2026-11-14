"""Repository documentation checks.

* Every relative Markdown link resolves to an existing file (and heading anchor).
* Every inline-code repository path (for example `docs/14_ROADMAP.md`) exists.
* All nine project skills are discoverable by Claude Code: one directory per skill
  under `.claude/skills/`, each with a `SKILL.md` whose frontmatter `name` matches the
  directory and whose `description` is non-empty.
* Every ADR file is listed in the ADR index.

No third-party YAML or Markdown parser is used, so these checks add no dependencies.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    "runs",  # local, git-ignored experiment outputs
    "artifacts",
}

EXPECTED_SKILLS = {
    "architecture-review",
    "simulation-core",
    "path-planning",
    "task-allocation",
    "marl-research",
    "perception",
    "experiment-analysis",
    "testing-and-review",
    "documentation",
}

LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
# Inline code that looks like a repository path: a known top-level directory or a
# top-level Markdown/config file.
REPO_PATH_RE = re.compile(
    r"^(?:(?:docs|src|tests|\.claude|\.github)/[\w./-]+"
    r"|(?:README|CLAUDE|MASTER_PROMPT)\.md|pyproject\.toml|LICENSE|\.gitignore)$"
)
FENCE_RE = re.compile(r"^(```|~~~).*?^\1", re.MULTILINE | re.DOTALL)


def _markdown_files() -> list[Path]:
    return sorted(
        p
        for p in ROOT.rglob("*.md")
        if not any(part in EXCLUDED_DIRS for part in p.relative_to(ROOT).parts)
    )


def _strip_fenced_code(text: str) -> str:
    return FENCE_RE.sub("", text)


def _github_slug(heading: str) -> str:
    """Approximate GitHub's heading-anchor algorithm."""
    slug = heading.strip().lower()
    slug = re.sub(r"[^\w\- ]", "", slug)
    return slug.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    text = _strip_fenced_code(path.read_text(encoding="utf-8"))
    return {_github_slug(m.group(1)) for m in re.finditer(r"^#{1,6}\s+(.+)$", text, re.MULTILINE)}


MARKDOWN_FILES = _markdown_files()


def test_markdown_files_found() -> None:
    assert len(MARKDOWN_FILES) >= 18


@pytest.mark.parametrize("md", MARKDOWN_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(md: Path) -> None:
    text = _strip_fenced_code(md.read_text(encoding="utf-8"))
    broken: list[str] = []
    for target in LINK_RE.findall(text):
        if re.match(r"^[a-z]+:", target):  # http:, https:, mailto:
            continue
        file_part, _, anchor = target.partition("#")
        resolved = (md.parent / file_part).resolve() if file_part else md
        if not resolved.exists():
            broken.append(f"{target} (missing file)")
        elif anchor and resolved.suffix == ".md" and anchor not in _anchors(resolved):
            broken.append(f"{target} (missing anchor)")
    assert not broken, f"Broken links in {md.relative_to(ROOT)}: {broken}"


@pytest.mark.parametrize("md", MARKDOWN_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_inline_repository_paths_exist(md: Path) -> None:
    text = _strip_fenced_code(md.read_text(encoding="utf-8"))
    missing = [
        token
        for token in INLINE_CODE_RE.findall(text)
        if REPO_PATH_RE.match(token) and not (ROOT / token.rstrip("/")).exists()
    ]
    assert not missing, f"Nonexistent repository paths in {md.relative_to(ROOT)}: {missing}"


def _frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, f"{path.relative_to(ROOT)} has no YAML frontmatter block"
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        assert sep, f"Unparseable frontmatter line in {path.relative_to(ROOT)}: {line!r}"
        fields[key.strip()] = value.strip()
    return fields


def test_exactly_the_nine_expected_skills_exist() -> None:
    skills_dir = ROOT / ".claude" / "skills"
    found = {p.name for p in skills_dir.iterdir() if p.is_dir()}
    assert found == EXPECTED_SKILLS


@pytest.mark.parametrize("name", sorted(EXPECTED_SKILLS))
def test_skill_is_discoverable(name: str) -> None:
    skill_file = ROOT / ".claude" / "skills" / name / "SKILL.md"
    assert skill_file.is_file()
    fields = _frontmatter(skill_file)
    assert fields.get("name") == name
    assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)
    assert len(name) <= 64
    description = fields.get("description", "")
    assert 50 <= len(description) <= 1024, "description must say what the skill does and when"
    assert "<" not in description
    assert ">" not in description
    # Unquoted YAML scalars break on ": " and " #"; the frontmatter must stay valid YAML.
    assert ": " not in description
    assert " #" not in description


def test_every_adr_is_indexed() -> None:
    decisions = ROOT / "docs" / "decisions"
    index = (decisions / "README.md").read_text(encoding="utf-8")
    adrs = sorted(p.name for p in decisions.glob("[0-9][0-9][0-9][0-9]-*.md"))
    assert len(adrs) >= 10
    unindexed = [name for name in adrs if f"({name})" not in index]
    assert not unindexed, f"ADRs missing from docs/decisions/README.md: {unindexed}"

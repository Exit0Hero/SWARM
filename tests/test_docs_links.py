"""Documentation check: every relative Markdown link in the repository resolves to a file."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
SKIP_DIRS = {".git", ".venv", "node_modules", ".claude", "runs"}


def _markdown_files() -> list[Path]:
    return [p for p in ROOT.rglob("*.md") if not SKIP_DIRS & set(p.relative_to(ROOT).parts)]


def test_relative_markdown_links_resolve() -> None:
    broken = []
    for md in _markdown_files():
        text = re.sub(r"```.*?```", "", md.read_text(encoding="utf-8"), flags=re.S)
        for target in LINK.findall(text):
            if re.match(r"^[a-z]+:|^#|^mailto:", target):
                continue
            path = target.split("#", 1)[0]
            if path and not (md.parent / path).exists():
                broken.append(f"{md.relative_to(ROOT)} -> {target}")
    assert not broken, "broken links:\n" + "\n".join(broken)

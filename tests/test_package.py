"""Packaging and dependency-policy checks for Phase 0."""

from __future__ import annotations

import re
import sys
from importlib.metadata import metadata, requires

import swarm


def test_version_matches_installed_metadata() -> None:
    assert swarm.__version__ == metadata("swarm")["Version"]


def test_numpy_is_the_only_runtime_dependency() -> None:
    """Dependency policy: NumPy only until a phase justifies more (ADR index)."""
    declared = requires("swarm") or []
    runtime = [req for req in declared if "extra ==" not in req]
    names = {re.split(r"[\s<>=!~;\[]", req, maxsplit=1)[0].lower() for req in runtime}
    assert names == {"numpy"}


def test_dev_extra_contains_required_tools() -> None:
    declared = requires("swarm") or []
    dev = {
        re.split(r"[\s<>=!~;\[]", req, maxsplit=1)[0].lower()
        for req in declared
        if re.search(r"""extra\s*==\s*['"]dev['"]""", req)
    }
    assert dev == {"pytest", "hypothesis", "ruff", "mypy"}


def test_running_interpreter_satisfies_requires_python() -> None:
    assert metadata("swarm")["Requires-Python"] == ">=3.12"
    assert sys.version_info >= (3, 12)

"""SWARM: research simulator for multi-agent disaster response and search & rescue.

Status: Phase 2 (navigation, prioritized planning, frontier exploration, assignment and an
oracle reference, on the Phase 1 deterministic core); see
docs/14_ROADMAP.md. Subpackages exist only when they contain tested implementations
(docs/decisions/README.md, "Repository conventions").
"""

from importlib.metadata import version as _version

__all__ = ["__version__"]

__version__: str = _version("swarm")

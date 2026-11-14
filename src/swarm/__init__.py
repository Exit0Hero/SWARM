"""SWARM: research simulator for multi-agent disaster response and search & rescue.

Status: Phase 1, milestone M1 (deterministic 2D core with a random baseline); see
docs/14_ROADMAP.md. Subpackages exist only when they contain tested implementations
(docs/decisions/README.md, "Repository conventions").
"""

from importlib.metadata import version as _version

__all__ = ["__version__"]

__version__: str = _version("swarm")

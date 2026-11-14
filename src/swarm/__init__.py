"""SWARM: research simulator for multi-agent disaster response and search & rescue.

Status: Phase 0 (foundation). No simulator code exists yet; see docs/14_ROADMAP.md.
Subpackages are created only when they contain real implementations
(docs/decisions/README.md, "Repository conventions").
"""

from importlib.metadata import version as _version

__all__ = ["__version__"]

__version__: str = _version("swarm")

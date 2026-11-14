"""SWARM: research simulator for multi-agent disaster response and search & rescue.

Status: Phase 1, milestone M2 (deterministic 2D core with hazards, noisy detection, mission
success and multi-seed evaluation; random baseline only); see
docs/14_ROADMAP.md. Subpackages exist only when they contain tested implementations
(docs/decisions/README.md, "Repository conventions").
"""

from importlib.metadata import version as _version

__all__ = ["__version__"]

__version__: str = _version("swarm")

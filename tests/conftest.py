"""Shared pytest configuration."""

from __future__ import annotations

from hypothesis import HealthCheck, settings

# Deterministic property tests: the same examples run locally and in CI (testing-and-review skill).
settings.register_profile(
    "swarm",
    derandomize=True,
    deadline=None,
    database=None,
    max_examples=60,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("swarm")

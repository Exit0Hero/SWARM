"""Golden-hash regression test for fixed-seed episodes (docs/10, ADR-0005, ADR-0006).

These hashes were produced by ``configs/m1_default.json`` with the ``random_valid`` policy on
NumPy 2.5.4 / Python 3.13 and are checked in CI on Python 3.12, 3.13 and 3.14. A mismatch means
the simulation's behaviour changed. If the change is intentional (a rule change, a new event
field, a different draw order), update the hashes in the same commit and say why in the commit
message. Never update them to silence an unexplained failure.

A NumPy release that changes ``PCG64``/``SeedSequence`` output would also change them; record
that in the commit if it ever happens (ADR-0006, consequence 2).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from swarm.experiments.runner import run_episode
from swarm.policies.random_valid import random_valid_factory
from swarm.scenarios.config import ScenarioConfig

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "m1_default.json"

# seed: (scenario_sha256, event_stream_sha256)
GOLDEN: dict[int, tuple[str, str]] = {
    0: (
        "516cadb8646d86feb1d2d02b2366d2ad126744bf755fb24556db332b2d92d01d",
        "7194a89b98260ce39bcb1dd9f2e9c17bb4435d2374e582c69b00aeb645aa2a5b",
    ),
    1: (
        "18187e00275c077cc12fbc03f3e3787c6d196c934958607d7bf3611faba3e1a7",
        "8f603097e5a35fe27ab1b45652abc08c391915d94c5fee982ba8506c86863f28",
    ),
    2: (
        "5af6f950ce327441691ae7a26895a725f9fd36e7944c23ec7755dfc1f8b2b7ad",
        "3a55340e7c3b1c5a297caf0265a74c76d07cb01f8765e90cf1cb090efe00c50a",
    ),
}


@pytest.mark.parametrize("seed", sorted(GOLDEN))
def test_fixed_seed_episode_matches_golden_hashes(seed: int) -> None:
    config = ScenarioConfig.from_json(CONFIG.read_text(encoding="utf-8"))
    result = run_episode(config, random_valid_factory, seed)
    scenario_hash, stream_hash = GOLDEN[seed]
    assert result.scenario_sha256 == scenario_hash, "scenario generation changed"
    assert result.event_stream_sha256 == stream_hash, "episode dynamics or event schema changed"


# Phase 1 M2: configs/m2_default.json (hazards, false positives, recall target 0.75) with the
# random_valid policy. Seed 0 ends in mission success; seeds 1-2 hit the time limit.
M2_CONFIG = CONFIG.parent / "m2_default.json"
GOLDEN_M2: dict[int, tuple[str, str]] = {
    0: (
        "780649c6f0d56aafd0d7d5a03229714e5137bf194c397e2355eb7af163842b64",
        "4a642d11cbdfbce9e3d0c41f657564ed420227abcf77d1b954f3233985fc00c4",
    ),
    1: (
        "9aab79bd7f127653a6ebc81d1d6646b47773e79eb3fa5607c75a6db866814606",
        "538055081ac8a3f12ba2ccffdcc4fada3c424b3148c9abbe7d094c8f0ec2a0e6",
    ),
    2: (
        "6132bf81ae792094a8264b7fc35b1a1e78fe27480b62588fdbcd1f2e3bcee634",
        "940fd50fc449dedeb8fb263ea04a3e97c5cc6cf17fa3436ae94816286cb5dc16",
    ),
}


@pytest.mark.parametrize("seed", sorted(GOLDEN_M2))
def test_m2_fixed_seed_episode_matches_golden_hashes(seed: int) -> None:
    config = ScenarioConfig.from_json(M2_CONFIG.read_text(encoding="utf-8"))
    result = run_episode(config, random_valid_factory, seed)
    scenario_hash, stream_hash = GOLDEN_M2[seed]
    assert result.scenario_sha256 == scenario_hash, "scenario generation changed"
    assert result.event_stream_sha256 == stream_hash, "episode dynamics or event schema changed"

"""Seeded scenario generation, RNG stream isolation and reachability (ADR-0002, ADR-0006)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from swarm.core.rng import episode_streams, seed_derivation, validate_seed
from swarm.core.types import Layer, Terrain
from swarm.scenarios.generator import (
    Scenario,
    ScenarioError,
    eligible_cells,
    generate_scenario,
    parse_ascii_map,
    reachable_from_base,
)
from tests.helpers import make_config

AGENTS = ["drone_0", "ground_0"]


def _generate(seed: int, **overrides: object) -> Scenario:
    config = make_config(**overrides)  # type: ignore[arg-type]
    return generate_scenario(config, episode_streams(seed, AGENTS).scenario)


def test_same_seed_same_scenario() -> None:
    assert _generate(7) == _generate(7)
    assert _generate(7).fingerprint() == _generate(7).fingerprint()


def test_different_seeds_give_different_scenarios() -> None:
    fingerprints = {_generate(seed, height=12, width=12).fingerprint() for seed in range(20)}
    assert len(fingerprints) == 20


def test_policy_stream_count_does_not_change_scenario_or_dynamics() -> None:
    """Isolation (ADR-0006): the roster's policy streams do not shift other streams."""
    a = episode_streams(3, ["a"])
    b = episode_streams(3, ["a", "b", "c", "d"])
    assert a.scenario.random() == b.scenario.random()
    assert a.dynamics.random() == b.dynamics.random()


def test_streams_are_independent_and_keyed_by_sorted_agent_id() -> None:
    s = episode_streams(0, ["z_0", "a_0"])
    assert list(s.policies) == ["a_0", "z_0"]
    draws = {s.scenario.random(), s.dynamics.random(), *(g.random() for g in s.policies.values())}
    assert len(draws) == 4


@pytest.mark.parametrize("bad", [-1, 1.5, "3", True, None])
def test_invalid_seeds_rejected(bad: object) -> None:
    with pytest.raises(ValueError, match="seed: must be a non-negative integer"):
        validate_seed(bad)


def test_seed_derivation_is_recorded() -> None:
    record = seed_derivation(5, ["b", "a"])
    assert record["root_seed"] == 5
    assert record["spawn_order"] == ["scenario", "dynamics", "policies"]
    assert record["policy_stream_order"] == ["a", "b"]


def test_base_is_unique_and_survivors_on_open_or_debris() -> None:
    s = _generate(1, height=15, width=15, survivors=20)
    assert int((s.terrain == Terrain.BASE).sum()) == 1
    assert s.terrain[s.base] == Terrain.BASE
    assert len(s.survivors) == 20
    assert all(s.terrain[p] in (Terrain.OPEN, Terrain.DEBRIS) for p in s.survivors)
    assert list(s.survivors) == sorted(s.survivors)


def test_terrain_is_read_only() -> None:
    s = _generate(0)
    with pytest.raises(ValueError, match="read-only"):
        s.terrain[0, 0] = Terrain.OPEN


def test_fully_blocked_map_without_survivors() -> None:
    s = _generate(0, structure=1.0, debris=0.0, survivors=0)
    assert int((s.terrain == Terrain.STRUCTURE).sum()) == s.terrain.size - 1
    assert s.survivors == ()


def test_fully_blocked_map_cannot_hold_survivors() -> None:
    with pytest.raises(ScenarioError, match="only 0 OPEN/DEBRIS cells"):
        _generate(0, structure=1.0, debris=0.0, survivors=1)


def test_zero_survivors() -> None:
    assert _generate(0, survivors=0).survivors == ()


def test_one_by_one_grid_is_just_the_base() -> None:
    s = _generate(0, height=1, width=1, survivors=0)
    assert s.terrain.tolist() == [[Terrain.BASE]]


def test_terrain_fractions_are_respected_statistically() -> None:
    s = _generate(0, height=200, width=200, debris=0.2, structure=0.3, survivors=0)
    frac_structure = float((s.terrain == Terrain.STRUCTURE).mean())
    frac_debris = float((s.terrain == Terrain.DEBRIS).mean())
    # 40,000 cells: standard error ~0.0023; bounds are > 8 standard errors.
    assert abs(frac_structure - 0.3) < 0.02
    assert abs(frac_debris - 0.2) < 0.02


def test_ascii_map_parsing() -> None:
    s = parse_ascii_map(["S.#", "~B.", "..s"])
    assert s.base == (1, 1)
    assert s.survivors == ((0, 0), (2, 2))
    assert s.terrain[0, 2] == Terrain.STRUCTURE
    assert s.terrain[2, 2] == Terrain.DEBRIS


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (["...", ".."], "equal length"),
        (["..."], "no base"),
        (["BB."], "exactly one BASE"),
        (["B?"], "unknown character"),
    ],
)
def test_ascii_map_errors(rows: list[str], message: str) -> None:
    with pytest.raises(ScenarioError, match=message):
        parse_ascii_map(rows)


def test_scenario_validation_rejects_survivor_on_structure() -> None:
    terrain = np.array([[Terrain.BASE, Terrain.STRUCTURE]], dtype=np.int8)
    with pytest.raises(ScenarioError, match="not on OPEN or DEBRIS"):
        Scenario(terrain=terrain, base=(0, 0), survivors=((0, 1),))


def test_reachability_is_layer_specific() -> None:
    s = parse_ascii_map(["B~.", "##.", "..."])
    air = reachable_from_base(s, Layer.AIR)
    ground = reachable_from_base(s, Layer.GROUND)
    assert air.tolist() == [
        [True, True, True],
        [False, False, True],
        [True, True, True],
    ]
    assert ground.tolist() == [
        [True, False, False],
        [False, False, False],
        [False, False, False],
    ]
    assert int(eligible_cells(s, [Layer.AIR, Layer.GROUND]).sum()) == 7


@given(seed=st.integers(0, 2**32), h=st.integers(1, 12), w=st.integers(1, 12))
def test_generation_invariants(seed: int, h: int, w: int) -> None:
    count = min(3, h * w - 1)
    try:
        s = _generate(seed, height=h, width=w, survivors=count)
    except ScenarioError:
        return  # legitimately too few open/debris cells for this seed
    assert s.shape == (h, w)
    assert s.base == (h // 2, w // 2)
    assert len(s.survivors) == count
    assert s == _generate(seed, height=h, width=w, survivors=count)

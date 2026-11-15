"""Phase 7: clustered survivor placement (ADR-0023)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from swarm.core.rng import episode_streams
from swarm.scenarios.config import ConfigError, ScenarioConfig, SurvivorLayout
from swarm.scenarios.generator import generate_scenario
from tests.helpers import make_config


def _scenario(config: ScenarioConfig, seed: int):  # type: ignore[no-untyped-def]
    return generate_scenario(config, episode_streams(seed, tuple(config.agents())).scenario)


def _spread(survivors: tuple[tuple[int, int], ...]) -> float:
    pts = np.array(survivors, dtype=np.float64)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float(d.min(axis=1).mean())


def test_layout_round_trips_and_leaves_uniform_configs_unchanged() -> None:
    plain = make_config(drones=1, grounds=1)
    assert "survivor_layout" not in plain.to_dict()
    clustered = replace(plain, survivor_layout=SurvivorLayout(2, 1.5))
    assert clustered.to_dict()["schema_version"] == 3
    assert ScenarioConfig.from_json(clustered.to_json()) == clustered
    a, b = _scenario(plain, 7), _scenario(clustered, 7)
    assert np.array_equal(a.terrain, b.terrain)  # terrain is drawn first, unchanged


@pytest.mark.parametrize(
    ("kw", "match"),
    [({"clusters": 0, "spread": 1.0}, "clusters"), ({"clusters": 1, "spread": 0}, "spread")],
)
def test_layout_validation(kw: dict[str, float], match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        SurvivorLayout(**kw)  # type: ignore[arg-type]


def test_clustered_survivors_are_valid_deterministic_and_closer_together() -> None:
    base = replace(make_config(drones=1, grounds=1), height=20, width=20, survivor_count=8)
    clustered = replace(base, survivor_layout=SurvivorLayout(2, 1.5))
    uniform_spread, clustered_spread = [], []
    for seed in range(20):
        s = _scenario(clustered, seed)
        assert s == _scenario(clustered, seed)
        assert len(s.survivors) == 8
        clustered_spread.append(_spread(s.survivors))
        uniform_spread.append(_spread(_scenario(base, seed).survivors))
    assert np.mean(clustered_spread) < 0.6 * np.mean(uniform_spread)

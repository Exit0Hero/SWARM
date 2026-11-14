"""Hidden-state boundary (ADR-0008): observations must not depend on unseen ground truth."""

from __future__ import annotations

from dataclasses import fields

from hypothesis import given
from hypothesis import strategies as st

from swarm.core.env import SwarmEnv
from swarm.core.observation import Observation, in_footprint
from swarm.core.types import Action, Layer
from swarm.scenarios.config import AgentSpec
from swarm.scenarios.generator import Scenario
from tests.helpers import config_for, spec

SIZE = 11
CENTER = (5, 5)
AIR = spec("air", Layer.AIR, radius=2, capacity=50)
GND = spec("gnd", Layer.GROUND, radius=1, capacity=50)
SCRIPT: list[dict[str, int]] = [
    {"air_0": Action.NORTH, "gnd_0": Action.EAST},
    {"air_0": Action.NORTH, "gnd_0": Action.STAY},
    {"air_0": Action.EAST, "gnd_0": Action.SOUTH},
    {"air_0": Action.STAY, "gnd_0": Action.WEST},
]
# Union of all footprints along the scripted trajectory (computed for the fixed script above).
TRAJECTORY = {
    "air_0": [(5, 5), (4, 5), (3, 5), (3, 6), (3, 6)],
    "gnd_0": [(5, 5), (5, 6), (5, 6), (6, 6), (6, 5)],
}
RADIUS = {"air_0": 2, "gnd_0": 1}


def _seen(cell: tuple[int, int]) -> bool:
    return any(in_footprint(p, cell, RADIUS[aid]) for aid, path in TRAJECTORY.items() for p in path)


UNSEEN = [(r, c) for r in range(SIZE) for c in range(SIZE) if not _seen((r, c))]
SEEN = [(r, c) for r in range(SIZE) for c in range(SIZE) if _seen((r, c)) and (r, c) != CENTER]


def _rows(survivor: tuple[int, int]) -> list[str]:
    rows = [["."] * SIZE for _ in range(SIZE)]
    rows[CENTER[0]][CENTER[1]] = "B"
    rows[survivor[0]][survivor[1]] = "S"
    return ["".join(r) for r in rows]


def _trace(survivor: tuple[int, int], seed: int = 0) -> list[dict[str, Observation]]:
    config, scenario = config_for(
        _rows(survivor), (AIR, GND), (AgentSpec("air"), AgentSpec("gnd")), p_detect=1.0
    )
    env = SwarmEnv(config)
    trace = [env.reset(seed, scenario=scenario)[0]]
    for actions in SCRIPT:
        trace.append(env.step({k: int(v) for k, v in actions.items()}).observations)
    return trace


def test_script_follows_the_expected_trajectory() -> None:
    trace = _trace(UNSEEN[0])
    for aid, path in TRAJECTORY.items():
        assert [obs[aid].position for obs in trace] == path


@given(a=st.sampled_from(UNSEEN), b=st.sampled_from(UNSEEN), seed=st.integers(0, 1000))
def test_unseen_survivor_positions_do_not_change_any_observation(
    a: tuple[int, int], b: tuple[int, int], seed: int
) -> None:
    assert _trace(a, seed) == _trace(b, seed)


def test_positive_control_a_seen_survivor_does_change_observations() -> None:
    """Guards against a vacuous leakage test: detections must show up when in range."""
    assert _trace(SEEN[0]) != _trace(UNSEEN[0])


def test_observation_has_no_ground_truth_fields() -> None:
    names = {f.name for f in fields(Observation)}
    assert not {n for n in names if "survivor" in n or "scenario" in n or "env" in n}
    obs = _trace(UNSEEN[0])[1]["air_0"]
    for f in fields(Observation):
        value = getattr(obs, f.name)
        assert not isinstance(value, Scenario | SwarmEnv)

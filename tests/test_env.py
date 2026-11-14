"""SwarmEnv behaviour on hand-built maps: rules, edge cases and the API contract."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from swarm.core.env import SwarmEnv
from swarm.core.events import (
    AgentDisabled,
    Detection,
    EnergySpent,
    EpisodeEnd,
    Event,
    InvalidAction,
    MoveConflict,
    Moved,
    Recharged,
)
from swarm.core.observation import OUT_OF_BOUNDS, UNKNOWN
from swarm.core.types import Action, AgentStatus, Layer, Terrain
from swarm.scenarios.config import AgentSpec
from tests.helpers import config_for, spec

S, N, E, SO, W = (int(a) for a in Action)  # STAY, NORTH, EAST, SOUTH, WEST

AIR = spec("air", Layer.AIR, capacity=10, move=1, stay=0)
GND = spec("gnd", Layer.GROUND, capacity=10, move=1, stay=0)


def make_env(
    rows: list[str], *roster: AgentSpec, types: Any = (AIR, GND), **kwargs: Any
) -> SwarmEnv:
    config, scenario = config_for(rows, types, tuple(roster), **kwargs)
    env = SwarmEnv(config)
    env.reset(0, scenario=scenario)
    return env


def kinds(events: tuple[Event, ...]) -> list[str]:
    return [e.kind for e in events]


# ----------------------------------------------------------------- reset


def test_reset_spawns_everyone_on_base_with_configured_energy() -> None:
    config, scenario = config_for(
        ["...", ".B.", "..."], (AIR, GND), (AgentSpec("air", 3), AgentSpec("gnd"))
    )
    env = SwarmEnv(config)
    observations, infos = env.reset(0, scenario=scenario)
    assert env.agents == ("air_0", "gnd_0")
    assert observations["air_0"].position == (1, 1)
    assert observations["air_0"].energy == 3
    assert observations["gnd_0"].energy == 10
    assert all(o.detections == () and o.step == 0 for o in observations.values())
    assert infos == {"air_0": {}, "gnd_0": {}}


def test_reset_starts_a_fresh_episode() -> None:
    env = make_env(["B."], AgentSpec("air"), time_limit=1)
    env.step({"air_0": E})
    assert env.snapshot().done
    env.reset(0)
    assert not env.snapshot().done
    assert env.step_count == 0
    assert env.snapshot().agents[0].position == (0, 0)


def test_reset_with_mismatched_scenario_raises() -> None:
    config, _ = config_for(["B.."], (AIR,), (AgentSpec("air"),))
    _, other = config_for(["B.", ".."], (AIR,), (AgentSpec("air"),))
    with pytest.raises(ValueError, match="scenario shape"):
        SwarmEnv(config).reset(0, scenario=other)


# ----------------------------------------------------------------- movement and terrain


def test_successful_move_emits_moved_and_energy_spent() -> None:
    env = make_env(["B.."], AgentSpec("gnd"))
    result = env.step({"gnd_0": E})
    assert result.events == (Moved(1, "gnd_0", (0, 0), (0, 1)), EnergySpent(1, "gnd_0", 1, 9))
    assert result.observations["gnd_0"].position == (0, 1)


def test_ground_cannot_enter_debris_but_air_can() -> None:
    env = make_env(["B~"], AgentSpec("air"), AgentSpec("gnd"))
    masks = {aid: env.action_mask(aid) for aid in env.agents}
    assert masks["air_0"][E] is True
    assert masks["gnd_0"][E] is False
    result = env.step({"air_0": E, "gnd_0": E})
    assert InvalidAction(1, "gnd_0", E, "impassable_terrain") in result.events
    assert Moved(1, "air_0", (0, 0), (0, 1)) in result.events
    assert result.observations["gnd_0"].position == (0, 0)


def test_nobody_enters_structure_or_leaves_the_grid() -> None:
    env = make_env(["B#"], AgentSpec("air"))
    assert env.action_mask("air_0") == (True, False, False, False, False)
    result = env.step({"air_0": N})
    assert result.events[0] == InvalidAction(1, "air_0", N, "out_of_bounds")
    result = env.step({"air_0": E})
    assert result.events[0] == InvalidAction(2, "air_0", E, "impassable_terrain")
    assert env.snapshot().agents[0].position == (0, 0)


def test_invalid_action_is_charged_the_stay_cost() -> None:
    hover = spec("air", Layer.AIR, capacity=10, move=3, stay=2)
    env = make_env([".B", ".."], AgentSpec("air"), types=(hover,))
    env.step({"air_0": W})  # valid move to (0, 0): costs 3
    result = env.step({"air_0": N})  # out of bounds -> STAY, costs 2
    assert kinds(result.events) == ["invalid_action", "energy_spent"]
    assert result.observations["air_0"].energy == 10 - 3 - 2


def test_insufficient_energy_masks_moves() -> None:
    env = make_env(["B.."], AgentSpec("gnd", initial_energy=0))
    assert env.action_mask("gnd_0") == (True, False, False, False, False)
    result = env.step({"gnd_0": E})
    assert result.events[0] == InvalidAction(1, "gnd_0", E, "insufficient_energy")


def test_vertex_conflict_in_environment() -> None:
    env = make_env(["...", ".B.", "..."], AgentSpec("gnd"), AgentSpec("gnd"))
    result = env.step({"gnd_0": N, "gnd_1": N})
    assert result.events == (
        MoveConflict(1, "gnd_0", (1, 1), (0, 1), "vertex"),
        MoveConflict(1, "gnd_1", (1, 1), (0, 1), "vertex"),
    )
    assert env.snapshot().agents[0].position == (1, 1)


def test_air_and_ground_share_a_cell() -> None:
    env = make_env(["B."], AgentSpec("air"), AgentSpec("gnd"))
    result = env.step({"air_0": E, "gnd_0": E})
    assert all(o.position == (0, 1) for o in result.observations.values())


def test_action_dict_order_does_not_matter() -> None:
    rows = ["....", ".B..", "...."]
    team = (AgentSpec("gnd"), AgentSpec("gnd"), AgentSpec("air"))
    a = make_env(rows, *team)
    b = make_env(rows, *team)
    ra = a.step({"gnd_0": N, "gnd_1": E, "air_0": N})
    rb = b.step({"air_0": N, "gnd_1": E, "gnd_0": N})
    assert ra.events == rb.events
    assert ra.observations == rb.observations


# ----------------------------------------------------------------- energy and disabling


def test_recharge_on_base_is_capped_at_capacity() -> None:
    charger = spec("gnd", Layer.GROUND, capacity=10, move=1, stay=0, recharge=4)
    env = make_env(["B."], AgentSpec("gnd", initial_energy=7), types=(charger,))
    first = env.step({"gnd_0": S})
    assert first.events == (Recharged(1, "gnd_0", 3, 10),)
    assert env.step({"gnd_0": S}).events == ()


def test_zero_energy_spawn_recharges_on_base_and_stays_active() -> None:
    charger = spec("gnd", Layer.GROUND, capacity=10, move=1, stay=0, recharge=2)
    env = make_env(["B."], AgentSpec("gnd", initial_energy=0), types=(charger,))
    result = env.step({"gnd_0": S})
    assert result.events == (Recharged(1, "gnd_0", 2, 2),)
    assert result.observations["gnd_0"].status is AgentStatus.ACTIVE
    assert env.action_mask("gnd_0")[E] is True


def test_zero_energy_spawn_without_recharge_survives_until_time_limit() -> None:
    env = make_env(["B."], AgentSpec("gnd", initial_energy=0), time_limit=3)
    for _ in range(3):
        result = env.step({"gnd_0": S})
    assert result.events == (EpisodeEnd(3, "time_limit", False, True),)
    assert result.truncations == {"gnd_0": True}


def test_agent_disabled_at_zero_energy_away_from_base() -> None:
    env = make_env(["B..."], AgentSpec("gnd", initial_energy=1), AgentSpec("gnd"))
    result = env.step({"gnd_0": E, "gnd_1": S})
    assert AgentDisabled(1, "gnd_0", (0, 1), "energy_depleted") in result.events
    assert result.terminations == {"gnd_0": True, "gnd_1": False}
    assert result.observations["gnd_0"].status is AgentStatus.DISABLED
    assert result.observations["gnd_0"].action_mask == (True, False, False, False, False)
    assert env.agents == ("gnd_1",)
    # The disabled agent still occupies its cell and blocks the other robot.
    blocked = env.step({"gnd_1": E})
    assert MoveConflict(2, "gnd_1", (0, 0), (0, 1), "blocked") in blocked.events
    # Disabled agents take no actions.
    with pytest.raises(ValueError, match="unexpected \\['gnd_0'\\]"):
        env.step({"gnd_0": S, "gnd_1": S})


def test_all_disabled_terminates_the_episode() -> None:
    env = make_env(["B.."], AgentSpec("gnd", initial_energy=1), AgentSpec("air", initial_energy=1))
    result = env.step({"gnd_0": E, "air_0": E})
    assert result.events[-1] == EpisodeEnd(1, "all_disabled", True, False)
    assert result.terminations == {"air_0": True, "gnd_0": True}
    assert result.truncations == {"air_0": False, "gnd_0": False}
    assert env.done
    assert env.agents == ()
    with pytest.raises(RuntimeError, match="episode ended at step 1 \\(all_disabled\\)"):
        env.step({})


def test_energy_never_negative_with_large_stay_cost() -> None:
    hover = spec("air", Layer.AIR, capacity=10, move=3, stay=5)
    env = make_env(["B...."], AgentSpec("air", initial_energy=4), types=(hover,))
    env.step({"air_0": E})  # 4 - 3 = 1
    result = env.step({"air_0": S})  # stay cost 5, only 1 left
    assert EnergySpent(2, "air_0", 1, 0) in result.events
    assert AgentDisabled(2, "air_0", (0, 1), "energy_depleted") in result.events


# ----------------------------------------------------------------- time limit


def test_time_limit_truncates_and_further_steps_raise() -> None:
    env = make_env(["B."], AgentSpec("air"), time_limit=2)
    assert env.step({"air_0": S}).events == ()
    result = env.step({"air_0": S})
    assert result.events == (EpisodeEnd(2, "time_limit", False, True),)
    assert result.truncations == {"air_0": True}
    assert result.terminations == {"air_0": False}
    with pytest.raises(RuntimeError, match="time_limit"):
        env.step({"air_0": S})


def test_disabled_on_last_step_is_terminated_not_truncated() -> None:
    env = make_env(["B.."], AgentSpec("gnd", initial_energy=1), AgentSpec("air"), time_limit=1)
    result = env.step({"gnd_0": E, "air_0": S})
    assert result.terminations == {"air_0": False, "gnd_0": True}
    assert result.truncations == {"air_0": True, "gnd_0": False}


# ----------------------------------------------------------------- sensing


def test_detection_within_euclidean_radius_only() -> None:
    # Radius 1: orthogonal neighbour is in range, diagonal (distance sqrt 2) is not.
    env = make_env([".S", "BS"], AgentSpec("gnd"), types=(GND,), p_detect=1.0)
    result = env.step({"gnd_0": S})
    assert result.events == (Detection(1, "gnd_0", (1, 1), True, 1),)
    assert result.observations["gnd_0"].detections == ((1, 1),)


def test_detection_probability_zero_never_reports() -> None:
    env = make_env(["BS"], AgentSpec("gnd"), types=(GND,), p_detect=0.0, time_limit=50)
    for _ in range(50):
        result = env.step({"gnd_0": S})
        assert not any(isinstance(e, Detection) for e in result.events)


def test_detection_probability_is_respected_statistically() -> None:
    env = make_env(["BS"], AgentSpec("gnd"), types=(GND,), p_detect=0.3, time_limit=4000)
    hits = 0
    for _ in range(4000):
        result = env.step({"gnd_0": S})
        hits += sum(isinstance(e, Detection) for e in result.events)
    # Binomial(4000, 0.3): sd ~ 29; the bound is ~4 sd and the seed is fixed (not flaky).
    assert abs(hits - 1200) < 120


def test_every_agent_reports_independently() -> None:
    env = make_env(["BS"], AgentSpec("gnd"), AgentSpec("air"), types=(AIR, GND))
    result = env.step({"gnd_0": S, "air_0": S})
    assert [e for e in result.events if isinstance(e, Detection)] == [
        Detection(1, "air_0", (0, 1), True, 0),
        Detection(1, "gnd_0", (0, 1), True, 0),
    ]


def test_zero_survivor_episode_runs_to_time_limit() -> None:
    env = make_env(["B.", ".."], AgentSpec("air"), time_limit=5)
    seen: list[Event] = []
    for _ in range(5):
        seen.extend(env.step({"air_0": S}).events)
    assert not any(isinstance(e, Detection) for e in seen)
    assert seen[-1] == EpisodeEnd(5, "time_limit", False, True)


def test_fully_blocked_map_masks_every_move() -> None:
    env = make_env(["###", "#B#", "###"], AgentSpec("air"), AgentSpec("gnd"), time_limit=3)
    for aid in env.agents:
        assert env.action_mask(aid) == (True, False, False, False, False)
    for _ in range(3):
        env.step({"air_0": N, "gnd_0": W})
    assert all(a.position == (1, 1) for a in env.snapshot().agents)


# ----------------------------------------------------------------- observation content


def test_local_terrain_view() -> None:
    env = make_env(["~.#", "B.."], AgentSpec("gnd"), types=(GND,))
    obs = env.step({"gnd_0": S}).observations["gnd_0"]
    # Radius 1 centred on (1, 0): a plus-shaped footprint.
    assert obs.local_terrain.tolist() == [
        [OUT_OF_BOUNDS, Terrain.DEBRIS, UNKNOWN],
        [OUT_OF_BOUNDS, Terrain.BASE, Terrain.OPEN],
        [OUT_OF_BOUNDS, OUT_OF_BOUNDS, OUT_OF_BOUNDS],
    ]
    with pytest.raises(ValueError, match="read-only"):
        obs.local_terrain[0, 0] = 0


def test_visible_agents_exclude_self_and_hide_energy() -> None:
    env = make_env(["B.."], AgentSpec("gnd"), AgentSpec("air"), types=(AIR, GND))
    obs = env.step({"gnd_0": S, "air_0": E}).observations["gnd_0"]
    assert [v.agent_id for v in obs.visible_agents] == ["air_0"]
    assert not hasattr(obs.visible_agents[0], "energy")


# ----------------------------------------------------------------- contract


def test_step_before_reset_raises() -> None:
    config, _ = config_for(["B."], (AIR,), (AgentSpec("air"),))
    with pytest.raises(RuntimeError, match="call reset"):
        SwarmEnv(config).step({"air_0": S})


@pytest.mark.parametrize(
    ("actions", "message"),
    [
        ({}, "missing \\['air_0'\\]"),
        ({"air_0": S, "ghost": S}, "unexpected \\['ghost'\\]"),
        ({"air_0": 5}, "must be in 0..4, got 5"),
        ({"air_0": -1}, "must be in 0..4, got -1"),
        ({"air_0": True}, "must be an int"),
        ({"air_0": 1.0}, "must be an int"),
        ({"air_0": "1"}, "must be an int"),
    ],
)
def test_contract_violations_raise(actions: dict[str, Any], message: str) -> None:
    env = make_env(["B."], AgentSpec("air"))
    with pytest.raises(ValueError, match=message):
        env.step(actions)


def test_numpy_integer_actions_are_accepted() -> None:
    env = make_env(["B."], AgentSpec("air"))
    result = env.step({"air_0": np.int64(E)})
    assert result.observations["air_0"].position == (0, 1)


def test_events_are_ordered_by_phase_then_agent() -> None:
    phase = {
        "invalid_action": 1,
        "moved": 2,
        "move_conflict": 2,
        "energy_spent": 3,
        "recharged": 3,
        "agent_disabled": 3,
        "detection": 5,
        "episode_end": 6,
    }
    rows = ["S...", ".B#.", "...S"]
    env = make_env(
        rows,
        AgentSpec("air", 1),
        AgentSpec("gnd"),
        AgentSpec("gnd"),
        types=(AIR, GND),
        time_limit=1,
    )
    result = env.step({"air_0": N, "gnd_0": E, "gnd_1": S})
    keys = [(phase[e.kind], getattr(e, "agent_id", "")) for e in result.events]
    assert keys == sorted(keys)
    assert {"invalid_action", "moved", "energy_spent", "agent_disabled"} <= set(
        kinds(result.events)
    )

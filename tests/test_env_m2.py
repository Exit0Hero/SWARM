"""Phase 1 M2 environment behaviour: hazards, false positives, mission success (ADR-0009/11/12)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from swarm.core.env import StepResult, SwarmEnv
from swarm.core.events import (
    AgentDisabled,
    Detection,
    EnergySpent,
    EpisodeEnd,
    Event,
    HazardExposure,
    MoveConflict,
)
from swarm.core.observation import Observation
from swarm.core.types import Action, AgentStatus, Layer
from swarm.scenarios.config import AgentSpec, HazardConfig
from tests.helpers import config_for, spec

S, N, E, SO, W = (int(a) for a in Action)

AIR = spec("air", Layer.AIR, capacity=20, move=1, stay=0)
GND = spec("gnd", Layer.GROUND, capacity=20, move=1, stay=0)


def hz(
    *, cost: int = 2, p: float = 0.0, layers: tuple[Layer, ...] = (Layer.GROUND,)
) -> HazardConfig:
    return HazardConfig(
        fraction=0.0, affected_layers=layers, energy_cost=cost, disable_probability=p
    )


def make_env(
    rows: list[str], *roster: AgentSpec, seed: int = 0, types: Any = (AIR, GND), **kw: Any
) -> SwarmEnv:
    config, scenario = config_for(rows, types, tuple(roster), **kw)
    env = SwarmEnv(config)
    env.reset(seed, scenario=scenario)
    return env


def of(result: StepResult, kind: type[Event]) -> list[Event]:
    return [e for e in result.events if isinstance(e, kind)]


# ================================================================= hazards (ADR-0011)


def test_entering_a_hazard_costs_energy_and_emits_exposure() -> None:
    env = make_env(["B.."], AgentSpec("gnd"), hazard_cells=((0, 1),), hazards=hz(cost=3))
    result = env.step({"gnd_0": E})
    assert result.events[-2:] == (
        EnergySpent(1, "gnd_0", 1, 19),
        HazardExposure(1, "gnd_0", (0, 1), 3, 16),
    )


def test_staying_in_a_hazard_is_exposure_every_step() -> None:
    env = make_env(["B.."], AgentSpec("gnd"), hazard_cells=((0, 1),), hazards=hz(cost=2))
    env.step({"gnd_0": E})
    for t in (2, 3):
        result = env.step({"gnd_0": S})
        assert of(result, HazardExposure) == [HazardExposure(t, "gnd_0", (0, 1), 2, 20 - 1 - 2 * t)]


def test_leaving_a_hazard_ends_exposure() -> None:
    env = make_env(["B.."], AgentSpec("gnd"), hazard_cells=((0, 1),), hazards=hz())
    env.step({"gnd_0": E})
    assert of(env.step({"gnd_0": E}), HazardExposure) == []


def test_unaffected_layer_ignores_hazards() -> None:
    env = make_env(
        ["B.."], AgentSpec("air"), AgentSpec("gnd"), hazard_cells=((0, 1),), hazards=hz()
    )
    result = env.step({"air_0": E, "gnd_0": E})
    assert [e.agent_id for e in of(result, HazardExposure)] == ["gnd_0"]  # type: ignore[attr-defined]


def test_both_layers_when_configured() -> None:
    env = make_env(
        ["B.."],
        AgentSpec("air"),
        AgentSpec("gnd"),
        hazard_cells=((0, 1),),
        hazards=hz(layers=(Layer.AIR, Layer.GROUND)),
    )
    result = env.step({"air_0": E, "gnd_0": E})
    assert [e.agent_id for e in of(result, HazardExposure)] == ["air_0", "gnd_0"]  # type: ignore[attr-defined]


def test_zero_cost_hazard_still_records_exposure() -> None:
    env = make_env(["B."], AgentSpec("gnd"), hazard_cells=((0, 1),), hazards=hz(cost=0))
    assert of(env.step({"gnd_0": E}), HazardExposure) == [HazardExposure(1, "gnd_0", (0, 1), 0, 19)]


def test_hazard_damage_is_capped_and_depletion_takes_precedence_over_the_draw() -> None:
    env = make_env(
        ["B.."],
        AgentSpec("gnd", initial_energy=3),
        hazard_cells=((0, 1),),
        hazards=hz(cost=5, p=1.0),
    )
    result = env.step({"gnd_0": E})
    assert of(result, HazardExposure) == [HazardExposure(1, "gnd_0", (0, 1), 2, 0)]
    assert of(result, AgentDisabled) == [AgentDisabled(1, "gnd_0", (0, 1), "energy_depleted")]
    assert result.observations["gnd_0"].energy == 0


def test_disable_probability_one_disables_with_reason_hazard() -> None:
    env = make_env(["B.."], AgentSpec("gnd"), hazard_cells=((0, 1),), hazards=hz(p=1.0))
    result = env.step({"gnd_0": E})
    assert of(result, AgentDisabled) == [AgentDisabled(1, "gnd_0", (0, 1), "hazard")]
    assert result.terminations == {"gnd_0": True}
    assert result.observations["gnd_0"].status is AgentStatus.DISABLED
    assert result.observations["gnd_0"].energy > 0  # destroyed, not depleted


def test_disable_probability_zero_never_disables() -> None:
    env = make_env(
        ["B.."], AgentSpec("gnd"), hazard_cells=((0, 1),), hazards=hz(cost=0), time_limit=500
    )
    env.step({"gnd_0": E})
    for _ in range(498):
        assert of(env.step({"gnd_0": S}), AgentDisabled) == []


def test_disable_probability_statistics_over_seeds() -> None:
    disabled = 0
    for seed in range(600):
        env = make_env(
            ["B."], AgentSpec("gnd"), seed=seed, hazard_cells=((0, 1),), hazards=hz(p=0.25)
        )
        disabled += bool(of(env.step({"gnd_0": E}), AgentDisabled))
    # Binomial(600, 0.25): mean 150, sd ~10.6; bound > 4.5 sd; fixed seeds (not flaky).
    assert abs(disabled - 150) < 48


def test_agent_disabled_by_energy_in_phase_3_is_not_exposed() -> None:
    env = make_env(
        ["B.."], AgentSpec("gnd", initial_energy=1), hazard_cells=((0, 1),), hazards=hz(p=1.0)
    )
    result = env.step({"gnd_0": E})
    assert of(result, HazardExposure) == []
    assert of(result, AgentDisabled) == [AgentDisabled(1, "gnd_0", (0, 1), "energy_depleted")]


def test_hazard_disabled_agent_blocks_its_cell_and_does_not_sense() -> None:
    env = make_env(
        ["BS..", "...S"],  # (1, 3) is never in range: success cannot end the episode
        AgentSpec("gnd"),
        AgentSpec("gnd"),
        hazard_cells=((1, 0),),
        hazards=hz(p=1.0),
        p_detect=1.0,
    )
    first = env.step({"gnd_0": SO, "gnd_1": S})
    # gnd_0 is disabled in phase 4 and does not report the survivor; gnd_1 on the base does.
    assert [e.agent_id for e in of(first, Detection)] == ["gnd_1"]  # type: ignore[attr-defined]
    blocked = env.step({"gnd_1": SO})
    assert MoveConflict(2, "gnd_1", (0, 0), (1, 0), "blocked") in blocked.events


def test_hazard_event_order_within_a_step() -> None:
    env = make_env(
        [".S..S", ".B..."],  # (0, 4) is never in range: success cannot end the episode
        AgentSpec("gnd"),
        hazard_cells=((0, 0),),
        hazards=hz(cost=1, p=0.0),
        p_detect=1.0,
    )
    env.step({"gnd_0": W})  # (1, 0)
    result = env.step({"gnd_0": N})  # into the hazard at (0, 0), next to the survivor
    assert [e.kind for e in result.events] == [
        "moved",
        "energy_spent",
        "hazard_exposure",
        "detection",
    ]


def test_hazards_ignored_when_config_has_no_hazard_layer() -> None:
    env = make_env(["B."], AgentSpec("gnd"), hazard_cells=((0, 1),), hazards=None)
    assert of(env.step({"gnd_0": E}), HazardExposure) == []


def test_hazards_visible_only_inside_the_footprint() -> None:
    env = make_env(["B..", "..."], AgentSpec("gnd"), hazard_cells=((0, 1), (1, 1)), hazards=hz())
    obs = env.step({"gnd_0": S}).observations["gnd_0"]
    # Radius 1 at (0, 0): (0, 1) is in the footprint; diagonal (1, 1) is not.
    assert obs.local_hazard.tolist() == [
        [False, False, False],
        [False, False, True],
        [False, False, False],
    ]
    with pytest.raises(ValueError, match="read-only"):
        obs.local_hazard[0, 0] = True


def test_hazards_do_not_change_the_action_mask() -> None:
    plain = make_env(["B.."], AgentSpec("gnd"))
    hazardous = make_env(["B.."], AgentSpec("gnd"), hazard_cells=((0, 1),), hazards=hz())
    assert plain.action_mask("gnd_0") == hazardous.action_mask("gnd_0")


# ================================================================= false positives (ADR-0012)


def test_false_positive_probability_one_reports_every_empty_footprint_cell() -> None:
    env = make_env(["...", ".BS", "..."], AgentSpec("gnd"), p_false_positive=1.0, p_detect=1.0)
    result = env.step({"gnd_0": S})
    detections = of(result, Detection)
    assert detections[0] == Detection(1, "gnd_0", (1, 2), True, 0)
    assert detections[1:] == [
        Detection(1, "gnd_0", cell, False, None) for cell in [(0, 1), (1, 0), (1, 1), (2, 1)]
    ]
    assert result.observations["gnd_0"].detections == ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))


def test_false_positives_never_land_on_survivor_cells() -> None:
    env = make_env(
        ["SSS", "SBS", "SSS"],
        AgentSpec("air"),
        types=(spec("air", Layer.AIR, radius=2),),
        p_detect=0.0,
        p_false_positive=1.0,
        time_limit=5,
    )
    result = env.step({"air_0": S})
    assert [e.cell for e in of(result, Detection)] == [(1, 1)]  # type: ignore[attr-defined]


def test_false_positive_rate_statistics() -> None:
    env = make_env(
        ["....", ".B..", "...."], AgentSpec("gnd"), p_false_positive=0.1, time_limit=3000
    )
    count = 0
    for _ in range(3000):
        count += len(of(env.step({"gnd_0": S}), Detection))
    # 5 footprint cells x 3000 steps x 0.1 = 1500 expected; sd ~ 36.7; bound > 4 sd.
    assert abs(count - 1500) < 150


def test_observations_cannot_distinguish_true_from_false_reports() -> None:
    """Same reported cells, different truth: the observations are identical."""
    truthful = make_env(
        [".....", ".BS..", "....S"], AgentSpec("gnd"), p_detect=1.0, p_false_positive=1.0
    )
    false_alarm = make_env(
        [".....", ".B...", "...SS"], AgentSpec("gnd"), p_detect=1.0, p_false_positive=1.0
    )
    a = truthful.step({"gnd_0": S}).observations["gnd_0"]
    b = false_alarm.step({"gnd_0": S}).observations["gnd_0"]
    assert (1, 2) in a.detections
    assert a == b
    assert not {f for f in Observation.__dataclass_fields__ if "true" in f or "confidence" in f}


def test_false_positive_sweep_keeps_true_detection_draws() -> None:
    """Common random numbers (ADR-0013): changing p_false_positive leaves true reports intact."""
    rows = [".....", ".SBS.", "....."]

    def true_reports(p_fp: float) -> list[Event]:
        env = make_env(rows, AgentSpec("gnd"), p_detect=0.5, p_false_positive=p_fp, time_limit=60)
        out: list[Event] = []
        for _ in range(60):
            out.extend(e for e in env.step({"gnd_0": S}).events if isinstance(e, Detection))
            if env.done:
                break
        return [e for e in out if e.true_positive]  # type: ignore[attr-defined]

    baseline = true_reports(0.0)
    assert baseline  # the comparison is not vacuous
    assert true_reports(0.4) == baseline[: len(true_reports(0.4))]


# ================================================================= mission success (ADR-0009)


def test_mission_success_ends_the_episode_after_sensing() -> None:
    env = make_env(["BS"], AgentSpec("gnd"), AgentSpec("air"), p_detect=1.0)
    result = env.step({"gnd_0": S, "air_0": S})
    assert result.events[-1] == EpisodeEnd(1, "mission_success", True, False)
    assert result.terminations == {"air_0": True, "gnd_0": True}
    assert result.truncations == {"air_0": False, "gnd_0": False}
    assert env.done
    with pytest.raises(RuntimeError, match="mission_success"):
        env.step({"gnd_0": S, "air_0": S})


def test_partial_recall_target() -> None:
    env = make_env(["BS...S"], AgentSpec("gnd"), p_detect=1.0, recall_target=0.5)
    assert env.step({"gnd_0": S}).events[-1] == EpisodeEnd(1, "mission_success", True, False)


def test_recall_target_compared_exactly() -> None:
    # 10 survivors, 3 in range of a radius-1 robot at the base; 3/10 must meet a 0.3 target even
    # though 0.3 * 10 == 3.0000000000000004 in floating point.
    rows = [".S.......", "SBS......", ".........", "SSSSSSS.."]
    env = make_env(rows, AgentSpec("gnd"), p_detect=1.0, recall_target=0.3)
    result = env.step({"gnd_0": S})
    assert len(of(result, Detection)) == 3
    assert result.events[-1] == EpisodeEnd(1, "mission_success", True, False)


def test_success_needs_the_target_not_just_progress() -> None:
    env = make_env(["BS...S"], AgentSpec("gnd"), p_detect=1.0, recall_target=0.51, time_limit=3)
    for _ in range(3):
        result = env.step({"gnd_0": S})
    assert result.events[-1] == EpisodeEnd(3, "time_limit", False, True)


def test_unreachable_survivor_makes_full_recall_impossible() -> None:
    env = make_env(["B.#S"], AgentSpec("gnd"), p_detect=1.0, time_limit=6)
    for _ in range(6):
        actions = {"gnd_0": E if env.snapshot().agents[0].position == (0, 0) else S}
        result = env.step(actions)
    assert result.events[-1] == EpisodeEnd(6, "time_limit", False, True)


def test_false_positives_never_count_toward_success() -> None:
    env = make_env(
        ["B....", "....S"], AgentSpec("gnd"), p_detect=1.0, p_false_positive=1.0, time_limit=2
    )
    env.step({"gnd_0": S})
    assert env.step({"gnd_0": S}).events[-1] == EpisodeEnd(2, "time_limit", False, True)


def test_zero_survivors_never_succeed() -> None:
    env = make_env(["B.."], AgentSpec("gnd"), recall_target=0.1, time_limit=2)
    env.step({"gnd_0": S})
    assert env.step({"gnd_0": S}).events[-1] == EpisodeEnd(2, "time_limit", False, True)


def test_success_takes_precedence_over_time_limit() -> None:
    env = make_env(["BS"], AgentSpec("gnd"), p_detect=1.0, time_limit=1)
    result = env.step({"gnd_0": S})
    assert result.events[-1] == EpisodeEnd(1, "mission_success", True, False)
    assert result.truncations == {"gnd_0": False}


def test_success_while_another_agent_is_disabled_terminates_both() -> None:
    env = make_env(
        ["B...", "S..."], AgentSpec("gnd"), AgentSpec("air", initial_energy=1), p_detect=1.0
    )
    result = env.step({"gnd_0": S, "air_0": E})
    assert AgentDisabled(1, "air_0", (0, 1), "energy_depleted") in result.events
    assert result.events[-1] == EpisodeEnd(1, "mission_success", True, False)
    assert result.terminations == {"air_0": True, "gnd_0": True}


def test_all_disabled_takes_precedence_over_time_limit() -> None:
    env = make_env(["B..S"], AgentSpec("gnd", initial_energy=1), time_limit=1)
    result = env.step({"gnd_0": E})
    assert result.events[-1] == EpisodeEnd(1, "all_disabled", True, False)
    assert result.truncations == {"gnd_0": False}


def test_m2_episode_is_deterministic() -> None:
    def trace(seed: int) -> list[Event]:
        config, scenario = config_for(
            ["......", ".S..S.", "..B...", "S....."],
            (AIR, GND),
            (AgentSpec("air"), AgentSpec("gnd"), AgentSpec("gnd")),
            hazard_cells=((1, 2), (2, 3), (3, 3)),
            hazards=hz(cost=1, p=0.3, layers=(Layer.AIR, Layer.GROUND)),
            p_detect=0.5,
            p_false_positive=0.05,
            time_limit=40,
        )
        env = SwarmEnv(config)
        observations, _ = env.reset(seed, scenario=scenario)
        rng = np.random.default_rng(seed)
        events: list[Event] = []
        while not env.done:
            acts = {}
            for aid in env.agents:
                valid = [a for a, ok in enumerate(observations[aid].action_mask) if ok]
                acts[aid] = valid[int(rng.integers(len(valid)))]
            result = env.step(acts)
            observations = result.observations
            events.extend(result.events)
        return events

    assert trace(3) == trace(3)
    assert trace(3) != trace(4)

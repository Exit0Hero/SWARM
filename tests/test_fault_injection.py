"""M3.5: scheduled failures, communication partitions and traffic metrics (ADR-0021)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from swarm.core.env import SwarmEnv
from swarm.core.events import (
    AgentDisabled,
    EpisodeEnd,
    Event,
    MessageDelivered,
    MessageDropped,
    MessageSent,
    Moved,
    event_stream_sha256,
)
from swarm.core.messages import Message
from swarm.core.types import Action, AgentId, Layer
from swarm.evaluation.comms_metrics import compute_comm_metrics
from swarm.experiments.runner import run_episode
from swarm.policies.registry import POLICIES
from swarm.scenarios.config import (
    AgentSpec,
    CommsConfig,
    ConfigError,
    FailureConfig,
    ScenarioConfig,
)
from swarm.scenarios.generator import generate_scenario
from tests.helpers import config_for, make_config, spec

GND = spec("gnd", Layer.GROUND, radius=1, capacity=100, move=1, stay=0)
STAY = int(Action.STAY)
COMMS = CommsConfig(ranges=(("drone", 8), ("ground", 5)))


def _trio(**comms: Any) -> SwarmEnv:
    """Three ground robots on one cell row, all within range of each other."""
    config, scenario = config_for(
        ["B....."], (GND,), (AgentSpec("gnd"), AgentSpec("gnd"), AgentSpec("gnd")), time_limit=30
    )
    env = SwarmEnv(replace(config, comms=CommsConfig(**({"ranges": {"gnd": 10}} | comms))))
    env.reset(0, scenario=scenario)
    return env


def _send_all(env: SwarmEnv, step: int) -> list[Event]:
    outbox = {aid: [Message(aid, step, "t", ())] for aid in env.agents}
    return list(env.step({aid: STAY for aid in env.agents}, outbox).events)


def _recipients(events: list[Event]) -> dict[AgentId, tuple[AgentId, ...]]:
    return {e.sender: e.recipients for e in events if isinstance(e, MessageSent)}


# ----------------------------------------------------------------- configuration


def test_failures_and_partitions_round_trip_and_leave_old_hashes_unchanged() -> None:
    base = replace(make_config(drones=1, grounds=2), comms=COMMS)
    assert "failures" not in base.to_dict()
    assert "partitions" not in base.to_dict()["comms"]
    config = replace(
        base,
        failures=FailureConfig(count=2, start=5, end=9),
        comms=replace(COMMS, partitions=((3, 6, ("ground_1", "drone_0")),)),
    )
    again = ScenarioConfig.from_json(config.to_json())
    assert again == config
    assert again.comms is not None
    assert again.comms.partitions == ((3, 6, ("drone_0", "ground_1")),)  # sorted
    assert again.config_hash() != base.config_hash()
    plain = make_config(drones=1, grounds=2)
    with_failures = replace(plain, failures=FailureConfig(1, 2, 3))
    assert with_failures.to_dict()["schema_version"] == 3
    assert with_failures.to_dict()["comms"] is None
    assert ScenarioConfig.from_json(with_failures.to_json()) == with_failures


@pytest.mark.parametrize(
    ("failures", "message"),
    [
        (FailureConfig(4, 1, 2), "failures.count"),
        (FailureConfig(1, 2, 999), "failures.end"),
    ],
)
def test_failure_config_is_validated_against_the_scenario(
    failures: FailureConfig, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        replace(make_config(drones=1, grounds=2), failures=failures)


def test_failure_window_must_be_ordered() -> None:
    with pytest.raises(ConfigError, match=r"failures\.end"):
        FailureConfig(1, 5, 4)


@pytest.mark.parametrize(
    ("window", "message"),
    [
        ((5, 5, ("drone_0",)), "end"),
        ((0, 5, ("drone_0",)), "start"),
        ((1, 5, ()), "group"),
        ((1, 5, ("drone_0", "drone_0")), "duplicate"),
    ],
)
def test_partition_windows_are_validated(window: Any, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        CommsConfig(ranges=(("drone", 8),), partitions=(window,))


def test_partition_groups_must_name_real_agents() -> None:
    comms = replace(COMMS, partitions=((1, 5, ("drone_7",)),))
    with pytest.raises(ConfigError, match="unknown agent IDs"):
        replace(make_config(drones=1, grounds=2), comms=comms)


# ----------------------------------------------------------------- scheduled failures


def test_injected_failures_disable_the_drawn_agents_at_their_steps() -> None:
    config = replace(
        make_config(drones=2, grounds=2, time_limit=60), failures=FailureConfig(2, 5, 9)
    )
    result = run_episode(config, POLICIES["frontier"], 3, designation="development")
    injected = [e for e in result.events if isinstance(e, AgentDisabled) and e.reason == "injected"]
    assert len(injected) == 2
    assert len({e.agent_id for e in injected}) == 2
    assert all(5 <= e.step <= 9 for e in injected)
    again = run_episode(config, POLICIES["frontier"], 3, designation="development")
    assert again.event_stream_sha256 == result.event_stream_sha256


def test_injection_does_not_change_the_episode_before_the_first_failure() -> None:
    plain = make_config(drones=2, grounds=2, time_limit=60)
    failing = replace(plain, failures=FailureConfig(1, 3, 3))
    a = run_episode(plain, POLICIES["frontier"], 5, designation="development").events
    b = run_episode(failing, POLICIES["frontier"], 5, designation="development").events
    assert [e for e in a if e.step < 3] == [e for e in b if e.step < 3]
    assert any(isinstance(e, AgentDisabled) and e.step == 3 for e in b)
    # the scenario stream is untouched by the failure stream
    s1 = generate_scenario(plain, _streams(plain, 5))
    s2 = generate_scenario(failing, _streams(failing, 5))
    assert (s1.survivors, s1.terrain.tobytes()) == (s2.survivors, s2.terrain.tobytes())


def _streams(config: ScenarioConfig, seed: int) -> Any:
    from swarm.core.rng import episode_streams

    return episode_streams(seed, tuple(config.agents())).scenario


def test_zero_failures_inject_nothing_and_match_no_failure_block() -> None:
    plain = make_config(drones=2, grounds=2, time_limit=40)
    zero = replace(plain, failures=FailureConfig(0, 1, 2))
    a = run_episode(plain, POLICIES["frontier"], 1, designation="development")
    b = run_episode(zero, POLICIES["frontier"], 1, designation="development")
    assert event_stream_sha256(a.events) == event_stream_sha256(b.events)


def test_an_agent_already_disabled_is_not_disabled_twice() -> None:
    config, scenario = config_for(["B..."], (GND,), (AgentSpec("gnd", initial_energy=1),))
    config = replace(config, failures=FailureConfig(1, 1, 1), time_limit=5)
    env = SwarmEnv(config)
    env.reset(0, scenario=scenario)
    events = env.step({"gnd_0": int(Action.EAST)}).events
    causes = [e.reason for e in events if isinstance(e, AgentDisabled)]
    assert causes == ["energy_depleted"]


# ----------------------------------------------------------------- partitions


def test_partition_cuts_links_across_the_group_then_recovers() -> None:
    env = _trio(partitions=((2, 4, ("gnd_0",)),))
    assert _recipients(_send_all(env, 1))["gnd_0"] == ("gnd_1", "gnd_2")  # step 1: before
    during = _recipients(_send_all(env, 2))  # step 2: gnd_0 isolated
    assert during == {"gnd_0": (), "gnd_1": ("gnd_2",), "gnd_2": ("gnd_1",)}
    assert _recipients(_send_all(env, 3))["gnd_1"] == ("gnd_2",)
    assert _recipients(_send_all(env, 4))["gnd_0"] == ("gnd_1", "gnd_2")  # recovered at end


def test_partition_keeps_links_inside_a_group() -> None:
    env = _trio(partitions=((1, 3, ("gnd_0", "gnd_1")),))
    assert _recipients(_send_all(env, 1)) == {
        "gnd_0": ("gnd_1",),
        "gnd_1": ("gnd_0",),
        "gnd_2": (),
    }


def test_messages_in_flight_before_a_partition_are_still_delivered() -> None:
    env = _trio(latency=2, partitions=((2, 9, ("gnd_0",)),))
    _send_all(env, 1)
    _send_all(env, 2)
    events = _send_all(env, 3)
    delivered = {(e.sender, e.receiver) for e in events if isinstance(e, MessageDelivered)}
    assert ("gnd_0", "gnd_1") in delivered


# ----------------------------------------------------------------- traffic metrics


def test_traffic_metrics_count_overhead_and_delivery_rate() -> None:
    config = replace(make_config(drones=1, grounds=2), comms=COMMS)
    scenario = generate_scenario(config, _streams(config, 0))
    events: list[Event] = [
        MessageSent(1, "drone_0", 1, "k", 4, ("ground_0", "ground_1")),
        MessageSent(1, "ground_0", 1, "k", 3, ()),
        MessageDelivered(1, "ground_0", "drone_0", 1, 1, 0),
        MessageDropped(1, "ground_1", "drone_0", 1, "random"),
        EpisodeEnd(10, "time_limit", False, True),
    ]
    m = compute_comm_metrics(config, scenario, events)
    assert (m["messages_sent"], m["message_units_sent"], m["messages_unaddressed"]) == (2, 7, 1)
    assert (m["links_addressed"], m["links_delivered"], m["links_dropped_random"]) == (2, 1, 1)
    assert m["message_delivery_rate"] == 0.5
    assert m["message_units_per_step"] == 0.7
    empty = compute_comm_metrics(config, scenario, [EpisodeEnd(3, "time_limit", False, True)])
    assert empty["message_delivery_rate"] is None
    assert empty["messages_sent"] == 0


def test_the_mission_brief_hides_partition_schedules() -> None:
    from swarm.policies.base import MissionBrief

    comms = replace(COMMS, partitions=((1, 5, ("drone_0",)),))
    config = replace(make_config(drones=1, grounds=2), comms=comms, failures=FailureConfig(1, 2, 3))
    brief = MissionBrief.from_config(config)
    assert brief.comms is not None
    assert brief.comms.partitions == ()
    assert brief.comms.ranges == COMMS.ranges
    assert "failure" not in repr(brief).lower()


# ----------------------------------------------------------------- replay viewer (Phase 3)


def test_replay_state_follows_moves_and_records_injected_disablement() -> None:
    from swarm.core.rng import episode_streams
    from swarm.visualization.replay import replay_state

    plain = make_config(drones=1, grounds=1, time_limit=40)
    config = replace(plain, failures=FailureConfig(1, 4, 4))
    result = run_episode(config, POLICIES["frontier"], 2, designation="development")
    agents = tuple(config.agents())
    scenario = generate_scenario(config, episode_streams(2, agents).scenario)
    full = replay_state(scenario, agents, result.events, until=None)
    early = replay_state(scenario, agents, result.events, until=2)
    for aid, trace in full.items():
        moves = [e for e in result.events if isinstance(e, Moved) and e.agent_id == aid]
        assert trace.path == [scenario.base] + [e.to_position for e in moves]
        assert len(early[aid].path) <= 3
    disabled = [t.disabled_at for t in full.values() if t.disabled_at is not None]
    assert (4, "injected") in {(step, reason) for step, _, reason in disabled}
    assert all(t.disabled_at is None for t in early.values())


def test_replay_renders_a_figure(tmp_path: Any) -> None:
    pytest.importorskip("matplotlib")
    from swarm.visualization.replay import main

    out = tmp_path / "replay.png"
    config = tmp_path / "c.json"
    config.write_text(make_config(drones=1, grounds=1, time_limit=20).to_json())
    assert (
        main(["--config", str(config), "--policy", "frontier", "--seed", "1", "--out", str(out)])
        == 0
    )
    assert out.stat().st_size > 1000

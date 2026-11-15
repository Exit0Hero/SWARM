"""Phase 3 M3.1: the communication channel (ADR-0017)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from swarm.core.comms import Channel
from swarm.core.env import SwarmEnv
from swarm.core.events import (
    Event,
    MessageDelivered,
    MessageDropped,
    MessageSent,
    event_stream_sha256,
)
from swarm.core.messages import HEADER_UNITS, Delivery, Message
from swarm.core.observation import Observation
from swarm.core.oracle import OracleView
from swarm.core.rng import episode_streams
from swarm.core.types import Action, AgentId, Layer
from swarm.experiments.runner import run_episode
from swarm.policies.base import (
    CommunicatingTeam,
    Diagnostics,
    MissionBrief,
    TeamFactory,
    TeamPolicy,
)
from swarm.policies.registry import POLICIES
from swarm.scenarios.config import AgentSpec, CommsConfig, ConfigError, ScenarioConfig
from tests.helpers import config_for, make_config, spec

ROOT = Path(__file__).resolve().parents[1]
GND = spec("gnd", Layer.GROUND, radius=1, capacity=100, move=1, stay=0)
STAY, EAST = int(Action.STAY), int(Action.EAST)


def _pair(
    *,
    rows: tuple[str, ...] = ("B.....",),
    energies: tuple[int | None, int | None] = (None, None),
    **comms: Any,
) -> tuple[SwarmEnv, dict[AgentId, Observation]]:
    """Two ground robots gnd_0 and gnd_1 on a one-row map, with a channel."""
    config, scenario = config_for(
        list(rows),
        (GND,),
        (
            AgentSpec("gnd", initial_energy=energies[0]),
            AgentSpec("gnd", initial_energy=energies[1]),
        ),
        time_limit=30,
    )
    settings: dict[str, Any] = {"ranges": {"gnd": 10}} | comms
    env = SwarmEnv(replace(config, comms=CommsConfig(**settings)))
    obs, _ = env.reset(0, scenario=scenario)
    return env, obs


def _msg(sender: AgentId, seq: int, *records: tuple[Any, ...]) -> Message:
    return Message(sender, seq, "test", tuple(records))


# ----------------------------------------------------------------- configuration


def test_comms_config_round_trips_and_writes_version_3() -> None:
    config = replace(
        make_config(),
        comms=CommsConfig(
            ranges=(("drone", 8), ("ground", 5)), p_drop=0.2, latency=1, jitter=1, bandwidth=96
        ),
    )
    data = json.loads(config.to_json())
    assert data["schema_version"] == 3
    assert data["comms"]["ranges"] == {"drone": 8, "ground": 5}
    restored = ScenarioConfig.from_json(config.to_json())
    assert restored == config
    assert restored.config_hash() == config.config_hash()
    assert config.config_hash() != make_config().config_hash()


@pytest.mark.parametrize(
    ("ranges", "kwargs", "match"),
    [
        ({"drone": 8}, {}, "exactly the agent types"),
        ({"drone": 8, "ground": 5, "boat": 3}, {}, "exactly the agent types"),
        ({"drone": -1, "ground": 5}, {}, r"ranges\.drone"),
        ({"drone": 8, "ground": 5}, {"p_drop": 1.5}, "p_drop"),
        ({"drone": 8, "ground": 5}, {"latency": -1}, "latency"),
        ({"drone": 8, "ground": 5}, {"bandwidth": 0}, "bandwidth"),
    ],
)
def test_comms_config_rejects_bad_values(
    ranges: dict[str, int], kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(ConfigError, match=match):
        replace(make_config(), comms=CommsConfig(ranges=tuple(ranges.items()), **kwargs))


def test_version_3_requires_the_comms_key() -> None:
    data = json.loads(make_config().to_json()) | {"schema_version": 3}
    with pytest.raises(ConfigError, match="comms"):
        ScenarioConfig.from_dict(data)


@pytest.mark.parametrize(
    ("name", "prefix"), [("m1_default", "fec0dca1c6f801fc"), ("m2_default", "8db09395238eb122")]
)
def test_existing_config_hashes_are_unchanged(name: str, prefix: str) -> None:
    config = ScenarioConfig.from_json((ROOT / "configs" / f"{name}.json").read_text("utf-8"))
    assert config.comms is None
    assert config.config_hash().startswith(prefix)
    assert json.loads(config.to_json())["schema_version"] == 2


# ----------------------------------------------------------------- reproducibility of Phase 1/2


def test_existing_rng_streams_are_unchanged() -> None:
    streams = episode_streams(7, ["a", "b"])
    _, dynamics, _ = np.random.SeedSequence(7).spawn(3)
    hazard_ref, fp_ref = dynamics.spawn(2)
    assert np.array_equal(streams.hazard.random(8), np.random.default_rng(hazard_ref).random(8))
    assert np.array_equal(streams.false_positive.random(8), np.random.default_rng(fp_ref).random(8))


@pytest.mark.parametrize(
    ("config_name", "policy", "seed"),
    [("m2", "frontier", 100), ("m1", "assignment", 128), ("m2", "assignment_oracle", 149)],
)
def test_phase2_validation_episodes_reproduce_exactly(
    config_name: str, policy: str, seed: int
) -> None:
    archive = ROOT / "docs" / "results" / "phase2_validation" / config_name / f"{policy}.json"
    record = json.loads(archive.read_text("utf-8"))
    episode = next(e for e in record["episodes"] if e["seed"] == seed)
    config = ScenarioConfig.from_json(
        (ROOT / "configs" / f"{config_name}_default.json").read_text("utf-8")
    )
    assert config.config_hash() == record["config_hash"]
    result = run_episode(config, POLICIES[policy], seed)
    assert result.event_stream_sha256 == episode["event_stream_sha256"]


def test_channel_without_messages_leaves_the_episode_unchanged() -> None:
    plain = make_config()
    channel = CommsConfig(ranges=(("drone", 8), ("ground", 5)), p_drop=0.5)
    with_channel = replace(plain, comms=channel)
    streams = []
    for config in (plain, with_channel):
        env = SwarmEnv(config)
        obs, _ = env.reset(3)
        rng = np.random.default_rng(0)
        events: list[Event] = []
        while not env.done:
            actions = {a: int(rng.choice(np.flatnonzero(obs[a].action_mask))) for a in env.agents}
            result = env.step(actions)
            events.extend(result.events)
            obs = result.observations
            assert all(o.inbox == () for o in obs.values())
        streams.append(event_stream_sha256(events))
    assert streams[0] == streams[1]


# ----------------------------------------------------------------- outbox validation


def test_messages_without_a_channel_are_rejected() -> None:
    config, scenario = config_for(["B..."], (GND,), (AgentSpec("gnd"),))
    env = SwarmEnv(config)
    env.reset(0, scenario=scenario)
    with pytest.raises(ValueError, match="no comms block"):
        env.step({"gnd_0": STAY}, {"gnd_0": [_msg("gnd_0", 0)]})
    assert env.step_count == 0


@pytest.mark.parametrize(
    ("outbox", "error", "match"),
    [
        ({"gnd_0": [_msg("gnd_1", 0)]}, ValueError, "does not match"),
        ({"gnd_0": [_msg("gnd_0", 0, (1.5,))]}, TypeError, "int, str or bool"),
        ({"gnd_0": [_msg("gnd_0", 0, (np.int64(3),))]}, TypeError, "int, str or bool"),
        ({"gnd_0": [Message("gnd_0", 0, "x", ([1, 2],))]}, TypeError, "must be a tuple"),  # type: ignore[arg-type]
        ({"gnd_0": [_msg("gnd_0", 0, tuple(range(17)))]}, ValueError, "at most 16"),
        ({"gnd_0": [_msg("gnd_0", 0, ("x" * 65,))]}, ValueError, "64 characters"),
        ({"gnd_0": [Message("gnd_0", 0, "")]}, ValueError, "kind"),
        ({"gnd_0": [_msg("gnd_0", 1), _msg("gnd_0", 1)]}, ValueError, "strictly increase"),
        ({"gnd_0": [_msg("gnd_0", 0, *[(k,) for k in range(20)])]}, ValueError, "bandwidth"),
        ({"ghost": [_msg("ghost", 0)]}, ValueError, "only active agents"),
        ({"gnd_0": "not a list"}, TypeError, "sequence of Message"),
        ({"gnd_0": [object()]}, TypeError, "expected Message"),
    ],
)
def test_bad_outboxes_raise_before_any_state_changes(
    outbox: Mapping[AgentId, Sequence[Message]], error: type[Exception], match: str
) -> None:
    env, _ = _pair(bandwidth=HEADER_UNITS + 10)
    with pytest.raises(error, match=match):
        env.step({"gnd_0": STAY, "gnd_1": STAY}, outbox)
    assert env.step_count == 0


def test_seq_must_increase_across_steps() -> None:
    env, _ = _pair()
    env.step({"gnd_0": STAY, "gnd_1": STAY}, {"gnd_0": [_msg("gnd_0", 5)]})
    with pytest.raises(ValueError, match="strictly increase"):
        env.step({"gnd_0": STAY, "gnd_1": STAY}, {"gnd_0": [_msg("gnd_0", 5)]})
    assert env.step_count == 1


# ----------------------------------------------------------------- link, drop and delay model


def test_link_is_symmetric_euclidean_and_uses_the_smaller_range() -> None:
    ch = Channel(CommsConfig(ranges=(("t", 1),)), {"a": 3, "b": 5, "c": None, "d": None})
    assert ch.in_range("a", (0, 0), "b", (0, 3))
    assert ch.in_range("b", (0, 3), "a", (0, 0))
    assert not ch.in_range("a", (0, 0), "b", (0, 4))
    assert ch.in_range("a", (0, 0), "b", (2, 2))  # 8 <= 9
    assert not ch.in_range("a", (0, 0), "b", (3, 1))  # 10 > 9
    assert ch.in_range("c", (0, 0), "b", (3, 4))
    assert not ch.in_range("c", (0, 0), "b", (4, 4))
    assert ch.in_range("c", (0, 0), "d", (500, 500))


def test_latency_delays_delivery_and_senders_do_not_hear_themselves() -> None:
    env, _ = _pair(latency=2)
    r1 = env.step({"gnd_0": STAY, "gnd_1": STAY}, {"gnd_0": [_msg("gnd_0", 0, (7,))]})
    sent = [e for e in r1.events if isinstance(e, MessageSent)]
    assert sent == [MessageSent(1, "gnd_0", 0, "test", HEADER_UNITS + 1, ("gnd_1",))]
    r2 = env.step({"gnd_0": STAY, "gnd_1": STAY})
    assert r2.observations["gnd_1"].inbox == ()
    r3 = env.step({"gnd_0": STAY, "gnd_1": STAY})
    assert r3.observations["gnd_1"].inbox == (Delivery(_msg("gnd_0", 0, (7,)), 1),)
    assert r3.observations["gnd_0"].inbox == ()
    assert [e for e in r3.events if isinstance(e, MessageDelivered)] == [
        MessageDelivered(3, "gnd_1", "gnd_0", 0, 1, 2)
    ]


def test_zero_latency_arrives_in_the_same_steps_observation() -> None:
    env, _ = _pair()
    r1 = env.step({"gnd_0": STAY, "gnd_1": STAY}, {"gnd_1": [_msg("gnd_1", 0)]})
    assert r1.observations["gnd_0"].inbox == (Delivery(_msg("gnd_1", 0), 1),)


@pytest.mark.parametrize(("p_drop", "delivered"), [(0.0, True), (1.0, False)])
def test_drop_probability_extremes(p_drop: float, delivered: bool) -> None:
    env, _ = _pair(p_drop=p_drop)
    r1 = env.step({"gnd_0": STAY, "gnd_1": STAY}, {"gnd_0": [_msg("gnd_0", 0)]})
    assert bool(r1.observations["gnd_1"].inbox) is delivered
    drops = [e for e in r1.events if isinstance(e, MessageDropped)]
    assert drops == ([] if delivered else [MessageDropped(1, "gnd_1", "gnd_0", 0, "random")])


def test_receivers_out_of_range_get_nothing() -> None:
    env, _ = _pair(ranges={"gnd": 1})
    r1 = env.step({"gnd_0": EAST, "gnd_1": STAY}, {"gnd_0": [_msg("gnd_0", 0)]})
    assert r1.observations["gnd_1"].inbox != ()  # distance 1
    r2 = env.step({"gnd_0": EAST, "gnd_1": STAY}, {"gnd_0": [_msg("gnd_0", 1)]})
    assert r2.observations["gnd_1"].inbox == ()  # distance 2 > range 1
    assert [e.recipients for e in r2.events if isinstance(e, MessageSent)] == [()]


def test_a_sender_disabled_during_the_step_reaches_no_one() -> None:
    env, _ = _pair(energies=(None, 1))
    r1 = env.step({"gnd_0": STAY, "gnd_1": EAST}, {"gnd_1": [_msg("gnd_1", 0)]})
    assert r1.terminations["gnd_1"]
    assert [e.recipients for e in r1.events if isinstance(e, MessageSent)] == [()]
    assert r1.observations["gnd_0"].inbox == ()


def test_a_receiver_disabled_before_delivery_loses_the_message() -> None:
    env, _ = _pair(energies=(None, 2), latency=2)
    env.step({"gnd_0": STAY, "gnd_1": EAST}, {"gnd_0": [_msg("gnd_0", 0)]})
    r2 = env.step({"gnd_0": STAY, "gnd_1": EAST})
    assert r2.terminations["gnd_1"]
    r3 = env.step({"gnd_0": STAY})
    assert [e for e in r3.events if isinstance(e, MessageDropped)] == [
        MessageDropped(3, "gnd_1", "gnd_0", 0, "receiver_disabled")
    ]


def test_jitter_draws_do_not_depend_on_the_drop_probability() -> None:
    """Two draws per in-range pair: delivery steps of surviving messages match across p_drop."""
    arrivals: list[dict[int, int]] = []
    for p_drop in (0.0, 0.5):
        env, _ = _pair(p_drop=p_drop, jitter=3)
        got: dict[int, int] = {}
        for seq in range(40):
            result = env.step({"gnd_0": STAY, "gnd_1": STAY}, {"gnd_0": [_msg("gnd_0", seq)]})
            for e in result.events:
                if isinstance(e, MessageDelivered):
                    got[e.seq] = e.step
            if env.done:
                break
        arrivals.append(got)
    full, lossy = arrivals
    assert lossy
    assert len(lossy) < len(full)
    assert all(full[seq] == step for seq, step in lossy.items())


def test_draw_count_does_not_depend_on_jitter() -> None:
    """Two uniforms per in-range pair for any jitter, so drop outcomes align across settings."""
    patterns: list[list[int]] = []
    for jitter in (0, 2):
        env, _ = _pair(p_drop=0.5, jitter=jitter)
        dropped: list[int] = []
        for seq in range(25):
            result = env.step({"gnd_0": STAY, "gnd_1": STAY}, {"gnd_0": [_msg("gnd_0", seq)]})
            dropped += [e.seq for e in result.events if isinstance(e, MessageDropped)]
        patterns.append(dropped)
    assert patterns[0]
    assert patterns[0] == patterns[1]


class _SubMessage(Message):
    """A subclass could smuggle extra attributes through the channel."""


class _SubTuple(tuple):  # type: ignore[type-arg]
    pass


@pytest.mark.parametrize(
    "bad",
    [
        _SubMessage("gnd_0", 0, "x"),
        Message("gnd_0", 0, "x", (_SubTuple((1,)),)),
        Message("gnd_0", 0, "x", _SubTuple()),
    ],
)
def test_subclasses_are_rejected(bad: Message) -> None:
    env, _ = _pair()
    with pytest.raises(TypeError):
        env.step({"gnd_0": STAY, "gnd_1": STAY}, {"gnd_0": [bad]})
    assert env.step_count == 0


def test_a_rejected_outbox_does_not_consume_sequence_numbers() -> None:
    env, _ = _pair()
    with pytest.raises(TypeError):
        env.step({"gnd_0": STAY, "gnd_1": STAY}, {"gnd_0": [_msg("gnd_0", 0, (1.5,))]})
    env.step({"gnd_0": STAY, "gnd_1": STAY}, {"gnd_0": [_msg("gnd_0", 0)]})
    assert env.step_count == 1


def test_every_recipient_gets_exactly_one_outcome() -> None:
    """Delivered + dropped (any reason) + still in flight = all (message, recipient) pairs."""
    config = _comms_config()
    env = SwarmEnv(config)
    obs, _ = env.reset(6)
    team = _BeaconTeam()
    rngs = episode_streams(6, env.possible_agents).policies
    sent: set[tuple[AgentId, int, AgentId]] = set()
    outcomes: list[tuple[AgentId, int, AgentId]] = []
    while not env.done:
        actions = team.act(obs, rngs)
        result = env.step(actions, team.outbox())
        for e in result.events:
            if isinstance(e, MessageSent):
                sent |= {(e.sender, e.seq, r) for r in e.recipients}
            elif isinstance(e, MessageDelivered | MessageDropped):
                outcomes.append((e.sender, e.seq, e.receiver))
        obs = result.observations
    assert len(outcomes) == len(set(outcomes))  # no pair resolved twice
    assert set(outcomes) <= sent
    channel = env._channel
    assert channel is not None
    assert len(sent) == len(outcomes) + channel.in_flight


# ----------------------------------------------------------------- runner integration


class _BeaconTeam:
    """Random walkers that broadcast their position every step (test-only)."""

    name = "test_beacon"
    version = "1"
    uses_oracle = False
    information = "decentralized"

    def __init__(self) -> None:
        self._out: dict[AgentId, list[Message]] = {}
        self._seq = 0
        self.heard: dict[AgentId, int] = {}

    def act(
        self,
        observations: Mapping[AgentId, Observation],
        rngs: Mapping[AgentId, np.random.Generator],
    ) -> dict[AgentId, int]:
        actions: dict[AgentId, int] = {}
        self._out = {}
        for aid, obs in sorted(observations.items()):
            if obs.status.value != "active":
                continue
            self.heard[aid] = self.heard.get(aid, 0) + len(obs.inbox)
            actions[aid] = int(rngs[aid].choice(np.flatnonzero(obs.action_mask)))
            self._out[aid] = [Message(aid, self._seq, "beacon", (obs.position,))]
            self._seq += 1
        return actions

    def outbox(self) -> Mapping[AgentId, Sequence[Message]]:
        return self._out

    def attach_oracle(self, view: OracleView) -> None:
        raise AssertionError("not an oracle")

    def diagnostics(self) -> Diagnostics:
        return {"messages_heard": sum(self.heard.values())}

    def decision_log(self) -> list[dict[str, Any]]:
        return []


BEACON = TeamFactory("test_beacon", lambda brief: _BeaconTeam())


def _comms_config() -> ScenarioConfig:
    return replace(
        make_config(),
        comms=CommsConfig(ranges=(("drone", 8), ("ground", 5)), p_drop=0.2, latency=1, jitter=1),
    )


def test_runner_passes_outboxes_and_runs_are_deterministic() -> None:
    team: TeamPolicy = _BeaconTeam()
    assert isinstance(team, CommunicatingTeam)
    a = run_episode(_comms_config(), BEACON, 4)
    b = run_episode(_comms_config(), BEACON, 4)
    c = run_episode(_comms_config(), BEACON, 5)
    assert a.event_stream_sha256 == b.event_stream_sha256 != c.event_stream_sha256
    kinds = {e.kind for e in a.events}
    assert {"message_sent", "message_delivered", "message_dropped"} <= kinds
    assert a.policy_diagnostics is not None
    assert a.policy_diagnostics["messages_heard"] > 0
    assert a.metadata["schema_versions"]["config"] == 3


def test_runner_rejects_a_communicating_team_without_a_channel() -> None:
    with pytest.raises(ValueError, match="no comms block"):
        run_episode(make_config(), BEACON, 0)


def test_mission_brief_shape_is_unchanged() -> None:
    """M3.1 adds no policy-facing information beyond the inbox (ADR-0008)."""
    brief = MissionBrief.from_config(_comms_config())
    assert brief == MissionBrief.from_config(make_config())

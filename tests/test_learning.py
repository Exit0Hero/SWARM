"""Phase 4 M4.1: learned sector agent, reward, PettingZoo adapters (ADR-0022)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from swarm.core.events import AgentDisabled, Detection, EpisodeEnd, Event, HazardExposure
from swarm.core.rng import episode_streams
from swarm.experiments.runner import run_episode
from swarm.learning.rewards import DEFAULT_WEIGHTS, TeamReward
from swarm.learning.sector import (
    FREE,
    N_ACTIONS,
    OBS_SIZE,
    RETURN,
    NumpyActor,
    SectorTeam,
    fixed_choice_policy,
    learned_policy,
)
from swarm.policies.base import MissionBrief, TeamFactory
from swarm.scenarios.config import CommsConfig, ScenarioConfig
from tests.helpers import make_config

COMMS = CommsConfig(ranges=(("drone", 8), ("ground", 5)))
ROOT = Path(__file__).resolve().parents[1]


def _config(**kw: Any) -> ScenarioConfig:
    return replace(make_config(drones=1, grounds=2, time_limit=60, **kw), comms=COMMS)


def _nominal() -> ScenarioConfig:
    return ScenarioConfig.from_json((ROOT / "configs/phase3/nominal.json").read_text())


# ----------------------------------------------------------------- features and team


def test_features_and_masks_are_well_formed_at_every_decision() -> None:
    config = _nominal()
    seen: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = []

    def record(_: str, feats: np.ndarray[Any, Any], mask: np.ndarray[Any, Any]) -> int:
        seen.append((feats, mask))
        return int(np.flatnonzero(mask)[0])

    team_factory = TeamFactory("probe", lambda brief: SectorTeam("probe", brief, record))
    result = run_episode(config, team_factory, 1000, designation="development")
    assert result.end_reason in {"mission_success", "time_limit", "all_disabled"}
    assert seen
    for feats, mask in seen:
        assert feats.shape == (OBS_SIZE,)
        assert feats.dtype == np.float32
        assert float(feats.min()) >= 0.0
        assert float(feats.max()) <= 1.0
        assert mask.shape == (N_ACTIONS,)
        assert mask.any()
    # at step 0 every agent is on the base: RETURN is masked
    assert not seen[0][1][RETURN]


def test_fixed_choice_policies_are_deterministic_and_differ_from_free_search() -> None:
    config = _nominal()
    a = run_episode(config, fixed_choice_policy("s5", 5), 1001, designation="development")
    b = run_episode(config, fixed_choice_policy("s5", 5), 1001, designation="development")
    free = run_episode(config, fixed_choice_policy("free", FREE), 1001, designation="development")
    assert a.event_stream_sha256 == b.event_stream_sha256
    assert a.event_stream_sha256 != free.event_stream_sha256
    assert a.policy_diagnostics is not None
    assert a.policy_diagnostics["macro_decisions"] > 0


def test_numpy_actor_respects_the_mask_and_round_trips(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    w0, b0 = rng.normal(size=(OBS_SIZE, 8)), rng.normal(size=8)
    w1, b1 = rng.normal(size=(8, N_ACTIONS)), rng.normal(size=N_ACTIONS)
    path = tmp_path / "actor.npz"
    np.savez(path, w0=w0, b0=b0, w1=w1, b1=b1)
    actor = NumpyActor.load(path)
    feats = rng.random(OBS_SIZE).astype(np.float32)
    mask = np.zeros(N_ACTIONS, dtype=np.bool_)
    mask[[3, 11]] = True
    choice = actor("a", feats, mask)
    assert choice in (3, 11)
    logits = np.tanh(feats @ w0 + b0) @ w1 + b1
    assert choice == (3 if logits[3] >= logits[11] else 11)
    result = run_episode(_nominal(), learned_policy("tiny", path), 1002, designation="development")
    assert result.policy is not None
    assert result.policy["name"] == "tiny"


def test_grid_sectors_partition_any_map() -> None:
    from swarm.learning.sector import grid_sectors

    for shape in ((20, 20), (7, 9), (4, 4), (33, 18)):
        masks = grid_sectors(shape)
        assert len(masks) == 16
        assert all(m.any() for m in masks)
        assert np.array_equal(np.sum(masks, axis=0), np.ones(shape))
    with pytest.raises(ValueError, match="smaller"):
        grid_sectors((3, 10))


def test_numpy_actor_rejects_wrong_shapes() -> None:
    with pytest.raises(ValueError, match="expected 73 inputs"):
        NumpyActor([(np.zeros((5, N_ACTIONS)), np.zeros(N_ACTIONS))])


def test_the_team_needs_a_channel() -> None:
    brief = MissionBrief.from_config(make_config(drones=1, grounds=1))
    with pytest.raises(ValueError, match="comms"):
        SectorTeam("x", brief)


def test_invalid_macro_action_raises() -> None:
    team = SectorTeam("x", MissionBrief.from_config(_config()))
    agent = next(iter(team.agents.values()))
    with pytest.raises(ValueError, match="macro action"):
        agent.apply(N_ACTIONS, 0)


def test_actor_features_ignore_hidden_survivors() -> None:
    """ADR-0008: features depend on observations only; moving unseen survivors changes nothing."""
    from swarm.core.env import SwarmEnv
    from swarm.scenarios.generator import Scenario, generate_scenario

    config = _nominal()
    agents = tuple(config.agents())
    scenario = generate_scenario(config, episode_streams(1003, agents).scenario)
    hidden = Scenario(scenario.terrain, scenario.base, ((0, 0),), scenario.hazards)
    feats = []
    for sc in (scenario, hidden):
        env = SwarmEnv(config)
        obs, _ = env.reset(1003, scenario=sc)
        team = SectorTeam("x", MissionBrief.from_config(config))
        feats.append(team.prepare(obs))
    for aid in feats[0]:
        assert np.array_equal(feats[0][aid][0], feats[1][aid][0])


# ----------------------------------------------------------------- reward


def test_reward_counts_first_true_detections_only() -> None:
    reward = TeamReward()
    events: list[Event] = [
        Detection(1, "a", (1, 1), True, 0),
        Detection(1, "b", (1, 1), True, 0),  # duplicate: same survivor
        Detection(1, "b", (2, 2), False, None),  # false positive
        AgentDisabled(1, "c", (0, 0), "energy_depleted"),
        AgentDisabled(1, "d", (0, 0), "injected"),  # scheduled failures are not penalized
        HazardExposure(1, "a", (3, 3), 2, 10),
        EpisodeEnd(1, "mission_success", True, False),
    ]
    out = reward.step(events, {})
    assert out["survivor"] == DEFAULT_WEIGHTS["survivor"]
    assert out["disabled"] == DEFAULT_WEIGHTS["disabled"]
    assert out["hazard"] == DEFAULT_WEIGHTS["hazard"]
    assert out["success"] == DEFAULT_WEIGHTS["success"]
    assert out["time"] == DEFAULT_WEIGHTS["time"]
    again = reward.step([Detection(2, "a", (1, 1), True, 0)], {})
    assert again["survivor"] == 0.0
    assert reward.totals["survivor"] == DEFAULT_WEIGHTS["survivor"]


def test_reward_rejects_unknown_components() -> None:
    with pytest.raises(ValueError, match="unknown components"):
        TeamReward({"bogus": 1.0})


# ----------------------------------------------------------------- PettingZoo adapters


def test_parallel_api_test_passes_for_both_adapters() -> None:
    pytest.importorskip("pettingzoo")
    from pettingzoo.test import parallel_api_test

    from swarm.learning.pz_env import SectorParallelEnv, SwarmParallelEnv

    parallel_api_test(SwarmParallelEnv(_config(), seeds=range(2000, 2100)), num_cycles=200)
    parallel_api_test(SectorParallelEnv(_config(), seeds=range(2000, 2100)), num_cycles=60)


def test_training_adapter_reproduces_the_runner_episode() -> None:
    """Train/evaluate consistency: the same choices give the same event stream."""
    pytest.importorskip("pettingzoo")
    from swarm.learning.pz_env import SectorParallelEnv

    config = _nominal()

    def rule(aid: str, feats: np.ndarray[Any, Any], mask: np.ndarray[Any, Any]) -> int:
        valid = np.flatnonzero(mask)
        return int(valid[(int(aid[-1]) * 7 + int(feats.sum() * 1000)) % len(valid)])

    reference = run_episode(
        config,
        TeamFactory("rule", lambda brief: SectorTeam("rule", brief, rule)),
        1004,
        designation="development",
    )
    env = SectorParallelEnv(config)
    obs, _ = env.reset(seed=1004)
    total = 0.0
    while env.agents:
        actions = {
            a: rule(a, o["observation"], o["action_mask"].astype(bool))
            for a, o in obs.items()
            if a in env.agents
        }
        obs, rewards, _, _, infos = env.step(actions)
        total += next(iter(rewards.values()))
        assert set(next(iter(infos.values()))["reward_components"]) == set(DEFAULT_WEIGHTS)
    from swarm.core.events import event_stream_sha256

    assert event_stream_sha256(env.events) == reference.event_stream_sha256
    assert env.end_reason == reference.end_reason
    assert env.state().shape == env.state_space.shape
    assert total == pytest.approx(sum(env.reward_model.totals.values()))


def test_training_seeds_come_from_the_given_range() -> None:
    pytest.importorskip("pettingzoo")
    from swarm.learning.pz_env import SectorParallelEnv

    env = SectorParallelEnv(_config(), seeds=[2005, 2006])
    env.reset()
    first = env.env.snapshot().scenario.fingerprint()
    env.reset()
    env.reset(seed=2005)
    assert env.env.snapshot().scenario.fingerprint() == first
    with pytest.raises(StopIteration):
        env.reset()

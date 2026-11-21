"""Phase 5: replaceable perception model (ADR-0025)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from swarm.core.env import SwarmEnv
from swarm.core.events import Detection, PerceptionError
from swarm.core.perception import NoisySensor, corrupt_view, make_sensor, report_confidence
from swarm.core.types import Terrain
from swarm.evaluation.perception_metrics import compute_perception_metrics
from swarm.experiments.runner import run_episode
from swarm.policies.base import MissionBrief
from swarm.policies.registry import POLICIES
from swarm.scenarios.config import CommsConfig, ConfigError, PerceptionConfig, ScenarioConfig
from tests.helpers import make_config

NOISY = PerceptionConfig(range_falloff=0.5, terrain_error=0.2, hazard_miss=0.3, hazard_false=0.1)


def _config(perception: PerceptionConfig | None = NOISY, **kw: object) -> ScenarioConfig:
    base = make_config(drones=1, grounds=1, time_limit=40, **kw)  # type: ignore[arg-type]
    return replace(base, perception=perception)


# ----------------------------------------------------------------- configuration


def test_perception_round_trips_and_is_absent_by_default() -> None:
    plain = make_config(drones=1, grounds=1)
    assert "perception" not in plain.to_dict()
    config = _config()
    again = ScenarioConfig.from_json(config.to_json())
    assert again == config
    assert again.config_hash() != plain.config_hash()


@pytest.mark.parametrize("field", ["range_falloff", "terrain_error", "hazard_miss", "prior"])
def test_perception_rates_are_validated(field: str) -> None:
    with pytest.raises(ConfigError, match=f"perception.{field}"):
        PerceptionConfig(**{field: 1.5})  # type: ignore[arg-type]


def test_unknown_sensor_model_is_rejected_with_the_known_names() -> None:
    with pytest.raises(ValueError, match="noisy"):
        make_sensor("lidar")
    with pytest.raises(ValueError, match=r"perception\.model"):
        SwarmEnv(_config(PerceptionConfig(model="lidar")))


# ----------------------------------------------------------------- sensor model


def test_detection_falls_with_distance_and_confidence_follows() -> None:
    s = NoisySensor(range_falloff=1.0, prior=0.1)
    near, far = s.detect_probability(0.8, 0, 3), s.detect_probability(0.8, 3, 3)
    assert near == pytest.approx(0.8)
    assert far == pytest.approx(0.2)
    assert s.confidence(0.8, 0.05, 0, 3) > s.confidence(0.8, 0.05, 3, 3)
    assert s.confidence(0.8, 0.0, 2, 3) == 1.0  # no false alarms: every report is true
    assert report_confidence(s, 0.8, 0.05, (0, 0), (0, 3), 3) == s.confidence(0.8, 0.05, 3, 3)


def test_corruption_follows_the_draws_and_never_touches_unknown_cells() -> None:
    s = NoisySensor(terrain_error=0.5, hazard_miss=0.5, hazard_false=0.5)
    terrain = np.array([[Terrain.OPEN, Terrain.STRUCTURE], [Terrain.DEBRIS, -1]], dtype=np.int8)
    hazard = np.array([[True, False], [False, False]])
    cells = [(0, 0), (0, 1), (1, 0)]
    draws = np.array([0.1, 0.1, 0.9, 0.9, 0.1, 0.9])
    t, h, errors = corrupt_view(s, terrain, hazard, cells, draws)
    assert t[0, 0] == Terrain.DEBRIS  # open misread
    assert not h[0, 0]  # hazard missed
    assert t[0, 1] == Terrain.STRUCTURE  # draw above the rate: unchanged
    assert t[1, 0] == Terrain.OPEN
    assert t[1, 1] == -1
    assert sorted(k for _, _, k in errors) == ["hazard_missed", "terrain", "terrain"]


# ----------------------------------------------------------------- environment integration


def test_without_perception_the_episode_is_unchanged() -> None:
    """Common streams: an all-zero noise model gives the original event stream."""
    plain = make_config(drones=1, grounds=1, time_limit=40)
    zero = replace(plain, perception=PerceptionConfig())
    a = run_episode(plain, POLICIES["frontier"], 4, designation="development")
    b = run_episode(zero, POLICIES["frontier"], 4, designation="development")
    assert a.event_stream_sha256 == b.event_stream_sha256


def test_noisy_perception_is_deterministic_logged_and_measured() -> None:
    config = _config()
    a = run_episode(config, POLICIES["frontier"], 4, designation="development")
    b = run_episode(config, POLICIES["frontier"], 4, designation="development")
    assert a.event_stream_sha256 == b.event_stream_sha256
    errors = [e for e in a.events if isinstance(e, PerceptionError)]
    assert errors
    assert {e.error for e in errors} <= {"terrain", "hazard_missed", "hazard_false"}
    assert a.comm_metrics is not None
    assert a.comm_metrics == compute_perception_metrics(a.events)
    assert a.comm_metrics["perception_terrain_errors"] == sum(e.error == "terrain" for e in errors)
    assert a.events[-1].kind == "episode_end"


def test_observations_differ_from_truth_only_where_errors_are_logged() -> None:
    config = _config(PerceptionConfig(terrain_error=0.5))
    env = SwarmEnv(config)
    observations, _ = env.reset(7)
    snap = env.snapshot()
    terrain = snap.scenario.terrain
    queued = [e for e in env._perception_errors if isinstance(e, PerceptionError)]
    wrong = set()
    for obs in observations.values():
        r0, c0 = obs.position
        rad = obs.sensor_radius
        for vr in range(obs.local_terrain.shape[0]):
            for vc in range(obs.local_terrain.shape[1]):
                code = int(obs.local_terrain[vr, vc])
                if code < 0:
                    continue
                cell = (r0 + vr - rad, c0 + vc - rad)
                if code != int(terrain[cell]):
                    wrong.add((obs.agent_id, cell))
    logged = {(e.agent_id, e.cell) for e in queued if e.error == "terrain"}
    assert wrong == logged
    assert wrong  # 50% misread rate produces errors
    for obs in observations.values():
        assert obs.position not in {c for a, c in wrong if a == obs.agent_id}  # own cell exact


def test_reset_errors_are_emitted_with_the_first_step() -> None:
    config = _config(PerceptionConfig(terrain_error=0.5))
    env = SwarmEnv(config)
    env.reset(7)
    pending = len(env._perception_errors)
    events = env.step({aid: 0 for aid in env.agents}).events
    assert sum(isinstance(e, PerceptionError) and e.step == 0 for e in events) == pending


def test_range_falloff_lowers_true_detections() -> None:
    def true_reports(falloff: float) -> int:
        config = _config(PerceptionConfig(range_falloff=falloff))
        total = 0
        for seed in range(1000, 1006):
            r = run_episode(config, POLICIES["frontier"], seed, designation="development")
            total += sum(isinstance(e, Detection) and e.true_positive for e in r.events)
        return total

    assert true_reports(1.0) < true_reports(0.0)


def test_shared_beliefs_and_auction_run_on_noisy_perception() -> None:
    comms = CommsConfig(ranges=(("drone", 8), ("ground", 5)))
    config = replace(_config(), comms=comms)
    for policy in ("frontier_shared", "auction_comm"):
        r = run_episode(config, POLICIES[policy], 2, designation="development")
        assert r.comm_metrics is not None
        assert "perception_terrain_errors" in r.comm_metrics
        assert "message_delivery_rate" in r.comm_metrics


def test_brief_exposes_only_the_public_sensor_model() -> None:
    brief = MissionBrief.from_config(_config())
    assert brief.perception == NOISY

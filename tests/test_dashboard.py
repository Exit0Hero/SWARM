"""Phase 6: dashboard service and HTTP API (ADR-0026)."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from swarm.core.events import Detection
from swarm.dashboard import service
from swarm.dashboard.server import serve
from swarm.dashboard.service import DashboardError, Store
from swarm.experiments.runner import run_episode
from swarm.policies.registry import POLICIES
from swarm.scenarios.config import CommsConfig, FailureConfig, PerceptionConfig
from tests.helpers import make_config

ROOT = Path(__file__).resolve().parents[1]


def _config() -> Any:
    base = make_config(drones=1, grounds=1, time_limit=30)
    return replace(
        base,
        comms=CommsConfig(ranges=(("drone", 8), ("ground", 5))),
        failures=FailureConfig(1, 5, 10),
        perception=PerceptionConfig(terrain_error=0.1),
    )


# ----------------------------------------------------------------- service


def test_recorded_frames_are_the_real_episode() -> None:
    config = _config()
    record = service.record_episode(config, "auction_comm", 3)
    plain = run_episode(config, POLICIES["auction_comm"], 3, designation="development")
    assert record["event_stream_sha256"] == plain.event_stream_sha256  # hook changes nothing
    assert len(record["frames"]) == plain.steps + 1
    assert record["metrics"]["recall"] == plain.metrics.recall
    last = record["frames"][-1]
    snap = plain.final_snapshot
    assert snap is not None
    assert [a["pos"] for a in last["agents"]] == [list(a.position) for a in snap.agents]
    true_found = {
        e.survivor_index for e in plain.events if isinstance(e, Detection) and e.true_positive
    }
    assert last["found"] == len(true_found)
    assert sum(len(f["disabled"]) for f in record["frames"]) == plain.metrics.agents_disabled
    assert "perception_terrain_errors" in record["extension_metrics"]
    assert record["scenario"]["survivors_ground_truth"]  # evaluation view only


def test_build_config_applies_overrides_and_reports_errors() -> None:
    base = make_config(drones=1, grounds=1).to_dict()
    config = service.build_config(
        base,
        {
            "comms": {
                "ranges": {"drone": 4, "ground": 3},
                "p_drop": 0.2,
                "latency": 0,
                "jitter": 0,
                "bandwidth": None,
            }
        },
    )
    assert config.comms is not None
    assert config.comms.p_drop == 0.2
    with pytest.raises(DashboardError, match="invalid configuration"):
        service.build_config(base, {"p_detect": 3})


def test_policy_catalog_marks_policies_that_need_a_channel() -> None:
    catalog = {p["name"]: p for p in service.policy_catalog(make_config(drones=1, grounds=1))}
    assert catalog["frontier"]["available"]
    assert not catalog["auction_comm"]["available"]
    assert catalog["assignment_oracle"]["oracle"]


def test_comparison_pairs_by_seed_and_limits_size() -> None:
    result = service.run_comparison(_config(), ["frontier_shared", "auction_comm"], [1000, 1001])
    assert set(result["per_policy"]) == {"frontier_shared", "auction_comm"}
    assert len(result["seed_rows"]) == 4
    assert {d["b"] for d in result["paired_vs_first"]} == {"auction_comm"}
    with pytest.raises(DashboardError, match="at most"):
        service.run_comparison(_config(), ["frontier"], list(range(700)))
    with pytest.raises(DashboardError, match="unknown policies"):
        service.run_comparison(_config(), ["nope"], [1])


def test_store_history_export_and_bad_ids(tmp_path: Path) -> None:
    store = Store(tmp_path)
    record = service.record_episode(_config(), "frontier", 1)
    store.save(record)
    assert store.history()[0]["id"] == record["id"]
    body, mime, name = service.export_run(store.load("runs", record["id"]), "csv")
    assert mime == "text/csv"
    assert "recall" in body
    assert name.endswith(".csv")
    events, _, _ = service.export_run(record, "events")
    assert len(events.splitlines()) == len(record["events_jsonl"].splitlines())
    with pytest.raises(DashboardError):
        store.load("runs", "../etc")
    with pytest.raises(DashboardError):
        service.export_run(record, "xml")
    store.delete("runs", record["id"])
    assert store.history() == []


# ----------------------------------------------------------------- HTTP


@pytest.fixture
def base_url(tmp_path: Path) -> Iterator[str]:
    server = serve(
        tmp_path / "data",
        [ROOT / "configs"],
        port=0,
        models_root=ROOT / "docs" / "results" / "phase4" / "models",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _call(url: str, method: str = "GET", body: Any = None) -> Any:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as res:
        raw = res.read()
        return json.loads(raw) if res.headers["Content-Type"] == "application/json" else raw


def test_http_end_to_end(base_url: str) -> None:
    page = _call(base_url + "/")
    assert b"SWARM" in page
    configs = _call(base_url + "/api/configs")
    assert "configs/phase3/nominal.json" in configs
    assert not any("study" in name for name in configs)  # study specs are not configs
    request = {
        "config_name": "configs/phase3/nominal.json",
        "overrides": {"perception": {"model": "noisy", "terrain_error": 0.05}},
    }
    checked = _call(base_url + "/api/validate", "POST", request)
    assert checked["config"]["perception"]["terrain_error"] == 0.05
    run = _call(base_url + "/api/runs", "POST", request | {"policy": "auction_comm", "seed": 1000})
    assert run["frames"][0]["step"] == 0
    again = _call(base_url + f"/api/runs/{run['id']}")
    assert again["event_stream_sha256"] == run["event_stream_sha256"]
    csv_body = _call(base_url + f"/api/runs/{run['id']}/export?format=csv")
    assert b"metric,value" in csv_body
    job = _call(
        base_url + "/api/experiments",
        "POST",
        request | {"policies": ["frontier_shared", "auction_comm"], "seeds": "1000-1001"},
    )
    for _ in range(240):
        state = _call(base_url + f"/api/jobs/{job['id']}")
        if state["status"] != "running":
            break
        threading.Event().wait(0.5)
    assert state["status"] == "done", state
    experiment = _call(base_url + f"/api/experiments/{state['result']['id']}")
    assert experiment["seeds"] == [1000, 1001]
    kinds = {i["kind"] for i in _call(base_url + "/api/history")}
    assert kinds == {"run", "experiment"}
    saved = _call(base_url + "/api/configs", "POST", request | {"name": "noisy_nominal"})
    assert saved["saved"] == "noisy_nominal"
    assert "saved/noisy_nominal.json" in _call(base_url + "/api/configs")
    assert _call(base_url + f"/api/runs/{run['id']}", "DELETE")["deleted"] == run["id"]


def test_http_errors_are_json_400(base_url: str) -> None:
    with pytest.raises(urllib.error.HTTPError) as info:
        _call(base_url + "/api/runs", "POST", {"config_name": "nope.json", "policy": "frontier"})
    with info.value as response:
        assert response.code == 400
        assert "unknown config" in json.loads(response.read())["error"]


def test_trained_marl_actors_run_through_the_dashboard() -> None:
    learned = service.learned_catalog(ROOT / "docs" / "results" / "phase4" / "models")
    assert "learned:mappo_s0" in learned
    catalog = {p["name"]: p for p in service.policy_catalog(_config(), learned)}
    assert catalog["learned:mappo_s0"]["available"]
    record = service.record_episode(_config(), "learned:mappo_s0", 5, learned)
    plain = run_episode(_config(), learned["learned:mappo_s0"], 5, designation="development")
    assert record["event_stream_sha256"] == plain.event_stream_sha256
    assert service.learned_catalog(ROOT / "no-such-dir") == {}


def test_scene3d_endpoint_returns_the_recorded_tracks(tmp_path: Path) -> None:
    store = Store(tmp_path)
    record = service.record_episode(_config(), "frontier", 2)
    store.save(record)
    out = service.scene3d(store.load("runs", record["id"]))
    tracks = out["scene"]["tracks"]
    assert set(tracks) == {a["id"] for a in record["frames"][0]["agents"]}
    assert all(len(t["points"]) == len(record["frames"]) for t in tracks.values())
    analysis = out["analysis"]
    assert "unavailable" in analysis or analysis["contacts"] == 0


def _raw(url: str, method: str, body: bytes, headers: dict[str, str]) -> int:
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as res:
            return int(res.status)
    except urllib.error.HTTPError as exc:
        with exc:
            return int(exc.code)


def test_http_hardening(base_url: str) -> None:
    run = base_url + "/api/runs"
    body = b'{"config_name": "examples/basic.json", "policy": "frontier", "seed": 1}'
    assert _raw(run, "POST", body, {"Content-Type": "text/plain"}) == 403  # no simple CSRF posts
    assert (
        _raw(run, "POST", body, {"Content-Type": "application/json", "Host": "evil.example"}) == 403
    )
    big = b"{" + b" " * 1_000_001 + b"}"
    assert _raw(run, "POST", big, {"Content-Type": "application/json"}) == 413
    assert _raw(run, "POST", body, {"Content-Type": "application/json"}) == 200  # packaged example
    with urllib.request.urlopen(base_url + "/", timeout=30) as res:
        assert res.headers["X-Content-Type-Options"] == "nosniff"
        assert res.headers["X-Frame-Options"] == "DENY"

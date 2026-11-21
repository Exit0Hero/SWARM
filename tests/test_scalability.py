"""Phase 7: the scalability benchmark varies one factor and records runtime and traffic."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from swarm.experiments import scalability
from swarm.experiments.scalability import main, variant
from swarm.scenarios.config import CommsConfig
from tests.helpers import make_config


def test_variants_change_one_factor() -> None:
    base = replace(
        make_config(drones=1, grounds=1, time_limit=50), height=10, width=10, survivor_count=4
    )
    assert len(variant(base, "team", 3).roster) == 6
    big = variant(base, "map", 20)
    assert (big.height, big.width, big.survivor_count, big.time_limit) == (20, 20, 16, 100)
    assert variant(base, "survivors", 7).survivor_count == 7
    with pytest.raises(ValueError, match="unknown factor"):
        variant(base, "weather", 1)


def test_benchmark_cli_writes_points(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scalability, "TEAM_SIZES", (1,))
    monkeypatch.setattr(scalability, "MAP_SIDES", (8,))
    monkeypatch.setattr(scalability, "SURVIVORS", (2,))
    base = replace(
        make_config(drones=1, grounds=1, time_limit=30),
        comms=CommsConfig(ranges=(("drone", 8), ("ground", 5))),
    )
    cfg = tmp_path / "c.json"
    cfg.write_text(base.to_json())
    out = tmp_path / "s.json"
    argv = [
        "--config",
        str(cfg),
        "--out",
        str(out),
        "--policies",
        "frontier_shared",
        "--episodes",
        "1",
        "--profile",
    ]
    assert main(argv) == 0
    points = json.loads(out.read_text())["points"]
    assert [p["factor"] for p in points] == ["team", "map", "survivors"]
    assert all(p["ms_per_step"] > 0 and "hotspots" in p for p in points)
    assert main(["--config", str(cfg), "--out", str(out), "--policies", "nope"]) == 2

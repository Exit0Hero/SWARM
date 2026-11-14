"""ScenarioConfig validation and JSON serialization (ADR-0007)."""

from __future__ import annotations

import json
import re
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from swarm.core.types import Layer
from swarm.scenarios.config import (
    DRONE,
    GROUND,
    SCHEMA_VERSION,
    AgentSpec,
    ConfigError,
    HazardConfig,
    ScenarioConfig,
    TerrainConfig,
)
from tests.helpers import make_config, spec

ROOT = Path(__file__).resolve().parents[1]


def test_round_trip_preserves_config_and_hash() -> None:
    config = make_config(base=(1, 2), roster=(AgentSpec("drone", initial_energy=5),))
    restored = ScenarioConfig.from_json(config.to_json())
    assert restored == config
    assert restored.config_hash() == config.config_hash()


def test_json_contains_schema_version() -> None:
    assert json.loads(make_config().to_json())["schema_version"] == SCHEMA_VERSION


def test_hash_changes_with_any_field() -> None:
    base = make_config()
    assert base.config_hash() != replace(base, time_limit=base.time_limit + 1).config_hash()
    assert base.config_hash() != replace(base, p_detect=0.5).config_hash()


def test_config_is_frozen() -> None:
    config = make_config()
    with pytest.raises(FrozenInstanceError):
        config.height = 3  # type: ignore[misc]


def test_agent_ids_are_per_type_counters_sorted() -> None:
    config = make_config(
        roster=(AgentSpec("ground"), AgentSpec("drone"), AgentSpec("ground"), AgentSpec("drone"))
    )
    assert list(config.agents()) == ["drone_0", "drone_1", "ground_0", "ground_1"]


def test_initial_energy_defaults_to_capacity() -> None:
    agents = make_config(roster=(AgentSpec("drone"), AgentSpec("ground", 0))).agents()
    assert agents["drone_0"] == (DRONE, DRONE.energy_capacity)
    assert agents["ground_0"] == (GROUND, 0)


def test_default_base_is_grid_centre() -> None:
    assert make_config(height=7, width=10).base == (3, 5)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"height": 0}, "height: must be >= 1"),
        ({"width": True}, "width: must be an integer"),
        ({"survivors": -1}, "survivor_count: must be >= 0"),
        ({"height": 2, "width": 2, "survivors": 4}, "survivor_count: at most 3"),
        ({"time_limit": 0}, "time_limit: must be >= 1"),
        ({"p_detect": 1.5}, "p_detect: must be in [0, 1]"),
        ({"p_detect": float("nan")}, "p_detect: must be in [0, 1]"),
        ({"debris": 0.6, "structure": 0.6}, "terrain: debris_fraction + structure_fraction"),
        ({"base": (8, 0)}, "base_position[0]: must be <= 7"),
        ({"roster": ()}, "roster: at least one agent"),
        ({"roster": (AgentSpec("tank"),)}, "roster[0].agent_type: unknown type 'tank'"),
        (
            {"roster": (AgentSpec("drone", 201),)},
            "roster[0].initial_energy: must be <= energy_capacity (200)",
        ),
        ({"agent_types": ()}, "agent_types: at least one"),
        ({"agent_types": (DRONE, DRONE)}, "agent_types: names must be unique"),
    ],
)
def test_invalid_configs_raise_with_field_path(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=re.escape(message)):
        make_config(**overrides)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"radius": 0}, "sensor_radius: must be >= 1"),
        ({"capacity": 0}, "energy_capacity: must be >= 1"),
        ({"move": 0}, "move_cost: must be >= 1"),
        ({"stay": -1}, "stay_cost: must be >= 0"),
        ({"capacity": 3, "move": 4}, "move_cost: must be <= energy_capacity"),
    ],
)
def test_invalid_agent_type_specs(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ConfigError, match=re.escape(message)):
        spec("probe", Layer.AIR, **kwargs)


def test_invalid_agent_type_name() -> None:
    with pytest.raises(ConfigError, match="name: must be lowercase"):
        spec("Drone", Layer.AIR)


def test_terrain_fraction_type_checked() -> None:
    with pytest.raises(ConfigError, match=r"terrain\.debris_fraction: must be a number"):
        TerrainConfig(debris_fraction="0.1", structure_fraction=0.0)  # type: ignore[arg-type]


def _dict() -> dict[str, Any]:
    return make_config().to_dict()


def test_from_dict_rejects_unknown_key() -> None:
    data = _dict() | {"wind_speed": 3}
    with pytest.raises(ConfigError, match=r"unknown key\(s\) \['wind_speed'\]"):
        ScenarioConfig.from_dict(data)


def test_from_dict_rejects_missing_key() -> None:
    data = _dict()
    del data["time_limit"]
    with pytest.raises(ConfigError, match=r"missing required key\(s\) \['time_limit'\]"):
        ScenarioConfig.from_dict(data)


@pytest.mark.parametrize("version", [0, 3, "2", True, None])
def test_from_dict_rejects_unreadable_schema_version(version: object) -> None:
    with pytest.raises(ConfigError, match="schema_version"):
        ScenarioConfig.from_dict(_dict() | {"schema_version": version})


def test_from_dict_rejects_bad_layer_and_nested_unknown_key() -> None:
    data = _dict()
    data["agent_types"][0]["layer"] = "water"
    with pytest.raises(ConfigError, match=r"agent_types\[0\].layer"):
        ScenarioConfig.from_dict(data)
    data = _dict()
    data["roster"][0]["speed"] = 2
    with pytest.raises(ConfigError, match=r"roster\[0\]: unknown key"):
        ScenarioConfig.from_dict(data)


def test_from_json_rejects_invalid_json() -> None:
    with pytest.raises(ConfigError, match="invalid JSON"):
        ScenarioConfig.from_json("{not json")


def test_m1_example_config_is_a_valid_version_1_file() -> None:
    """configs/m1_default.json is kept at schema v1 on purpose: it is the migration fixture."""
    text = (ROOT / "configs" / "m1_default.json").read_text(encoding="utf-8")
    assert json.loads(text)["schema_version"] == 1
    config = ScenarioConfig.from_json(text)
    assert config.hazards is None
    assert config.p_false_positive == 0.0
    assert config.recall_target == 1.0


def test_m2_example_config_is_canonical() -> None:
    path = ROOT / "configs" / "m2_default.json"
    config = ScenarioConfig.from_json(path.read_text(encoding="utf-8"))
    assert config.to_json() == path.read_text(encoding="utf-8"), "keep the file canonical"
    assert config.hazards is not None


@given(
    height=st.integers(1, 30),
    width=st.integers(1, 30),
    debris=st.floats(0, 0.5),
    structure=st.floats(0, 0.5),
    p=st.floats(0, 1),
    limit=st.integers(1, 500),
)
def test_round_trip_property(
    height: int, width: int, debris: float, structure: float, p: float, limit: int
) -> None:
    config = make_config(
        height=height,
        width=width,
        debris=debris,
        structure=structure,
        survivors=0,
        p_detect=p,
        time_limit=limit,
    )
    restored = ScenarioConfig.from_json(config.to_json())
    assert restored == config
    assert restored.config_hash() == config.config_hash()


# ----------------------------------------------------------------- schema v2 (Phase 1 M2)


def _hazards(**overrides: Any) -> HazardConfig:
    values: dict[str, Any] = {
        "fraction": 0.1,
        "affected_layers": (Layer.GROUND,),
        "energy_cost": 3,
        "disable_probability": 0.05,
    }
    values.update(overrides)
    return HazardConfig(**values)


def test_v1_files_migrate_to_m1_equivalent_defaults() -> None:
    v1 = _dict()
    v1["schema_version"] = 1
    for key in ("hazards", "p_false_positive", "recall_target"):
        del v1[key]
    migrated = ScenarioConfig.from_dict(v1)
    assert migrated == make_config()
    assert migrated.to_dict()["schema_version"] == SCHEMA_VERSION == 2


def test_v1_files_may_not_contain_v2_keys_missing_from_v2() -> None:
    v2 = _dict()
    del v2["recall_target"]
    with pytest.raises(ConfigError, match=r"missing required key\(s\) \['recall_target'\]"):
        ScenarioConfig.from_dict(v2)


def test_hazard_config_round_trip_and_layer_order() -> None:
    hazards = _hazards(affected_layers=(Layer.GROUND, Layer.AIR))
    assert hazards.affected_layers == (Layer.AIR, Layer.GROUND)
    config = replace(make_config(), hazards=hazards, p_false_positive=0.01, recall_target=0.5)
    restored = ScenarioConfig.from_json(config.to_json())
    assert restored == config
    assert restored.to_dict()["hazards"]["affected_layers"] == ["air", "ground"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"fraction": 1.2}, "hazards.fraction: must be in [0, 1]"),
        ({"disable_probability": -0.1}, "hazards.disable_probability: must be in [0, 1]"),
        ({"energy_cost": -1}, "hazards.energy_cost: must be >= 0"),
        ({"affected_layers": ()}, "hazards.affected_layers: at least one layer"),
        ({"affected_layers": (Layer.AIR, Layer.AIR)}, "hazards.affected_layers: duplicate"),
        ({"affected_layers": ("air",)}, "hazards.affected_layers[0]: must be a Layer"),
    ],
)
def test_invalid_hazard_configs(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=re.escape(message)):
        _hazards(**overrides)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("recall_target", 0.0, "recall_target: must be in (0, 1]"),
        ("recall_target", 1.01, "recall_target: must be in [0, 1]"),
        ("recall_target", True, "recall_target: must be a number"),
        ("p_false_positive", -0.01, "p_false_positive: must be in [0, 1]"),
        ("p_false_positive", float("inf"), "p_false_positive: must be in [0, 1]"),
    ],
)
def test_invalid_m2_scalars(field: str, value: object, message: str) -> None:
    kwargs: dict[str, Any] = {field: value}
    with pytest.raises(ConfigError, match=re.escape(message)):
        replace(make_config(), **kwargs)


def test_hazards_json_rejects_unknown_layer_and_keys() -> None:
    data = replace(make_config(), hazards=_hazards()).to_dict()
    data["hazards"]["affected_layers"] = ["sea"]
    with pytest.raises(ConfigError, match=re.escape("hazards.affected_layers[0]")):
        ScenarioConfig.from_dict(data)
    data = replace(make_config(), hazards=_hazards()).to_dict()
    data["hazards"]["damage"] = 1
    with pytest.raises(ConfigError, match=r"hazards: unknown key"):
        ScenarioConfig.from_dict(data)

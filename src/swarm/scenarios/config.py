"""Frozen, validated scenario configuration with versioned JSON serialization (ADR-0007).

All configuration objects are immutable ``dataclass(frozen=True, slots=True)`` instances that
validate themselves in ``__post_init__`` and raise :class:`ConfigError` with the field path and the
offending value. JSON input is strict: unknown keys, missing keys, wrong types and an unsupported
``schema_version`` are all errors.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from swarm.core.types import AgentId, Layer, Position

SCHEMA_VERSION: Final[int] = 3
"""Newest version of the scenario-config JSON schema this code writes.

Version 3 adds the optional ``comms`` block (ADR-0017). A config without a channel is written as
version 2, with no ``comms`` key, so its JSON and :meth:`ScenarioConfig.config_hash` are exactly
as before. Version 1 (Phase 1 M1) files are still read: they are migrated by adding
``hazards = null``, ``p_false_positive = 0.0`` and ``recall_target = 1.0``, which reproduces M1
behaviour exactly (ADR-0007, ADR-0013).
"""

READABLE_SCHEMA_VERSIONS: Final[frozenset[int]] = frozenset({1, 2, 3})
MAX_COMMS_DELAY: Final[int] = 1000

MAX_GRID_SIDE: Final[int] = 1024
MAX_TIME_LIMIT: Final[int] = 1_000_000
_NAME_RE: Final = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


class ConfigError(ValueError):
    """A configuration value is missing, has the wrong type, or violates a constraint."""


# --------------------------------------------------------------------------- validation helpers


def _check_int(value: object, path: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path}: must be an integer, got {value!r}")
    if value < minimum:
        raise ConfigError(f"{path}: must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{path}: must be <= {maximum}, got {value}")
    return value


def _check_fraction(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{path}: must be a number, got {value!r}")
    as_float = float(value)
    if not math.isfinite(as_float) or not 0.0 <= as_float <= 1.0:
        raise ConfigError(f"{path}: must be in [0, 1], got {value!r}")
    return as_float


def _check_name(value: object, path: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.match(value):
        raise ConfigError(
            f"{path}: must be lowercase letters, digits or '_', starting with a letter "
            f"(max 32 chars), got {value!r}"
        )
    return value


def _check_layer(value: object, path: str) -> Layer:
    if not isinstance(value, Layer):
        raise ConfigError(f"{path}: must be a Layer, got {value!r}")
    return value


def _require_mapping(data: object, path: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: must be a JSON object, got {type(data).__name__}")
    return data


def _check_keys(data: Mapping[str, Any], path: str, required: set[str], optional: set[str]) -> None:
    keys = set(data)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise ConfigError(f"{path}: missing required key(s) {sorted(missing)}")
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {sorted(unknown)}")


def _parse_layer(value: object, path: str) -> Layer:
    if not isinstance(value, str):
        raise ConfigError(f"{path}: must be 'air' or 'ground', got {value!r}")
    try:
        return Layer[value.upper()]
    except KeyError:
        raise ConfigError(f"{path}: must be 'air' or 'ground', got {value!r}") from None


def _parse_list(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"{path}: must be a JSON array, got {type(value).__name__}")
    return value


# --------------------------------------------------------------------------- config objects


@dataclass(frozen=True, slots=True)
class AgentTypeSpec:
    """Capabilities of one agent type. Agent types are data, not subclasses (ADR-0002).

    Energy is in integer *energy units*. ``move_cost`` is charged for a successful move;
    ``stay_cost`` for staying, for a move that failed in conflict resolution, and for an
    invalid action (ADR-0003, ADR-0004). ``recharge_rate`` is added per step on the base cell.
    """

    name: str
    layer: Layer
    sensor_radius: int
    energy_capacity: int
    move_cost: int
    stay_cost: int
    recharge_rate: int

    def __post_init__(self) -> None:
        p = f"agent_types[{self.name!r}]"
        _check_name(self.name, f"{p}.name")
        _check_layer(self.layer, f"{p}.layer")
        # Radius >= 1 so the action mask never reveals unseen terrain (ADR-0004, ADR-0008).
        _check_int(self.sensor_radius, f"{p}.sensor_radius", minimum=1, maximum=MAX_GRID_SIDE)
        _check_int(self.energy_capacity, f"{p}.energy_capacity", minimum=1)
        _check_int(self.move_cost, f"{p}.move_cost", minimum=1)
        _check_int(self.stay_cost, f"{p}.stay_cost", minimum=0)
        _check_int(self.recharge_rate, f"{p}.recharge_rate", minimum=0)
        if self.move_cost > self.energy_capacity:
            raise ConfigError(
                f"{p}.move_cost: must be <= energy_capacity ({self.energy_capacity}), "
                f"got {self.move_cost}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "layer": self.layer.name.lower(),
            "sensor_radius": self.sensor_radius,
            "energy_capacity": self.energy_capacity,
            "move_cost": self.move_cost,
            "stay_cost": self.stay_cost,
            "recharge_rate": self.recharge_rate,
        }

    @classmethod
    def from_dict(cls, data: object, path: str = "agent_types[]") -> AgentTypeSpec:
        d = _require_mapping(data, path)
        _check_keys(
            d,
            path,
            {
                "name",
                "layer",
                "sensor_radius",
                "energy_capacity",
                "move_cost",
                "stay_cost",
                "recharge_rate",
            },
            set(),
        )
        return cls(
            name=d["name"],
            layer=_parse_layer(d["layer"], f"{path}.layer"),
            sensor_radius=d["sensor_radius"],
            energy_capacity=d["energy_capacity"],
            move_cost=d["move_cost"],
            stay_cost=d["stay_cost"],
            recharge_rate=d["recharge_rate"],
        )


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """One roster entry: an agent of type ``agent_type``.

    ``initial_energy`` defaults to the type's full capacity when ``None``.
    """

    agent_type: str
    initial_energy: int | None = None

    def __post_init__(self) -> None:
        _check_name(self.agent_type, "roster[].agent_type")
        if self.initial_energy is not None:
            _check_int(self.initial_energy, "roster[].initial_energy", minimum=0)

    def to_dict(self) -> dict[str, Any]:
        return {"agent_type": self.agent_type, "initial_energy": self.initial_energy}

    @classmethod
    def from_dict(cls, data: object, path: str = "roster[]") -> AgentSpec:
        d = _require_mapping(data, path)
        _check_keys(d, path, {"agent_type"}, {"initial_energy"})
        return cls(agent_type=d["agent_type"], initial_energy=d.get("initial_energy"))


@dataclass(frozen=True, slots=True)
class TerrainConfig:
    """Independent per-cell terrain probabilities for the generator (ADR-0002).

    Each non-base cell is ``STRUCTURE`` with probability ``structure_fraction``, otherwise
    ``DEBRIS`` with probability ``debris_fraction``, otherwise ``OPEN``.
    """

    debris_fraction: float
    structure_fraction: float

    def __post_init__(self) -> None:
        d = _check_fraction(self.debris_fraction, "terrain.debris_fraction")
        s = _check_fraction(self.structure_fraction, "terrain.structure_fraction")
        object.__setattr__(self, "debris_fraction", d)
        object.__setattr__(self, "structure_fraction", s)
        if d + s > 1.0:
            raise ConfigError(
                f"terrain: debris_fraction + structure_fraction must be <= 1, got {d + s}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "debris_fraction": self.debris_fraction,
            "structure_fraction": self.structure_fraction,
        }

    @classmethod
    def from_dict(cls, data: object, path: str = "terrain") -> TerrainConfig:
        d = _require_mapping(data, path)
        _check_keys(d, path, {"debris_fraction", "structure_fraction"}, set())
        return cls(debris_fraction=d["debris_fraction"], structure_fraction=d["structure_fraction"])


@dataclass(frozen=True, slots=True)
class HazardConfig:
    """Static hazard layer (ADR-0011).

    Each ``OPEN``/``DEBRIS`` non-base cell is hazardous with probability ``fraction``. An active
    agent of an ``affected_layers`` layer that ends a step in a hazardous cell loses
    ``energy_cost`` energy (capped at what it has) and, unless that empties it, is disabled with
    probability ``disable_probability``.
    """

    fraction: float
    affected_layers: tuple[Layer, ...]
    energy_cost: int
    disable_probability: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "fraction", _check_fraction(self.fraction, "hazards.fraction"))
        object.__setattr__(
            self,
            "disable_probability",
            _check_fraction(self.disable_probability, "hazards.disable_probability"),
        )
        _check_int(self.energy_cost, "hazards.energy_cost", minimum=0)
        layers = tuple(self.affected_layers)
        if not layers:
            raise ConfigError("hazards.affected_layers: at least one layer is required")
        for i, layer in enumerate(layers):
            _check_layer(layer, f"hazards.affected_layers[{i}]")
        if len(set(layers)) != len(layers):
            raise ConfigError(f"hazards.affected_layers: duplicate layers in {layers}")
        object.__setattr__(self, "affected_layers", tuple(sorted(layers)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "fraction": self.fraction,
            "affected_layers": [layer.name.lower() for layer in self.affected_layers],
            "energy_cost": self.energy_cost,
            "disable_probability": self.disable_probability,
        }

    @classmethod
    def from_dict(cls, data: object, path: str = "hazards") -> HazardConfig:
        d = _require_mapping(data, path)
        _check_keys(
            d, path, {"fraction", "affected_layers", "energy_cost", "disable_probability"}, set()
        )
        layers = _parse_list(d["affected_layers"], f"{path}.affected_layers")
        return cls(
            fraction=d["fraction"],
            affected_layers=tuple(
                _parse_layer(v, f"{path}.affected_layers[{i}]") for i, v in enumerate(layers)
            ),
            energy_cost=d["energy_cost"],
            disable_probability=d["disable_probability"],
        )


@dataclass(frozen=True, slots=True)
class CommsConfig:
    """Communication channel (ADR-0017).

    ``ranges`` maps every agent type to its communication range in cells (``None`` = unlimited).
    A link from i to j exists when both are active and their Euclidean distance after movement is
    at most ``min(range_i, range_j)``. Each (message, receiver) pair in range is lost with
    probability ``p_drop``; otherwise it is delivered ``latency + J`` steps after sending, with
    ``J`` uniform on ``0..jitter``. ``bandwidth`` caps the units an agent may send per step
    (``None`` = unlimited).

    ``partitions`` are scheduled link cuts (ADR-0021): each ``(start, end, group)`` cuts every
    link between an agent in ``group`` and one outside it during steps ``start <= t < end``, as if
    out of range. A group of one isolates that agent. Links return at step ``end`` (recovery).
    """

    ranges: tuple[tuple[str, int | None], ...]
    p_drop: float = 0.0
    latency: int = 0
    jitter: int = 0
    bandwidth: int | None = None
    partitions: tuple[tuple[int, int, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        raw = self.ranges.items() if isinstance(self.ranges, Mapping) else self.ranges
        entries = tuple(tuple(e) for e in raw)
        if not entries:
            raise ConfigError("comms.ranges: at least one agent type is required")
        names: list[str] = []
        for entry in entries:
            if len(entry) != 2 or not isinstance(entry[0], str):
                raise ConfigError(f"comms.ranges: entries must be (name, range), got {entry!r}")
            name, value = entry
            if value is not None:
                _check_int(value, f"comms.ranges.{name}", minimum=0, maximum=MAX_GRID_SIDE * 2)
            names.append(name)
        if len(set(names)) != len(names):
            raise ConfigError(f"comms.ranges: duplicate agent types in {names}")
        object.__setattr__(self, "ranges", tuple(sorted(entries)))
        object.__setattr__(self, "p_drop", _check_fraction(self.p_drop, "comms.p_drop"))
        _check_int(self.latency, "comms.latency", minimum=0, maximum=MAX_COMMS_DELAY)
        _check_int(self.jitter, "comms.jitter", minimum=0, maximum=MAX_COMMS_DELAY)
        if self.bandwidth is not None:
            _check_int(self.bandwidth, "comms.bandwidth", minimum=1)
        windows: list[tuple[int, int, tuple[str, ...]]] = []
        for k, window in enumerate(self.partitions):
            where = f"comms.partitions[{k}]"
            if len(window) != 3:
                raise ConfigError(f"{where}: must be (start, end, group), got {window!r}")
            start, end, group = window
            _check_int(start, f"{where}.start", minimum=1, maximum=MAX_TIME_LIMIT)
            _check_int(end, f"{where}.end", minimum=start + 1, maximum=MAX_TIME_LIMIT + 1)
            members = tuple(group)
            if not members or not all(isinstance(m, str) for m in members):
                raise ConfigError(f"{where}.group: must be a non-empty list of agent IDs")
            if len(set(members)) != len(members):
                raise ConfigError(f"{where}.group: duplicate agent IDs in {list(members)}")
            windows.append((start, end, tuple(sorted(members))))
        object.__setattr__(self, "partitions", tuple(windows))

    def range_for(self, agent_type: str) -> int | None:
        """Communication range of ``agent_type`` (``None`` = unlimited)."""
        for name, value in self.ranges:
            if name == agent_type:
                return value
        raise KeyError(agent_type)

    def cut(self, step: int, a: str, b: str) -> bool:
        """Whether a scheduled partition cuts the link between ``a`` and ``b`` at ``step``."""
        return any(
            start <= step < end and ((a in group) != (b in group))
            for start, end, group in self.partitions
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ranges": dict(self.ranges),
            "p_drop": self.p_drop,
            "latency": self.latency,
            "jitter": self.jitter,
            "bandwidth": self.bandwidth,
        }
        if self.partitions:  # omitted when empty, so existing config hashes are unchanged
            out["partitions"] = [[s, e, list(g)] for s, e, g in self.partitions]
        return out

    @classmethod
    def from_dict(cls, data: object, path: str = "comms") -> CommsConfig:
        d = _require_mapping(data, path)
        _check_keys(d, path, {"ranges", "p_drop", "latency", "jitter", "bandwidth"}, {"partitions"})
        ranges = _require_mapping(d["ranges"], f"{path}.ranges")
        windows = []
        for k, raw in enumerate(_parse_list(d.get("partitions", []), f"{path}.partitions")):
            items = _parse_list(raw, f"{path}.partitions[{k}]")
            if len(items) != 3:
                raise ConfigError(f"{path}.partitions[{k}]: must be [start, end, group]")
            group = _parse_list(items[2], f"{path}.partitions[{k}].group")
            windows.append((items[0], items[1], tuple(group)))
        return cls(
            ranges=tuple((str(k), v) for k, v in ranges.items()),
            p_drop=d["p_drop"],
            latency=d["latency"],
            jitter=d["jitter"],
            bandwidth=d["bandwidth"],
            partitions=tuple(windows),
        )


@dataclass(frozen=True, slots=True)
class FailureConfig:
    """Scheduled agent failures for controlled experiments (ADR-0021).

    At reset, ``count`` distinct agents are drawn uniformly without replacement from the roster,
    each with a failure step drawn uniformly from ``start..end`` (inclusive), using the episode's
    ``failure`` stream. An agent still active at the end of its failure step is disabled with
    cause ``"injected"``. ``count = 0`` injects nothing.
    """

    count: int
    start: int
    end: int

    def __post_init__(self) -> None:
        _check_int(self.count, "failures.count", minimum=0)
        _check_int(self.start, "failures.start", minimum=1, maximum=MAX_TIME_LIMIT)
        _check_int(self.end, "failures.end", minimum=self.start, maximum=MAX_TIME_LIMIT)

    def to_dict(self) -> dict[str, Any]:
        return {"count": self.count, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, data: object, path: str = "failures") -> FailureConfig:
        d = _require_mapping(data, path)
        _check_keys(d, path, {"count", "start", "end"}, set())
        return cls(count=d["count"], start=d["start"], end=d["end"])


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """A scenario family. Together with a root seed it fully determines an episode's scenario.

    Agent IDs are ``"<agent_type>_<k>"`` where ``k`` counts that type in roster order.
    ``base_position`` defaults to the grid centre ``(height // 2, width // 2)``.
    ``hazards`` is ``None`` for no hazard layer (ADR-0011). ``p_false_positive`` is the per
    footprint-cell false-report probability (ADR-0012). ``recall_target`` in ``(0, 1]`` is the
    global recall at which the mission succeeds (ADR-0009). ``comms`` is ``None`` for no
    communication channel (ADR-0017). The defaults reproduce M1 exactly.
    """

    height: int
    width: int
    terrain: TerrainConfig
    survivor_count: int
    agent_types: tuple[AgentTypeSpec, ...]
    roster: tuple[AgentSpec, ...]
    time_limit: int
    p_detect: float
    base_position: Position | None = None
    hazards: HazardConfig | None = None
    p_false_positive: float = 0.0
    recall_target: float = 1.0
    comms: CommsConfig | None = None
    failures: FailureConfig | None = None

    def __post_init__(self) -> None:
        _check_int(self.height, "height", minimum=1, maximum=MAX_GRID_SIDE)
        _check_int(self.width, "width", minimum=1, maximum=MAX_GRID_SIDE)
        if not isinstance(self.terrain, TerrainConfig):
            raise ConfigError(f"terrain: must be a TerrainConfig, got {self.terrain!r}")
        _check_int(self.survivor_count, "survivor_count", minimum=0)
        if self.survivor_count > self.height * self.width - 1:
            raise ConfigError(
                f"survivor_count: at most {self.height * self.width - 1} cells exist besides the "
                f"base, got {self.survivor_count}"
            )
        _check_int(self.time_limit, "time_limit", minimum=1, maximum=MAX_TIME_LIMIT)
        object.__setattr__(self, "p_detect", _check_fraction(self.p_detect, "p_detect"))
        object.__setattr__(
            self, "p_false_positive", _check_fraction(self.p_false_positive, "p_false_positive")
        )
        target = _check_fraction(self.recall_target, "recall_target")
        if target <= 0.0:
            raise ConfigError(f"recall_target: must be in (0, 1], got {self.recall_target!r}")
        object.__setattr__(self, "recall_target", target)
        if self.hazards is not None and not isinstance(self.hazards, HazardConfig):
            raise ConfigError(f"hazards: must be a HazardConfig or None, got {self.hazards!r}")

        object.__setattr__(self, "agent_types", tuple(self.agent_types))
        object.__setattr__(self, "roster", tuple(self.roster))
        if not self.agent_types:
            raise ConfigError("agent_types: at least one agent type is required")
        names = [t.name for t in self.agent_types]
        for i, spec in enumerate(self.agent_types):
            if not isinstance(spec, AgentTypeSpec):
                raise ConfigError(f"agent_types[{i}]: must be an AgentTypeSpec, got {spec!r}")
        if len(set(names)) != len(names):
            raise ConfigError(f"agent_types: names must be unique, got {names}")
        if not self.roster:
            raise ConfigError("roster: at least one agent is required")
        capacities = {t.name: t.energy_capacity for t in self.agent_types}
        for i, entry in enumerate(self.roster):
            if not isinstance(entry, AgentSpec):
                raise ConfigError(f"roster[{i}]: must be an AgentSpec, got {entry!r}")
            if entry.agent_type not in capacities:
                raise ConfigError(
                    f"roster[{i}].agent_type: unknown type {entry.agent_type!r}; "
                    f"defined types are {sorted(capacities)}"
                )
            cap = capacities[entry.agent_type]
            if entry.initial_energy is not None and entry.initial_energy > cap:
                raise ConfigError(
                    f"roster[{i}].initial_energy: must be <= energy_capacity ({cap}), "
                    f"got {entry.initial_energy}"
                )

        if self.comms is not None:
            if not isinstance(self.comms, CommsConfig):
                raise ConfigError(f"comms: must be a CommsConfig or None, got {self.comms!r}")
            typed = {name for name, _ in self.comms.ranges}
            if typed != set(names):
                raise ConfigError(
                    f"comms.ranges: must give a range for exactly the agent types {sorted(names)}, "
                    f"got {sorted(typed)}"
                )

        if self.comms is not None and self.comms.partitions:
            ids = set(self.agents())
            for k, (_, _, group) in enumerate(self.comms.partitions):
                unknown = sorted(set(group) - ids)
                if unknown:
                    raise ConfigError(
                        f"comms.partitions[{k}].group: unknown agent IDs {unknown}; "
                        f"agents are {sorted(ids)}"
                    )
        if self.failures is not None:
            if not isinstance(self.failures, FailureConfig):
                raise ConfigError(
                    f"failures: must be a FailureConfig or None, got {self.failures!r}"
                )
            if self.failures.count > len(self.roster):
                raise ConfigError(
                    f"failures.count: at most {len(self.roster)} agents exist, "
                    f"got {self.failures.count}"
                )
            if self.failures.end > self.time_limit:
                raise ConfigError(
                    f"failures.end: must be <= time_limit ({self.time_limit}), "
                    f"got {self.failures.end}"
                )

        if self.base_position is not None:
            bp = self.base_position
            if not (isinstance(bp, tuple) and len(bp) == 2):
                raise ConfigError(f"base_position: must be a (row, col) tuple, got {bp!r}")
            _check_int(bp[0], "base_position[0]", minimum=0, maximum=self.height - 1)
            _check_int(bp[1], "base_position[1]", minimum=0, maximum=self.width - 1)

    # ----------------------------------------------------------------- derived values

    @property
    def base(self) -> Position:
        """The base cell (explicit ``base_position`` or the grid centre)."""
        if self.base_position is not None:
            return self.base_position
        return (self.height // 2, self.width // 2)

    def type_spec(self, name: str) -> AgentTypeSpec:
        """Return the :class:`AgentTypeSpec` called ``name``."""
        for spec in self.agent_types:
            if spec.name == name:
                return spec
        raise KeyError(name)

    def agents(self) -> dict[AgentId, tuple[AgentTypeSpec, int]]:
        """Map each agent ID to its type spec and initial energy, sorted by agent ID."""
        counters: dict[str, int] = {}
        result: dict[AgentId, tuple[AgentTypeSpec, int]] = {}
        for entry in self.roster:
            k = counters.get(entry.agent_type, 0)
            counters[entry.agent_type] = k + 1
            spec = self.type_spec(entry.agent_type)
            energy = spec.energy_capacity if entry.initial_energy is None else entry.initial_energy
            result[f"{entry.agent_type}_{k}"] = (spec, energy)
        return dict(sorted(result.items()))

    # ----------------------------------------------------------------- serialization

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready representation, including ``schema_version``.

        Without a channel or failures the result is the version-2 form (no ``comms`` key), so
        existing config hashes are unchanged. ``failures`` is written only when set.
        """
        v2 = self.comms is None and self.failures is None
        out: dict[str, Any] = {
            "schema_version": 2 if v2 else SCHEMA_VERSION,
            "height": self.height,
            "width": self.width,
            "base_position": None if self.base_position is None else list(self.base_position),
            "terrain": self.terrain.to_dict(),
            "survivor_count": self.survivor_count,
            "agent_types": [t.to_dict() for t in self.agent_types],
            "roster": [r.to_dict() for r in self.roster],
            "time_limit": self.time_limit,
            "p_detect": self.p_detect,
            "hazards": None if self.hazards is None else self.hazards.to_dict(),
            "p_false_positive": self.p_false_positive,
            "recall_target": self.recall_target,
        }
        if not v2:
            out["comms"] = None if self.comms is None else self.comms.to_dict()
        if self.failures is not None:
            out["failures"] = self.failures.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: object) -> ScenarioConfig:
        """Build a config from its JSON representation. Strict: see module docstring.

        Schema version 1 is migrated to version 2 (see :data:`SCHEMA_VERSION`).
        """
        d = _require_mapping(data, "config")
        version = d.get("schema_version")
        if version not in READABLE_SCHEMA_VERSIONS or isinstance(version, bool):
            raise ConfigError(
                f"schema_version: this code reads versions {sorted(READABLE_SCHEMA_VERSIONS)}, "
                f"got {version!r}"
            )
        required = {
            "schema_version",
            "height",
            "width",
            "terrain",
            "survivor_count",
            "agent_types",
            "roster",
            "time_limit",
            "p_detect",
        }
        if version >= 2:
            required |= {"hazards", "p_false_positive", "recall_target"}
        if version >= 3:
            required |= {"comms"}
        optional = {"base_position"} | ({"failures"} if version >= 3 else set())
        _check_keys(d, "config", required, optional)
        base_raw = d.get("base_position")
        base: Position | None = None
        if base_raw is not None:
            items = _parse_list(base_raw, "base_position")
            if len(items) != 2:
                raise ConfigError(f"base_position: must be [row, col], got {base_raw!r}")
            base = (items[0], items[1])
        hazards_raw = d.get("hazards")
        return cls(
            height=d["height"],
            width=d["width"],
            base_position=base,
            terrain=TerrainConfig.from_dict(d["terrain"]),
            survivor_count=d["survivor_count"],
            agent_types=tuple(
                AgentTypeSpec.from_dict(t, f"agent_types[{i}]")
                for i, t in enumerate(_parse_list(d["agent_types"], "agent_types"))
            ),
            roster=tuple(
                AgentSpec.from_dict(r, f"roster[{i}]")
                for i, r in enumerate(_parse_list(d["roster"], "roster"))
            ),
            time_limit=d["time_limit"],
            p_detect=d["p_detect"],
            hazards=None if hazards_raw is None else HazardConfig.from_dict(hazards_raw),
            p_false_positive=d.get("p_false_positive", 0.0),
            recall_target=d.get("recall_target", 1.0),
            comms=None if d.get("comms") is None else CommsConfig.from_dict(d["comms"]),
            failures=None if d.get("failures") is None else FailureConfig.from_dict(d["failures"]),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> ScenarioConfig:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"config: invalid JSON ({exc})") from exc
        return cls.from_dict(data)

    def config_hash(self) -> str:
        """SHA-256 of the canonical JSON form (ADR-0007). Identifies the scenario family."""
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- defaults

DRONE: Final[AgentTypeSpec] = AgentTypeSpec(
    name="drone",
    layer=Layer.AIR,
    sensor_radius=3,
    energy_capacity=200,
    move_cost=2,
    stay_cost=1,
    recharge_rate=25,
)
"""Default aerial agent: wide sensing, small battery, hovering costs energy (ADR-0002 defaults)."""

GROUND: Final[AgentTypeSpec] = AgentTypeSpec(
    name="ground",
    layer=Layer.GROUND,
    sensor_radius=1,
    energy_capacity=600,
    move_cost=1,
    stay_cost=0,
    recharge_rate=25,
)
"""Default ground agent: short-range sensing, large battery, idle is free (ADR-0002 defaults)."""

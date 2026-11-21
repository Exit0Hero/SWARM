"""Dashboard back end (Phase 6, ADR-0026): every number here comes from a real simulator run.

The service never simulates anything itself. Episodes go through
:func:`swarm.experiments.runner.run_episode`. Frames are recorded with its read-only ``on_step``
hook. Comparisons go through :func:`swarm.experiments.experiment.run_experiment`. Results are
stored as JSON under a data directory, so the run history survives restarts and can be exported.

The dashboard is an *evaluation* view. It shows ground truth (survivor positions, true/false
detection labels), clearly marked as such. That truth never reaches a policy (ADR-0008).
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from swarm.core.env import StepResult, SwarmEnv
from swarm.core.events import (
    AgentDisabled,
    Detection,
    MessageDelivered,
    MessageDropped,
    MessageSent,
    PerceptionError,
    events_to_jsonl,
)
from swarm.core.perception import SENSOR_MODELS
from swarm.core.types import AgentStatus
from swarm.evaluation.statistics import BootstrapSpec, bootstrap_mean_ci, sign_flip_p_value
from swarm.experiments.experiment import run_experiment
from swarm.experiments.runner import EpisodeResult, describe_policy, run_episode
from swarm.experiments.study import deep_merge
from swarm.policies.registry import POLICIES
from swarm.scenarios.config import ConfigError, ScenarioConfig

MAX_EXPERIMENT_EPISODES: Final[int] = 600
"""Cap on policies x seeds per dashboard experiment, to keep the server responsive."""

COMPARE_METRICS: Final[tuple[str, ...]] = (
    "recall",
    "mission_success",
    "time_to_recall_target",
    "coverage_sensed",
    "energy_used",
    "agents_disabled",
    "precision",
)


class DashboardError(ValueError):
    """A request the dashboard cannot serve; the message is shown to the user."""


def learned_catalog(models_root: Path | None) -> dict[str, Any]:
    """Trained sector-selection actors (Phase 4) under ``models_root`` as policy factories.

    Each ``<name>/model.json`` with a validated ``best`` actor is exposed as ``learned:<name>``,
    in the mode chosen on validation seeds. Inference is NumPy-only (no learning extra needed).
    """
    from swarm.learning.sector import learned_policy

    found: dict[str, Any] = {}
    if models_root is None or not models_root.is_dir():
        return found
    for model in sorted(models_root.glob("*/model.json")):
        best = json.loads(model.read_text(encoding="utf-8")).get("best")
        actor = model.parent / "best_actor.npz"
        if best and actor.exists():
            name = f"learned:{model.parent.name}"
            found[name] = learned_policy(name, actor, sample=best.get("mode") == "sampled")
    return found


def resolve(policy: str, learned: dict[str, Any] | None = None) -> Any:
    """A registered or learned policy factory; raises ``DashboardError`` if unknown."""
    table = {**POLICIES, **(learned or {})}
    if policy not in table:
        raise DashboardError(f"unknown policy {policy!r}")
    return table[policy]


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "value") and not isinstance(value, int | float | str | bool):
        return value.value
    return value


# ----------------------------------------------------------------- configs


def build_config(base: dict[str, Any], overrides: dict[str, Any] | None) -> ScenarioConfig:
    """Validate ``base`` with ``overrides`` deep-merged in (``null`` removes a block)."""
    merged = deep_merge(base, overrides or {})
    if any(merged.get(k) is not None for k in ("comms", "failures", "perception")):
        merged["schema_version"] = max(int(merged.get("schema_version", 3)), 3)
        merged.setdefault("comms", None)
    try:
        return ScenarioConfig.from_dict(merged)
    except (ConfigError, KeyError, TypeError) as exc:
        raise DashboardError(f"invalid configuration: {exc}") from exc


def policy_catalog(
    config: ScenarioConfig, learned: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Every registered and learned policy, whether it can run on ``config``, its oracle flag."""
    out = []
    table = {**POLICIES, **(learned or {})}
    for name in sorted(table):
        try:
            identity = describe_policy(table[name], config)
            out.append(
                {
                    "name": name,
                    "available": True,
                    "oracle": bool(identity["uses_oracle"]),
                    "information": identity["information"],
                }
            )
        except Exception as exc:  # policies refuse configs they cannot use (e.g. no comms)
            out.append({"name": name, "available": False, "reason": str(exc)})
    return out


# ----------------------------------------------------------------- episode recording


@dataclass
class _Recorder:
    frames: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, env: SwarmEnv, result: StepResult | None) -> None:
        snap = env.snapshot()
        events = () if result is None else result.events
        frame: dict[str, Any] = {
            "step": snap.step,
            "agents": [
                {
                    "id": a.agent_id,
                    "type": a.agent_type,
                    "layer": a.layer.name.lower(),
                    "pos": list(a.position),
                    "energy": a.energy,
                    "capacity": a.energy_capacity,
                    "radius": a.sensor_radius,
                    "active": a.status is AgentStatus.ACTIVE,
                }
                for a in snap.agents
            ],
            "detections": [
                {"agent": e.agent_id, "cell": list(e.cell), "true": e.true_positive}
                for e in events
                if isinstance(e, Detection)
            ],
            "messages": [
                {"sender": e.sender, "recipients": list(e.recipients), "size": e.size}
                for e in events
                if isinstance(e, MessageSent)
            ],
            "delivered": sum(isinstance(e, MessageDelivered) for e in events),
            "dropped": sum(isinstance(e, MessageDropped) for e in events),
            "disabled": [
                {"agent": e.agent_id, "reason": e.reason}
                for e in events
                if isinstance(e, AgentDisabled)
            ],
            "perception_errors": sum(isinstance(e, PerceptionError) for e in events),
            "found": 0,  # filled from the event stream after the run
        }
        self.frames.append(frame)


def _scenario_view(env_snapshot: Any) -> dict[str, Any]:
    sc = env_snapshot.scenario
    return {
        "shape": list(sc.shape),
        "terrain": sc.terrain.tolist(),
        "hazard": sc.hazard_grid.astype(int).tolist(),
        "base": list(sc.base),
        "survivors_ground_truth": [list(s) for s in sc.survivors],
    }


def record_episode(
    config: ScenarioConfig, policy: str, seed: int, learned: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run one episode and return its full dashboard record (frames, metrics, scenario)."""
    factory = resolve(policy, learned)
    recorder = _Recorder()
    try:
        result: EpisodeResult = run_episode(
            config, factory, seed, designation="development", on_step=recorder
        )
    except ValueError as exc:
        raise DashboardError(str(exc)) from exc
    found: set[int] = set()
    for frame, step_events in zip(recorder.frames, _events_by_step(result), strict=False):
        for e in step_events:
            if isinstance(e, Detection) and e.true_positive and e.survivor_index is not None:
                found.add(e.survivor_index)
        frame["found"] = len(found)
    assert result.final_snapshot is not None
    return {
        "id": uuid.uuid4().hex[:12],
        "kind": "run",
        "created": _now(),
        "policy": policy,
        "seed": seed,
        "config": config.to_dict(),
        "config_hash": config.config_hash(),
        "steps": result.steps,
        "end_reason": result.end_reason,
        "event_stream_sha256": result.event_stream_sha256,
        "metrics": _jsonable(result.metrics.to_dict()),
        "extension_metrics": _jsonable(result.comm_metrics or {}),
        "diagnostics": _jsonable(result.policy_diagnostics or {}),
        "scenario": _scenario_view(result.final_snapshot),
        "frames": recorder.frames,
        "events_jsonl": events_to_jsonl(result.events),
    }


def _events_by_step(result: EpisodeResult) -> list[list[Any]]:
    """Events grouped as the recorder saw them: [] for reset, then one list per step."""
    groups: list[list[Any]] = [[] for _ in range(result.steps + 1)]
    for e in result.events:
        # perception errors from the reset observation carry step 0 but arrive with step 1
        groups[min(max(e.step, 1), result.steps)].append(e)
    return groups


# ----------------------------------------------------------------- experiments


def run_comparison(
    config: ScenarioConfig,
    policies: list[str],
    seeds: list[int],
    learned: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run every policy on the same seeds; summaries plus paired differences vs the first."""
    if not policies or not seeds:
        raise DashboardError("choose at least one policy and one seed")
    unknown = sorted(set(policies) - set(POLICIES) - set(learned or {}))
    if unknown:
        raise DashboardError(f"unknown policies {unknown}")
    if len(policies) * len(seeds) > MAX_EXPERIMENT_EPISODES:
        raise DashboardError(
            f"{len(policies) * len(seeds)} episodes requested; the dashboard runs at most "
            f"{MAX_EXPERIMENT_EPISODES}. Use python -m swarm.experiments.study for larger runs."
        )
    spec = BootstrapSpec(resamples=2000)
    per_policy: dict[str, Any] = {}
    values: dict[str, dict[str, list[Any]]] = {}
    for policy in policies:
        try:
            result = run_experiment(
                config, resolve(policy, learned), seeds, bootstrap=spec, designation="development"
            )
        except ValueError as exc:
            raise DashboardError(f"{policy}: {exc}") from exc
        record = json.loads(result.to_json())
        summary = {m: record["summary"][m] for m in COMPARE_METRICS if m in record["summary"]}
        per_policy[policy] = {
            "summaries": summary,
            "oracle": bool(record["policy"]["uses_oracle"]),
            "information": record["policy"]["information"],
            "failed": sum(ep["status"] != "completed" for ep in record["episodes"]),
            "duration_seconds": record["metadata"]["run"]["duration_seconds"],
        }
        values[policy] = {
            m: [(ep.get("metrics") or {}).get(m) for ep in record["episodes"]]
            for m in COMPARE_METRICS
        }
    baseline = policies[0]
    paired = []
    for policy in policies[1:]:
        for m in COMPARE_METRICS:
            diffs = [
                float(b) - float(a)
                for a, b in zip(values[baseline][m], values[policy][m], strict=True)
                if a is not None and b is not None
            ]
            ci = bootstrap_mean_ci(np.array(diffs), spec, f"{baseline}|{policy}|{m}")
            paired.append(
                {
                    "a": baseline,
                    "b": policy,
                    "metric": m,
                    "n": len(diffs),
                    "mean_difference": float(np.mean(diffs)) if diffs else None,
                    "ci_low": None if ci is None else ci[0],
                    "ci_high": None if ci is None else ci[1],
                    "p_value_unadjusted": sign_flip_p_value(diffs, resamples=2000),
                }
            )
    rows = [
        {"policy": p, "seed": s} | {m: values[p][m][k] for m in COMPARE_METRICS}
        for p in policies
        for k, s in enumerate(seeds)
    ]
    return {
        "id": uuid.uuid4().hex[:12],
        "kind": "experiment",
        "created": _now(),
        "config": config.to_dict(),
        "config_hash": config.config_hash(),
        "policies": policies,
        "seeds": seeds,
        "designation": "development",
        "note": "Exploratory dashboard run: no pre-registration or multiple-comparison correction.",
        "per_policy": per_policy,
        "paired_vs_first": paired,
        "seed_rows": rows,
    }


# ----------------------------------------------------------------- storage


class Store:
    """Run and experiment records as JSON files under ``root``; thread-safe."""

    def __init__(self, root: Path) -> None:
        self.root = root
        for sub in ("runs", "experiments", "configs"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, kind: str, rid: str) -> Path:
        if not rid.isalnum():
            raise DashboardError(f"bad id {rid!r}")
        return self.root / kind / f"{rid}.json"

    def save(self, record: dict[str, Any]) -> None:
        kind = "runs" if record["kind"] == "run" else "experiments"
        with self._lock:
            self._path(kind, record["id"]).write_text(json.dumps(record), encoding="utf-8")

    def load(self, kind: str, rid: str) -> dict[str, Any]:
        path = self._path(kind, rid)
        if not path.exists():
            raise DashboardError(f"no {kind[:-1]} {rid!r}")
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data

    def delete(self, kind: str, rid: str) -> None:
        path = self._path(kind, rid)
        if not path.exists():
            raise DashboardError(f"no {kind[:-1]} {rid!r}")
        path.unlink()

    def history(self) -> list[dict[str, Any]]:
        items = []
        for kind in ("runs", "experiments"):
            for path in (self.root / kind).glob("*.json"):
                r = json.loads(path.read_text(encoding="utf-8"))
                items.append(
                    {
                        "id": r["id"],
                        "kind": r["kind"],
                        "created": r["created"],
                        "config_hash": r["config_hash"][:12],
                        "policy": r.get("policy") or ", ".join(r.get("policies", [])),
                        "seed": r.get("seed"),
                        "seeds": len(r.get("seeds", [])) or None,
                        "end_reason": r.get("end_reason"),
                        "recall": (r.get("metrics") or {}).get("recall"),
                    }
                )
        return sorted(items, key=lambda i: i["created"], reverse=True)

    def save_config(self, name: str, config: ScenarioConfig) -> None:
        if not name.replace("_", "").replace("-", "").isalnum():
            raise DashboardError("config name: letters, digits, '-' and '_' only")
        (self.root / "configs" / f"{name}.json").write_text(config.to_json(), encoding="utf-8")


def list_configs(roots: list[Path]) -> dict[str, dict[str, Any]]:
    """Every JSON file under ``roots`` that parses as a scenario config, keyed by a label."""
    found: dict[str, dict[str, Any]] = {}
    for root in roots:
        for path in sorted(root.rglob("*.json")):
            try:
                config = ScenarioConfig.from_json(path.read_text(encoding="utf-8"))
            except Exception:  # study specs and other JSON files are not scenario configs
                continue
            found[str(path.relative_to(root.parent))] = config.to_dict()
    return found


def export_run(record: dict[str, Any], fmt: str) -> tuple[str, str, str]:
    """(content, mime type, filename) for a run export."""
    rid = record["id"]
    if fmt == "json":
        slim = {k: v for k, v in record.items() if k != "events_jsonl"}
        return json.dumps(slim, indent=2), "application/json", f"swarm-run-{rid}.json"
    if fmt == "events":
        return record["events_jsonl"], "application/x-ndjson", f"swarm-run-{rid}-events.jsonl"
    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["metric", "value"])
        for k, v in {**record["metrics"], **record["extension_metrics"]}.items():
            writer.writerow([k, v])
        return buf.getvalue(), "text/csv", f"swarm-run-{rid}-metrics.csv"
    raise DashboardError(f"unknown export format {fmt!r} (json, events, csv)")


def export_experiment(record: dict[str, Any], fmt: str) -> tuple[str, str, str]:
    rid = record["id"]
    if fmt == "json":
        return json.dumps(record, indent=2), "application/json", f"swarm-experiment-{rid}.json"
    if fmt == "csv":
        buf = io.StringIO()
        rows = record["seed_rows"]
        writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        return buf.getvalue(), "text/csv", f"swarm-experiment-{rid}.csv"
    raise DashboardError(f"unknown export format {fmt!r} (json, csv)")


def meta() -> dict[str, Any]:
    return {"policies": sorted(POLICIES), "sensor_models": sorted(SENSOR_MODELS)}

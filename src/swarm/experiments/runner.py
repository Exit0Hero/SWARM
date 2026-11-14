"""Run one episode and record a reproducible result (ADR-0010, docs/11).

The runner is the only place where policies, the environment and the evaluator meet. Policies
see only their own observation and RNG stream; the evaluator sees the scenario and events after
the episode.

Reproducibility contract: for a fixed config, seed, policy and code revision, the event stream
(``event_stream_sha256``) and all metric values are identical across runs. Only the fields under
``metadata.run`` (run ID, timestamps, duration, host) differ.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

import swarm
from swarm.core.env import EnvSnapshot, SwarmEnv
from swarm.core.events import EVENT_SCHEMA_VERSION, Event, event_stream_sha256, events_to_jsonl
from swarm.core.rng import episode_streams, seed_derivation, validate_seed
from swarm.evaluation.metrics import (
    METRIC_DEFINITIONS,
    METRIC_SCHEMA_VERSION,
    EpisodeMetrics,
    compute_metrics,
)
from swarm.policies.base import Policy, PolicyFactory
from swarm.scenarios.config import SCHEMA_VERSION as CONFIG_SCHEMA_VERSION
from swarm.scenarios.config import ScenarioConfig

RESULT_SCHEMA_VERSION: Final[int] = 1
DESIGNATIONS: Final[frozenset[str]] = frozenset({"evaluation", "development"})


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Everything needed to audit and reproduce one episode."""

    config: ScenarioConfig
    seed: int
    policy: dict[str, Any]
    designation: str
    steps: int
    end_reason: str
    terminated: bool
    truncated: bool
    scenario_sha256: str
    event_stream_sha256: str
    metrics: EpisodeMetrics
    events: tuple[Event, ...]
    metadata: dict[str, Any]
    final_snapshot: EnvSnapshot
    """Final state including ground truth; for rendering and checks, not serialized."""

    def to_dict(self) -> dict[str, Any]:
        """Versioned JSON-ready record. Events are written separately (JSON Lines)."""
        return {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "designation": self.designation,
            "seed": self.seed,
            "policy": self.policy,
            "config": self.config.to_dict(),
            "config_hash": self.config.config_hash(),
            "episode": {
                "steps": self.steps,
                "end_reason": self.end_reason,
                "terminated": self.terminated,
                "truncated": self.truncated,
                "event_count": len(self.events),
            },
            "scenario_sha256": self.scenario_sha256,
            "event_stream_sha256": self.event_stream_sha256,
            "metrics": self.metrics.to_dict(),
            "metric_definitions": {
                name: {
                    "definition": d.definition,
                    "unit": d.unit,
                    "undefined_when": d.undefined_when,
                }
                for name, d in METRIC_DEFINITIONS.items()
            },
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def events_jsonl(self) -> str:
        return events_to_jsonl(self.events)


def run_episode(
    config: ScenarioConfig,
    policy_factory: PolicyFactory,
    seed: int,
    *,
    designation: str = "evaluation",
) -> EpisodeResult:
    """Run one episode of ``config`` with one policy per agent built by ``policy_factory``.

    Raises:
        ValueError: on an invalid seed or designation, a policy that requires oracle access
            (not available before Phase 2), or a policy that returns an invalid action type.
    """
    validate_seed(seed)
    if designation not in DESIGNATIONS:
        raise ValueError(f"designation: must be one of {sorted(DESIGNATIONS)}, got {designation!r}")
    started_wall = dt.datetime.now(dt.UTC)
    started = time.perf_counter()

    env = SwarmEnv(config)
    agents = config.agents()
    policies: dict[str, Policy] = {
        aid: policy_factory(aid, spec) for aid, (spec, _) in agents.items()
    }
    _check_policies(policies)
    policy_rngs = episode_streams(seed, env.possible_agents).policies

    observations, _ = env.reset(seed)
    events: list[Event] = []
    while not env.done:
        actions = {
            aid: policies[aid].act(observations[aid], policy_rngs[aid]) for aid in env.agents
        }
        result = env.step(actions)
        events.extend(result.events)
        observations = result.observations

    snapshot = env.snapshot()
    assert snapshot.end_reason is not None
    metrics = compute_metrics(config, snapshot.scenario, events)
    first = next(iter(policies.values()))
    return EpisodeResult(
        config=config,
        seed=seed,
        policy={"name": first.name, "version": first.version, "uses_oracle": first.uses_oracle},
        designation=designation,
        steps=snapshot.step,
        end_reason=snapshot.end_reason,
        terminated=snapshot.end_reason == "all_disabled",
        truncated=snapshot.end_reason == "time_limit",
        scenario_sha256=snapshot.scenario.fingerprint(),
        event_stream_sha256=event_stream_sha256(events),
        metrics=metrics,
        events=tuple(events),
        metadata=_metadata(seed, env.possible_agents, started_wall, time.perf_counter() - started),
        final_snapshot=snapshot,
    )


def write_result(result: EpisodeResult, out: Path, events_out: Path | None = None) -> None:
    """Write the JSON result (and optionally the JSON Lines event stream)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.to_json(), encoding="utf-8")
    if events_out is not None:
        events_out.parent.mkdir(parents=True, exist_ok=True)
        events_out.write_text(result.events_jsonl(), encoding="utf-8")


def _check_policies(policies: dict[str, Policy]) -> None:
    identities = {(p.name, p.version, p.uses_oracle) for p in policies.values()}
    if len(identities) != 1:
        raise ValueError(
            f"policy_factory: all agents must use one policy identity in M1, got {identities}"
        )
    if any(p.uses_oracle for p in policies.values()):
        raise ValueError("oracle policies need an OracleView, which arrives in Phase 2 (ADR-0008)")


def _git_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    if not (root / "pyproject.toml").is_file() or not (root / ".git").exists():
        return {"commit": None, "dirty": None}  # installed package: no source checkout
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def environment_metadata() -> dict[str, Any]:
    """Software versions and source revision, shared by episode and experiment records."""
    return {
        "software": {
            "swarm": swarm.__version__,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "code": _git_state(),
    }


def _metadata(
    seed: int, agent_ids: tuple[str, ...], started: dt.datetime, duration: float
) -> dict[str, Any]:
    return {
        "seed_derivation": seed_derivation(seed, agent_ids),
        "schema_versions": {
            "config": CONFIG_SCHEMA_VERSION,
            "events": EVENT_SCHEMA_VERSION,
            "metrics": METRIC_SCHEMA_VERSION,
            "result": RESULT_SCHEMA_VERSION,
        },
        **environment_metadata(),
        "run": {
            "run_id": uuid.uuid4().hex,
            "started_at": started.isoformat(timespec="seconds"),
            "duration_seconds": round(duration, 6),
            "platform": platform.platform(),
        },
    }

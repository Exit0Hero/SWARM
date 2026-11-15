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
from swarm.policies.base import (
    CommunicatingTeam,
    MissionBrief,
    PolicyFactory,
    TeamFactory,
    TeamPolicy,
    build_team,
    policy_identity,
)
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
    policy_diagnostics: dict[str, Any] | None = None
    decision_log: tuple[dict[str, Any], ...] = ()

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
            "policy_diagnostics": self.policy_diagnostics or {},
            "decision_log": list(self.decision_log),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def events_jsonl(self) -> str:
        return events_to_jsonl(self.events)


def run_episode(
    config: ScenarioConfig,
    policy_factory: PolicyFactory | TeamFactory,
    seed: int,
    *,
    designation: str = "evaluation",
) -> EpisodeResult:
    """Run one episode of ``config`` with the policy built by ``policy_factory``.

    ``policy_factory`` is either a per-agent :data:`~swarm.policies.base.PolicyFactory` (wrapped
    in an ``IndependentTeam``, Phase 1 behaviour) or a :class:`~swarm.policies.base.TeamFactory`
    (ADR-0014). Oracle teams receive :meth:`SwarmEnv.oracle_view` after reset; no other policy
    does. A team that implements :class:`~swarm.policies.base.CommunicatingTeam` has its outbox
    passed to the environment each step; this requires a ``comms`` block in ``config``.

    Raises:
        ValueError: on an invalid seed or designation, an inconsistent policy identity, or
            actions for the wrong set of agents.
    """
    validate_seed(seed)
    if designation not in DESIGNATIONS:
        raise ValueError(f"designation: must be one of {sorted(DESIGNATIONS)}, got {designation!r}")
    started_wall = dt.datetime.now(dt.UTC)
    started = time.perf_counter()

    env = SwarmEnv(config)
    team = build_team(policy_factory, MissionBrief.from_config(config))
    policy_rngs = episode_streams(seed, env.possible_agents).policies

    observations, _ = env.reset(seed)
    if team.uses_oracle:
        team.attach_oracle(env.oracle_view())
    events: list[Event] = []
    while not env.done:
        actions = team.act(observations, policy_rngs)
        if set(actions) != set(env.agents):
            raise ValueError(
                f"{team.name}: returned actions for {sorted(actions)}, expected {list(env.agents)}"
            )
        messages = team.outbox() if isinstance(team, CommunicatingTeam) else None
        if messages is not None and config.comms is None:
            raise ValueError(
                f"{team.name}: sends messages, but the config has no comms block (ADR-0017)"
            )
        result = env.step(actions, messages)
        events.extend(result.events)
        observations = result.observations

    snapshot = env.snapshot()
    assert snapshot.end_reason is not None
    metrics = compute_metrics(config, snapshot.scenario, events)
    return EpisodeResult(
        config=config,
        seed=seed,
        policy=policy_identity(team),
        designation=designation,
        steps=snapshot.step,
        end_reason=snapshot.end_reason,
        terminated=snapshot.end_reason != "time_limit",
        truncated=snapshot.end_reason == "time_limit",
        scenario_sha256=snapshot.scenario.fingerprint(),
        event_stream_sha256=event_stream_sha256(events),
        metrics=metrics,
        events=tuple(events),
        metadata=_metadata(
            config, seed, env.possible_agents, started_wall, time.perf_counter() - started
        ),
        final_snapshot=snapshot,
        policy_diagnostics=dict(team.diagnostics()),
        decision_log=tuple(team.decision_log()),
    )


def describe_policy(
    policy_factory: PolicyFactory | TeamFactory, config: ScenarioConfig
) -> dict[str, Any]:
    """The policy identity record, without running an episode."""
    team: TeamPolicy = build_team(policy_factory, MissionBrief.from_config(config))
    return policy_identity(team)


def write_result(result: EpisodeResult, out: Path, events_out: Path | None = None) -> None:
    """Write the JSON result (and optionally the JSON Lines event stream)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.to_json(), encoding="utf-8")
    if events_out is not None:
        events_out.parent.mkdir(parents=True, exist_ok=True)
        events_out.write_text(result.events_jsonl(), encoding="utf-8")


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
    config: ScenarioConfig,
    seed: int,
    agent_ids: tuple[str, ...],
    started: dt.datetime,
    duration: float,
) -> dict[str, Any]:
    return {
        "seed_derivation": seed_derivation(seed, agent_ids),
        "schema_versions": {
            "config": config.to_dict()["schema_version"],
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

"""Multi-seed experiments and paired policy comparisons (ADR-0009 revision 2, docs/05).

An experiment is one scenario config, one policy and an explicit list of evaluation seeds. Every
seed produces an :class:`EpisodeRecord`, completed or failed; nothing is dropped. Summaries are
computed over completed episodes and report how many seeds failed or gave undefined values.

A comparison requires the same config hash and the same seed list (same order), and uses paired
per-seed differences.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import time
import traceback
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from swarm.core.rng import validate_seed
from swarm.evaluation.metrics import METRIC_DEFINITIONS, METRIC_SCHEMA_VERSION
from swarm.evaluation.statistics import (
    BootstrapSpec,
    MetricSummary,
    PairedSummary,
    paired_difference,
    summarize,
)
from swarm.experiments.runner import (
    DESIGNATIONS,
    environment_metadata,
    run_episode,
)
from swarm.policies.base import PolicyFactory
from swarm.scenarios.config import SCHEMA_VERSION as CONFIG_SCHEMA_VERSION
from swarm.scenarios.config import ScenarioConfig

EXPERIMENT_SCHEMA_VERSION: Final[int] = 1
COMPARISON_SCHEMA_VERSION: Final[int] = 1


class ComparisonError(ValueError):
    """Two experiments cannot be compared (different seeds, config or designation)."""


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """One seed's outcome. ``status`` is ``"completed"`` or ``"failed"``."""

    seed: int
    status: str
    end_reason: str | None = None
    steps: int | None = None
    scenario_sha256: str | None = None
    event_stream_sha256: str | None = None
    metrics: dict[str, Any] | None = None
    error: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "status": self.status,
            "end_reason": self.end_reason,
            "steps": self.steps,
            "scenario_sha256": self.scenario_sha256,
            "event_stream_sha256": self.event_stream_sha256,
            "metrics": self.metrics,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EpisodeRecord:
        return cls(**d)


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    config: ScenarioConfig
    policy: dict[str, Any]
    seeds: tuple[int, ...]
    designation: str
    bootstrap: BootstrapSpec
    episodes: tuple[EpisodeRecord, ...]
    summary: dict[str, MetricSummary]
    metadata: dict[str, Any]

    @property
    def complete(self) -> bool:
        """``True`` only if every seed's episode completed."""
        return all(e.status == "completed" for e in self.episodes)

    @property
    def config_hash(self) -> str:
        return self.config.config_hash()

    def to_dict(self) -> dict[str, Any]:
        failed = [e.seed for e in self.episodes if e.status == "failed"]
        return {
            "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
            "designation": self.designation,
            "policy": self.policy,
            "config": self.config.to_dict(),
            "config_hash": self.config_hash,
            "seeds": list(self.seeds),
            "complete": self.complete,
            "failed_seeds": failed,
            "end_reasons": dict(
                sorted(Counter(e.end_reason or "failed" for e in self.episodes).items())
            ),
            "bootstrap": self.bootstrap.to_dict(),
            "summary": {name: s.to_dict() for name, s in self.summary.items()},
            "metric_definitions": {
                name: {
                    "definition": d.definition,
                    "unit": d.unit,
                    "undefined_when": d.undefined_when,
                    "role": d.role,
                }
                for name, d in METRIC_DEFINITIONS.items()
            },
            "episodes": [e.to_dict() for e in self.episodes],
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentResult:
        """Rebuild a result from its JSON form (for comparisons of stored experiments)."""
        if d.get("experiment_schema_version") != EXPERIMENT_SCHEMA_VERSION:
            raise ValueError(
                f"experiment_schema_version: expected {EXPERIMENT_SCHEMA_VERSION}, "
                f"got {d.get('experiment_schema_version')!r}"
            )
        config = ScenarioConfig.from_dict(d["config"])
        if config.config_hash() != d["config_hash"]:
            raise ValueError("config_hash does not match the embedded config")
        b = d["bootstrap"]
        return cls(
            config=config,
            policy=dict(d["policy"]),
            seeds=tuple(d["seeds"]),
            designation=d["designation"],
            bootstrap=BootstrapSpec(b["resamples"], b["confidence"], b["seed"]),
            episodes=tuple(EpisodeRecord.from_dict(e) for e in d["episodes"]),
            summary={name: MetricSummary(**s) for name, s in d["summary"].items()},
            metadata=dict(d["metadata"]),
        )


def parse_seeds(text: str) -> list[int]:
    """Parse ``"0-9,20,25-29"`` into an ordered list of seeds (ranges inclusive)."""
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"seeds: empty item in {text!r}")
        lo, sep, hi = part.partition("-")
        try:
            start = int(lo)
            stop = int(hi) if sep else start
        except ValueError:
            raise ValueError(f"seeds: cannot parse {part!r}") from None
        if stop < start:
            raise ValueError(f"seeds: descending range {part!r}")
        seeds.extend(range(start, stop + 1))
    return seeds


def _validate_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    if not seeds:
        raise ValueError("seeds: at least one evaluation seed is required")
    for s in seeds:
        validate_seed(s)
    duplicates = sorted(s for s, k in Counter(seeds).items() if k > 1)
    if duplicates:
        raise ValueError(f"seeds: duplicates would double-count episodes: {duplicates}")
    return tuple(seeds)


def run_experiment(
    config: ScenarioConfig,
    policy_factory: PolicyFactory,
    seeds: Sequence[int],
    *,
    bootstrap: BootstrapSpec | None = None,
    designation: str = "evaluation",
) -> ExperimentResult:
    """Run one episode per seed and summarize every metric (ADR-0009).

    Any exception raised while running a seed's episode is recorded as a failed episode
    (type, message, traceback) and counted in every summary; the experiment then reports
    ``complete = False``. Invalid *inputs* (seeds, designation, an oracle policy) raise instead.
    """
    seed_list = _validate_seeds(seeds)
    spec = bootstrap or BootstrapSpec()
    if designation not in DESIGNATIONS:
        raise ValueError(f"designation: must be one of {sorted(DESIGNATIONS)}, got {designation!r}")
    agents = config.agents()
    first_id, (first_spec, _) = next(iter(agents.items()))
    probe = policy_factory(first_id, first_spec)
    if probe.uses_oracle:
        raise ValueError("oracle policies need an OracleView, which arrives in Phase 2 (ADR-0008)")
    policy = {"name": probe.name, "version": probe.version, "uses_oracle": probe.uses_oracle}

    started_wall = dt.datetime.now(dt.UTC)
    started = time.perf_counter()
    episodes: list[EpisodeRecord] = []
    for seed in seed_list:
        try:
            result = run_episode(config, policy_factory, seed, designation=designation)
        except Exception as exc:  # recorded, never dropped (ADR-0009)
            episodes.append(
                EpisodeRecord(
                    seed=seed,
                    status="failed",
                    error={
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(limit=5),
                    },
                )
            )
            continue
        episodes.append(
            EpisodeRecord(
                seed=seed,
                status="completed",
                end_reason=result.end_reason,
                steps=result.steps,
                scenario_sha256=result.scenario_sha256,
                event_stream_sha256=result.event_stream_sha256,
                metrics=result.metrics.to_dict(),
            )
        )

    completed = [e for e in episodes if e.status == "completed"]
    n_failed = len(episodes) - len(completed)
    summary = {
        name: summarize(
            [e.metrics[name] for e in completed if e.metrics is not None],
            n_failed=n_failed,
            spec=spec,
            metric=name,
        )
        for name in METRIC_DEFINITIONS
    }
    metadata = {
        "schema_versions": {
            "config": CONFIG_SCHEMA_VERSION,
            "metrics": METRIC_SCHEMA_VERSION,
            "experiment": EXPERIMENT_SCHEMA_VERSION,
        },
        **environment_metadata(),
        "run": {
            "run_id": uuid.uuid4().hex,
            "started_at": started_wall.isoformat(timespec="seconds"),
            "duration_seconds": round(time.perf_counter() - started, 6),
            "platform": platform.platform(),
        },
    }
    return ExperimentResult(
        config=config,
        policy=policy,
        seeds=seed_list,
        designation=designation,
        bootstrap=spec,
        episodes=tuple(episodes),
        summary=summary,
        metadata=metadata,
    )


def compare_experiments(
    baseline: ExperimentResult,
    candidate: ExperimentResult,
    *,
    bootstrap: BootstrapSpec | None = None,
) -> dict[str, Any]:
    """Paired comparison ``candidate - baseline`` per metric (ADR-0009).

    Raises:
        ComparisonError: if the config hashes, seed lists (including order) or designations
            differ. Seeds that failed in either experiment are excluded and listed.
    """
    if baseline.config_hash != candidate.config_hash:
        raise ComparisonError(
            f"config hashes differ: {baseline.config_hash[:12]} vs {candidate.config_hash[:12]}"
        )
    if baseline.seeds != candidate.seeds:
        raise ComparisonError("seed lists differ; comparisons need identical evaluation seeds")
    if baseline.designation != candidate.designation:
        raise ComparisonError(
            f"designations differ: {baseline.designation} vs {candidate.designation}"
        )
    spec = bootstrap or BootstrapSpec()
    a_eps = {e.seed: e for e in baseline.episodes}
    b_eps = {e.seed: e for e in candidate.episodes}
    usable = [
        s
        for s in baseline.seeds
        if a_eps[s].status == "completed" and b_eps[s].status == "completed"
    ]
    excluded = [s for s in baseline.seeds if s not in usable]
    metrics: dict[str, PairedSummary] = {}
    for name in METRIC_DEFINITIONS:
        a_vals = [(a_eps[s].metrics or {}).get(name) for s in usable]
        b_vals = [(b_eps[s].metrics or {}).get(name) for s in usable]
        metrics[name] = paired_difference(a_vals, b_vals, spec=spec, metric=name)
    return {
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "difference": "candidate - baseline, paired by seed",
        "baseline_policy": baseline.policy,
        "candidate_policy": candidate.policy,
        "config_hash": baseline.config_hash,
        "designation": baseline.designation,
        "seeds": list(baseline.seeds),
        "excluded_failed_seeds": excluded,
        "bootstrap": spec.to_dict(),
        "metrics": {name: s.to_dict() for name, s in metrics.items()},
    }

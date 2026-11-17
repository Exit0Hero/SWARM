"""Training entry point (Phase 4, ADR-0022). Needs ``[learning]``. Evaluation is separate.

Usage::

    python -m swarm.learning.train --config configs/phase4/train_mappo.json --seed 0 \\
        --out runs/phase4/mappo_s0 [--resume]

Outputs in ``--out``:

- ``train_log.jsonl``: one line per update (episodes, decisions, losses, entropy, KL, mean team
  return and reward components, end reasons) and one per validation.
- ``checkpoint.pt``: the full learner state, used for ``--resume``.
- ``last_actor.npz``: the latest actor, exported for NumPy inference.
- ``best_actor.npz``: the actor with the best validation score, selected on the validation
  seeds and never on evaluation seeds. ``model.json`` records whether it is best used greedily
  or sampled; both modes are validated, and the mode is part of the selection.
- ``model.json``: model version record. It holds the training config and its hash, the scenario
  config hash, the code revision, software versions, the training-seed range, the validation
  history and the selected update.

Validation:
- Every ``validate_every`` updates (and at the end), the exported actor is evaluated greedily with
  the normal runner on ``validation_seeds``.
- The score is the mean ``time_censored`` (lower is better). Ties are broken by higher mean
  recall.

Training scenario seeds are ``train_seed_start + 5000 * seed`` onward, all within 10000-59999.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from swarm.experiments.experiment import parse_seeds
from swarm.experiments.runner import environment_metadata, run_episode
from swarm.learning.ppo import EpisodeStats, Learner, PPOConfig, train_loop
from swarm.learning.pz_env import SectorParallelEnv
from swarm.learning.rewards import DEFAULT_WEIGHTS, TeamReward
from swarm.learning.sector import DECISION_INTERVAL, N_ACTIONS, OBS_SIZE, learned_policy
from swarm.scenarios.config import ScenarioConfig

TRAIN_SEED_MIN, TRAIN_SEED_MAX = 10_000, 59_999
TRAIN_SEED_STRIDE = 5_000


@dataclass(frozen=True, slots=True)
class TrainConfig:
    scenario: str  # path to a ScenarioConfig JSON, relative to the training config file
    ppo: PPOConfig = field(default_factory=PPOConfig)
    total_episodes: int = 1000
    episodes_per_update: int = 16
    validate_every: int = 10
    validation_seeds: str = "3000-3019"
    train_seed_start: int = TRAIN_SEED_MIN
    decision_interval: int = DECISION_INTERVAL
    reward_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    wall_clock_limit_s: float | None = None

    def __post_init__(self) -> None:
        if self.total_episodes < 1 or self.episodes_per_update < 1 or self.validate_every < 1:
            raise ValueError("total_episodes, episodes_per_update and validate_every must be >= 1")
        seeds = parse_seeds(self.validation_seeds)
        if any(TRAIN_SEED_MIN <= s <= TRAIN_SEED_MAX for s in seeds):
            raise ValueError("validation_seeds overlap the training seed range")

    @classmethod
    def from_json(cls, text: str) -> TrainConfig:
        raw = json.loads(text)
        ppo = PPOConfig(**raw.pop("ppo", {}))
        return cls(ppo=ppo, **raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def training_seeds(config: TrainConfig, seed: int) -> range:
    start = config.train_seed_start + TRAIN_SEED_STRIDE * seed
    if not TRAIN_SEED_MIN <= start <= TRAIN_SEED_MAX - TRAIN_SEED_STRIDE + 1:
        raise ValueError(f"training seed {seed}: scenario seeds would leave 10000-59999")
    return range(start, start + TRAIN_SEED_STRIDE)


def validate(
    scenario: ScenarioConfig, actor_path: Path, seeds: Sequence[int], *, sample: bool = False
) -> dict[str, float]:
    """Evaluate an exported actor on validation seeds via the normal runner."""
    factory = learned_policy("validation", actor_path, sample=sample)
    times, recalls, successes = [], [], []
    for s in seeds:
        r = run_episode(scenario, factory, s, designation="development")
        m = r.metrics.to_dict()
        success = bool(m["mission_success"])
        reached = m["time_to_recall_target"]
        times.append(reached if success and reached is not None else scenario.time_limit + 1)
        recalls.append(float(m["recall"]))
        successes.append(float(success))
    return {
        "time_censored": float(np.mean(times)),
        "recall": float(np.mean(recalls)),
        "mission_success": float(np.mean(successes)),
    }


def _summary(stats: Sequence[EpisodeStats]) -> dict[str, Any]:
    comps: dict[str, float] = {}
    for s in stats:
        for k, v in s.components.items():
            comps[k] = comps.get(k, 0.0) + v / len(stats)
    reasons: dict[str, int] = {}
    for s in stats:
        reasons[str(s.end_reason)] = reasons.get(str(s.end_reason), 0) + 1
    return {
        "team_return": float(np.mean([s.team_return for s in stats])),
        "reward_components": comps,
        "end_reasons": reasons,
    }


def save_actor(path: Path, weights: dict[str, Any]) -> None:
    np.savez(path, **weights)


def train(
    config: TrainConfig, config_dir: Path, seed: int, out: Path, *, resume: bool
) -> dict[str, Any]:
    scenario = ScenarioConfig.from_json((config_dir / config.scenario).read_text("utf-8"))
    seeds = training_seeds(config, seed)
    env = SectorParallelEnv(
        scenario,
        seeds=seeds,
        reward=TeamReward(config.reward_weights),
        decision_interval=config.decision_interval,
    )
    learner = Learner(env, config.ppo, seed=seed)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.jsonl"
    record: dict[str, Any] = {
        "obs_size": OBS_SIZE,
        "n_actions": N_ACTIONS,
        "training_seed": seed,
        "train_config": config.to_dict(),
        "train_config_hash": config.config_hash(),
        "scenario_config_hash": scenario.config_hash(),
        "training_scenario_seeds": [
            training_seeds(config, seed).start,
            training_seeds(config, seed).stop - 1,
        ],
        "validation_seeds": config.validation_seeds,
        "validation": [],
        "best": None,
        "torch_version": torch.__version__,
        **environment_metadata(),
    }
    if resume and (out / "checkpoint.pt").exists():
        state = torch.load(out / "checkpoint.pt", weights_only=False)
        learner.load_state_dict(state["learner"])
        # the env consumes training seeds in order: skip those already used
        for _ in range(learner.episodes):
            next(env._seeds)
        record = state["record"]
    elif log_path.exists():
        log_path.unlink()
    started = time.perf_counter()
    val_seeds = parse_seeds(config.validation_seeds)

    def checkpoint() -> None:
        save_actor(out / "last_actor.npz", learner.model.export_actor())
        torch.save({"learner": learner.state_dict(), "record": record}, out / "checkpoint.pt")

    def run_validation() -> None:
        checkpoint()
        for mode in ("greedy", "sampled"):
            scores = validate(scenario, out / "last_actor.npz", val_seeds, sample=mode == "sampled")
            entry: dict[str, Any] = {
                "update": learner.updates,
                "episodes": learner.episodes,
                "mode": mode,
            }
            entry |= scores
            record["validation"].append(entry)
            best = record["best"]
            key = (scores["time_censored"], -scores["recall"])
            if best is None or key < (best["time_censored"], -best["recall"]):
                record["best"] = entry
                save_actor(out / "best_actor.npz", learner.model.export_actor())
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"kind": "validation"} | entry) + "\n")
        (out / "model.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    def on_update(update: int, logs: dict[str, float], stats: list[EpisodeStats]) -> bool:
        line = (
            {
                "kind": "update",
                "update": update,
                "episodes": learner.episodes,
                "decisions": learner.decisions,
                "seconds": round(time.perf_counter() - started, 3),
            }
            | logs
            | _summary(stats)
        )
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")
        if update % config.validate_every == 0:
            run_validation()
        limit = config.wall_clock_limit_s
        return limit is None or time.perf_counter() - started < limit

    train_loop(
        learner,
        total_episodes=config.total_episodes,
        episodes_per_update=config.episodes_per_update,
        on_update=on_update,
    )
    if not record["validation"] or record["validation"][-1]["update"] != learner.updates:
        run_validation()
    record["stopped"] = {
        "updates": learner.updates,
        "episodes": learner.episodes,
        "decisions": learner.decisions,
        "reason": "total_episodes" if learner.episodes >= config.total_episodes else "wall_clock",
        "seconds_this_session": round(time.perf_counter() - started, 1),
    }
    checkpoint()
    (out / "model.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m swarm.learning.train")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = TrainConfig.from_json(args.config.read_text("utf-8"))
        record = train(config, args.config.parent, args.seed, args.out, resume=args.resume)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    best = record["best"]
    print(json.dumps({"stopped": record["stopped"], "best": best}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase 4 M4.2: PPO learner sanity checks, export and the training entry point (ADR-0022)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

import torch
from gymnasium import spaces

from swarm.learning.ppo import Learner, PPOConfig, Trajectory, Transition, train_loop
from swarm.learning.sector import N_ACTIONS, OBS_SIZE, NumpyActor
from swarm.learning.train import TrainConfig, main, training_seeds

ROOT = Path(__file__).resolve().parents[1]
GOOD = 3


class Bandit:
    """A one-decision two-agent task: the team earns 1 for each agent choosing action 3.

    Action 5 is always masked. A correct learner must end up choosing 3 and never 5."""

    def __init__(self) -> None:
        self.possible_agents = ["a", "b"]
        self.agents: list[str] = []
        self.state_space = spaces.Box(0.0, 1.0, (2,), np.float32)
        self.rng = np.random.default_rng(0)

    def state(self) -> np.ndarray[Any, Any]:
        return np.ones(2, dtype=np.float32)

    def _obs(self) -> dict[str, dict[str, Any]]:
        mask = np.ones(N_ACTIONS, dtype=np.int8)
        mask[5] = 0
        feats = self.rng.random(OBS_SIZE).astype(np.float32)
        return {a: {"observation": feats, "action_mask": mask} for a in self.agents}

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self.agents = list(self.possible_agents)
        return self._obs(), {a: {} for a in self.agents}

    def step(self, actions: dict[str, int]) -> tuple[Any, ...]:
        assert all(int(v) != 5 for v in actions.values()), "masked action sampled"
        team = float(sum(int(v) == GOOD for v in actions.values()))
        acting = list(self.agents)
        self.agents = []
        obs = self._obs() | {
            a: {
                "observation": np.zeros(OBS_SIZE, np.float32),
                "action_mask": np.ones(N_ACTIONS, np.int8),
            }
            for a in acting
        }
        return (
            obs,
            dict.fromkeys(acting, team),
            dict.fromkeys(acting, True),
            dict.fromkeys(acting, False),
            {a: {} for a in acting},
        )


@pytest.mark.parametrize("algorithm", ["mappo", "ippo"])
def test_ppo_learns_a_tiny_known_task(algorithm: str) -> None:
    learner = Learner(Bandit(), PPOConfig(algorithm=algorithm, lr=3e-3, minibatch=64), seed=0)
    train_loop(learner, total_episodes=400, episodes_per_update=16)
    actor = NumpyActor([(w, b) for w, b in _pairs(learner.model.export_actor())])
    mask = np.ones(N_ACTIONS, dtype=np.bool_)
    mask[5] = False
    feats = np.random.default_rng(1).random(OBS_SIZE).astype(np.float32)
    assert actor("a", feats, mask) == GOOD
    with torch.no_grad():
        p = torch.exp(
            learner.model.log_probs(torch.from_numpy(feats[None]), torch.from_numpy(mask[None]))
        )
    assert float(p[0, GOOD]) > 0.8
    assert float(p[0, 5]) == 0.0


def _pairs(weights: dict[str, Any]) -> list[tuple[Any, Any]]:
    return [(weights[f"w{i}"], weights[f"b{i}"]) for i in range(len(weights) // 2)]


def test_numpy_export_matches_torch_logits() -> None:
    learner = Learner(Bandit(), PPOConfig(), seed=3)
    actor = NumpyActor(_pairs(learner.model.export_actor()))
    x = np.random.default_rng(2).random(OBS_SIZE).astype(np.float32)
    with torch.no_grad():
        ref = learner.model.actor(torch.from_numpy(x[None]))[0].double().numpy()
    assert np.allclose(actor.logits(x), ref, atol=1e-5)


def test_gae_matches_hand_computation() -> None:
    learner = Learner(Bandit(), PPOConfig(gamma=0.5, gae_lambda=1.0), seed=0)
    z = np.zeros(1, np.float32)
    m = np.ones(1, np.bool_)
    traj = Trajectory(
        [Transition(z, m, z, 0, 0.0, 1.0, 1.0), Transition(z, m, z, 0, 0.0, 2.0, 3.0)]
    )
    traj.bootstrap = 4.0
    _, adv, ret = learner._advantages([traj])
    # returns: step 2 = 3 + 0.5*4 = 5; step 1 = 1 + 0.5*5 = 3.5
    assert ret.tolist() == pytest.approx([3.5, 5.0])
    assert adv.tolist() == pytest.approx([2.5, 3.0])


def test_ppo_config_validation() -> None:
    with pytest.raises(ValueError, match="algorithm"):
        PPOConfig(algorithm="dqn")
    with pytest.raises(ValueError, match="overlap"):
        TrainConfig(scenario="x.json", validation_seeds="10000-10002")
    with pytest.raises(ValueError, match="10000-59999"):
        training_seeds(TrainConfig(scenario="x.json"), 10)


def test_train_cli_writes_versioned_outputs_and_resumes(tmp_path: Path) -> None:
    cfg = {
        "scenario": str(ROOT / "configs/phase3/nominal.json"),
        "total_episodes": 4,
        "episodes_per_update": 2,
        "validate_every": 1,
        "validation_seeds": "1000-1001",
        "ppo": {"hidden": 16, "epochs": 1},
    }
    path = tmp_path / "train.json"
    path.write_text(json.dumps(cfg))
    out = tmp_path / "run"
    assert main(["--config", str(path), "--seed", "0", "--out", str(out)]) == 0
    record = json.loads((out / "model.json").read_text())
    assert record["stopped"]["episodes"] == 4
    assert record["training_scenario_seeds"] == [10000, 14999]
    assert record["best"] is not None
    assert len(record["validation"]) == 4  # 2 validations x (greedy, sampled)
    assert record["best"]["mode"] in ("greedy", "sampled")
    for name in ("best_actor.npz", "last_actor.npz", "checkpoint.pt", "train_log.jsonl"):
        assert (out / name).exists()
    NumpyActor.load(out / "best_actor.npz")
    cfg["total_episodes"] = 6
    path.write_text(json.dumps(cfg))
    assert main(["--config", str(path), "--seed", "0", "--out", str(out), "--resume"]) == 0
    assert json.loads((out / "model.json").read_text())["stopped"]["episodes"] == 6
    lines = [json.loads(x) for x in (out / "train_log.jsonl").read_text().splitlines()]
    assert [x["episodes"] for x in lines if x["kind"] == "update"] == [2, 4, 6]

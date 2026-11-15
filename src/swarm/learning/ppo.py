"""IPPO and MAPPO for the sector-selection environment (ADR-0022). Needs ``[learning]``.

One shared actor for all agents (parameter sharing; the observation carries the agent's layer).
Invalid actions are masked to probability 0 before sampling and in the loss.

- **MAPPO** (centralized critic): the critic sees ``env.state()`` plus a one-hot of the agent's
  roster index.
- **IPPO** (decentralized critic, the diagnostic): the critic sees the agent's own observation.

Per agent, transitions are its decision epochs, discounted by ``gamma`` per epoch, with GAE.
- An agent that stops acting (disabled, or the mission ends) gets no bootstrap.
- At a time-limit truncation the value of its final observation is bootstrapped.
- Advantages are normalized per update. Rewards are not normalized.

Determinism: ``torch.manual_seed`` and NumPy generators are seeded from ``TrainConfig.seed``.
PyTorch runs single-threaded with deterministic algorithms, so a training run is reproducible on
one machine and software stack. It is not guaranteed across PyTorch versions or CPUs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from swarm.learning.sector import N_ACTIONS, OBS_SIZE

ALGORITHMS = ("mappo", "ippo")


@dataclass(frozen=True, slots=True)
class PPOConfig:
    algorithm: str = "mappo"
    hidden: int = 64
    lr: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 8
    minibatch: int = 128

    def __post_init__(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm: must be one of {ALGORITHMS}, got {self.algorithm!r}")
        if self.hidden < 1 or self.epochs < 1 or self.minibatch < 1:
            raise ValueError("hidden, epochs and minibatch must be >= 1")
        if not (0 < self.gamma <= 1 and 0 <= self.gae_lambda <= 1 and self.clip > 0):
            raise ValueError("need 0 < gamma <= 1, 0 <= gae_lambda <= 1 and clip > 0")


def mlp(n_in: int, n_out: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_in, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, n_out),
    )


class ActorCritic(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, critic_size: int, hidden: int) -> None:
        super().__init__()
        self.actor = mlp(obs_size, n_actions, hidden)
        self.critic = mlp(critic_size, 1, hidden)
        with torch.no_grad():  # small initial policy logits: near-uniform over valid actions
            last = self.actor[-1]
            assert isinstance(last, nn.Linear)
            last.weight.mul_(0.01)
            last.bias.zero_()

    def log_probs(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Log-probabilities over actions; invalid actions get probability 0 (logit -1e9)."""
        logits: torch.Tensor = self.actor(obs).masked_fill(~mask, -1e9)
        return torch.log_softmax(logits, dim=-1)

    def value(self, critic_in: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.critic(critic_in).squeeze(-1)
        return out

    def export_actor(self) -> dict[str, npt.NDArray[np.float64]]:
        """Weights for :class:`swarm.learning.sector.NumpyActor` (``x @ w + b``)."""
        out: dict[str, npt.NDArray[np.float64]] = {}
        linears = [m for m in self.actor if isinstance(m, nn.Linear)]
        for i, layer in enumerate(linears):
            out[f"w{i}"] = layer.weight.detach().double().numpy().T.copy()
            out[f"b{i}"] = layer.bias.detach().double().numpy().copy()
        return out


@dataclass(slots=True)
class Transition:
    obs: npt.NDArray[np.float32]
    mask: npt.NDArray[np.bool_]
    critic_in: npt.NDArray[np.float32]
    action: int
    logp: float
    value: float
    reward: float = 0.0


@dataclass(slots=True)
class Trajectory:
    steps: list[Transition] = field(default_factory=list)
    bootstrap: float = 0.0  # value after the last step (0 when terminated)


@dataclass(slots=True)
class EpisodeStats:
    team_return: float
    components: dict[str, float]
    end_reason: str | None
    decisions: int


class Learner:
    """Collects episodes from a PettingZoo ``ParallelEnv`` and runs PPO updates."""

    def __init__(
        self,
        env: Any,
        config: PPOConfig,
        *,
        seed: int,
        obs_size: int = OBS_SIZE,
        n_actions: int = N_ACTIONS,
    ) -> None:
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.env = env
        self.config = config
        self.agents = list(env.possible_agents)
        self.n = len(self.agents)
        self.obs_size = obs_size
        if config.algorithm == "mappo":
            critic_size = int(np.prod(env.state_space.shape)) + self.n
        else:
            critic_size = obs_size
        self.model = ActorCritic(obs_size, n_actions, critic_size, config.hidden)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, eps=1e-5)
        self.rng = np.random.default_rng(seed)
        self.updates = 0
        self.episodes = 0
        self.decisions = 0

    # ----------------------------------------------------------------- collection

    def _critic_input(self, agent: str, obs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        if self.config.algorithm == "ippo":
            return obs
        onehot = np.zeros(self.n, dtype=np.float32)
        onehot[self.agents.index(agent)] = 1.0
        return np.concatenate([self.env.state(), onehot]).astype(np.float32)

    @torch.no_grad()
    def _act(
        self, observations: dict[str, dict[str, Any]], agents: Sequence[str]
    ) -> tuple[dict[str, int], dict[str, Transition]]:
        ids = list(agents)
        obs = np.stack([observations[a]["observation"] for a in ids]).astype(np.float32)
        mask = np.stack([observations[a]["action_mask"] for a in ids]).astype(bool)
        crit = np.stack([self._critic_input(a, observations[a]["observation"]) for a in ids])
        log_probs = self.model.log_probs(torch.from_numpy(obs), torch.from_numpy(mask))
        probs = torch.exp(log_probs).numpy().astype(np.float64)
        actions: dict[str, int] = {}
        out: dict[str, Transition] = {}
        values = self.model.value(torch.from_numpy(crit)).numpy()
        for i, a in enumerate(ids):
            p = probs[i] / probs[i].sum()
            action = int(self.rng.choice(len(p), p=p))
            actions[a] = action
            out[a] = Transition(
                obs[i],
                mask[i],
                crit[i],
                action,
                float(np.log(max(p[action], 1e-12))),
                float(values[i]),
            )
        return actions, out

    def collect(self, episodes: int) -> tuple[list[Trajectory], list[EpisodeStats]]:
        trajectories: list[Trajectory] = []
        stats: list[EpisodeStats] = []
        for _ in range(episodes):
            observations, _ = self.env.reset()
            open_traj: dict[str, Trajectory] = {a: Trajectory() for a in self.env.agents}
            team_return = 0.0
            components: dict[str, float] = {}
            end_reason = None
            decisions = 0
            while self.env.agents:
                acting = list(self.env.agents)
                actions, trans = self._act(observations, acting)
                decisions += len(acting)
                observations, rewards, terms, truncs, infos = self.env.step(actions)
                team = float(next(iter(rewards.values())))
                team_return += team
                info = next(iter(infos.values()))
                for k, v in info.get("reward_components", {}).items():
                    components[k] = components.get(k, 0.0) + v
                end_reason = info.get("end_reason")
                for a in acting:
                    t = trans[a]
                    t.reward = float(rewards[a])
                    open_traj.setdefault(a, Trajectory()).steps.append(t)
                    if terms[a] or truncs[a] or a not in self.env.agents:
                        traj = open_traj.pop(a)
                        if truncs[a] and not terms[a]:
                            final = observations[a]["observation"]
                            crit = self._critic_input(a, final)
                            with torch.no_grad():
                                traj.bootstrap = float(
                                    self.model.value(torch.from_numpy(crit[None]))[0]
                                )
                        trajectories.append(traj)
            for traj in open_traj.values():
                trajectories.append(traj)
            self.episodes += 1
            self.decisions += decisions
            stats.append(EpisodeStats(team_return, components, end_reason, decisions))
        return trajectories, stats

    # ----------------------------------------------------------------- update

    def _advantages(
        self, trajectories: Sequence[Trajectory]
    ) -> tuple[list[Transition], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        g, lam = self.config.gamma, self.config.gae_lambda
        flat: list[Transition] = []
        adv: list[float] = []
        ret: list[float] = []
        for traj in trajectories:
            next_value, gae = traj.bootstrap, 0.0
            a_rev, r_rev = [], []
            for t in reversed(traj.steps):
                delta = t.reward + g * next_value - t.value
                gae = delta + g * lam * gae
                a_rev.append(gae)
                r_rev.append(gae + t.value)
                next_value = t.value
            flat.extend(traj.steps)
            adv.extend(reversed(a_rev))
            ret.extend(reversed(r_rev))
        return flat, np.array(adv, np.float32), np.array(ret, np.float32)

    def update(self, trajectories: Sequence[Trajectory]) -> dict[str, float]:
        flat, adv, ret = self._advantages(trajectories)
        if not flat:
            return {}
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        obs = torch.from_numpy(np.stack([t.obs for t in flat]))
        mask = torch.from_numpy(np.stack([t.mask for t in flat]))
        crit = torch.from_numpy(np.stack([t.critic_in for t in flat]))
        act = torch.tensor([t.action for t in flat])
        old_logp = torch.tensor([t.logp for t in flat], dtype=torch.float32)
        adv_t, ret_t = torch.from_numpy(adv), torch.from_numpy(ret)
        c = self.config
        n = len(flat)
        logs: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl": [],
        }
        for _ in range(c.epochs):
            order = self.rng.permutation(n)
            for start in range(0, n, c.minibatch):
                idx = torch.from_numpy(order[start : start + c.minibatch])
                all_logp = self.model.log_probs(obs[idx], mask[idx])
                logp = all_logp.gather(1, act[idx][:, None]).squeeze(1)
                ratio = torch.exp(logp - old_logp[idx])
                surr = torch.min(
                    ratio * adv_t[idx], torch.clamp(ratio, 1 - c.clip, 1 + c.clip) * adv_t[idx]
                )
                policy_loss = -surr.mean()
                value_loss = ((self.model.value(crit[idx]) - ret_t[idx]) ** 2).mean()
                entropy = -(torch.exp(all_logp) * all_logp).sum(-1).mean()
                loss = policy_loss + c.value_coef * value_loss - c.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                nn.utils.clip_grad_norm_(self.model.parameters(), c.max_grad_norm)
                self.optimizer.step()
                logs["policy_loss"].append(policy_loss.item())
                logs["value_loss"].append(value_loss.item())
                logs["entropy"].append(entropy.item())
                logs["kl"].append((old_logp[idx] - logp).mean().item())
        self.updates += 1
        return {k: float(np.mean(v)) for k, v in logs.items()} | {"transitions": float(n)}

    # ----------------------------------------------------------------- checkpoints

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": self.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "updates": self.updates,
            "episodes": self.episodes,
            "decisions": self.decisions,
            "config": self.config,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"])
        self.updates = state["updates"]
        self.episodes = state["episodes"]
        self.decisions = state["decisions"]


def train_loop(
    learner: Learner,
    *,
    total_episodes: int,
    episodes_per_update: int,
    on_update: Callable[[int, dict[str, float], list[EpisodeStats]], bool] | None = None,
) -> None:
    """Collect and update until ``total_episodes``. ``on_update`` returning False stops early."""
    while learner.episodes < total_episodes:
        n = min(episodes_per_update, total_episodes - learner.episodes)
        trajectories, stats = learner.collect(n)
        logs = learner.update(trajectories)
        if on_update is not None and not on_update(learner.updates, logs, stats):
            break

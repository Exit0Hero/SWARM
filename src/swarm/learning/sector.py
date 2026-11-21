"""Learned sector selection: the decentralized agent a MARL policy controls (ADR-0022).

The learned policy does not output primitive moves. Every ``decision_interval`` steps (and at
step 0), each active agent picks one *macro action*:

- ``0 .. 15``: search sector ``k`` of a 4 x 4 grid of sectors over the map (row-major; unlike the
  auction's fixed 4 x 4-cell sectors, the count is fixed, so the same network fits any map size);
- ``16`` (:data:`FREE`): search the nearest frontier anywhere;
- ``17`` (:data:`RETURN`): return to base and recharge.

Between decisions the Phase 2 energy-aware navigator executes the choice. It searches frontier
cells inside the chosen sector, falls back to the nearest frontier anywhere when that sector has
no reachable frontier left, and still returns to base by itself when energy requires it.

The agent shares its belief exactly like ``frontier_shared`` (ADR-0018). It also broadcasts one
``intent`` record (its latest choice), which is the only coordination signal it sends.

Every actor feature comes from the agent's own observation, its own belief map and records it
received (ADR-0008). Nothing here reads hidden state. :func:`actor_features` documents the
layout.

:class:`SectorTeam` is used unchanged by the training adapter
(:class:`swarm.learning.pz_env.SectorParallelEnv`) and by evaluation through the runner. With the
same choices, both produce the same event stream (tested).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from swarm.core.messages import Message
from swarm.core.observation import Observation
from swarm.core.oracle import OracleView
from swarm.core.types import AgentId, AgentStatus, Layer
from swarm.navigation.astar import UNREACHABLE, cost_field
from swarm.policies.base import DECENTRALIZED_COMM, Diagnostics, MissionBrief, TeamFactory
from swarm.policies.decentralized import NavigatorAgent
from swarm.policies.planning import Counters, feasible_mask
from swarm.policies.shared import MESSAGE_KIND, SharedNavigatorAgent
from swarm.scenarios.config import AgentTypeSpec

N_SECTORS: Final[int] = 16
FREE: Final[int] = 16
RETURN: Final[int] = 17
N_ACTIONS: Final[int] = 18
SECTOR_FEATURES: Final[int] = 4
OWN_FEATURES: Final[int] = 9
OBS_SIZE: Final[int] = N_SECTORS * SECTOR_FEATURES + OWN_FEATURES  # 73
DECISION_INTERVAL: Final[int] = 8
INTENT_TAG: Final[str] = "intent"
HEARD_WINDOW: Final[int] = 10

Chooser = Callable[
    [AgentId, npt.NDArray[np.float32], npt.NDArray[np.bool_], np.random.Generator], int
]


def grid_sectors(shape: tuple[int, int], per_side: int = 4) -> list[npt.NDArray[np.bool_]]:
    """``per_side`` x ``per_side`` sector masks, row-major. Boundaries are spread as evenly as
    possible, so every sector is non-empty when the map is at least ``per_side`` cells wide."""
    h, w = shape
    if h < per_side or w < per_side:
        raise ValueError(f"grid_sectors: map {shape} is smaller than {per_side} x {per_side}")
    rows = [round(i * h / per_side) for i in range(per_side + 1)]
    cols = [round(j * w / per_side) for j in range(per_side + 1)]
    masks = []
    for i in range(per_side):
        for j in range(per_side):
            mask = np.zeros(shape, dtype=np.bool_)
            mask[rows[i] : rows[i + 1], cols[j] : cols[j + 1]] = True
            masks.append(mask)
    return masks


class SectorAgent(SharedNavigatorAgent):
    """A shared-belief navigator whose search scope is set by a learned macro action."""

    def __init__(
        self, agent_id: AgentId, spec: AgentTypeSpec, brief: MissionBrief, counters: Counters
    ) -> None:
        super().__init__(agent_id, spec, brief, counters)
        self.spec = spec
        self.sectors = grid_sectors(brief.grid_shape)
        self.teammates = tuple(sorted(a for a in brief.agents if a != agent_id))
        self.choice = FREE
        self.intents: dict[AgentId, tuple[int, int]] = {}  # agent -> (step, choice)
        self.last_heard: dict[AgentId, int] = {}
        self.decisions = 0

    # ----------------------------------------------------------------- information

    def observe(self, obs: Observation) -> None:
        """Merge the inbox (beliefs, intents, contact times), then the agent's own observation."""
        for delivery in obs.inbox:
            sender = delivery.message.sender
            self.last_heard[sender] = max(self.last_heard.get(sender, 0), obs.step)
            self.belief.merge_records(delivery.message.records)
            for record in delivery.message.records:
                if not (record and record[0] == INTENT_TAG and len(record) == 4):
                    continue
                _, agent, choice, step = record
                if not (isinstance(agent, str) and isinstance(choice, int)):
                    continue
                if not isinstance(step, int) or agent == self.agent_id:
                    continue
                if step >= self.intents.get(agent, (-1, 0))[0]:
                    self.intents[agent] = (step, choice)
        self.belief.update(obs)

    def features(self, obs: Observation) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        """Actor observation and action mask (see :func:`actor_features`)."""
        passable = self.cache.passable_for(self.known, self.layer)
        to_base = self.cache.to_base_for(self.known, self.model)
        feasible = feasible_mask(
            self.model, obs.energy, passable, obs.position, self.known, to_base
        )
        dist = cost_field(passable, obs.position, self._costs())
        targetable = feasible & self.cache.frontier_for(self.known, self.layer)
        targetable &= dist != UNREACHABLE
        return actor_features(self, obs, targetable, dist)

    # ----------------------------------------------------------------- acting

    def apply(self, choice: int, step: int) -> None:
        """Adopt a macro action chosen at ``step``."""
        if not 0 <= choice < N_ACTIONS:
            raise ValueError(f"{self.agent_id}: macro action {choice} outside 0..{N_ACTIONS - 1}")
        self.decisions += 1
        if choice != self.choice:
            self.target, self.path = None, []
        self.choice = choice
        if choice == RETURN:
            self.target, self.path = None, []
            self.returning = True
        self.intents[self.agent_id] = (step, choice)

    def move(self, obs: Observation, rng: np.random.Generator) -> int:
        return NavigatorAgent.act(self, obs, rng)

    def _scope(self) -> npt.NDArray[np.bool_] | None:
        if self.choice < N_SECTORS:
            mask = self.sectors[self.choice]
            frontier = self.cache.frontier_for(self.known, self.layer)
            if bool((frontier & mask).any()):
                return mask
        return None

    def compose(self) -> list[Message]:
        if self.max_records == 0:
            return []
        step, choice = self.intents.get(self.agent_id, (0, self.choice))
        intent = ((INTENT_TAG, self.agent_id, choice, step),)
        room = None if self.max_records is None else self.max_records - 1
        beliefs = self.belief.outbox_records(room) if room is None or room > 0 else ()
        message = Message(self.agent_id, self.seq, MESSAGE_KIND, intent + beliefs)
        self.seq += 1
        return [message]


def actor_features(
    agent: SectorAgent,
    obs: Observation,
    targetable: npt.NDArray[np.bool_],
    dist: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """The 73-value actor observation and the 18-action mask.

    Per sector ``k`` (indices ``4k .. 4k+3``), all in ``[0, 1]``:

    0. fraction of the sector unknown in the agent's belief;
    1. path cost to the nearest targetable frontier cell in it, divided by ``rows + cols``
       (1 when none);
    2. cells in it with survivor evidence in the belief, divided by 4, capped at 1;
    3. fraction of teammates whose latest received intent (at most ``2 * DECISION_INTERVAL``
       steps old) is ``k``.

    Own features (indices 64..72):
    - energy fraction;
    - row and column, normalized;
    - is-air and is-ground flags;
    - fraction of the time limit elapsed;
    - fraction of teammates heard within the last ``HEARD_WINDOW`` steps;
    - fraction of teammates known to be disabled;
    - at-base flag.

    Mask:
    - sector ``k`` is valid when it has a targetable frontier cell;
    - ``FREE`` is valid when any cell is targetable;
    - ``RETURN`` is valid away from the base.
    - If nothing is valid, ``FREE`` is allowed (the navigator then idles).
    """
    h, w = obs.grid_shape
    known = agent.known.known
    evidence = np.zeros((h, w), dtype=np.bool_)
    for cell in agent.belief.survivor_evidence():
        evidence[cell] = True
    n_mates = max(1, len(agent.teammates))
    fresh = {
        a: choice
        for a, (step, choice) in agent.intents.items()
        if a != agent.agent_id and obs.step - step <= 2 * DECISION_INTERVAL
    }
    feats = np.zeros(OBS_SIZE, dtype=np.float32)
    mask = np.zeros(N_ACTIONS, dtype=np.bool_)
    scale = float(h + w)
    for k, sector in enumerate(agent.sectors):
        cells = targetable & sector
        base = SECTOR_FEATURES * k
        feats[base] = 1.0 - float(known[sector].mean())
        if bool(cells.any()):
            feats[base + 1] = min(1.0, float(dist[cells].min()) / scale)
            mask[k] = True
        else:
            feats[base + 1] = 1.0
        feats[base + 2] = min(1.0, float(evidence[sector].sum()) / 4.0)
        feats[base + 3] = sum(1 for c in fresh.values() if c == k) / n_mates
    own = N_SECTORS * SECTOR_FEATURES
    heard = sum(
        1 for a in agent.teammates if obs.step - agent.last_heard.get(a, -(10**9)) <= HEARD_WINDOW
    )
    disabled = agent.belief.disabled_agents()
    feats[own : own + OWN_FEATURES] = (
        obs.energy / max(1, obs.energy_capacity),
        obs.position[0] / max(1, h - 1),
        obs.position[1] / max(1, w - 1),
        float(agent.layer is Layer.AIR),
        float(agent.layer is Layer.GROUND),
        min(1.0, obs.step / max(1, agent.brief.time_limit)),
        heard / n_mates,
        sum(1 for a in agent.teammates if a in disabled) / n_mates,
        float(obs.position == obs.base_position),
    )
    mask[FREE] = bool(targetable.any())
    mask[RETURN] = obs.position != obs.base_position
    if not mask.any():
        mask[FREE] = True
    return feats, mask


class SectorTeam:
    """One :class:`SectorAgent` per agent; macro actions come from ``chooser``.

    :meth:`prepare` and :meth:`advance` split :meth:`act` so the training adapter can return
    features to the learner between them. Agents decide when ``step % decision_interval == 0``.
    """

    information = DECENTRALIZED_COMM
    uses_oracle = False
    version = "1"

    def __init__(
        self,
        name: str,
        brief: MissionBrief,
        chooser: Chooser | None = None,
        *,
        decision_interval: int = DECISION_INTERVAL,
    ) -> None:
        if brief.comms is None:
            raise ValueError(f"{name}: needs a scenario with a comms block (ADR-0017)")
        if decision_interval < 1:
            raise ValueError(f"{name}: decision_interval must be >= 1")
        self.name = name
        self.brief = brief
        self.chooser = chooser
        self.decision_interval = decision_interval
        self.counters = Counters()
        self.agents = {
            aid: SectorAgent(aid, spec, brief, self.counters) for aid, spec in brief.agents.items()
        }
        self._outbox: dict[AgentId, list[Message]] = {}
        self.choices: dict[int, int] = {}

    def prepare(
        self, observations: Mapping[AgentId, Observation]
    ) -> dict[AgentId, tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]]:
        """Observe; return features and masks for the agents that decide this step."""
        out = {}
        for aid, obs in sorted(observations.items()):
            if obs.status is not AgentStatus.ACTIVE:
                continue
            agent = self.agents[aid]
            agent.observe(obs)
            if obs.step % self.decision_interval == 0:
                out[aid] = agent.features(obs)
        return out

    def advance(
        self,
        observations: Mapping[AgentId, Observation],
        rngs: Mapping[AgentId, np.random.Generator],
        choices: Mapping[AgentId, int],
    ) -> dict[AgentId, int]:
        """Adopt ``choices`` (decision steps only), then move every active agent one step."""
        actions: dict[AgentId, int] = {}
        self._outbox = {}
        for aid, obs in sorted(observations.items()):
            if obs.status is not AgentStatus.ACTIVE:
                continue
            agent = self.agents[aid]
            if aid in choices:
                agent.apply(int(choices[aid]), obs.step)
                self.choices[int(choices[aid])] = self.choices.get(int(choices[aid]), 0) + 1
            actions[aid] = agent.move(obs, rngs[aid])
            messages = agent.compose()
            if messages:
                self._outbox[aid] = messages
        return actions

    def act(
        self,
        observations: Mapping[AgentId, Observation],
        rngs: Mapping[AgentId, np.random.Generator],
    ) -> dict[AgentId, int]:
        if self.chooser is None:
            raise RuntimeError(f"{self.name}: no chooser (the training adapter drives prepare)")
        decide = self.prepare(observations)
        choices = {aid: self.chooser(aid, f, m, rngs[aid]) for aid, (f, m) in decide.items()}
        return self.advance(observations, rngs, choices)

    def outbox(self) -> Mapping[AgentId, Sequence[Message]]:
        return self._outbox

    def attach_oracle(self, view: OracleView) -> None:
        raise RuntimeError(f"{self.name} is decentralized and never uses the oracle")

    def diagnostics(self) -> Diagnostics:
        out: Diagnostics = dict(self.counters.as_dict())
        out["macro_decisions"] = sum(a.decisions for a in self.agents.values())
        out["macro_free"] = self.choices.get(FREE, 0)
        out["macro_return"] = self.choices.get(RETURN, 0)
        return out

    def decision_log(self) -> list[dict[str, Any]]:
        return []


# ----------------------------------------------------------------- inference (NumPy only)


class NumpyActor:
    """A tanh MLP actor loaded from an ``.npz`` export (``w0, b0, w1, b1, ...``).

    ``sample=False`` (greedy): argmax over valid actions, ties to the lowest index.
    ``sample=True``: one draw from the masked softmax using the agent's own policy stream
    (ADR-0006), so sampled evaluation is still deterministic per seed. Needs no PyTorch.
    """

    def __init__(
        self,
        weights: Sequence[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
        *,
        sample: bool = False,
    ):
        self.sample = sample
        if not weights:
            raise ValueError("NumpyActor: no layers")
        if weights[0][0].shape[0] != OBS_SIZE or weights[-1][0].shape[1] != N_ACTIONS:
            raise ValueError(
                f"NumpyActor: expected {OBS_SIZE} inputs and {N_ACTIONS} outputs, got "
                f"{weights[0][0].shape[0]} and {weights[-1][0].shape[1]}"
            )
        self.layers = [(np.asarray(w, np.float64), np.asarray(b, np.float64)) for w, b in weights]

    @classmethod
    def load(cls, path: Path, *, sample: bool = False) -> NumpyActor:
        with np.load(path) as data:
            n = len([k for k in data.files if k.startswith("w")])
            return cls([(data[f"w{i}"], data[f"b{i}"]) for i in range(n)], sample=sample)

    def logits(self, features: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        x = np.asarray(features, dtype=np.float64)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = np.tanh(x)
        return x

    def __call__(
        self,
        agent_id: AgentId,
        features: npt.NDArray[np.float32],
        mask: npt.NDArray[np.bool_],
        rng: np.random.Generator | None = None,
    ) -> int:
        logits = np.where(mask, self.logits(features), -np.inf)
        if not self.sample:
            return int(np.argmax(logits))
        if rng is None:
            raise ValueError("NumpyActor(sample=True) needs the agent's policy generator")
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return int(rng.choice(len(p), p=p))


def learned_policy(name: str, checkpoint: Path, *, sample: bool = False) -> TeamFactory:
    """A registry-compatible factory for a trained actor (evaluated with the normal runner)."""
    actor = NumpyActor.load(checkpoint, sample=sample)
    return TeamFactory(name, lambda brief: SectorTeam(name, brief, actor))


def fixed_choice_policy(name: str, choice: int) -> TeamFactory:
    """Every agent always picks ``choice`` when valid, else ``FREE`` (tests and sanity checks)."""

    def choose(
        _: AgentId, __: npt.NDArray[np.float32], mask: npt.NDArray[np.bool_], ___: Any
    ) -> int:
        return choice if mask[choice] else FREE

    return TeamFactory(name, lambda brief: SectorTeam(name, brief, choose))

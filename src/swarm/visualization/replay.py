"""Lightweight replay viewer (Phase 3 acceptance; matplotlib, optional ``viz`` extra).

The viewer is a read-only consumer of an episode's event stream and its scenario (docs/12).
:func:`replay_state` reconstructs, up to any step, each agent's path, detections and
disablement from the events alone. It needs NumPy only and is what the tests check.
:func:`render_replay` draws that state.

Survivors are drawn as **evaluation ground truth** and labelled as such. Agents never saw them.

Example::

    python -m swarm.visualization.replay --config configs/phase3/nominal.json \\
        --policy auction_comm --seed 1000 --step 40 --out replay.png
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swarm.core.events import AgentDisabled, Detection, Event, Moved
from swarm.core.types import AgentId, Position, Terrain
from swarm.scenarios.generator import Scenario


@dataclass(slots=True)
class AgentTrace:
    path: list[Position]
    detections: list[tuple[int, Position, bool]] = field(default_factory=list)
    disabled_at: tuple[int, Position, str] | None = None


def replay_state(
    scenario: Scenario, agents: Sequence[AgentId], events: Iterable[Event], *, until: int | None
) -> dict[AgentId, AgentTrace]:
    """Every agent's trace from the base through step ``until`` (``None`` = whole episode)."""
    traces = {aid: AgentTrace([scenario.base]) for aid in sorted(agents)}
    for e in events:
        if until is not None and e.step > until:
            break
        if isinstance(e, Moved) and e.agent_id in traces:
            traces[e.agent_id].path.append(e.to_position)
        elif isinstance(e, Detection) and e.agent_id in traces:
            traces[e.agent_id].detections.append((e.step, e.cell, e.true_positive))
        elif isinstance(e, AgentDisabled) and e.agent_id in traces:
            traces[e.agent_id].disabled_at = (e.step, e.position, e.reason)
    return traces


def render_replay(
    scenario: Scenario,
    traces: dict[AgentId, AgentTrace],
    path: Path,
    *,
    title: str = "",
) -> Path:
    """Draw terrain, hazards, ground-truth survivors and every agent's trace to ``path``."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise RuntimeError("the replay viewer needs matplotlib: pip install -e .[viz]") from exc

    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
    terrain_colors = {
        Terrain.OPEN: "#fcfcfb",
        Terrain.DEBRIS: "#d9d7cf",
        Terrain.STRUCTURE: "#52514e",
        Terrain.BASE: "#cde2fb",
    }
    h, w = scenario.terrain.shape
    fig, ax = plt.subplots(figsize=(6.4, 6.4 * h / max(w, 1) + 0.6))
    cmap = ListedColormap([terrain_colors[Terrain(k)] for k in range(4)])
    ax.imshow(scenario.terrain, cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
    hz = scenario.hazard_grid
    rows, cols = hz.nonzero()
    ax.scatter(cols, rows, marker="x", s=18, color="#d03b3b", lw=0.8, label="hazard")
    if scenario.survivors:
        sr, sc = zip(*scenario.survivors, strict=True)
        ax.scatter(
            sc,
            sr,
            marker="*",
            s=90,
            facecolor="none",
            edgecolor="#0b0b0b",
            lw=0.9,
            label="survivor (ground truth, evaluation only)",
        )
    for k, (aid, trace) in enumerate(traces.items()):
        color = palette[k % len(palette)]
        pr, pc = zip(*trace.path, strict=True)
        ax.plot(pc, pr, color=color, lw=1.6, alpha=0.85, label=aid)
        ax.plot(pc[-1], pr[-1], "o", color=color, ms=7, markeredgecolor="#fcfcfb")
        hits = [cell for _, cell, true in trace.detections if true]
        if hits:
            hr, hc = zip(*hits, strict=True)
            ax.scatter(hc, hr, s=26, color=color, marker="o", zorder=3)
        if trace.disabled_at is not None:
            _, (dr, dc), _ = trace.disabled_at
            ax.plot(dc, dr, marker="X", ms=11, color=color, markeredgecolor="#0b0b0b")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=9)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    from swarm.core.rng import episode_streams
    from swarm.experiments.runner import run_episode
    from swarm.policies.registry import POLICIES
    from swarm.scenarios.config import ScenarioConfig
    from swarm.scenarios.generator import generate_scenario

    parser = argparse.ArgumentParser(prog="python -m swarm.visualization.replay")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy", choices=sorted(POLICIES), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--step", type=int, help="draw the state after this step (default: end)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    config = ScenarioConfig.from_json(args.config.read_text(encoding="utf-8"))
    result: Any = run_episode(config, POLICIES[args.policy], args.seed, designation="development")
    agents = tuple(config.agents())
    scenario = generate_scenario(config, episode_streams(args.seed, agents).scenario)
    traces = replay_state(scenario, agents, result.events, until=args.step)
    shown = result.steps if args.step is None else min(args.step, result.steps)
    title = f"{args.policy}, seed {args.seed}, step {shown} of {result.steps} ({result.end_reason})"
    print(render_replay(scenario, traces, args.out, title=title))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

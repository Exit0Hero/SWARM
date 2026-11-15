"""Tables and figures for the Phase 4 MARL study (docs/19). Reads files only; runs nothing.

Usage (from the repository root)::

    python docs/results/phase4/report.py --analysis docs/results/phase4/analysis.json \\
        --models docs/results/phase4/models --out docs/results/phase4/results.md \\
        --plots docs/results/phase4/figures

Reuses the Phase 3 renderer for the common tables, then adds the training summary. Figures need
the optional ``viz`` extra.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "phase3_report", HERE.parent / "phase3" / "report.py"
)
assert _spec is not None
assert _spec.loader is not None
phase3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(phase3)

GROUPS = {"mappo": "#1baf7a", "ippo": "#2a78d6", "mappo_nocov": "#eb6834"}


def _load(models: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for d in sorted(p for p in models.iterdir() if (p / "model.json").exists()):
        lines = [json.loads(x) for x in (d / "train_log.jsonl").read_text().splitlines()]
        out[d.name] = {"model": json.loads((d / "model.json").read_text()), "log": lines}
    return out


def training_table(runs: dict[str, dict[str, Any]]) -> str:
    rows = [
        "## Training (validation seeds 3000-3019; selection only, not results)",
        "",
        "| Model | Episodes | Decisions | Selected update | Mode | Validation time_censored |"
        " Update 25, greedy |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in runs.items():
        m = r["model"]
        first = next(v for v in m["validation"] if v["mode"] == "greedy")
        rows.append(
            f"| {name} | {m['stopped']['episodes']} | {m['stopped']['decisions']} | "
            f"{m['best']['update']} | {m['best']['mode']} | {m['best']['time_censored']:.2f} | "
            f"{first['time_censored']:.2f} |"
        )
    return "\n".join(rows) + "\n"


def plot_training(runs: dict[str, dict[str, Any]], directory: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, r in runs.items():
        group = name.rsplit("_s", 1)[0]
        color = GROUPS[group]
        upd = [x for x in r["log"] if x["kind"] == "update"]
        weight = r["model"]["train_config"]["reward_weights"]["time"]
        steps = [x["reward_components"]["time"] / weight for x in upd]
        window = 10
        smooth = [
            sum(steps[max(0, i - window + 1) : i + 1]) / len(steps[max(0, i - window + 1) : i + 1])
            for i in range(len(steps))
        ]
        axes[0].plot(
            [x["episodes"] for x in upd],
            smooth,
            color=color,
            lw=1.4,
            alpha=0.8,
            label=group if name.endswith("_s0") else None,
        )
        val = [v for v in r["model"]["validation"] if v["mode"] == "greedy"]
        axes[1].plot(
            [v["episodes"] for v in val],
            [v["time_censored"] for v in val],
            color=color,
            marker="o",
            ms=4,
            lw=1.2,
            alpha=0.8,
            label=group if name.endswith("_s0") else None,
        )
    axes[0].set_title("Training episodes: steps per episode (10-update mean)", loc="left")
    axes[0].set_xlabel("training episodes")
    axes[0].set_ylabel("steps (sampled policy)")
    axes[1].set_title("Validation seeds 3000-3019: greedy time_censored", loc="left")
    axes[1].set_xlabel("training episodes")
    axes[1].set_ylabel("time_censored (steps)")
    for ax in axes:
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "training_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plots", type=Path)
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    runs = _load(args.models)
    args.out.write_text(phase3.render(analysis) + "\n" + training_table(runs), encoding="utf-8")
    if args.plots is not None:
        try:
            print(plot_training(runs, args.plots))
            for p in phase3.plot(analysis, args.plots):
                print(p)
        except ImportError:
            print("matplotlib not installed; figures skipped (pip install -e .[viz])")


if __name__ == "__main__":
    main()

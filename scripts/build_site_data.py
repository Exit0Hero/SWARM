"""Generate the data files of the public website (``web/data/``) from real SWARM runs.

The site is a static, read-only showcase (ADR-0028). Everything it shows comes from one of two
sources:

- **Prerecorded episodes:** run here with the normal runner, through the dashboard recorder.
  Each one's event-stream hash is stored, so ``tests/test_site_data.py`` can check that the
  current simulator still reproduces it exactly.
- **Validated results:** copied from the archived pre-registered analyses in ``docs/results/``
  (Phase 3 and Phase 7) and from the exploratory benchmark, each with its source path.

Run from the repository root (MuJoCo, from the ``sim3d`` extra, is needed for the 3D analysis):

    python scripts/build_site_data.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import swarm
from swarm.dashboard.service import learned_catalog, record_episode, scene3d
from swarm.examples import example_text
from swarm.scenarios.config import ScenarioConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "data"
SEED = 1000

EPISODES: list[dict[str, str]] = [
    {
        "id": "coordinated",
        "title": "Coordinated search",
        "config": "basic",
        "policy": "auction_comm",
        "summary": "Two drones and two ground robots share belief maps and divide the map into "
        "sectors with a lease auction, over a channel with an 8 / 5 cell range.",
    },
    {
        "id": "independent",
        "title": "Independent search",
        "config": "basic",
        "policy": "frontier",
        "summary": "Same map and seed, but every robot explores on its own information only. "
        "Compare the time to the recall target with the coordinated run.",
    },
    {
        "id": "faults",
        "title": "Faults and noisy sensing",
        "config": "advanced",
        "policy": "auction_comm",
        "summary": "20% message loss, 1-step latency with jitter, two scheduled robot failures, "
        "more hazards and a noisy sensor that misreads terrain and hazards.",
    },
    {
        "id": "learned",
        "title": "Learned allocation (MAPPO)",
        "config": "basic",
        "policy": "learned:mappo_s0",
        "summary": "A sector-selection policy trained with MAPPO (Phase 4), run with NumPy "
        "inference. Same map and seed as the coordinated run.",
    },
]


def _commit() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def episodes() -> list[dict[str, Any]]:
    learned = learned_catalog(ROOT / "docs" / "results" / "phase4" / "models")
    index = []
    for spec in EPISODES:
        config = ScenarioConfig.from_json(example_text(spec["config"]))
        record = record_episode(config, spec["policy"], SEED, learned)
        record.pop("events_jsonl")
        record.pop("diagnostics", None)
        three = scene3d(record)
        record["scene3d"] = three
        record["id"] = spec["id"]
        (OUT / f"episode-{spec['id']}.json").write_text(
            json.dumps(record, separators=(",", ":")), encoding="utf-8"
        )
        m = record["metrics"]
        index.append(
            spec
            | {
                "seed": SEED,
                "config_hash": record["config_hash"],
                "event_stream_sha256": record["event_stream_sha256"],
                "steps": record["steps"],
                "end_reason": record["end_reason"],
                "recall": m["recall"],
                "time_to_recall_target": m["time_to_recall_target"],
                "agents_disabled": m["agents_disabled"],
                "file": f"episode-{spec['id']}.json",
            }
        )
        print(f"{spec['id']}: {record['steps']} steps, {record['end_reason']}")
    return index


def _summary(s: dict[str, Any], key: str, metric: str) -> dict[str, Any] | None:
    m = s[key]["metrics"].get(metric)
    if not m:
        return None
    return {k: m.get(k) for k in ("mean", "ci_low", "ci_high", "n")}


def results() -> dict[str, Any]:
    p7 = json.loads((ROOT / "docs/results/phase7/analysis.json").read_text("utf-8"))
    p3 = json.loads((ROOT / "docs/results/phase3/analysis.json").read_text("utf-8"))
    bench = json.loads((ROOT / "docs/results/benchmarks/v0.1.0-benchmark.json").read_text("utf-8"))
    s = p7["summaries"]
    policies = ["frontier", "frontier_shared", "mappo", "auction_comm", "assignment"]
    nominal = [
        {
            "policy": p,
            "time_censored": _summary(s, f"nominal/{p}", "time_censored"),
            "mission_success": _summary(s, f"nominal/{p}", "mission_success"),
        }
        for p in policies
    ]
    maps = [
        {
            "condition": c,
            "success": {p: _summary(s, f"{c}/{p}", "mission_success") for p in policies},
        }
        for c in ("map_15", "nominal", "map_30", "map_40")
    ]
    families: dict[str, dict[str, int]] = {}
    for r in p7["confirmatory"]:
        fam = r["id"].split(":")[0]
        families.setdefault(fam, {})
        families[fam][r["verdict"]] = families[fam].get(r["verdict"], 0) + 1
    phase3 = [
        {
            k: r[k]
            for k in (
                "id",
                "a",
                "b",
                "metric",
                "mean_a",
                "mean_b",
                "mean_difference",
                "ci_low",
                "ci_high",
                "p_adjusted",
                "verdict",
            )
        }
        for r in p3["confirmatory"]
    ]
    return {
        "phase7": {
            "source": "docs/results/phase7/analysis.json",
            "seeds": f"{p7['seeds'][0]}-{p7['seeds'][-1]}",
            "nominal": nominal,
            "maps": maps,
            "families": families,
        },
        "phase3": {"source": "docs/results/phase3/analysis.json", "confirmatory": phase3},
        "benchmark": {
            "source": "docs/results/benchmarks/v0.1.0-benchmark.json",
            "environment": bench["environment"],
            "robots": [
                {"case": r["case"], "ms_per_step": r["ms_per_step"]}
                for r in bench["results"]
                if r["group"] == "robots"
            ],
        },
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "swarm_version": swarm.__version__,
        "commit": _commit(),
        "kind": "prerecorded",
        "note": "Episodes were run ahead of time with the SWARM simulator and are replayed "
        "in the browser. Nothing on this site runs a simulation.",
        "episodes": episodes(),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(results(), indent=1), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

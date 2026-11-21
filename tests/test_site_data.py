"""Public website (ADR-0028): its data are real SWARM output, and its deployment is static."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swarm.dashboard.service import learned_catalog
from swarm.examples import example_text
from swarm.experiments.runner import run_episode
from swarm.policies.registry import POLICIES
from swarm.scenarios.config import ScenarioConfig

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
MANIFEST = json.loads((WEB / "data" / "manifest.json").read_text())


@pytest.mark.parametrize("episode", MANIFEST["episodes"], ids=lambda e: e["id"])
def test_every_prerecorded_episode_is_reproduced_by_the_current_simulator(
    episode: dict[str, object],
) -> None:
    config = ScenarioConfig.from_json(example_text(str(episode["config"])))
    assert config.config_hash() == episode["config_hash"]
    table = {**POLICIES, **learned_catalog(ROOT / "docs" / "results" / "phase4" / "models")}
    result = run_episode(
        config, table[str(episode["policy"])], int(str(episode["seed"])), designation="development"
    )
    assert result.event_stream_sha256 == episode["event_stream_sha256"]
    record = json.loads((WEB / "data" / str(episode["file"])).read_text())
    assert record["event_stream_sha256"] == episode["event_stream_sha256"]
    assert len(record["frames"]) == result.steps + 1
    assert record["metrics"]["recall"] == result.metrics.recall
    assert "events_jsonl" not in record  # only what the page needs is published


def test_site_results_match_the_archived_analyses() -> None:
    site = json.loads((WEB / "data" / "results.json").read_text())
    archive = json.loads((ROOT / site["phase7"]["source"]).read_text())
    for row in site["phase7"]["nominal"]:
        archived = archive["summaries"][f"nominal/{row['policy']}"]["metrics"]["time_censored"]
        assert row["time_censored"]["mean"] == archived["mean"]
    counts: dict[str, int] = {}
    for r in archive["confirmatory"]:
        counts[r["id"].split(":")[0]] = counts.get(r["id"].split(":")[0], 0) + 1
    assert {k: sum(v.values()) for k, v in site["phase7"]["families"].items()} == counts


def test_vercel_serves_only_the_static_site_with_a_strict_policy() -> None:
    config = json.loads((ROOT / "vercel.json").read_text())
    assert config["outputDirectory"] == "web"
    assert config["buildCommand"] == ""  # no build step: nothing executes on Vercel
    assert not (ROOT / "api").exists()  # no serverless functions
    headers = {h["key"]: h["value"] for r in config["headers"] for h in r["headers"]}
    csp = headers["Content-Security-Policy"]
    assert "script-src 'self'" in csp
    assert "unsafe-eval" not in csp
    assert "frame-ancestors 'none'" in csp
    lines = (ROOT / ".vercelignore").read_text().splitlines()
    rules = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    assert rules[:2] == ["*", "!web"]  # the upload is limited to web/ and vercel.json


def test_site_references_only_files_that_exist() -> None:
    html = (WEB / "index.html").read_text()
    for asset in ("assets/site.css", "assets/app.js", "assets/favicon.svg"):
        assert asset in html
        assert (WEB / asset).exists()
    for episode in MANIFEST["episodes"]:
        assert (WEB / "data" / episode["file"]).exists()
    assert MANIFEST["kind"] == "prerecorded"

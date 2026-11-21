"""The ``swarm`` command: every documented subcommand, its output and its exit codes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import swarm
from swarm.cli import EXIT_OK, EXIT_USAGE, main
from swarm.examples import NAMES, example_text, resolve_config_text
from swarm.scenarios.config import ScenarioConfig

ROOT = Path(__file__).resolve().parents[1]


def test_help_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--help"])
    assert info.value.code == 0
    out = capsys.readouterr().out
    for command in (
        "demo",
        "run",
        "validate",
        "policies",
        "compare",
        "dashboard",
        "sim3d",
        "study",
        "analyze",
        "benchmark",
    ):
        assert command in out
    with pytest.raises(SystemExit):
        main(["--version"])
    assert capsys.readouterr().out.strip() == f"swarm {swarm.__version__}"


def test_packaged_examples_are_valid_and_basic_is_the_validated_nominal_config() -> None:
    for name in NAMES:
        ScenarioConfig.from_json(example_text(name))
    nominal = ScenarioConfig.from_json((ROOT / "configs/phase3/nominal.json").read_text())
    assert ScenarioConfig.from_json(example_text("basic")).config_hash() == nominal.config_hash()
    with pytest.raises(FileNotFoundError, match="not an example"):
        resolve_config_text("nope")


def test_validate_and_policies(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", "--config", "advanced"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "valid config" in out
    assert "failures: yes" in out
    assert main(["policies", "--config", "basic"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "assignment_oracle (ORACLE" in out
    assert "learned:mappo_s0" in out  # run from the repo root: trained actors are found


def test_run_writes_a_reproducible_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out, events = tmp_path / "r.json", tmp_path / "e.jsonl"
    args = ["run", "--config", "basic", "--policy", "auction_comm", "--seed", "1000"]
    assert main([*args, "--out", str(out), "--events", str(events)]) == EXIT_OK
    assert "survivors found" in capsys.readouterr().out
    first = json.loads(out.read_text())
    assert main([*args, "--out", str(out)]) == EXIT_OK
    assert json.loads(out.read_text())["event_stream_sha256"] == first["event_stream_sha256"]
    assert first["metadata"]["code"]["commit"]
    lines = events.read_text().splitlines()
    assert json.loads(lines[-1])["kind"] == "episode_end"


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["run", "--config", "missing.json"], "no such file"),
        (["run", "--policy", "bogus"], "unknown policy"),
        (["run", "--config", "basic", "--seed", "-1"], "seed"),
        (["compare", "--policies", "frontier", "--seeds", "x"], "seed"),
    ],
)
def test_errors_exit_2_with_one_line(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == EXIT_USAGE
    err = capsys.readouterr().err
    assert message in err
    assert "Traceback" not in err


def test_bad_config_file_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema_version": 2}')
    assert main(["validate", "--config", str(bad)]) == EXIT_USAGE
    assert "bad.json" in capsys.readouterr().err


def test_compare(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "c.json"
    argv = ["compare", "--policies", "frontier", "auction_comm", "--seeds", "1000-1001"]
    assert main([*argv, "--out", str(out)]) == EXIT_OK
    assert "exploratory" in capsys.readouterr().out
    assert json.loads(out.read_text())["seeds"] == [1000, 1001]


def test_demo_produces_real_results_for_the_dashboard(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "demo"
    assert main(["demo", "--out-dir", str(out), "--compare-seeds", "2"]) == EXIT_OK
    text = capsys.readouterr().out
    assert "[basic]" in text
    assert "[advanced]" in text
    for name in (
        "basic_run.json",
        "basic_events.jsonl",
        "basic_metrics.csv",
        "advanced_run.json",
        "advanced_scene3d.json",
        "advanced_comparison.json",
    ):
        assert (out / name).stat().st_size > 0
    assert len(list((out / "dashboard" / "runs").glob("*.json"))) == 2
    run = json.loads((out / "basic_run.json").read_text())
    assert run["metrics"]["recall"] is not None
    assert len(run["frames"]) == run["steps"] + 1


@pytest.mark.parametrize("command", ["dashboard", "study", "analyze", "benchmark"])
def test_delegated_commands_show_their_own_help(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as info:
        main([command, "--help"])
    assert info.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_benchmark_reports_environment_and_cases(tmp_path: Path) -> None:
    out = tmp_path / "bench.json"
    assert main(["benchmark", "--seeds", "1000", "--repetitions", "1", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["environment"]["python"] == sys.version.split()[0]
    groups = {r["group"] for r in report["results"]}
    assert groups == {"overhead", "robots", "map", "duration"}
    assert all(r["ms_per_step"] > 0 for r in report["results"])


def test_sim3d_on_a_packaged_example(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("mujoco")
    scene = tmp_path / "scene.json"
    argv = ["sim3d", "--config", "basic", "--policy", "frontier", "--seed", "1000"]
    assert main([*argv, "--scene-out", str(scene)]) == EXIT_OK
    assert "0 contacts" in capsys.readouterr().out
    assert json.loads(scene.read_text())["scene"]["tracks"]


def test_sim3d_without_mujoco_explains_the_extra(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "mujoco", None)  # import now raises ImportError
    assert main(["sim3d", "--config", "basic"]) == EXIT_USAGE
    assert "swarm[sim3d]" in capsys.readouterr().err

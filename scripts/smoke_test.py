"""End-to-end smoke test of an installed SWARM, as a new user would run it.

Run it with the Python of an environment where SWARM was installed from a wheel, from a directory
outside the repository, so that nothing comes from the source tree:

    python scripts/smoke_test.py [--sim3d]

Steps: CLI help and version; validate a packaged example; run an episode and export it; run the
demo; start the dashboard, load the demo run over HTTP and fetch its replay frames, 3D scene and
CSV export; optionally run the MuJoCo 3D analysis. Every step must succeed (exit code 0).
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


def swarm(*args: str, cwd: Path) -> str:
    exe = shutil.which("swarm")
    cmd = [exe, *args] if exe else [sys.executable, "-m", "swarm.cli", *args]
    print("$ swarm " + " ".join(args), flush=True)
    done = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=600)
    if done.returncode != 0:
        print(done.stdout, done.stderr, sep="\n")
        raise SystemExit(f"FAILED (exit {done.returncode}): swarm {' '.join(args)}")
    return done.stdout


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return bytes(response.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim3d", action="store_true", help="also run the MuJoCo 3D analysis")
    args = parser.parse_args()
    work = Path(tempfile.mkdtemp(prefix="swarm-smoke-"))
    assert "usage: swarm" in swarm("--help", cwd=work)
    print(swarm("--version", cwd=work).strip())
    assert "valid config" in swarm("validate", "--config", "advanced", cwd=work)
    out = swarm(
        "run",
        "--config",
        "basic",
        "--policy",
        "auction_comm",
        "--seed",
        "1000",
        "--out",
        "run.json",
        "--events",
        "run.jsonl",
        cwd=work,
    )
    assert "survivors found" in out
    result = json.loads((work / "run.json").read_text())
    assert result["metrics"]["recall"] is not None
    again = swarm(
        "run",
        "--config",
        "basic",
        "--policy",
        "auction_comm",
        "--seed",
        "1000",
        "--out",
        "run2.json",
        cwd=work,
    )
    assert again.splitlines()[0] == out.splitlines()[0]
    assert (
        json.loads((work / "run2.json").read_text())["event_stream_sha256"]
        == result["event_stream_sha256"]
    ), "same seed and config must reproduce the event stream"
    print(swarm("demo", "--compare-seeds", "2", cwd=work))

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    exe = shutil.which("swarm")
    cmd = [exe] if exe else [sys.executable, "-m", "swarm.cli"]
    server = subprocess.Popen(
        [*cmd, "dashboard", "--port", str(port), "--data-dir", "swarm-demo/dashboard"], cwd=work
    )
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(100):
            try:
                get(base + "/api/meta")
                break
            except OSError:
                time.sleep(0.2)
        assert b"SWARM" in get(base + "/")
        configs = json.loads(get(base + "/api/configs"))
        assert any(name.endswith("examples/basic.json") for name in configs), configs.keys()
        history = json.loads(get(base + "/api/history"))
        run_id = next(h["id"] for h in history if h["kind"] == "run")
        run = json.loads(get(f"{base}/api/runs/{run_id}"))
        assert len(run["frames"]) == run["steps"] + 1
        assert json.loads(get(f"{base}/api/runs/{run_id}/scene3d"))["scene"]["tracks"]
        assert b"metric,value" in get(f"{base}/api/runs/{run_id}/export?format=csv")
        print(
            f"dashboard: {len(history)} stored items, run {run_id} replayed ({run['steps']} steps)"
        )
    finally:
        server.terminate()
        server.wait(timeout=30)

    if args.sim3d:
        out3d = swarm(
            "sim3d",
            "--config",
            "basic",
            "--policy",
            "frontier",
            "--seed",
            "1000",
            "--scene-out",
            "scene.json",
            cwd=work,
        )
        assert " 0 contacts" in out3d, out3d
        print(out3d.strip())
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

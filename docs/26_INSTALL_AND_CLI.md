# Installation and CLI guide

## Requirements
- Python 3.12, 3.13 or 3.14.
- Runtime dependency: NumPy only.
- Optional extras:

| Extra | Adds | Needed for |
|---|---|---|
| `dev` | pytest, Hypothesis, Ruff, mypy | tests and the quality gate |
| `viz` | matplotlib | `swarm.visualization.replay` and study figures |
| `sim3d` | MuJoCo | 3D contact and line-of-sight analysis (`swarm sim3d`). The dashboard's 3D view works without it. |
| `learning` | PyTorch, PettingZoo, Gymnasium | training MARL models. Running the trained models needs only NumPy. |

**What is verified, and where:**
- **Linux:** the CI matrix runs the full test suite on Python 3.12, 3.13 and 3.14, plus jobs with
  the learning and sim3d extras.
- **macOS (Python 3.13) and Windows (Python 3.13):** CI builds the wheel, installs it into a clean
  virtual environment, and runs the end-to-end smoke test (`scripts/smoke_test.py`).
- **Not run on macOS or Windows:** the full test suite, and the learning and sim3d extras.

## Install

### Linux and macOS
```bash
git clone https://github.com/Exit0Hero/SWARM.git
cd SWARM
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .              # or ".[dev]", ".[sim3d]", ".[dev,sim3d,viz]"
swarm --version
```

### Windows (PowerShell)
```powershell
git clone https://github.com/Exit0Hero/SWARM.git
cd SWARM
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1               # if blocked: Set-ExecutionPolicy -Scope Process RemoteSigned
python -m pip install --upgrade pip
python -m pip install -e .
swarm --version
```

### From a release wheel (no clone)
```bash
python -m pip install swarm-0.1.0-py3-none-any.whl            # downloaded from the GitHub release
python -m pip install "swarm-0.1.0-py3-none-any.whl[sim3d]"   # with the 3D extra
```
A wheel install includes the packaged examples (`basic` and `advanced`), the dashboard and the CLI.
Three things come only with the source: the archived results, the per-phase configs and the trained
models (`docs/results/phase4/models`). Without the models, `learned:*` policies are not listed.

### MARL training (optional)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU-only PyTorch
pip install -e ".[learning]"
```

## The `swarm` command

`swarm --help` lists the commands, and `swarm <command> --help` shows each one's options.

**Configs.** `--config` accepts either a packaged example or a JSON file:
- `basic`: the validated Phase 3 nominal scenario. Same config hash as
  `configs/phase3/nominal.json`.
- `advanced`: adds loss and delay, two failures, more hazards and noisy perception.
- a path to any JSON file, for example `configs/m2_default.json`.

| Command | What it does |
|---|---|
| `swarm demo [--out-dir swarm-demo] [--seed 1000] [--compare-seeds 5]` | Runs both examples with `auction_comm`. Writes run JSON, event logs, metrics CSV and a 3D scene (with MuJoCo analysis if installed). Runs an exploratory comparison of three policies. Fills `swarm-demo/dashboard/`. |
| `swarm run --config basic --policy auction_comm --seed 1000 [--out r.json] [--events e.jsonl]` | Runs one episode and prints a summary. Optionally writes the full versioned result and the event stream. |
| `swarm validate --config FILE` | Validates a config, prints its hash and the optional blocks, and lists the policies that can run it. |
| `swarm policies --config FILE` | Lists every policy, oracle labels, and why a policy cannot run a config (for example, no channel). |
| `swarm compare --config advanced --policies frontier_shared auction_comm --seeds 1000-1009 [--out c.json]` | Exploratory comparison on matched seeds: means, 95% intervals and paired differences. |
| `swarm dashboard [--port 8765] [--data-dir DIR] [--host 127.0.0.1]` | Starts the web dashboard ([docs/23](23_DASHBOARD.md)). |
| `swarm sim3d --config basic --policy frontier --seed 1000 --scene-out scene.json` | 3D kinematic co-simulation of one episode ([docs/24](24_3D_SIMULATION.md)). Needs `sim3d`. |
| `swarm study --spec configs/phase3/study.json --out-dir runs/p3 --jobs 2 [--resume]` | Runs a pre-registered study spec. |
| `swarm analyze --spec … --runs … --out analysis.json` | Runs a study's pre-registered tests. |
| `swarm benchmark [--seeds 1000-1004] [--repetitions 3] [--out b.json] [--markdown b.md]` | Timing benchmark ([docs/28](28_BENCHMARKS.md)). |

**Exit codes:**
- `0`: success;
- `1`: the command ran, but an episode failed, an experiment was incomplete, or the 3D check found
  contacts;
- `2`: bad usage, config, policy, seed or input. The error is printed on one line on stderr.

The older module commands (`python -m swarm.experiments.run_episode`, `run_experiment`,
`run_baselines` and `python -m swarm.dashboard`) still work unchanged.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `swarm: command not found` | Activate the virtual environment, or use `python -m swarm.cli`. |
| `auction_comm: needs a scenario with a comms block` | Use `basic`, `advanced`, or a config with `comms`. `swarm policies --config X` shows what can run. |
| `the 3D analysis needs MuJoCo` | `pip install "swarm[sim3d]"` (or `-e ".[sim3d]"` from a clone). |
| Dashboard port in use | `swarm dashboard --port 9000`. |

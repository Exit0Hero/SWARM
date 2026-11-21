# Phase 4 (MARL) results

The pre-registration, verdicts and interpretation are in `docs/19_PHASE4_MARL_STUDY.md`.

| Path | Contents |
|---|---|
| `models/<name>/` | Trained models. `best_actor.npz` is the validation-selected actor and the one evaluated. `last_actor.npz`, `train_log.jsonl` and `model.json` hold the version record: configs and hashes, commit, software, seeds, validation history, and the selected update and mode. |
| `runs.tar.gz` | Per-cell evaluation results (seeds 400–449) and `seed_level.csv`. |
| `analysis.json`, `results.md`, `figures/` | Tests, tables, and figures (training curves, nominal policies). |
| `report.py` | Renders `results.md` and the figures from the files above. |

To reproduce the training, install `pip install -e .[learning]` and run, for each config and seed:
```
python -m swarm.learning.train --config configs/phase4/train_mappo.json --seed 0 --out docs/results/phase4/models/mappo_s0
```
Training is deterministic for a given machine and software stack, but not across different PyTorch builds or CPUs.

To reproduce the evaluation, which needs NumPy only:
```
python -m swarm.experiments.study --spec configs/phase4/study.json --out-dir docs/results/phase4/runs --jobs 2
python -m swarm.experiments.study_analysis --spec configs/phase4/study.json --runs docs/results/phase4/runs --out docs/results/phase4/analysis.json
python docs/results/phase4/report.py --analysis docs/results/phase4/analysis.json --models docs/results/phase4/models --out docs/results/phase4/results.md --plots docs/results/phase4/figures
```

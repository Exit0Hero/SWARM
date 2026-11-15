# Phase 3 (M3.5) coordination study results

The pre-registration, verdicts and interpretation are in `docs/18_PHASE3_COORDINATION_STUDY.md`.

| File | Contents |
|---|---|
| `runs.tar.gz` | Study A (seeds 200–249). One result file per (condition, policy) with per-episode records, plus `seed_level.csv` (one row per episode, every metric). |
| `analysis.json`, `results.md`, `figures/` | Study A tests, tables and figures. |
| `runs_b.tar.gz`, `analysis_b.json`, `results_b.md` | Study B (seeds 300–349), the same layout. |
| `report.py` | Renders `results*.md` and the figures from an analysis file. It reads files only. |

Reproduce study A (study B: `study_b.json`, `runs_b`, `analysis_b.json`):
```
python -m swarm.experiments.study --spec configs/phase3/study.json --out-dir docs/results/phase3/runs --jobs 2
python -m swarm.experiments.study_analysis --spec configs/phase3/study.json --runs docs/results/phase3/runs --out docs/results/phase3/analysis.json
python docs/results/phase3/report.py --analysis docs/results/phase3/analysis.json --out docs/results/phase3/results.md --plots docs/results/phase3/figures
```

- **Determinism:** episodes are deterministic, so a rerun reproduces every event-stream hash in the
  archive.
- **Not deterministic:** wall-clock runtimes.
- **Extracting the archive:** run `tar xzf runs.tar.gz` inside this directory to rerun the
  analysis without rerunning episodes.
- **Figures:** need `pip install -e .[viz]`.

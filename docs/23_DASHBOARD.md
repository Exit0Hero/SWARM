# Dashboard (Phase 6)

The design is in [ADR-0026](decisions/0026-dashboard.md).

```bash
python -m swarm.dashboard            # http://127.0.0.1:8765/
python -m swarm.dashboard --port 9000 --data-dir runs-ui --models docs/results/phase4/models
```
Run it from the repository root, so `configs/` and the trained models are found. It needs no extra
packages.

## Simulate tab
- **Choose:**
  - a base config (any scenario JSON under `configs/`, or one you saved);
  - a policy and a seed;
  - changes to the time limit and `p_detect`, communication (ranges, loss, latency, jitter,
    bandwidth), scheduled failures and noisy perception.
- **Validate** shows the effective config and its hash. The full JSON is editable under
  "Effective config".
- **Run episode** runs it on the server and shows the replay:
  - terrain, hazards, the base and robots (▲ drone, ■ ground, ✕ disabled);
  - sensor ranges, trails, message links, and detection reports (green true, red false);
  - survivors (☆, ground truth).
- **Replay controls:** play, pause, step, jump to the first or last step, scrub, and speed.
- **Side panels:**
  - mission progress: survivors truly found against the target, and active robots;
  - per-robot energy and status;
  - per-step communication, detection and perception counts, and failures;
  - the final metrics.
- **Exports:** run JSON, events (JSON Lines) and metrics CSV.
- **Save config** stores the current settings as `saved/<name>.json`.

## Experiments tab
- Runs the selected policies on the same seeds with the current settings, at most 600 episodes.
- Shows mean and 95% intervals per metric, a bar chart, and paired differences against the
  first-ticked policy.
- Exports a seed-level CSV and JSON.
- These comparisons are exploratory. Use `python -m swarm.experiments.study` for pre-registered
  studies.

## History tab
Lists every stored run and experiment. Each can be opened back into its view, deleted or exported.

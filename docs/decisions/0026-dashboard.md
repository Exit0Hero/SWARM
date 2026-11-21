# ADR-0026: Dashboard: a standard-library server over the real runner

- **Status:** Accepted
- **Date:** 2026-11-21
- **Phase:** 6

## Context
Phase 6 requires live and replay views, policy comparison and run summaries. All of them must be
read-only consumers of events and snapshots, with headless mode preserved. The user asked for
working controls and no fabricated data or movement.

## Decision
1. **Back end.** `swarm.dashboard.server` uses `http.server` from the standard library (no new
   dependencies) and binds to `127.0.0.1` by default.
   - Routes: `/api/meta`, `/api/configs` (GET/POST), `/api/validate`, `/api/runs` (POST/GET/DELETE),
     `/api/experiments` (background job, POST/GET/DELETE), `/api/jobs/<id>`, `/api/history` and
     `/api/<runs|experiments>/<id>/export?format=…`.
2. **One simulation path.** `run_episode` gains an optional read-only `on_step(env, result)` hook.
   - It is called after reset and after every step. When omitted, behaviour is unchanged.
   - The dashboard records each frame from `env.snapshot()` through this hook, so frames and metrics
     come from the same run. A test checks that the event hash equals an unhooked run's.
   - Comparisons call `run_experiment`. The browser never simulates anything.
3. **Policies.** Registered policies, plus the Phase 4 trained actors as `learned:<name>`.
   - Learned actors are loaded from `docs/results/phase4/models` with NumPy inference only.
   - Policies that cannot run on a config (for example the auction without `comms`) are shown as
     unavailable, with the reason.
4. **Evaluation view.** The page shows ground truth labelled as such: survivor stars and
   true/false report colours. Policies never receive it (ADR-0008).
5. **Storage.** Runs, experiments and saved configs are JSON files under `--data-dir` (default
   `.swarm-dashboard/`). Exports: run JSON, event JSON Lines, metrics CSV, experiment JSON and
   seed CSV.
6. **Statistics label.** Dashboard comparisons are labelled exploratory: development designation,
   paired bootstrap intervals, unadjusted sign-flip p-values. Pre-registered studies stay in
   `swarm.experiments.study`.
7. **Size cap.** At most 600 episodes per dashboard experiment.

## Consequences
- **Replay, not live.** Episodes finish in well under a second, so "live" means replaying recorded
  frames (play, pause, step, scrub, speed). There is no server-side step-by-step session.
- **Tests:**
  - service and HTTP tests in `tests/test_dashboard.py`;
  - every control was exercised in headless Chrome during development. That browser check is not
    in CI, which has no browser.

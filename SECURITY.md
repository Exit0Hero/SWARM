# Security policy

SWARM is a local research simulator. It has no network service other than the optional dashboard,
which binds to localhost, and it handles no credentials.

## Supported versions
Security fixes are applied to the latest release (0.1.x) and to `main`.

## Reporting a vulnerability
Please use GitHub's private vulnerability reporting ("Security" tab → "Report a vulnerability")
rather than a public issue. Include steps to reproduce and the affected version.

## Review summary (v0.1.0, 2026-11-22)

**Dependencies**
- `pip-audit` was run against the resolved environment: core, `dev`, `viz`, `sim3d` and
  `learning`, 65 packages, including NumPy 2.5.4, PyTorch 2.15.0, MuJoCo 4.0.1 and matplotlib
  3.11.2.
- Result: **no known vulnerabilities**.
- The only runtime dependency is NumPy.

**Code**
- **Deserialization.**
  - Configs and results are JSON, parsed with `json` and validated into frozen dataclasses.
  - Trained actors are loaded with `numpy.load`, without pickle.
  - There is no `eval`, `exec`, `yaml.load` or `shell=True`.
  - `subprocess` is called only with fixed `git` argument lists, to record provenance.
- **Secrets and environment:** none are read or stored. The package reads no environment
  variables.

**Dashboard** (`swarm dashboard`)
- **Binding:** it binds to `127.0.0.1` by default and prints a warning when bound to any other
  address. It has **no authentication**, so do not expose it to a network you do not trust.
- **Cross-site and DNS-rebinding protection** when bound to localhost:
  - requests whose `Host` header is not localhost are refused (403);
  - request bodies must be `application/json`, so a plain HTML form on another site cannot post to
    it.
- **Size limits:** request bodies are capped at 1 MB (413). Experiments are capped at 600 episodes
  each.
- **Files:**
  - Record IDs must be alphanumeric, and configs are chosen from a server-side list.
  - Saved config names are restricted to `[A-Za-z0-9_-]`.
  - Writes go only under `--data-dir`. Exports are generated in memory.
- **Headers:** `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
  `Referrer-Policy: no-referrer` and `Cache-Control: no-store`.
- **Known limitation:** any local process can call the API. The server can run simulations
  (CPU use) and write JSON files under its data directory.

## Known limitation: training checkpoints
`swarm.learning.train` resumes training with `torch.load(..., weights_only=False)`, which uses
pickle. **Only resume from checkpoints you created.** Evaluation and the dashboard never load these
files: they use the NumPy `best_actor.npz`.

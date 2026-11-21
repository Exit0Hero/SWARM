# Public website and Vercel deployment

The architecture and its rationale are in [ADR-0028](decisions/0028-public-website.md). In short,
the website is a **static, read-only showcase**: it replays real, prerecorded SWARM runs and shows
archived results. It never runs a simulation. New simulations run locally with `swarm dashboard`.

## Layout
| Path | Purpose |
|---|---|
| `web/index.html`, `web/assets/` | The page, its styles and its script (no build step, no framework) |
| `web/data/manifest.json` | The recorded episodes: scenario, policy, seed, config hash, event-stream hash, commit |
| `web/data/episode-*.json` | Frames recorded by the simulator, plus each episode's 3D scene and MuJoCo analysis |
| `web/data/results.json` | Numbers copied from `docs/results/` archives, each with its source path |
| `vercel.json`, `.vercelignore` | Static hosting from `web/`, security headers, upload limited to `web/` |
| `scripts/build_site_data.py` | Regenerates `web/data/` from real runs |
| `scripts/serve_site.py` | Local preview with the same headers as Vercel |

## Preview locally
```bash
python scripts/serve_site.py          # http://127.0.0.1:8811/
```

## Regenerate the data
Do this after any change that alters simulator behaviour. CI fails until it is done.
```bash
pip install -e ".[sim3d]"
python scripts/build_site_data.py
pytest tests/test_site_data.py
```

## Deploy on Vercel (one-time)
1. Sign in at vercel.com with the GitHub account that owns `Exit0Hero/SWARM`.
2. Choose **Add New → Project**, then **Import** `Exit0Hero/SWARM`. Allow the Vercel GitHub app
   access to this repository if asked.
3. Leave the settings as `vercel.json` sets them:
   - Framework Preset: **Other**;
   - Root Directory: `./`;
   - Build Command and Install Command: empty (`vercel.json` sets them);
   - Output Directory: `web`.

   No environment variables are needed.
4. Choose **Deploy**. Vercel assigns a `*.vercel.app` URL. Later pushes to `main` redeploy
   automatically, and pull requests get preview deployments.

With the Vercel CLI the same can be done with `vercel` (link the project) and `vercel --prod`.

## Verified before release
- **Browser checks** (headless Chrome, against `scripts/serve_site.py` with the production
  headers), desktop 1440 px and mobile 390 px:
  - autoplay, pause, stepping, timeline, scenario switching and the 3D view;
  - robot, failure and communication panels, and the results tables and charts;
  - no broken anchors, no horizontal overflow and no CSP violations;
  - `prefers-reduced-motion` disables autoplay.
- **Error states:** if a data file fails to load, the page says so instead of showing anything.
- **`tests/test_site_data.py`:** reproduction of every episode, results against the archives,
  static-only configuration and CSP.

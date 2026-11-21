"""HTTP front of the dashboard (Phase 6, ADR-0026). Standard library only; binds to localhost.

Run::

    python -m swarm.dashboard --data-dir .swarm-dashboard --port 8765

Then open http://127.0.0.1:8765/. Long experiments run in a background thread. The page polls
``/api/jobs/<id>`` until they finish.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import traceback
import uuid
from collections.abc import Callable, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from swarm.dashboard import service
from swarm.dashboard.service import DashboardError, Store

MAX_BODY_BYTES = 1_000_000
"""Largest accepted request body; configs and experiment requests are a few kilobytes."""

LOOPBACK = {"127.0.0.1", "localhost", "::1", "[::1]"}


def packaged_examples() -> Path:
    """Directory of the packaged example configs (``swarm.examples``)."""
    return Path(str(resources.files("swarm.examples")))


class Dashboard:
    """Request routing and the job table; independent of the HTTP transport for testing."""

    def __init__(
        self, store: Store, config_roots: list[Path], models_root: Path | None = None
    ) -> None:
        self.store = store
        self.config_roots = config_roots
        self.learned = service.learned_catalog(models_root)
        self.loopback_only = True  # set by serve(); False only for an explicit non-local bind
        self.jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    # -- helpers
    def _configs(self) -> dict[str, dict[str, Any]]:
        saved = service.list_configs([self.store.root / "configs"])
        return service.list_configs(self.config_roots) | {
            "saved/" + name.split("/", 1)[1]: c for name, c in saved.items()
        }

    def _config_from(self, body: dict[str, Any]) -> Any:
        if "config" in body and isinstance(body["config"], dict):
            base = body["config"]
        else:
            configs = self._configs()
            name = body.get("config_name")
            if name not in configs:
                raise DashboardError(f"unknown config {name!r}")
            base = configs[name]
        return service.build_config(base, body.get("overrides"))

    def _start_job(self, label: str, work: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        jid = uuid.uuid4().hex[:12]
        job: dict[str, Any] = {"id": jid, "label": label, "status": "running", "result": None}
        with self._lock:
            self.jobs[jid] = job

        def run() -> None:
            try:
                record = work()
                self.store.save(record)
                job.update(status="done", result={"kind": record["kind"], "id": record["id"]})
            except DashboardError as exc:
                job.update(status="error", error=str(exc))
            except Exception as exc:  # report, never crash the server
                job.update(status="error", error=f"{type(exc).__name__}: {exc}")
                traceback.print_exc()

        threading.Thread(target=run, daemon=True).start()
        return job

    # -- routes
    def handle(self, method: str, path: str, query: dict[str, str], body: Any) -> Any:
        parts = [p for p in path.split("/") if p][1:]  # drop "api"
        match method, parts:
            case "GET", ["meta"]:
                return service.meta() | {"learned": sorted(self.learned)}
            case "GET", ["configs"]:
                return self._configs()
            case "POST", ["configs"]:
                config = self._config_from(body)
                self.store.save_config(str(body.get("name", "")), config)
                return {"saved": body["name"], "config_hash": config.config_hash()}
            case "POST", ["validate"]:
                config = self._config_from(body)
                return {
                    "config": config.to_dict(),
                    "config_hash": config.config_hash(),
                    "policies": service.policy_catalog(config, self.learned),
                }
            case "POST", ["runs"]:
                config = self._config_from(body)
                policy, seed = str(body.get("policy")), int(body.get("seed", 0))
                record = service.record_episode(config, policy, seed, self.learned)
                self.store.save(record)
                return {k: v for k, v in record.items() if k != "events_jsonl"}
            case "GET", ["runs", rid, "scene3d"]:
                return service.scene3d(self.store.load("runs", rid))
            case "GET", ["runs", rid]:
                record = self.store.load("runs", rid)
                return {k: v for k, v in record.items() if k != "events_jsonl"}
            case "DELETE", [("runs" | "experiments") as kind, rid]:
                self.store.delete(kind, rid)
                return {"deleted": rid}
            case "POST", ["experiments"]:
                config = self._config_from(body)
                policies = [str(p) for p in body.get("policies", [])]
                seeds = _seeds(body.get("seeds", "1000-1009"))
                return self._start_job(
                    f"{len(policies)} policies x {len(seeds)} seeds",
                    lambda: service.run_comparison(config, policies, seeds, self.learned),
                )
            case "GET", ["experiments", rid]:
                return self.store.load("experiments", rid)
            case "GET", ["jobs", jid]:
                if jid not in self.jobs:
                    raise DashboardError(f"no job {jid!r}")
                return self.jobs[jid]
            case "GET", ["history"]:
                return self.store.history()
        raise DashboardError(f"no route {method} {path}")

    def export(self, kind: str, rid: str, fmt: str) -> tuple[str, str, str]:
        if kind == "runs":
            return service.export_run(self.store.load("runs", rid), fmt)
        if kind == "experiments":
            return service.export_experiment(self.store.load("experiments", rid), fmt)
        raise DashboardError(f"cannot export {kind!r}")


def _seeds(spec: Any) -> list[int]:
    from swarm.experiments.experiment import parse_seeds

    try:
        return list(parse_seeds(str(spec)))
    except ValueError as exc:
        raise DashboardError(f"seeds: {exc}") from exc


def make_handler(app: Dashboard) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "SwarmDashboard/1"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
            pass

        def _send(
            self, status: int, body: bytes, mime: str, extra: dict[str, str] | None = None
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data: Any) -> None:
            self._send(status, json.dumps(data).encode("utf-8"), "application/json")

        def _allowed(self, method: str) -> str | None:
            """Why a request is refused, or ``None``. See docs/SECURITY.md."""
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].lower()
            if app.loopback_only and host not in LOOPBACK:
                return "host header is not localhost (DNS-rebinding protection)"
            has_body = self.headers.get("Content-Length") not in (None, "0")
            json_body = (self.headers.get("Content-Type") or "").startswith("application/json")
            if method in ("POST", "DELETE") and has_body and not json_body:
                return "requests with a body must be application/json"
            return None

        def _dispatch(self, method: str) -> None:
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            refusal = self._allowed(method)
            if refusal is not None:
                self._json(HTTPStatus.FORBIDDEN, {"error": refusal})
                return
            try:
                if not url.path.startswith("/api/"):
                    if method != "GET":
                        raise DashboardError("not found")
                    page = resources.files("swarm.dashboard").joinpath("static/index.html")
                    self._send(200, page.read_bytes(), "text/html; charset=utf-8")
                    return
                parts = [p for p in url.path.split("/") if p]
                if method == "GET" and len(parts) == 4 and parts[3] == "export":
                    content, mime, name = app.export(
                        parts[1], parts[2], query.get("format", "json")
                    )
                    disposition = {"Content-Disposition": f'attachment; filename="{name}"'}
                    self._send(200, content.encode("utf-8"), mime, disposition)
                    return
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY_BYTES:
                    self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request too large"})
                    return
                body = json.loads(self.rfile.read(length) or b"{}") if length else {}
                self._json(200, app.handle(method, url.path, query, body))
            except DashboardError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except (ValueError, KeyError, TypeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": f"bad request: {exc}"})

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        def do_DELETE(self) -> None:
            self._dispatch("DELETE")

    return Handler


def serve(
    data_dir: Path,
    config_roots: list[Path],
    host: str = "127.0.0.1",
    port: int = 8765,
    models_root: Path | None = None,
) -> ThreadingHTTPServer:
    roots = [r for r in config_roots if r.is_dir()] + [packaged_examples()]
    app = Dashboard(Store(data_dir), roots, models_root)
    app.loopback_only = host in LOOPBACK
    return ThreadingHTTPServer((host, port), make_handler(app))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m swarm.dashboard")
    parser.add_argument("--data-dir", type=Path, default=Path(".swarm-dashboard"))
    parser.add_argument("--configs", type=Path, nargs="*", default=[Path("configs")])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--models",
        type=Path,
        default=Path("docs/results/phase4/models"),
        help="trained Phase 4 actors to offer as learned:<name> policies",
    )
    args = parser.parse_args(argv)
    if args.host not in LOOPBACK:
        print(
            f"WARNING: binding to {args.host} exposes the dashboard to your network. It has no "
            "authentication and can run simulations and write files under --data-dir "
            "(docs/SECURITY.md).",
            file=sys.stderr,
        )
    server = serve(args.data_dir, args.configs, args.host, args.port, args.models)
    print(f"SWARM dashboard on http://{args.host}:{server.server_address[1]}/ (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0

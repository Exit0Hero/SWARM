"""Serve web/ locally with the headers from vercel.json, to test the site as Vercel serves it.

python scripts/serve_site.py [--port 8811]
"""

from __future__ import annotations

import argparse
import functools
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_server(port: int) -> ThreadingHTTPServer:
    rules = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))["headers"]
    extra = [h for r in rules if r["source"] == "/(.*)" for h in r["headers"]]

    class Handler(SimpleHTTPRequestHandler):
        def end_headers(self) -> None:
            for h in extra:
                self.send_header(h["key"], h["value"])
            super().end_headers()

        def log_message(self, *args: object) -> None:
            pass

    handler = functools.partial(Handler, directory=str(ROOT / "web"))
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8811)
    server = make_server(p.parse_args().port)
    print(f"http://127.0.0.1:{server.server_address[1]}/")
    server.serve_forever()

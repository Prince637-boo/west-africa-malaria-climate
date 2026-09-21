"""Serve the results table locally.

    .\\.venv314\\Scripts\\python.exe ui\\serve.py
    -> http://127.0.0.1:8765/ui/
"""

from __future__ import annotations

import shutil
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = Path(__file__).resolve().parent
PORT = 8765


def sync_metrics() -> None:
    src = ROOT / "reports" / "model_training_metrics.json"
    dst = UI / "model_training_metrics.json"
    if src.exists():
        shutil.copy2(src, dst)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)


def main() -> None:
    sync_metrics()
    print(f"http://127.0.0.1:{PORT}/ui/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()

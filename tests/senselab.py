"""The real srv-senselab (../srv-senselab/srv.py) in a thread, on a data folder of the test's own."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

SRV = Path(os.environ.get("SRV_SENSELAB", Path(__file__).resolve().parents[2] / "srv-senselab")) / "srv.py"


def _module():
    spec = importlib.util.spec_from_file_location("srv_senselab", SRV)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Senselab:
    """`with Senselab(tmp) as s:` serves on s.url; s.store reads what it holds; s.calls are the
    requests it answered, as (method, path)."""

    def __init__(self, data: Path) -> None:
        self.srv = _module()
        self.data = Path(data)
        self.calls: list[tuple[str, str]] = []

    def __enter__(self):
        srv, calls = self.srv, self.calls
        handler = srv.make_handler(srv.Store(self.data))

        class Logged(handler):
            def log_message(self, fmt, *args):
                calls.append((self.command, self.path))

        self.httpd = srv.Server(("127.0.0.1", 0), Logged)
        import threading

        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.store = srv.Store(self.data)
        return self

    def __exit__(self, *_exc):
        self.httpd.shutdown()
        self.httpd.server_close()

    def manifest(self, session_id: str) -> dict:
        return self.store.session(session_id)["manifest"]

    def file(self, sha: str) -> bytes:
        return self.store.file_path(sha).read_bytes()

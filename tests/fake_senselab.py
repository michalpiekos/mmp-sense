"""An in-process stand-in for srv-senselab's protocol 0.1, as much of it as a
workspace uses: sessions, context, uploads in parts, and close."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOKEN = "mmps_test_token"


class FakeSenselab:
    def __init__(self, part_size: int = 1024) -> None:
        self.part_size = part_size
        self.sessions: dict[str, dict] = {}
        self.contexts: dict[str, dict] = {}
        self.uploads: dict[str, dict] = {}
        self.samples: dict[str, dict] = {}
        self.closed: dict[str, dict] = {}
        self.calls: list[tuple[str, str]] = []
        #: (method, path pattern) to fail once with a non-retryable 500.
        self.fail_once: set[tuple[str, str]] = set()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._server.shutdown()
        self._server.server_close()

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def _reply(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _route(self, method):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                path = self.path.removeprefix("/v0.1")
                fake.calls.append((method, path))
                if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                    return self._reply(401, _error("unauthorized", "bad token"))
                for wanted in list(fake.fail_once):
                    if wanted[0] == method and re.fullmatch(wanted[1], path):
                        fake.fail_once.discard(wanted)
                        return self._reply(500, _error("internal", "injected"))
                status, body = fake.handle(method, path, raw)
                self._reply(status, body)

            def do_GET(self):
                self._route("GET")

            def do_PUT(self):
                self._route("PUT")

            def do_POST(self):
                self._route("POST")

        return Handler

    def handle(self, method, path, raw):
        body = None if not raw or re.fullmatch(r"/uploads/[^/]+/parts/\d+", path) else json.loads(raw)
        if (m := re.fullmatch(r"/capture-sessions/([^/]+)", path)) and method == "PUT":
            sid = m[1]
            if sid in self.sessions and self.sessions[sid] != body:
                return 409, _error("conflict", "different declaration")
            self.sessions[sid] = body
            return 200, {"capture_session_id": sid, "state": "open"}
        if (m := re.fullmatch(r"/capture-sessions/([^/]+)/context", path)) and method == "PUT":
            current = self.contexts.get(m[1])
            if current and body["revision"] == current["revision"] and body != current:
                return 409, _error("conflict", "same revision, different content")
            if not current or body["revision"] >= current["revision"]:
                self.contexts[m[1]] = body
            return 200, self.contexts[m[1]]
        if path == "/uploads" and method == "POST":
            return self._declare(body)
        if (m := re.fullmatch(r"/uploads/([^/]+)/parts/(\d+)", path)) and method == "PUT":
            upload = self.uploads[m[1]]
            part = int(m[2])
            if part > upload["next_part"]:
                return 409, _error("part_out_of_order", "skip", {"next_part": upload["next_part"]})
            if part == upload["next_part"]:
                upload["data"] += raw
                upload["next_part"] += 1
            return 200, _public(upload)
        if (m := re.fullmatch(r"/uploads/([^/]+)", path)) and method == "GET":
            return 200, _public(self.uploads[m[1]])
        if (m := re.fullmatch(r"/uploads/([^/]+)/complete", path)) and method == "POST":
            upload = self.uploads[m[1]]
            declaration = upload["declaration"]
            if hashlib.sha256(upload["data"]).hexdigest() != declaration["payload"]["sha256"]:
                return 422, _error("hash_mismatch", "hash")
            upload["state"] = "received"
            self.samples[declaration["sample_id"]] = {"declaration": declaration, "data": upload["data"]}
            return 200, _public(upload)
        if (m := re.fullmatch(r"/capture-sessions/([^/]+)/close", path)) and method == "POST":
            self.closed[m[1]] = body
            missing = [s for s in body["sample_ids"] if s not in self.samples]
            return 200, {"capture_session_id": m[1], "state": "closed", "missing_samples": missing}
        return 404, _error("not_found", path)

    def _declare(self, declaration):
        if declaration["capture_session_id"] not in self.sessions:
            return 404, _error("not_found", "no such session")
        for upload in self.uploads.values():
            if upload["declaration"]["sample_id"] == declaration["sample_id"]:
                return 200, _public(upload)
        size = declaration["payload"]["bytes"]
        upload = {"upload_id": str(uuid.uuid4()), "declaration": declaration, "data": b"",
                  "next_part": 0, "total_parts": -(-size // self.part_size),
                  "part_size_bytes": self.part_size, "state": "created"}
        self.uploads[upload["upload_id"]] = upload
        return 200, _public(upload)


def _public(upload):
    return {k: v for k, v in upload.items() if k not in ("data", "declaration")}


def _error(code, message, details=None):
    return {"error": {"code": code, "message": message, "retryable": False, "details": details or {}}}

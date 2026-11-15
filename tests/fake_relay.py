"""An in-process stand-in for a gateway's relay (srv-senselab protocol/stream.md, "Relay"):
`GET /live` lists the sessions, `GET /live/{id}` sends a session's messages."""

from __future__ import annotations

import json
import queue
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HEADER = struct.Struct("<BBHI")
TOKEN = "mmps_relay_token"
GATEWAY_CLOCK = 255


def msg(kind: str, body: bytes = b"", stream: int = 0, flags: int = 0) -> bytes:
    return HEADER.pack(ord(kind), flags, stream, len(body)) + body


def data(first_frame: int, t: int, counts: np.ndarray, fmt: str = "s16", stream: int = 1) -> bytes:
    from mmp_sense.streaming.relay import encode

    return msg("D", struct.pack("<Qq", first_frame, t) + encode(counts, fmt), stream)


def clock(*readings) -> bytes:
    """C with (clock, value, uncertainty_ns) readings."""
    body = bytes([0, len(readings)]) + b"".join(struct.pack("<BqI", c, v, u) for c, v, u in readings)
    return msg("C", body)


def event(t: int, obj: dict) -> bytes:
    return msg("N", struct.pack("<q", t) + json.dumps(obj).encode())


def listing(sid: str, *, model="tetrahedral", channels=("S", "N", "W", "E"), fmt="s16", rate=48_000,
            utc=True, valid_bits=None, scale=None, unit=None) -> dict:
    clocks = [{"id": 0, "name": "mono", "kind": "monotonic", "hz": 1_000_000, "timeline": True},
              {"id": 1, "name": "i2s", "kind": "sample", "hz": rate}]
    if utc:
        clocks.append({"id": 2, "name": "utc", "kind": "utc", "hz": 1_000_000_000, "source": "ntp"})
    clocks.append({"id": GATEWAY_CLOCK, "name": "gateway", "kind": "utc", "hz": 1_000_000_000,
                   "source": "ntp", "domain": "ntp:stream-gateway"})
    stream = {"id": 1, "name": "mics", "kind": "pcm", "sample_clock": 1, "rate_hz": rate, "format": fmt,
              "channels": [{"name": c, **({"scale": scale} if scale else {}), **({"unit": unit} if unit else {})}
                           for c in channels]}
    if valid_bits:
        stream["valid_bits"] = valid_bits
    return {"capture_session_id": sid, "source_id": None, "device": {"model": model}, "trigger": None,
            "transport": "tcp", "stored": False, "started_utc": "2026-11-15T08:00:00.000Z",
            "clocks": clocks, "streams": [stream]}


def hello(item: dict) -> bytes:
    return msg("H", json.dumps({"v": 2, "clocks": item["clocks"], "streams": item["streams"],
                                "relay": {"capture_session_id": item["capture_session_id"],
                                          "gateway_clock": GATEWAY_CLOCK}}).encode())


class FakeRelay:
    """`sessions[id] = listing`; `push(id, bytes)` sends to that session's
    watchers (and to later ones); `end(id)` closes it."""

    def __init__(self, token: str | None = None) -> None:
        self.token = token
        self.sessions: dict[str, dict] = {}
        self.sent: dict[str, list[bytes]] = {}
        self.watchers: dict[str, list[queue.Queue]] = {}
        self.requests: list[tuple[str, str | None]] = []
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._server.daemon_threads = True
        self.address = f"127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        for sid in list(self.watchers):
            self.end(sid)
        self._server.shutdown()
        self._server.server_close()

    def add(self, item: dict) -> dict:
        self.sessions[item["capture_session_id"]] = item
        self.sent[item["capture_session_id"]] = [hello(item)]
        return item

    def push(self, sid: str, *messages: bytes) -> None:
        with self._lock:
            for m in messages:
                self.sent[sid].append(m)
                for q in self.watchers.get(sid, []):
                    q.put(m)

    def end(self, sid: str) -> None:
        with self._lock:
            for q in self.watchers.pop(sid, []):
                q.put(None)
            self.sessions.pop(sid, None)

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def _json(self, status, body):
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _chunk(self, b: bytes):
                self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n")
                self.wfile.flush()

            def do_GET(self):
                auth = self.headers.get("Authorization")
                token = auth.removeprefix("Bearer ") if auth else None
                fake.requests.append((self.path, token))
                if fake.token and token != fake.token:
                    return self._json(401, {"error": {"code": "unauthorized", "message": "a source token is needed"}})
                if self.path == "/live":
                    return self._json(200, {"streams": list(fake.sessions.values())})
                sid = self.path.removeprefix("/live/")
                if sid not in fake.sessions:
                    return self._json(404, {"error": {"code": "not_found", "message": "no such live stream"}})
                q: queue.Queue = queue.Queue()
                with fake._lock:
                    for m in fake.sent[sid]:
                        q.put(m)
                    fake.watchers.setdefault(sid, []).append(q)
                self.send_response(200)
                self.send_header("Content-Type", "application/x-mmp-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                try:
                    while (m := q.get()) is not None:
                        self._chunk(m)
                    self._chunk(b"")
                except OSError:
                    pass

        return Handler

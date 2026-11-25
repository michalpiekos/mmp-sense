"""An in-process stand-in for an mmp-stream receiver's relay (format 3), whose every message a
test writes itself: `GET /live` lists the sessions, `GET /live/{id}` sends a session's messages.
tests/test_live_ui.py runs the real receiver too."""

from __future__ import annotations

import json
import queue
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HEADER = struct.Struct("<BBHI")
RECEIVER_CLOCK = 255


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
    stream = {"id": 1, "name": "mics", "kind": "pcm", "sample_clock": 1, "rate_hz": rate, "format": fmt,
              "channels": [{"name": c, **({"scale": scale} if scale else {}), **({"unit": unit} if unit else {})}
                           for c in channels]}
    if valid_bits:
        stream["valid_bits"] = valid_bits
    return {"session_id": sid, "node": model, "device": {"model": model}, "trigger": None, "title": None,
            "tags": [], "saved": False, "started_utc_ns": 1_794_729_600_000_000_000,
            "clocks": clocks, "streams": [stream]}


def hello(item: dict) -> bytes:
    receiver = {"id": RECEIVER_CLOCK, "name": "receiver", "kind": "utc", "hz": 1_000_000_000,
                "source": "ntp", "domain": "ntp:mmp-stream-receiver"}
    return msg("H", json.dumps({"v": 3, "node": item["node"], "clocks": item["clocks"] + [receiver],
                                "streams": item["streams"],
                                "relay": {"session_id": item["session_id"], "saved": False,
                                          "receiver_clock": RECEIVER_CLOCK}}).encode())


class FakeRelay:
    """`sessions[id] = listing`; `push(id, bytes)` sends to that session's
    watchers (and to later ones); `end(id)` closes it."""

    def __init__(self) -> None:
        self.sessions: dict[str, dict] = {}
        self.sent: dict[str, list[bytes]] = {}
        self.watchers: dict[str, list[queue.Queue]] = {}
        self.requests: list[str] = []
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
        self.sessions[item["session_id"]] = item
        self.sent[item["session_id"]] = [hello(item)]
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
                fake.requests.append(self.path)
                if self.path == "/live":
                    return self._json(200, {"streams": list(fake.sessions.values())})
                sid = self.path.removeprefix("/live/")
                if sid not in fake.sessions:
                    return self._json(404, {"error": "not_found", "message": "no such open stream"})
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

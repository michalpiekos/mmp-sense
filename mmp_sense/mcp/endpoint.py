"""A running session, answerable from another process.

`session.serve()` opens a Unix socket and answers the semantic tools on
it, so the MCP server -- started by an AI client, in a process of its own
-- can ask about the window the user is actually looking at.

    {"id": 1, "tool": "get_current_context", "args": {}}
    {"id": 1, "ok": true, "result": {...}}

One JSON object per line, both ways. The socket is local, created for
this user only, and the tools it serves are read-only; nothing here
opens a port, reads a file the caller names, or runs code it is sent.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import socket
import threading
from pathlib import Path

from . import tools

__all__ = ["Endpoint", "EndpointClient", "default_directory", "endpoints"]

log = logging.getLogger("mmp_sense.endpoint")


def default_directory() -> Path:
    """Where sockets live: the runtime directory, or a private one in /tmp."""
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(runtime) if runtime else Path(f"/tmp/mmp-sense-{os.getuid()}")
    directory = base / "mmp-sense"
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    return directory


def endpoints() -> list[Path]:
    """The sockets of the sessions serving on this machine, newest first.

    A stale socket -- a session that died -- is removed as it is found,
    so a client is not offered one that cannot answer.
    """
    found = []
    for path in sorted(default_directory().glob("*.sock")):
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.connect(str(path))
            found.append(path)
        except OSError as exc:
            if exc.errno in (errno.ECONNREFUSED, errno.ENOENT):
                path.unlink(missing_ok=True)
        finally:
            probe.close()
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


class Endpoint:
    """Serves one session's tools on a Unix socket, on a thread of its own."""

    def __init__(self, session, path: str | os.PathLike | None = None) -> None:
        self.session = session
        self.path = Path(path) if path else default_directory() / f"{os.getpid()}.sock"
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._connections: list[socket.socket] = []

    def start(self) -> "Endpoint":
        if self._socket is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.path))
        os.chmod(self.path, 0o600)
        server.listen(4)
        server.settimeout(0.2)
        self._socket = server
        self._thread = threading.Thread(
            target=self._accept, name="mmp-sense-endpoint", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        for connection in list(self._connections):
            try:
                connection.close()
            except OSError:
                pass
        self._connections.clear()
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self.path.unlink(missing_ok=True)

    # -- serving -------------------------------------------------------

    def _accept(self) -> None:
        while not self._stopping.is_set():
            try:
                connection, _ = self._socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self._connections.append(connection)
            threading.Thread(
                target=self._serve, args=(connection,),
                name="mmp-sense-endpoint-conn", daemon=True,
            ).start()

    def _serve(self, connection: socket.socket) -> None:
        connection.settimeout(0.5)
        buffer = b""
        try:
            while not self._stopping.is_set():
                try:
                    chunk = connection.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if line.strip():
                        connection.sendall(self.handle(line) + b"\n")
        finally:
            if connection in self._connections:
                self._connections.remove(connection)
            connection.close()

    def handle(self, line: bytes) -> bytes:
        """One request to one response. Public, so a test needs no socket."""
        try:
            request = json.loads(line)
        except ValueError as exc:
            return json.dumps({"ok": False, "error": f"not JSON: {exc}"}).encode()
        reply = {"id": request.get("id")}
        name = request.get("tool", "")
        try:
            if name == "list_tools":
                reply |= {"ok": True, "result": {"tools": tools.describe_tools()}}
            else:
                reply |= {
                    "ok": True,
                    "result": tools.call(self.session, name, request.get("args") or {}),
                }
        except Exception as exc:                             # noqa: BLE001
            log.debug("tool %s failed", name, exc_info=True)
            reply |= {"ok": False, "error": str(exc), "type": type(exc).__name__}
        try:
            return json.dumps(reply, allow_nan=False).encode()
        except ValueError as exc:
            # A result that will not serialise is this end's fault, and
            # saying so beats dropping the connection.
            return json.dumps({
                "id": reply.get("id"), "ok": False,
                "error": f"{name} produced something that is not JSON: {exc}",
                "type": "ValueError",
            }).encode()


class EndpointClient:
    """The other side: asks a serving session for an answer."""

    def __init__(self, path: str | os.PathLike, timeout: float = 10.0) -> None:
        self.path = Path(path)
        self.timeout = timeout
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.settimeout(timeout)
        self._socket.connect(str(self.path))
        self._file = self._socket.makefile("rwb")
        self._lock = threading.Lock()
        self._next_id = 0

    def call(self, name: str, args: dict | None = None) -> dict:
        with self._lock:
            self._next_id += 1
            request = {"id": self._next_id, "tool": name, "args": args or {}}
            self._file.write(json.dumps(request).encode() + b"\n")
            self._file.flush()
            line = self._file.readline()
        if not line:
            raise ConnectionError(f"{self.path}: the session closed the connection")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "the session refused the call"))
        return reply["result"]

    def tools(self) -> list[dict]:
        return self.call("list_tools")["tools"]

    def close(self) -> None:
        try:
            self._file.close()
        finally:
            self._socket.close()

    def __enter__(self) -> "EndpointClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

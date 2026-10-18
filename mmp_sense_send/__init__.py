"""Send Python code to an mmp-sense window's console: `mmp-sense-send`.

    printf 'x = 1\\n' | mmp-sense-send
    :'<,'>w !mmp-sense-send                       (from nvim)

A window started with `--listen` takes code on a Unix socket and runs it
in its console as if it had been typed there. The protocol is the code,
then end of input; the window answers one line, `ok` or `error: ...`.

Standard library only, and apart from the `mmp_sense` package, so an editor
calling it on every keypress does not wait for NumPy to import.
"""

from __future__ import annotations

import argparse
import errno
import os
import socket
import sys
import textwrap
from pathlib import Path

__all__ = ["directory", "listeners", "send", "main", "MAX_BYTES"]

#: More than any selection; a bound so a stray pipe cannot fill memory.
MAX_BYTES = 4 * 1024 * 1024


def directory() -> Path:
    """Where the windows' sockets live: private to this user."""
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(runtime) if runtime else Path(f"/tmp/mmp-sense-{os.getuid()}")
    path = base / "mmp-sense" / "console"
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    os.chmod(path, 0o700)
    return path


def listeners() -> list[Path]:
    """The sockets of the windows listening, newest first; stale ones removed."""
    found = []
    for path in directory().glob("*.sock"):
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.connect(str(path))
            found.append((path.stat().st_mtime, path))
        except OSError as exc:
            if exc.errno in (errno.ECONNREFUSED, errno.ENOENT):
                path.unlink(missing_ok=True)
        finally:
            probe.close()
    return [path for _mtime, path in sorted(found, reverse=True)]


def send(code: str, path: str | os.PathLike | None = None, timeout: float = 5.0) -> str:
    """Send `code` to the window at `path`, or the newest; returns its answer."""
    if path is None:
        path = os.environ.get("MMP_SENSE_SOCKET") or next(iter(listeners()), None)
    if path is None:
        raise ConnectionError("no mmp-sense console listening (start mmp-sense with --listen)")
    data = code.encode()
    if len(data) > MAX_BYTES:
        raise ValueError(f"{len(data)} bytes is more than {MAX_BYTES}")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        try:
            client.connect(str(path))
        except OSError as exc:
            raise ConnectionError(f"no mmp-sense console at {path}: {exc.strerror}") from None
        client.sendall(data)
        client.shutdown(socket.SHUT_WR)
        answer = b""
        while chunk := client.recv(4096):
            answer += chunk
    return answer.decode(errors="replace").strip() or "error: no answer"


def prepare(code: str) -> str:
    """The selection as something to run: common indentation removed."""
    return textwrap.dedent(code.expandtabs(4)).strip("\n") + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="mmp-sense-send",
        description="Run Python code from stdin in an mmp-sense window's console "
                    "(a window started with --listen).",
    )
    parser.add_argument("--socket", help="the window's socket; default "
                        "$MMP_SENSE_SOCKET, else the newest window listening")
    parser.add_argument("--list", action="store_true",
                        help="list the windows listening and exit")
    args = parser.parse_args(argv)
    try:
        if args.list:
            for path in listeners():
                print(path)
            return 0
        code = prepare(sys.stdin.read())
        if not code.strip():
            return 0
        answer = send(code, args.socket)
    except (ConnectionError, ValueError, OSError) as exc:
        print(f"mmp-sense-send: {exc}", file=sys.stderr)
        return 1
    if answer != "ok":
        print(f"mmp-sense-send: {answer}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Code from an editor, run in the window's console: the `--listen` socket.

A thread accepts connections on a Unix socket private to this user and
reads each one's code to the end; the code crosses to the Qt thread by a
queued signal, and the console runs it as if it had been typed -- shown,
run, in its history. Code that arrives while the console is busy waits
its turn, and whatever was half typed at the prompt is put back after.

The socket runs what it is sent, so it is off unless asked for, and only
this user can connect to it. `mmp-sense-send` is the client.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
from collections import deque
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

from mmp_sense_send import MAX_BYTES, directory

__all__ = ["ConsoleListener", "ConsoleRunner"]

log = logging.getLogger("mmp_sense.remote")


class ConsoleRunner(QObject):
    """Runs code in a qtconsole widget, one piece at a time, as if typed."""

    #: Each sent piece's outcome: ok, and the error ("NameError: ...") if not.
    finished = Signal(bool, str)

    def __init__(self, console, parent=None) -> None:
        super().__init__(parent)
        self.console = console
        self._queue: deque[str] = deque()
        self._saved: str | None = None
        console.executed.connect(self._finished)
        # The prompt comes back a moment after the reply; poll for it
        # rather than depend on qtconsole's private ordering.
        self._pump_timer = QTimer(self)
        self._pump_timer.setInterval(50)
        self._pump_timer.timeout.connect(self._pump)

    @property
    def pending(self) -> int:
        return len(self._queue)

    def submit(self, code: str) -> None:
        self._queue.append(code)
        self._pump()

    def _busy(self) -> bool:
        console = self.console
        return bool(console._executing or console._reading or self._saved is not None)

    def _pump(self) -> None:
        if not self._queue:
            self._pump_timer.stop()
            return
        if self._busy():
            self._pump_timer.start()
            return
        self._pump_timer.stop()
        self._saved = self.console.input_buffer
        # A trailing newline would draw an empty continuation prompt.
        self.console.execute(self._queue.popleft().rstrip())

    def _finished(self, reply=None) -> None:
        if self._saved is None:
            return                # typed at the prompt, not sent
        content = (reply or {}).get("content", {})
        if content.get("status") == "error":
            self.finished.emit(False, f"{content.get('ename', 'Error')}: {content.get('evalue', '')}")
        else:
            self.finished.emit(True, "")
        # After the reply the console draws a fresh prompt; restore onto it.
        QTimer.singleShot(0, self._restore)

    def _restore(self) -> None:
        saved, self._saved = self._saved, None
        if saved:
            self.console.input_buffer = saved
        self._pump()


class ConsoleListener(QObject):
    """The socket; hands what it reads to a `ConsoleRunner` on the Qt thread."""

    received = Signal(str)

    def __init__(self, runner: ConsoleRunner, path=None, parent=None) -> None:
        super().__init__(parent)
        self.runner = runner
        self.path = Path(path) if path else directory() / f"{os.getpid()}.sock"
        self.received.connect(runner.submit)        # queued: across threads
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()

    def start(self) -> "ConsoleListener":
        if self._socket is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.path))
        os.chmod(self.path, 0o600)
        server.listen(4)
        server.settimeout(0.2)
        self._socket = server
        self._thread = threading.Thread(target=self._accept, name="mmp-sense-listen", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self.path.unlink(missing_ok=True)

    def _accept(self) -> None:
        while not self._stopping.is_set():
            try:
                connection, _ = self._socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with connection:
                self._serve(connection)

    def _serve(self, connection: socket.socket) -> None:
        connection.settimeout(2.0)
        data = b""
        try:
            while chunk := connection.recv(65536):
                data += chunk
                if len(data) > MAX_BYTES:
                    connection.sendall(f"error: more than {MAX_BYTES} bytes\n".encode())
                    return
        except OSError as exc:
            log.warning("listen: dropped a connection: %s", exc)
            return
        if not data:
            return            # a probe: mmp-sense-send looking for live windows
        try:
            code = data.decode()
        except UnicodeDecodeError:
            connection.sendall(b"error: not UTF-8\n")
            return
        self.received.emit(code)
        try:
            connection.sendall(b"ok\n")
        except OSError:
            pass

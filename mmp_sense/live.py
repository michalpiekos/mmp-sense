"""The console's names, for a script written in an editor and sent to it.

    from mmp_sense.live import session, signal, show, ui

In a file, the line tells a linter and an editor what these are -- the
running `Session`, `signal()`, the window -- so it can check and
complete them. Sent to the mmp-sense console (`mmp-sense-send`), it gives that
console's own objects: nothing is created, so the line is safe to send
again and again, and a selection without it still runs, the names being
in the console already.

Anywhere else there is no window to take them from, and the import says
so.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["session", "signal", "show", "ui"]

if TYPE_CHECKING:
    from .core.graph.node import signal as signal
    from .core.session import Session
    from .ui.main_window import MainWindow

    def _running() -> Session: ...

    # Taken from a Session, so the methods keep their signatures.
    session = _running()
    show = session.show
    ui: MainWindow

_namespace: dict[str, Any] | None = None


def attach(namespace: dict[str, Any]) -> None:
    """Serve these objects: what the window gives its console."""
    global _namespace
    _namespace = namespace


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module 'mmp_sense.live' has no attribute {name!r}")
    if _namespace is None or name not in _namespace:
        raise ImportError(
            f"mmp_sense.live.{name}: no mmp-sense console in this process; run this in the "
            "mmp-sense window's console, e.g. send it there with mmp-sense-send"
        )
    return _namespace[name]

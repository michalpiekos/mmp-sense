"""The session's changes, as Qt signals.

The core knows nothing about Qt, so something has to translate. This is
that something, and it is the only place where the two meet: a listener
on the session, and one Qt signal per kind of change.

A change made on the GUI thread -- from a pane, or from the console --
is forwarded at once. One made anywhere else -- a source's worker, the
MCP endpoint -- goes through a queued connection, so it still arrives on
the GUI thread.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot

__all__ = ["SessionBridge"]


class SessionBridge(QObject):
    """One Qt signal per kind of session event."""

    source_changed = Signal(dict)
    data_arrived = Signal(dict)
    signals_changed = Signal(dict)
    cursor_moved = Signal(float)
    frequency_moved = Signal(object)
    selection_changed = Signal(object)
    analysis_changed = Signal(float)
    playback_changed = Signal(dict)
    intake_changed = Signal(dict)
    plot_requested = Signal(str, object)
    detections_changed = Signal(dict)
    bands_changed = Signal(dict)

    _event = Signal(str, dict)

    def __init__(self, session, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.session = session
        self._event.connect(self._dispatch, Qt.QueuedConnection)
        self._unlisten = session.listen(self._on_session_event)

    def _on_session_event(self, kind: str, payload: dict) -> None:
        if QThread.currentThread() is self.thread():
            # The console and the panes run here, and a change made from
            # one should be on the screen this tick, not the next.
            self._dispatch(kind, payload)
        else:
            self._event.emit(kind, payload)

    @Slot(str, dict)
    def _dispatch(self, kind: str, payload: dict) -> None:
        if kind == "source":
            self.source_changed.emit(payload)
        elif kind == "data":
            self.data_arrived.emit(payload)
        elif kind == "signals":
            self.signals_changed.emit(payload)
        elif kind == "cursor":
            self.cursor_moved.emit(float(payload.get("cursor", 0.0)))
        elif kind == "frequency":
            self.frequency_moved.emit(payload.get("frequency_cursor"))
        elif kind == "selection":
            self.selection_changed.emit(payload.get("selection"))
        elif kind == "analysis":
            self.analysis_changed.emit(float(payload.get("analysis_window", 0.0)))
        elif kind == "playback":
            self.playback_changed.emit(payload)
        elif kind == "intake":
            self.intake_changed.emit(payload)
        elif kind == "detections":
            self.detections_changed.emit(payload)
        elif kind == "bands":
            self.bands_changed.emit(payload)
        elif kind == "plot":
            self.plot_requested.emit(payload.get("signal", ""), payload.get("view"))

    def close(self) -> None:
        self._unlisten()

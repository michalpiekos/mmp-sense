"""File -> Open stream and Save selection to workspace, and the Live streams dock.

Like the File menu's other actions, each takes what its dialog would ask as
arguments too, so a script or a test can run it without one.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtGui import QAction

from ..settings import Settings
from ..streaming import LiveStreams, RelayError, cut
from .panes.base import ALL_CHANNELS
from .widgets.live_dock import LiveDock

__all__ = ["LiveControl", "relay_token"]

#: How often the streams' alignment and the dock are brought up to date.
ALIGN_MS = 500


def _host(address: str) -> str | None:
    from urllib.parse import urlsplit

    return urlsplit(address if "://" in address else "http://" + address).hostname


def relay_token(settings: Settings, relay: str) -> str | None:
    """The server's source token, for a relay on the server's own host, or on
    this machine (an ssh tunnel to it): no other host is given it."""
    host = _host(relay)
    if not settings.server or host not in (_host(settings.server), "localhost", "127.0.0.1", "::1"):
        return None
    return settings.token()


class LiveControl(QObject):
    def __init__(self, window) -> None:
        super().__init__(window)
        self.window = window
        self.live = LiveStreams(window.session)
        files = window.files
        self.open_action = QAction("Open s&tream...", window)
        self.open_action.setStatusTip("Watch live streams from a gateway relay")
        self.open_action.triggered.connect(lambda _checked=False: self.open_streams())
        self.save_action = QAction("Save s&election to workspace", window)
        self.save_action.setStatusTip("Keep the selected stretch of every live stream as workspace recordings")
        self.save_action.triggered.connect(lambda _checked=False: self.save_selection())
        actions = files.menu.actions()
        files.menu.insertAction(actions[actions.index(files.open_action) + 1], self.open_action)
        files.menu.insertAction(actions[actions.index(files.save_all_action) + 1], self.save_action)

        self.dock = LiveDock()
        self.dock.trimmed.connect(self.set_trim)
        self.dock.close_requested.connect(self.close_stream)
        self.live_dock = window._make_dock("Live streams", self.dock, Qt.LeftDockWidgetArea)
        window.set_dock_collapsed(self.live_dock, True)
        self.timer = QTimer(self)
        self.timer.setInterval(ALIGN_MS)
        self.timer.timeout.connect(self.update)
        self.timer.start()

    # -- what a dialog would ask; replaced in tests ---------------------------

    def ask_streams(self, relay: str, token_for):
        """(relay, the relayed sessions ticked) or None."""
        from .stream_dialog import OpenStreamDialog

        dialog = OpenStreamDialog(relay, token_for, self.window)
        try:
            if dialog.exec() != OpenStreamDialog.Accepted or not dialog.chosen:
                return None
            return dialog.listed_relay, dialog.chosen
        finally:
            dialog.deleteLater()

    # -- actions --------------------------------------------------------------

    def open_streams(self, listings=None, relay: str | None = None, token: str | None = None,
                     replace: bool = True) -> list[str]:
        """Watch relayed sessions (`listings`, as `GET /live` lists them).
        The first replaces what the window shows; with `replace=False` they join it."""
        settings = Settings.load()
        relay = relay or settings.relay_address()

        def token_for(address: str) -> str | None:
            return token or relay_token(settings, address)

        if listings is None:
            asked = self.ask_streams(relay, token_for)
            if asked is None:
                return []
            chosen_relay, listings = asked
            if chosen_relay and chosen_relay != relay:
                relay = settings.relay = chosen_relay
                try:
                    settings.save()
                except OSError:
                    pass
        replace = replace or not self.live.streams
        try:
            if replace:
                self.window.player.stop(quietly=True)
            names = self.live.open(relay, listings, token_for(relay), history_s=settings.live_history_s,
                                   replace=replace)
        except (RelayError, ValueError, KeyError) as exc:
            self.window.files.tell("Open stream", str(exc), problem=True)
            return []
        if replace:
            self.window.files.shown = None
        self._lay_out()
        self.window.set_dock_collapsed(self.live_dock, False)
        self.dock.show_streams(self.live)
        self.window.files._changed()
        return names

    def save_selection(self) -> list:
        """Every live stream's frames in the selection, as unsaved workspace recordings."""
        files = self.window.files
        selection = self.window.session.selection
        if not self.live.streams:
            files.tell("Save selection", "No live stream is being watched.", problem=True)
            return []
        if selection is None:
            files.tell("Save selection", "Select a stretch first: shift-drag on a chart.", problem=True)
            return []
        try:
            fragments = cut(self.live, *selection)
        except (ValueError, OSError) as exc:
            files.tell("Save selection", str(exc), problem=True)
            return []
        from ..io.files import read_file

        added = []
        for fragment in fragments:
            recording = files.workspace.add(fragment.path, series=read_file(fragment.path))
            recording.meta = fragment.meta
            recording.title = fragment.title
            recording.started_at_utc = fragment.started_at_utc
            recording.duration_s = fragment.duration_s
            added.append(recording)
        files._changed()
        self.window.statusBar().showMessage(
            f"{len(added)} recording{'s' if len(added) != 1 else ''} added to the workspace, not saved yet", 8000)
        return added

    def set_trim(self, name: str, seconds: float) -> None:
        if name in self.live.streams:
            self.live.set_trim(name, seconds)
            self.window._mark("data", "signals")

    def close_stream(self, name: str) -> None:
        if name not in self.live.streams:
            return
        self.live.close(name)
        self._lay_out()
        self.dock.show_streams(self.live)

    def leave(self) -> bool:
        """Stop watching, before the window shows a recording; True if it was watching."""
        if not self.live.streams:
            return False
        self.live.close_all()
        self.dock.show_streams(self.live)
        self.window.set_dock_collapsed(self.live_dock, True)
        return True

    def restore_panes(self) -> None:
        """The window's usual charts, after watching."""
        window = self.window
        for pane in list(window.panes):
            window.remove_pane(pane)
        window.focus_signal = window.session.channels[0]
        window._build_panes()

    def update(self) -> None:
        if not self.live.streams:
            return
        if self.live.align():
            self.window._mark("data", "signals")
        self.dock.refresh()

    # -- charts -----------------------------------------------------------------

    def _lay_out(self) -> None:
        """One amplitude chart per stream, then the spectrum and spectrogram of the first."""
        window = self.window
        session = window.session
        for pane in list(window.panes):
            window.remove_pane(pane)
        names = list(session.streams)
        window.focus_signal = session.channels[0]
        for name in names:
            window.add_pane("signal", name if len(names) > 1 else ALL_CHANNELS)
        window.add_pane("spectrum", window.focus_signal)
        window.add_pane("spectrogram", window.focus_signal)
        window.splitter.setSizes([3] * len(names) + [2, 2])

"""File -> Open stream, Receive streams and Save selection to workspace, and the Live streams dock.

Like the File menu's other actions, each takes what its dialog would ask as
arguments too, so a script or a test can run it without one.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QLabel

from ..settings import Settings
from ..streaming import LiveStreams, RelayError, cut
from .panes.base import ALL_CHANNELS
from .widgets.live_dock import LiveDock

__all__ = ["LiveControl", "address"]

#: How often the streams' alignment and the dock are brought up to date.
ALIGN_MS = 500


def address(text: str, default_port: int) -> tuple[str, int]:
    """HOST:PORT as a (host, port) to bind."""
    host, _, port = text.strip().rpartition(":")
    return (host or "0.0.0.0", int(port)) if port.isdigit() else (text.strip() or "0.0.0.0", default_port)


class LiveControl(QObject):
    #: A stream the receiver finished saving: its session folder (sent from the receiver's thread).
    received = Signal(object)

    def __init__(self, window) -> None:
        super().__init__(window)
        self.window = window
        self.live = LiveStreams(window.session)
        #: The mmp_stream.Receiver of File > Receive streams, while it runs.
        self.receiver = None
        self.received.connect(self._received)
        files = window.files
        self.open_action = QAction("Open s&tream...", window)
        self.open_action.setStatusTip("Watch live streams from a relay")
        self.open_action.triggered.connect(lambda _checked=False: self.open_streams())
        self.receive_action = QAction("&Receive streams", window)
        self.receive_action.setCheckable(True)
        self.receive_action.setStatusTip("Take node streams here, watch them, and keep each one whole in the workspace")
        self.receive_action.toggled.connect(self._receive_toggled)
        self.receiving_label = QLabel()
        window.statusBar().addPermanentWidget(self.receiving_label)
        self.save_action = QAction("Save s&election to workspace", window)
        self.save_action.setStatusTip("Keep the selected stretch of every live stream as workspace recordings")
        self.save_action.triggered.connect(lambda _checked=False: self.save_selection())
        actions = files.menu.actions()
        files.menu.insertAction(actions[actions.index(files.open_action) + 1], self.open_action)
        files.menu.insertAction(actions[actions.index(files.open_action) + 1], self.receive_action)
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

    def ask_streams(self, relay: str):
        """(relay, the relayed sessions ticked) or None."""
        from .stream_dialog import OpenStreamDialog

        dialog = OpenStreamDialog(relay, self.window)
        try:
            if dialog.exec() != OpenStreamDialog.Accepted or not dialog.chosen:
                return None
            return dialog.listed_relay, dialog.chosen
        finally:
            dialog.deleteLater()

    # -- actions --------------------------------------------------------------

    def open_streams(self, listings=None, relay: str | None = None, replace: bool = True) -> list[str]:
        """Watch relayed sessions (`listings`, as `GET /live` lists them): from this window's own
        relay while it receives, else the one in the settings. The first replaces what the
        window shows; with `replace=False` they join it."""
        settings = Settings.load()
        own = self.receiving_relay()
        relay = relay or own or settings.relay_address()
        if listings is None:
            asked = self.ask_streams(relay)
            if asked is None:
                return []
            chosen_relay, listings = asked
            if chosen_relay and chosen_relay != relay and chosen_relay != own:
                relay = settings.relay = chosen_relay
                try:
                    settings.save()
                except OSError:
                    pass
        replace = replace or not self.live.streams
        try:
            if replace:
                self.window.player.stop(quietly=True)
            names = self.live.open(relay, listings, history_s=settings.live_history_s, replace=replace)
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

    # -- receiving --------------------------------------------------------------

    def receiving_relay(self) -> str | None:
        """HOST:PORT of this window's own relay while it receives."""
        if self.receiver is None or self.receiver.relay_address is None:
            return None
        host, port = self.receiver.relay_address
        return f"{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"

    def _receive_toggled(self, on: bool) -> None:
        if on and self.receiver is None:
            self.start_receiving()
        elif not on and self.receiver is not None:
            self.stop_receiving()

    def start_receiving(self, bind: str | None = None, relay: str | None = None) -> bool:
        """Take node streams on `bind` and relay them on `relay` (by default the settings'); each
        finished stream is saved whole, in the workspace's received/ while it is in a folder."""
        if self.receiver is not None:
            return True
        import mmp_stream

        settings = Settings.load()
        workspace = self.window.files.workspace
        out = (workspace.folder / "received") if workspace.folder is not None else Path(
            tempfile.mkdtemp(prefix="mmp-sense-received-"))
        try:
            self.receiver = mmp_stream.Receiver(address(bind or settings.receive_bind, 8090),
                                                address(relay or settings.receive_relay, 8091), out,
                                                on_saved=self.received.emit, log=lambda _line: None).start()
        except (OSError, ValueError) as exc:
            self.receiver = None
            self.window.files.tell("Receive streams", f"Cannot receive: {exc}", problem=True)
            self._show_receiving()
            return False
        self._show_receiving()
        return True

    def stop_receiving(self) -> None:
        receiver, self.receiver = self.receiver, None
        if receiver is not None:
            receiver.stop()
        self._show_receiving()

    def _show_receiving(self) -> None:
        on = self.receiver is not None
        self.receive_action.blockSignals(True)
        self.receive_action.setChecked(on)
        self.receive_action.blockSignals(False)
        if on:
            host, port = self.receiver.address
            self.receiving_label.setText(f"Receiving streams on {host}:{port}, relayed on {self.receiving_relay()}")
        else:
            self.receiving_label.setText("")

    def _received(self, folder) -> None:
        """A stream the receiver saved: into the workspace, whole."""
        files = self.window.files
        try:
            recording = files.workspace.adopt(Path(folder))
        except (OSError, ValueError) as exc:
            files.tell("Receive streams", f"{Path(folder).name}: {exc}", problem=True)
            return
        files._changed()
        self.window.statusBar().showMessage(f"Received {recording.label}, kept in the workspace", 8000)

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

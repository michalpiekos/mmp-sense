"""The File menu: the window's workspace, its recordings, and sending them to srv-senselab.

Every action takes what a dialog would ask as arguments too, so a script
or a test can run it without one.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QFileDialog, QInputDialog, QLineEdit, QMessageBox

from ..settings import Settings
from ..workspace import Workspace, WorkspaceError
from ..workspace import sync as syncing
from ..workspace.loading import load_into, series_of
from ..io.session_export import SESSION_FILE
from .widgets import WorkspaceDock

__all__ = ["WorkspaceControl"]

log = logging.getLogger("mmp_sense.ui")


class WorkspaceControl(QObject):
    progressed = Signal(str)
    synchronized = Signal(object)

    def __init__(self, window, workspace: Workspace | None = None, shown=None) -> None:
        super().__init__(window)
        self.window = window
        self.workspace = workspace or Workspace.untitled()
        self.shown = shown
        self.syncing: threading.Thread | None = None
        self.last_results: list = []
        self.progressed.connect(self._progressed)
        self.synchronized.connect(self._synchronized)

        self.dock = WorkspaceDock()
        self.dock.chosen.connect(self._chosen)
        self.workspace_dock = window._make_dock("Workspace", self.dock, Qt.LeftDockWidgetArea)
        self._build_menu()
        self._changed()
        # Empty until a recording is opened, which opens it.
        if not self.workspace.recordings:
            window.set_dock_collapsed(self.workspace_dock, True)

    # -- the menu --------------------------------------------------------

    def _build_menu(self) -> None:
        menu = self.window.menuBar().addMenu("&File")

        def add(text, slot, keys=None, tip=""):
            action = QAction(text, self.window)
            if keys:
                action.setShortcut(QKeySequence(keys))
            action.setToolTip(tip)
            action.setStatusTip(tip)
            action.triggered.connect(lambda _checked=False: slot())
            menu.addAction(action)
            return action

        self.new_action = add("&New", lambda: None, tip="A new recording, streamed into; "
                              "comes with direct streaming")
        self.new_action.setEnabled(False)
        self.open_action = add("&Open...", self.open_file, QKeySequence.Open,
                               "Add a file or a srv-senselab export to the workspace and show it")
        self.open_workspace_action = add("Open &workspace...", self.open_workspace,
                                         tip="Work on a workspace folder")
        self.create_workspace_action = add("&Create workspace...", self.create_workspace,
                                           tip="Keep the workspace in a new folder")
        menu.addSeparator()
        self.save_action = add("&Save", self.save, QKeySequence.Save,
                               "Copy the shown recording into the workspace")
        self.save_workspace_action = add("Save wor&kspace", self.save_workspace,
                                         tip="Write the workspace and its recordings' metadata")
        self.save_workspace_as_action = add("Save workspace &as...", self.save_workspace_as,
                                            tip="Copy all of the workspace to another folder, "
                                                "as a new workspace, and work there")
        self.save_all_action = add("Save a&ll", self.save_all, "Ctrl+Shift+S",
                                   "Save every recording and the workspace")
        menu.addSeparator()
        self.sync_action = add("S&ynchronize", self.synchronize,
                               tip="Send the saved recordings and their metadata to srv-senselab")
        self.menu = menu

        edit = self.window.menuBar().addMenu("&Edit")
        self.settings_action = QAction("&Settings...", self.window)
        self.settings_action.setStatusTip("The srv-senselab Synchronize sends to, and its token")
        self.settings_action.triggered.connect(lambda _checked=False: self.edit_settings())
        edit.addAction(self.settings_action)
        self.edit_menu = edit

    # -- what a dialog would ask; replaced in tests --------------------------

    def ask_file(self) -> str | None:
        path, _ = QFileDialog.getOpenFileName(
            self.window, "Open a recording", "",
            "Recordings (*.mcap *.wav *.m4a *.aac *.mp3 *.ogg *.opus *.flac *.webm *.3gp *.amr "
            f"*.csv *.tsv *.txt *.npy *.npz {SESSION_FILE});;All files (*)")
        return path or None

    def ask_folder(self, caption: str) -> str | None:
        return QFileDialog.getExistingDirectory(self.window, caption) or None

    def ask_text(self, caption: str, label: str, text: str = "", secret: bool = False) -> str | None:
        mode = QLineEdit.Password if secret else QLineEdit.Normal
        value, ok = QInputDialog.getText(self.window, caption, label, mode, text)
        return value.strip() if ok and value.strip() else None

    def ask_settings(self, settings: Settings) -> bool:
        from .settings_dialog import SettingsDialog

        return SettingsDialog(settings, self.window).exec() == SettingsDialog.Accepted

    def tell(self, caption: str, text: str, problem: bool = False) -> None:
        (QMessageBox.warning if problem else QMessageBox.information)(self.window, caption, text)

    # -- actions -------------------------------------------------------------

    def open_file(self, path=None) -> None:
        path = path or self.ask_file()
        if not path:
            return
        path = Path(path)
        if path.name == SESSION_FILE:
            path = path.parent
        from ..io.files import read_file

        try:
            series = read_file(path)
            recording = self.workspace.add(path, series=series)
        except Exception as exc:                              # noqa: BLE001
            self.tell("Open", f"{path.name}: {exc}", problem=True)
            return
        self._show(recording, series)

    def open_workspace(self, folder=None) -> None:
        folder = folder or self.ask_folder("Open a workspace")
        if not folder:
            return
        try:
            workspace = Workspace.open(folder)
        except WorkspaceError as exc:
            self.tell("Open workspace", str(exc), problem=True)
            return
        self.workspace = workspace
        self.shown = None
        if workspace.recordings:
            self._show(workspace.recordings[0])
        self._changed()

    def create_workspace(self, folder=None, title=None) -> bool:
        """Put an untitled workspace, with what it holds, in a folder; or
        start a new, empty one there."""
        folder = folder or self.ask_folder("Create a workspace in an empty folder")
        if not folder:
            return False
        if title is None:
            title = self.ask_text("Create workspace", "Title", Path(folder).name)
            if title is None:
                return False
        try:
            if self.workspace.folder is None:
                self.workspace.settle(folder, title)
            else:
                self.workspace = Workspace.create(folder, title)
                self.shown = None
        except (WorkspaceError, OSError) as exc:
            self.tell("Create workspace", str(exc), problem=True)
            return False
        self._changed()
        return True

    def edit_settings(self) -> None:
        settings = Settings.load()
        if not self.ask_settings(settings):
            return
        try:
            settings.save()
        except OSError as exc:
            self.tell("Settings", str(exc), problem=True)

    def save(self) -> None:
        if self.shown is None:
            self.window.statusBar().showMessage("No recording of the workspace is shown", 5000)
            return
        self._saving(lambda: self.workspace.save_recording(self.shown))

    def save_workspace(self) -> None:
        self._saving(self.workspace.save)

    def save_all(self) -> None:
        self._saving(self.workspace.save_all)

    def save_workspace_as(self, folder=None) -> None:
        folder = folder or self.ask_folder("Save the workspace as, in an empty folder")
        if not folder:
            return
        self._saving(lambda: self.workspace.save_as(folder), needs_folder=False)

    def _saving(self, step, needs_folder: bool = True) -> None:
        if needs_folder and self.workspace.folder is None and not self.create_workspace():
            return
        try:
            step()
        except (WorkspaceError, OSError) as exc:
            self.tell("Save", str(exc), problem=True)
        self._changed()

    def synchronize(self, server=None, token=None, wait: bool = False) -> None:
        """Send the workspace to srv-senselab, in the background unless `wait`, then here."""
        if self.syncing is not None:
            return
        workspace = self.workspace
        if workspace.folder is None:
            self.tell("Synchronize", "Create the workspace and save its recordings first.",
                      problem=True)
            return
        settings = Settings.load()
        if not server:
            if not settings.server:
                self.edit_settings()
                settings = Settings.load()
            server = settings.server
        if not server:
            return
        server = server.rstrip("/")
        sent = any(r.sync.declared for r in workspace.recordings)
        if sent and workspace.server and workspace.server != server:
            self.tell("Synchronize", f"This workspace was sent to {workspace.server}, and the "
                      f"settings name {server}. Set that address back in Edit > Settings, or "
                      "use File > Save workspace as for a copy to send to the new one.",
                      problem=True)
            return
        from_file = not token and bool(settings.token_file)
        if from_file:
            try:
                token = settings.read_token()
            except (OSError, ValueError) as exc:
                self.tell("Synchronize", f"The token file in Edit > Settings: {exc}", problem=True)
                return
        token = token or syncing.load_token(server) or self.ask_text(
            "Synchronize", f"Source token for {server}\n(sample source create --type import)",
            secret=True)
        if not token:
            return
        try:
            if not from_file:
                syncing.store_token(server, token)
            workspace.server = server
            workspace.save()
        except OSError as exc:
            self.tell("Synchronize", str(exc), problem=True)
            return
        client = syncing.Client(server, token)
        self._sending_to = server
        self._token_from_file = from_file

        def run():
            try:
                results = syncing.synchronize(
                    workspace, client,
                    lambda recording, text: self.progressed.emit(f"{recording.label}: {text}"))
            except Exception as exc:                          # noqa: BLE001
                log.exception("synchronize failed")
                results = exc
            self.synchronized.emit(results)

        self._busy(True)
        if wait:                  # here, so the signals reach their slots directly
            self.syncing = threading.current_thread()
            run()
            return
        self.syncing = threading.Thread(target=run, name="mmp-sense-sync", daemon=True)
        self.syncing.start()

    def _progressed(self, text: str) -> None:
        self.window.statusBar().showMessage(text, 5000)

    def _synchronized(self, results) -> None:
        self.syncing = None
        self._busy(False)
        self._changed()
        if isinstance(results, Exception):
            self.last_results = []
            self.tell("Synchronize", str(results), problem=True)
            return
        self.last_results = results
        lines = [f"{r.recording.label}: {r.outcome}" + (f" -- {r.message}" if r.message else "")
                 for r in results] or ["The workspace has no recordings."]
        if any(r.status == 401 for r in results):
            if self._token_from_file:
                lines.append("The token in the token file of Edit > Settings was refused.")
            else:
                syncing.forget_token(self._sending_to)
        failed = any(r.outcome == "failed" for r in results)
        self.tell("Synchronize", "\n".join(lines), problem=failed)

    def _busy(self, on: bool) -> None:
        """While a Synchronize runs, the workspace it sends stays as it is."""
        self.dock.setEnabled(not on)
        for action in [*self.menu.actions(), self.settings_action]:
            if action is not self.new_action:
                action.setEnabled(not on)

    # -- showing -----------------------------------------------------------

    def _chosen(self, recording_id: str) -> None:
        recording = self.workspace.get(recording_id)
        if recording is not self.shown:
            self._show(recording)

    def _show(self, recording, series=None) -> None:
        try:
            if series is None:
                series = series_of(self.workspace, recording)
            self.window.player.stop(quietly=True)
            watching = self.window.live.leave()
            load_into(self.window.session, series)
        except Exception as exc:                              # noqa: BLE001
            self.tell("Open", f"{recording.label}: {exc}", problem=True)
            return
        if watching:
            self.window.live.restore_panes()
        self.shown = recording
        self.window.focus_signal = self.window.session.channels[0]
        self.window._mark("source", "signals", "data", "cursor", "selection")
        self._changed()

    def _changed(self) -> None:
        self.dock.show_workspace(self.workspace, self.shown)
        if self.workspace.recordings and self.workspace_dock.collapsed:
            self.window.set_dock_collapsed(self.workspace_dock, False)
        title = self.workspace.title or "untitled"
        self.window.setWindowTitle(f"mmp-sense -- {title} -- {self.window.session.dataset}")

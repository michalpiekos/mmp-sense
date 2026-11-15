"""The workspace's recordings, and the metadata of the one chosen."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

__all__ = ["WorkspaceDock"]


class WorkspaceDock(QWidget):
    """A recording is chosen here to be shown; its title, tags, notes and
    start are edited here. The start is fixed once the server has it."""

    chosen = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.workspace = None
        self._filling = False

        self.header = QLabel()
        self.header.setWordWrap(True)
        self.list = QListWidget()
        self.list.setMinimumHeight(40)
        self.list.currentItemChanged.connect(self._current_changed)

        self.title = QLineEdit()
        self.tags = QLineEdit()
        self.tags.setPlaceholderText("kind:knock, env:door")
        self.started = QLineEdit()
        self.started.setToolTip("When the recording started, in UTC (RFC 3339)")
        self.notes = QPlainTextEdit()
        self.notes.setMinimumHeight(36)
        self.notes.setMaximumHeight(90)
        self.state = QLabel()
        self.state.setWordWrap(True)
        self.title.textEdited.connect(lambda text: self._edit("title", text))
        self.tags.textEdited.connect(lambda text: self._edit("tags", _tags(text)))
        self.started.textEdited.connect(lambda text: self._edit("started_at_utc", text.strip()))
        self.notes.textChanged.connect(lambda: self._edit("notes", self.notes.toPlainText()))

        form = QFormLayout()
        form.addRow("Title", self.title)
        form.addRow("Tags", self.tags)
        form.addRow("Started", self.started)
        form.addRow("Notes", self.notes)
        self.form = QWidget()
        self.form.setLayout(form)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.header)
        layout.addWidget(self.list, 1)
        layout.addWidget(self.form)
        layout.addWidget(self.state)

    def show_workspace(self, workspace, shown=None) -> None:
        self.workspace = workspace
        self.refresh(shown)

    def refresh(self, shown=None) -> None:
        workspace = self.workspace
        where = str(workspace.folder) if workspace.folder else "not saved yet"
        self.header.setText(f"<b>{workspace.title or 'Untitled'}</b><br>{where}")
        self._filling = True
        try:
            self.list.clear()
            for recording in workspace.recordings:
                item = QListWidgetItem(_line(recording))
                item.setData(Qt.UserRole, recording.id)
                self.list.addItem(item)
                if shown is not None and recording.id == shown.id:
                    self.list.setCurrentItem(item)
        finally:
            self._filling = False
        self._fill(shown)

    def relabel(self) -> None:
        for row in range(self.list.count()):
            item = self.list.item(row)
            item.setText(_line(self.workspace.get(item.data(Qt.UserRole))))
        self._fill(self.current())

    def current(self):
        item = self.list.currentItem()
        return self.workspace.get(item.data(Qt.UserRole)) if item is not None else None

    def _current_changed(self, item, _previous) -> None:
        if self._filling or item is None:
            return
        recording = self.workspace.get(item.data(Qt.UserRole))
        self._fill(recording)
        self.chosen.emit(recording.id)

    def _fill(self, recording) -> None:
        self._filling = True
        try:
            self.form.setEnabled(recording is not None)
            self.title.setText(recording.title if recording else "")
            self.tags.setText(", ".join(recording.tags) if recording else "")
            self.started.setText(recording.started_at_utc if recording else "")
            self.notes.setPlainText(recording.notes if recording else "")
            fixed = recording is not None and (recording.origin == "server" or recording.sync.declared)
            self.started.setReadOnly(fixed)
            self.state.setText(_state(recording) if recording else "")
        finally:
            self._filling = False

    def _edit(self, field: str, value) -> None:
        recording = self.current()
        if self._filling or recording is None:
            return
        setattr(recording, field, value)
        self.list.currentItem().setText(_line(recording))


def _tags(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def _line(recording) -> str:
    marks = []
    if not recording.saved:
        marks.append("unsaved")
    if recording.origin == "server":
        marks.append("from the server")
    elif recording.sync.closed:
        marks.append("on the server")
    return recording.label + (f"  ({', '.join(marks)})" if marks else "")


def _state(recording) -> str:
    if recording.origin == "server":
        return f"From srv-senselab session {recording.server_session_id}; not sent again."
    sync = recording.sync
    if sync.closed:
        changed = recording.context() != sync.context_sent
        return ("On the server" + ("; changed metadata to send" if changed else "")
                + f" as session {sync.capture_session_id}.")
    if sync.declared:
        return "Partly sent; Synchronize again to finish."
    return "Not on the server yet."

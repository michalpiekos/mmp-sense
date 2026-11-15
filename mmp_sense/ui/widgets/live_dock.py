"""The live streams being watched: how each is doing and how it is timed."""

from __future__ import annotations

import datetime

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

__all__ = ["LiveDock"]


def _utc(seconds: float) -> str:
    return datetime.datetime.fromtimestamp(seconds, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def timing(live) -> str:
    """Where a stream's place in time comes from, and how sure it is."""
    utc = live.relayed.utc_offset()
    if utc is None:
        return "waiting for UTC"
    where = "gateway arrival" if utc.source == "gateway" else f"node {utc.source}"
    if utc.uncertainty_s is None:
        return f"{where} (UTC timeline)"
    return f"{where} ±{utc.uncertainty_s * 1e3:.0f} ms"


class LiveDock(QWidget):
    """A row per stream; the trim and Close act on the chosen one."""

    trimmed = Signal(str, float)          # stream, seconds
    close_requested = Signal(str)

    COLUMNS = ["Stream", "State", "Lost", "Timing", "Trim"]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.live = None
        self.epoch = QLabel()
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(self.COLUMNS)
        self.tree.setRootIsDecorated(False)
        self.tree.setMinimumHeight(40)
        self.tree.currentItemChanged.connect(lambda *_: self._fill_trim())
        self.trim = QDoubleSpinBox()
        self.trim.setRange(-60_000.0, 60_000.0)
        self.trim.setDecimals(1)
        self.trim.setSuffix(" ms")
        self.trim.setKeyboardTracking(False)            # applied once typed, not per keystroke
        self.trim.setToolTip("Move the chosen stream later (or earlier, negative) on the time axis")
        self.trim.valueChanged.connect(self._trim_changed)
        close = QPushButton("Close")
        close.setToolTip("Stop watching the chosen stream")
        close.clicked.connect(lambda: self.current() and self.close_requested.emit(self.current()))
        row = QHBoxLayout()
        row.addWidget(QLabel("Trim"))
        row.addWidget(self.trim, 1)
        row.addWidget(close)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.epoch)
        layout.addWidget(self.tree, 1)
        layout.addLayout(row)
        self._filling = False

    def current(self) -> str | None:
        item = self.tree.currentItem()
        return item.data(0, Qt.UserRole) if item is not None else None

    def show_streams(self, live) -> None:
        """Rows for `live`'s streams, kept in place when they are the same ones."""
        self.live = live
        names = list(live.streams)
        rows = [self.tree.topLevelItem(i).data(0, Qt.UserRole) for i in range(self.tree.topLevelItemCount())]
        if rows != names:
            chosen = self.current()
            self.tree.clear()
            for name in names:
                item = QTreeWidgetItem([name])
                item.setData(0, Qt.UserRole, name)
                self.tree.addTopLevelItem(item)
                if name == chosen:
                    self.tree.setCurrentItem(item)
            if self.tree.currentItem() is None and names:
                self.tree.setCurrentItem(self.tree.topLevelItem(0))
        self.refresh()

    def refresh(self) -> None:
        live = self.live
        if live is None:
            return
        self.epoch.setText(f"Time 0 is {_utc(live.epoch)}" if live.epoch is not None else "Waiting for UTC")
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            entry = live.streams.get(item.data(0, Qt.UserRole))
            if entry is None:
                continue
            relayed = entry.relayed
            state = relayed.state if relayed.state != "ended" else f"ended: {relayed.end}"
            values = [state, str(entry.source.pcm.lost_frames), timing(entry), f"{entry.trim * 1e3:+.1f} ms"]
            for column, text in enumerate(values, start=1):
                if item.text(column) != text:
                    item.setText(column, text)
        self._fill_trim()

    def _fill_trim(self) -> None:
        name = self.current()
        entry = self.live.streams.get(name) if self.live is not None and name else None
        self._filling = True
        try:
            self.trim.setEnabled(entry is not None)
            if entry is not None and not self.trim.hasFocus():
                self.trim.setValue(entry.trim * 1e3)
        finally:
            self._filling = False

    def _trim_changed(self, ms: float) -> None:
        name = self.current()
        if not self._filling and name:
            self.trimmed.emit(name, ms / 1e3)

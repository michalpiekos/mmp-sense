"""The signals of the session, and the ops that can be added to it."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...core.graph.ops import menu_ops

__all__ = ["SignalDock"]


class _Menu(QComboBox):
    def __init__(self, before_popup) -> None:
        super().__init__()
        self._before_popup = before_popup
        # Its longest entry would otherwise set the width of the whole column.
        self.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(10)

    def showPopup(self) -> None:
        self._before_popup()
        super().showPopup()


class SignalDock(QWidget):
    """What exists, and how to get one on the screen.

    Double-click a signal to plot it. A feature chosen here becomes a
    signal like any other -- the same objects the console makes.
    """

    plot_requested = Signal(str)
    drop_requested = Signal(str)
    op_requested = Signal(str)

    def __init__(self, session, parent=None) -> None:
        super().__init__(parent)
        self.session = session

        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Signal", "Kind", "Units"])
        self.tree.setRootIsDecorated(False)
        self.tree.itemDoubleClicked.connect(self._double_clicked)

        self.features = _Menu(self._fill_features)
        self._fill_features()
        add = QPushButton("Add feature")
        add.clicked.connect(self._add_clicked)
        drop = QPushButton("Drop")
        drop.clicked.connect(self._drop_selected)

        row = QHBoxLayout()
        row.addWidget(self.features, 1)
        row.addWidget(add)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(QLabel("Double-click a signal to plot it"))
        layout.addWidget(self.tree, 1)
        layout.addLayout(row)
        layout.addWidget(drop)
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the list -- only when the signals have changed."""
        selected = self.selected()
        self.tree.clear()
        signals = self.session.signals
        several = len(signals.streams) > 1
        parents = {}
        if several:
            # A stream's item stands for its channels: double-click plots them all.
            for stream in signals.streams.values():
                item = QTreeWidgetItem([stream.name, "stream", stream.buffer.units or ""])
                item.setToolTip(0, getattr(stream.source, "name", "") or stream.name)
                self.tree.addTopLevelItem(item)
                item.setExpanded(True)
                parents[stream.name] = item
        for name in signals.names:
            signal = signals.get(name)
            item = QTreeWidgetItem([name, signal.kind, signal.units or ""])
            item.setToolTip(0, signal.description or signal.source or name)
            if several:
                parents[signals.stream_of(name)].addChild(item)
            else:
                self.tree.addTopLevelItem(item)
            if name == selected:
                self.tree.setCurrentItem(item)
        for column in range(3):
            self.tree.resizeColumnToContents(column)

    def _fill_features(self) -> None:
        """The ops offered to the menu -- refilled as it opens, so one
        registered from the console is there."""
        current = self.features.currentText()
        self.features.clear()
        for name, spec in menu_ops().items():
            # The ellipsis: a form asks for its parameters first.
            self.features.addItem(f"{name}\u2026", ("op", name))
            self.features.setItemData(self.features.count() - 1, spec.description,
                                      Qt.ToolTipRole)
        if current:
            self.features.setCurrentText(current)

    def _add_clicked(self) -> None:
        chosen = self.features.currentData()
        if isinstance(chosen, tuple) and chosen[0] == "op":
            self.op_requested.emit(chosen[1])

    def selected(self) -> str | None:
        item = self.tree.currentItem()
        return item.text(0) if item is not None else None

    def _double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        self.plot_requested.emit(item.text(0))

    def _drop_selected(self) -> None:
        name = self.selected()
        if name:
            self.drop_requested.emit(name)

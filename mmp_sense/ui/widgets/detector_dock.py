"""The graph nodes the session shows, and what each has found."""

from __future__ import annotations

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QLabel,
    QPlainTextEdit,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from ...core.graph.series import format_seconds

__all__ = ["DetectorDock"]

#: The columns; the box comes first so a narrow dock still shows it.
SHOW, NODE, OUTPUT, FOUND, LATENCY = range(5)


class DetectorDock(QWidget):
    """Every shown node with its count and latency, and the selected
    one's graph. Nodes are made and changed in the console; the dock only
    draws an event node's marks or not."""

    def __init__(self, session, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self._listing: list = []
        self._syncing = False

        self.tree = QTreeWidget()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(["Show", "Node", "Output", "Found", "Latency"])
        self.tree.setRootIsDecorated(False)
        self.tree.currentItemChanged.connect(lambda *_: self._show_graph())
        self.tree.itemChanged.connect(self._hide_toggled)

        self.graph = QPlainTextEdit()
        self.graph.setReadOnly(True)
        self.graph.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.graph.setLineWrapMode(QPlainTextEdit.NoWrap)

        split = QSplitter(Qt.Vertical)
        split.addWidget(self.tree)
        split.addWidget(self.graph)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(QLabel("session.show(node) from the console"))
        layout.addWidget(split, 1)
        self.refresh()

    def refresh(self, window=None) -> None:
        """Relist the shown nodes; count their events over `window`."""
        selected = self.selected()
        shown = self.session.shown
        listing = [(name, node.key) for name, node in shown.items()]
        self._syncing = True
        try:
            if listing != self._listing:
                self._listing = listing
                self.tree.clear()
                for name, node in shown.items():
                    item = QTreeWidgetItem(
                        ["", name, node.output, "", format_seconds(node.latency)])
                    item.setToolTip(NODE, node.label)
                    if node.output == "events":
                        # A box never given a state is not drawn at all.
                        item.setCheckState(SHOW, Qt.Checked)
                        item.setToolTip(SHOW, "Draw its marks; unticked it is still computed")
                    self.tree.addTopLevelItem(item)
                    if name == selected:
                        self.tree.setCurrentItem(item)
            for i in range(self.tree.topLevelItemCount()):
                item = self.tree.topLevelItem(i)
                name = item.text(NODE)
                node = shown[name]
                item.setText(FOUND, self._found(name, node, window))
                if node.output == "events":
                    drawn = Qt.Checked if self.session.drawn(name) else Qt.Unchecked
                    if item.checkState(SHOW) != drawn:
                        item.setCheckState(SHOW, drawn)
        finally:
            self._syncing = False
        for column in range(5):
            self.tree.resizeColumnToContents(column)
        self._show_graph()

    def _hide_toggled(self, item, column) -> None:
        if self._syncing or column != SHOW:
            return
        name = item.text(NODE)
        try:
            self.session.set_drawn(name, item.checkState(SHOW) == Qt.Checked)
        except KeyError:
            pass                  # gone from the session since it was listed

    def shown(self, name: str) -> bool | None:
        """Whether the box on `name`'s row is ticked; None if it has none."""
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item.text(NODE) == name:
                if item.data(SHOW, Qt.CheckStateRole) is None:
                    return None
                return item.checkState(SHOW) == Qt.Checked
        return None

    def _found(self, name, node, window) -> str:
        if node.output != "events":
            return "-"
        if window is None:
            return ""
        try:
            return str(len(self.session.evaluator.evaluate(node, window)))
        except Exception as exc:                              # noqa: BLE001
            return f"error: {exc}"[:60]

    def selected(self) -> str | None:
        item = self.tree.currentItem()
        return item.text(NODE) if item is not None else None

    def _show_graph(self) -> None:
        name = self.selected()
        node = self.session.shown.get(name) if name else None
        text = node.graph() if node is not None else ""
        if self.graph.toPlainText() != text:
            self.graph.setPlainText(text)

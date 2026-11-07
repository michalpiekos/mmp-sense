"""The session's frequency bands, and the form that adds or changes one."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...core.bands import check_band

__all__ = ["BandDock", "BandDialog"]

NAME, LOW, HIGH = range(3)
#: The highest edge the form takes; far above any sound card.
MAX_HZ = 1e7


class BandDialog(QDialog):
    """Low and high in Hz, typed; OK adds the band, or changes `band`.

    `nyquist` is the focus signal's (name, highest frequency it holds),
    said in the form so an edge past it is no surprise.
    """

    def __init__(self, session, band=None, nyquist: tuple[str, float] | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self.band = band
        self.setWindowTitle("Edit band" if band else "Add band")

        self.name = QLineEdit(band.name if band else "")
        self.name.setPlaceholderText("band N")
        self.low = self._spin(band.low_hz if band else 0.0)
        self.high = self._spin(band.high_hz if band else min(1000.0, _top(nyquist)))
        self.limit = QLabel("")
        if nyquist:
            self.limit.setText(f"{nyquist[0]} holds up to {nyquist[1]:.6g} Hz")
        self.problem = QLabel("")
        self.problem.setStyleSheet("color: #c03030;")
        self.problem.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Name", self.name)
        form.addRow("Low", self.low)
        form.addRow("High", self.high)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.limit)
        layout.addWidget(self.problem)
        layout.addWidget(self.buttons)
        for spin in (self.low, self.high):
            spin.valueChanged.connect(lambda _v: self._check())
        self._check()

    @staticmethod
    def _spin(value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, MAX_HZ)
        spin.setDecimals(3)
        spin.setSuffix(" Hz")
        spin.setSingleStep(10.0)
        spin.setValue(float(value))
        return spin

    def _check(self) -> bool:
        try:
            check_band(self.low.value(), self.high.value())
        except ValueError as exc:
            self.problem.setText(str(exc))
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
            return False
        self.problem.setText("")
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(True)
        return True

    def accept(self) -> None:
        if not self._check():
            return
        name = self.name.text().strip() or None
        try:
            if self.band is None:
                self.session.add_band(self.low.value(), self.high.value(), name)
            else:
                self.session.edit_band(self.band.name, low_hz=self.low.value(),
                                       high_hz=self.high.value(), rename=name)
        except (ValueError, KeyError) as exc:
            self.problem.setText(str(exc).strip("'\""))
            return
        super().accept()


def _top(nyquist) -> float:
    return float(nyquist[1]) if nyquist and nyquist[1] > 0 else MAX_HZ


class BandDock(QWidget):
    """Every band with its edges; the ticked ones are what Play sounds.

    `nyquist` gives the focus signal's name and highest frequency, for
    the form to show.
    """

    def __init__(self, session, nyquist: Callable[[], tuple[str, float] | None] = lambda: None,
                 parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self.nyquist = nyquist
        self.dialog: BandDialog | None = None
        self._listing: list = []
        self._syncing = False

        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Band", "Low Hz", "High Hz"])
        self.tree.setRootIsDecorated(False)
        self.tree.itemChanged.connect(self._toggled)
        self.tree.itemDoubleClicked.connect(lambda item, _c: self.edit(item.text(NAME)))

        rows = []
        # Two rows: four buttons in one would widen the whole left column.
        for pair in ((("Add…", self.add), ("Edit…", self.edit)),
                     (("Remove", self.remove), ("Clear", session.clear_bands))):
            row = QHBoxLayout()
            for text, slot in pair:
                button = QPushButton(text)
                button.clicked.connect(lambda _c=False, s=slot: s())
                row.addWidget(button)
            rows.append(row)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        hint = QLabel("Play sounds the ticked bands; none ticked, all of it")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addWidget(self.tree, 1)
        for row in rows:
            layout.addLayout(row)
        self.refresh()

    def refresh(self) -> None:
        bands = self.session.bands
        if bands == self._listing:
            return
        selected = self.selected()
        self._listing = bands
        self._syncing = True
        try:
            self.tree.clear()
            for band in bands:
                item = QTreeWidgetItem([band.name, f"{band.low_hz:.6g}", f"{band.high_hz:.6g}"])
                item.setCheckState(NAME, Qt.Checked if band.on else Qt.Unchecked)
                item.setToolTip(NAME, "Ticked: Listen plays it. Double-click to edit")
                self.tree.addTopLevelItem(item)
                if band.name == selected:
                    self.tree.setCurrentItem(item)
        finally:
            self._syncing = False
        for column in range(3):
            self.tree.resizeColumnToContents(column)

    def selected(self) -> str | None:
        item = self.tree.currentItem()
        return item.text(NAME) if item is not None else None

    def _toggled(self, item, column) -> None:
        if self._syncing or column != NAME:
            return
        try:
            self.session.set_band_on(item.text(NAME), item.checkState(NAME) == Qt.Checked)
        except KeyError:
            pass                  # gone from the session since it was listed

    def add(self) -> BandDialog:
        """Open the form for a new band; returns it."""
        return self._open(None)

    def edit(self, name: str | None = None) -> BandDialog | None:
        """Open the form on `name`, else the selected band; returns it."""
        name = name or self.selected()
        try:
            band = self.session.band(name) if name else None
        except KeyError:
            band = None
        return self._open(band) if band is not None else None

    def _open(self, band) -> BandDialog:
        self.dialog = BandDialog(self.session, band, self.nyquist(), self)
        self.dialog.open()
        return self.dialog

    def remove(self) -> None:
        name = self.selected()
        if name:
            try:
                self.session.remove_band(name)
            except KeyError:
                pass

"""What every pane is.

A pane draws signals from the session and computes nothing of its own:
whatever it needs, it asks the session for, which is what keeps the same
answer on the screen and in the console.

Panes are told what changed -- `refresh({"data", "cursor"})` -- and redraw
only that. Nothing is rebuilt that has not changed, and curves are given
new data rather than recreated, because at a thousand samples a second
the cost of a redraw is the cost of the application.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

__all__ = ["Pane", "COLOURS", "SIDE_WIDTH"]


class PaneHeader(QWidget):
    """The row above a plot; a double-click on it, off its controls, is reported."""

    double_clicked = Signal()

    def mouseDoubleClickEvent(self, event) -> None:
        child = self.childAt(event.position().toPoint())
        if child is None or isinstance(child, QLabel):
            self.double_clicked.emit()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)


#: One colour per curve, from pyqtgraph's palette.
COLOURS = ["#4e9a06", "#3465a4", "#cc0000", "#f57900", "#75507b", "#06989a"]

#: The collapse button's face, open and collapsed.
EXPANDED_MARK = "\u25be"
COLLAPSED_MARK = "\u25b8"
QWIDGETSIZE_MAX = (1 << 24) - 1

#: The width of the column beside a time chart, where its distribution goes.
SIDE_WIDTH = 110

#: The entry that means "every channel the source produces".
ALL_CHANNELS = "channels"


class Pane(QWidget):
    """A plot of one signal, with a chooser and a close button."""

    kind = "pane"
    #: Whether the pane's x axis is time and should be linked to the others.
    time_axis = True

    signal_changed = Signal(str)
    closed = Signal(object)
    expand_toggled = Signal(object)
    collapse_toggled = Signal(object)

    def __init__(self, session, signal: str = ALL_CHANNELS, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self.signal = signal
        self._updating = False
        self.collapsed = False

        self.chooser = QComboBox()
        self.chooser.setMinimumWidth(160)
        self.chooser.currentTextChanged.connect(self._chosen)

        close = QToolButton()
        close.setText("x")
        close.setToolTip("Close this pane")
        close.clicked.connect(lambda: self.closed.emit(self))

        self.collapse_button = QToolButton()
        self.collapse_button.setText(EXPANDED_MARK)
        self.collapse_button.setToolTip("Collapse this pane to its title")
        self.collapse_button.clicked.connect(lambda: self.set_collapsed(not self.collapsed))

        self.status = QLabel("")
        self.status.setStyleSheet("color: #888888;")

        self.header = PaneHeader()
        self.header.double_clicked.connect(lambda: self.expand_toggled.emit(self))
        header = QHBoxLayout(self.header)
        header.setContentsMargins(6, 2, 6, 2)
        header.addWidget(QLabel(self.kind))
        header.addWidget(self.chooser)
        header.addWidget(self.status, 1)
        header.addWidget(self.collapse_button)
        header.addWidget(close)

        self.plot = pg.PlotWidget(viewBox=self.make_viewbox())
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.side = self.make_side()
        if self.side is not None:
            self.side.setFixedWidth(SIDE_WIDTH)
            self.side.hide()

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self.plot)
        if self.side is not None:
            body.addWidget(self.side)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.header)
        layout.addLayout(body)

        self.update_choices()

    # -- collapsing ----------------------------------------------------

    def set_collapsed(self, collapsed: bool) -> None:
        """Shrink to the header, or open again; the window lays out the rest.

        The plot goes to zero height rather than hidden: a hidden plot has
        no place on screen, and the time axes are linked by pixel.
        """
        collapsed = bool(collapsed)
        if collapsed == self.collapsed:
            return
        self.collapsed = collapsed
        drawn = [w for w in (self.plot, self.side) if w is not None]
        if collapsed:
            for widget in drawn:
                widget.setMinimumHeight(0)
                widget.setMaximumHeight(0)
            self.setMaximumHeight(self.header.sizeHint().height())
        else:
            for widget in drawn:
                widget.setMaximumHeight(QWIDGETSIZE_MAX)
            self.setMaximumHeight(QWIDGETSIZE_MAX)
        self.collapse_button.setText(COLLAPSED_MARK if collapsed else EXPANDED_MARK)
        self.collapse_button.setToolTip(
            "Open this pane" if collapsed else "Collapse this pane to its title")
        self.collapse_toggled.emit(self)

    # -- the signal it draws -------------------------------------------

    def make_viewbox(self):
        """The view box to plot in; pyqtgraph's own unless overridden."""
        return None

    def make_side(self):
        """The column beside the plot: on a time chart an empty one, so its
        time axis stays as wide as those of the charts that fill theirs."""
        return QWidget() if self.time_axis else None

    @property
    def distribution_shown(self) -> bool:
        return self.side is not None and not self.side.isHidden()

    def set_distribution_shown(self, shown: bool) -> None:
        """Show the column beside the plot, or take it away."""
        if self.side is not None:
            self.side.setVisible(bool(shown))

    def choices(self) -> list[str]:
        """What this pane can be pointed at."""
        groups = [g for g in self.session.groups if g != ALL_CHANNELS]
        return [ALL_CHANNELS, *groups, *self.session.signals.names]

    def update_choices(self) -> None:
        """Refill the chooser, keeping what is selected."""
        wanted = self.signal
        choices = self.choices()
        current = [self.chooser.itemText(i) for i in range(self.chooser.count())]
        if choices == current:
            return
        self._updating = True
        try:
            self.chooser.clear()
            self.chooser.addItems(choices)
            if wanted in choices:
                self.chooser.setCurrentText(wanted)
            elif choices:
                self.signal = choices[0]
                self.chooser.setCurrentText(self.signal)
        finally:
            self._updating = False

    def _chosen(self, name: str) -> None:
        if self._updating or not name or name == self.signal:
            return
        self.signal = name
        self.signal_changed.emit(name)
        # With the data now, not at the next sample: stopped data has none coming.
        self.refresh({"signals", "data", "cursor", "selection"}, self.session.window())

    def signal_names(self) -> list[str]:
        """The signals to draw: a group's members, or the chosen one."""
        members = self.session.members(self.signal)
        if members is not None:
            return members
        return [self.signal] if self.signal in self.session.signals else []

    # -- drawing -------------------------------------------------------

    def refresh(self, reasons: set[str], window=None) -> None:
        """Redraw what `reasons` invalidated, given the tick's window.

        `window` is the one `Window` the application built for this tick,
        so a signal several panes draw is computed once.
        """
        raise NotImplementedError

    def describe(self) -> dict:
        return {"kind": self.kind, "signal": self.signal}

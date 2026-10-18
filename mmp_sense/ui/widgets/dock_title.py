"""A dock's title bar with a collapse button beside float and close.

Qt's own title bar cannot take another button, so the docks get this one.
It lays itself out across, or down with the title turned on its side for
a dock squeezed into a thin strip.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QBoxLayout,
    QDockWidget,
    QLabel,
    QSizePolicy,
    QStyle,
    QToolButton,
    QWidget,
)

from ..panes.base import COLLAPSED_MARK, EXPANDED_MARK

__all__ = ["DockTitle"]


class _SideLabel(QLabel):
    """A label that reads bottom to top, for a vertical title bar."""

    def sizeHint(self) -> QSize:
        size = super().sizeHint()
        return QSize(size.height(), size.width())

    def minimumSizeHint(self) -> QSize:
        size = super().minimumSizeHint()
        return QSize(size.height(), size.width())

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.translate(0, self.height())
        painter.rotate(-90)
        painter.drawText(0, 0, self.height(), self.width(),
                         Qt.AlignLeft | Qt.AlignVCenter, self.text())


class DockTitle(QWidget):
    """Title, collapse, float and close, for one `QDockWidget`."""

    collapse_clicked = Signal()

    def __init__(self, dock: QDockWidget) -> None:
        super().__init__(dock)
        self.dock = dock
        self.vertical = False
        self.error: str | None = None
        self.label = QLabel(dock.windowTitle())
        self.side_label = _SideLabel(dock.windowTitle())
        self.side_label.hide()

        self.collapse_button = self._button(EXPANDED_MARK, "Collapse to the title")
        self.collapse_button.clicked.connect(self.collapse_clicked)
        style = self.style()
        self.float_button = self._button("", "Float or dock it")
        self.float_button.setIcon(style.standardIcon(QStyle.SP_TitleBarNormalButton))
        self.float_button.clicked.connect(lambda: dock.setFloating(not dock.isFloating()))
        self.close_button = self._button("", "Close it")
        self.close_button.setIcon(style.standardIcon(QStyle.SP_TitleBarCloseButton))
        self.close_button.clicked.connect(dock.close)

        self.box = QBoxLayout(QBoxLayout.LeftToRight, self)
        self.box.setContentsMargins(4, 1, 2, 1)
        self.box.setSpacing(2)
        self._fill()

    def _button(self, text: str, tip: str) -> QToolButton:
        button = QToolButton(self)
        button.setText(text)
        button.setToolTip(tip)
        button.setAutoRaise(True)
        button.setFocusPolicy(Qt.NoFocus)
        return button

    def _fill(self) -> None:
        while self.box.count():
            self.box.takeAt(0)
        buttons = [self.collapse_button, self.float_button, self.close_button]
        if self.vertical:
            self.box.setDirection(QBoxLayout.TopToBottom)
            for button in buttons:
                self.box.addWidget(button, 0, Qt.AlignHCenter)
            self.box.addWidget(self.side_label, 1, Qt.AlignHCenter)
            self.label.hide()
            self.side_label.show()
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        else:
            self.box.setDirection(QBoxLayout.LeftToRight)
            self.box.addWidget(self.label, 1)
            for button in buttons:
                self.box.addWidget(button)
            self.side_label.hide()
            self.label.show()
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_vertical(self, vertical: bool) -> None:
        if vertical != self.vertical:
            self.vertical = vertical
            self._fill()
            self.updateGeometry()

    def set_error(self, error: str | None) -> None:
        """Red and bold, with the error as the tip; None to clear it."""
        style = "color: #e03c31; font-weight: bold;" if error else ""
        for label in (self.label, self.side_label):
            label.setStyleSheet(style)
            label.setToolTip(error or "")
        self.error = error

    def set_collapsed(self, collapsed: bool) -> None:
        self.collapse_button.setText(COLLAPSED_MARK if collapsed else EXPANDED_MARK)
        self.collapse_button.setToolTip("Open it" if collapsed else "Collapse to the title")

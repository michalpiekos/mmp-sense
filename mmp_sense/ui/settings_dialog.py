"""Edit -> Settings: the user's settings, for every window."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
)

from ..settings import Settings

__all__ = ["SettingsDialog"]


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.settings = settings
        self.server = QLineEdit(settings.server)
        self.server.setPlaceholderText("http://127.0.0.1:8080")
        self.server.setToolTip("The srv-senselab that File > Synchronize sends to")
        self.relay = QLineEdit(settings.relay)
        self.relay.setPlaceholderText(f"{settings.relay_address()} (the server's host)")
        self.relay.setToolTip("The relay File > Open stream lists live streams from, HOST:PORT")
        self.receive_bind = QLineEdit(settings.receive_bind)
        self.receive_bind.setToolTip("Where File > Receive streams takes node streams, HOST:PORT")
        self.receive_relay = QLineEdit(settings.receive_relay)
        self.receive_relay.setToolTip("Where File > Receive streams relays them, HOST:PORT")
        self.history = QDoubleSpinBox()
        self.history.setRange(1.0, 600.0)
        self.history.setDecimals(0)
        self.history.setSuffix(" s")
        self.history.setValue(settings.live_history_s)
        self.history.setToolTip("How much of each live stream is kept")
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form = QFormLayout(self)
        form.addRow("srv-senselab address", self.server)
        form.addRow("Live streams from", self.relay)
        form.addRow("Receive streams on", self.receive_bind)
        form.addRow("Relay received streams on", self.receive_relay)
        form.addRow("Keep of each live stream", self.history)
        form.addRow(buttons)
        self.resize(420, self.sizeHint().height())

    def accept(self) -> None:
        self.settings.server = self.server.text()
        self.settings.receive_bind = self.receive_bind.text()
        self.settings.receive_relay = self.receive_relay.text()
        self.settings.relay = self.relay.text()
        self.settings.live_history_s = self.history.value()
        super().accept()

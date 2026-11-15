"""Edit -> Settings: the user's settings, for every window."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
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
        self.token_file = QLineEdit(settings.token_file)
        self.token_file.setPlaceholderText("asked at the first Synchronize when empty")
        self.token_file.setToolTip("A file holding the source token, as written by "
                                   "sample source create --token-file")
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        token_row = QHBoxLayout()
        token_row.addWidget(self.token_file, 1)
        token_row.addWidget(browse)
        self.relay = QLineEdit(settings.relay)
        self.relay.setPlaceholderText(f"{settings.relay_address()} (the server's host)")
        self.relay.setToolTip("The gateway relay File > Open stream lists live streams from, HOST:PORT")
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
        form.addRow("Token file", token_row)
        form.addRow("Live streams from", self.relay)
        form.addRow("Keep of each live stream", self.history)
        form.addRow(buttons)
        self.resize(420, self.sizeHint().height())

    def accept(self) -> None:
        self.settings.server = self.server.text()
        self.settings.token_file = self.token_file.text()
        self.settings.relay = self.relay.text()
        self.settings.live_history_s = self.history.value()
        super().accept()

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "The source token file", self.token_file.text())
        if path:
            self.token_file.setText(path)

"""File -> Open stream: the live streams a relay has open, to pick some to watch."""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from ..streaming import RelayError, list_streams

__all__ = ["OpenStreamDialog", "describe"]


def describe(listing: dict) -> str:
    """One line for a relayed session: its node, start, streams and whether the server keeps it."""
    model = (listing.get("device") or {}).get("model") or "stream"
    started = (listing.get("started_utc") or "")[11:19]
    streams = ", ".join(
        f"{s['name']} {len(s.get('channels') or [])} ch {s.get('rate_hz', '')} Hz".replace(" ch  Hz", "")
        for s in listing.get("streams") or [] if s.get("kind") == "pcm")
    kept = "stored" if listing.get("stored") else "relayed only"
    return f"{model}  {listing['capture_session_id'][:8]}  since {started} UTC  {streams}  ({kept})"


class _Lister(QObject):
    done = Signal(int, str, object, object)    # request, relay, streams or None, error or None


class OpenStreamDialog(QDialog):
    """Lists a relay's streams in the background; `chosen` are the ticked ones."""

    def __init__(self, relay: str, token_for, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Open stream")
        #: The token to send a relay, by its address: None for a relay that is not the server's.
        self.token_for = token_for
        self.chosen: list[dict] = []
        #: The relay the list shown came from; Watch uses it, not the text edited since.
        self.listed_relay = ""
        self._request = 0
        self.relay = QLineEdit(relay)
        self.relay.setToolTip("A gateway relay, HOST:PORT: the server's, or senselab-gateway's on a laptop")
        self.relay.returnPressed.connect(self.refresh)
        again = QPushButton("Refresh")
        again.clicked.connect(self.refresh)
        top = QHBoxLayout()
        top.addWidget(QLabel("Relay"))
        top.addWidget(self.relay, 1)
        top.addWidget(again)
        self.list = QListWidget()
        self.state = QLabel()
        self.state.setWordWrap(True)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Watch")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.list, 1)
        layout.addWidget(self.state)
        layout.addWidget(self.buttons)
        self.resize(640, 320)
        self._lister = _Lister(self)
        self._lister.done.connect(self._listed)
        self.refresh()

    def refresh(self) -> None:
        self.state.setText("Asking the relay...")
        self.list.clear()
        self.listed_relay = ""
        self._request += 1
        request, relay = self._request, self.relay.text().strip()
        token = self.token_for(relay)
        lister = self._lister

        def ask():
            try:
                result = (list_streams(relay, token), None)
            except (RelayError, ValueError, KeyError, TypeError) as exc:
                result = (None, str(exc) or type(exc).__name__)
            try:
                lister.done.emit(request, relay, *result)
            except RuntimeError:
                pass                      # the dialog closed meanwhile

        threading.Thread(target=ask, name="relay-list", daemon=True).start()

    def _listed(self, request: int, relay: str, streams, error) -> None:
        if request != self._request:
            return                        # an answer to an earlier Refresh
        if error:
            self.state.setText(error)
            return
        self.listed_relay = relay
        self.show_streams(streams)

    def show_streams(self, streams: list[dict]) -> None:
        self.list.clear()
        for listing in streams:
            item = QListWidgetItem(describe(listing))
            item.setData(Qt.UserRole, listing)
            has_pcm = any(s.get("kind") == "pcm" for s in listing.get("streams") or [])
            item.setFlags((item.flags() | Qt.ItemIsUserCheckable) if has_pcm else Qt.NoItemFlags)
            item.setCheckState(Qt.Unchecked)
            self.list.addItem(item)
        if len(streams) == 1 and self.list.item(0).flags() & Qt.ItemIsUserCheckable:
            self.list.item(0).setCheckState(Qt.Checked)
        self.state.setText(f"{len(streams)} open" if streams else "No stream is open on this relay.")

    def accept(self) -> None:
        if self.relay.text().strip() != self.listed_relay:
            self.refresh()                # the address changed: list that relay first
            return
        self.chosen = [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())
                       if self.list.item(i).checkState() == Qt.Checked]
        super().accept()

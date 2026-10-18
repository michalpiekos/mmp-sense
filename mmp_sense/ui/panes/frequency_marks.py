"""The session's frequency cursor, and its harmonics, over a chart whose x
is frequency.

A click places the cursor on the nearest peak, refined between bins;
Alt+click places it where clicked. Shift-drag stays the band's, and a
drag still pans: pyqtgraph reports a click only when the mouse did not
move. On a logarithmic axis the marks sit at log10 of their frequency,
as pyqtgraph puts everything else there.
"""

from __future__ import annotations

import math

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from ...dsp.peaks import snap_to_peak
from .band_strips import LOG_FLOOR_HZ

__all__ = ["FrequencyMarks", "CURSOR_COLOUR", "MAX_HARMONICS"]

#: Apart from the time cursor's orange, and a colour viridis never uses,
#: so it shows on the spectrogram too.
CURSOR_COLOUR = "#ff4fd8"
HARMONIC_PEN = pg.mkPen((255, 79, 216, 110), width=1, style=Qt.DashLine)
#: Enough for a comb to the top of a chart; a cursor near 0 Hz draws no more.
MAX_HARMONICS = 100


def add_readout(pane) -> QLabel:
    """A label in the pane's header, after its status, for the cursor's values."""
    readout = QLabel("")
    readout.setStyleSheet(f"color: {CURSOR_COLOUR};")
    layout = pane.header.layout()
    layout.insertWidget(layout.indexOf(pane.status) + 1, readout)
    return readout


class FrequencyMarks:
    """Draws `session.frequency_cursor` on `pane`, sets it from clicks, and
    keeps the readout. The pane gives `frequency_values()` -- (bin
    frequencies, dB) or None -- and `readout_text(hz)`."""

    def __init__(self, pane) -> None:
        self.pane = pane
        self.session = pane.session
        self.log = False
        self.harmonics_shown = False
        self.top_hz = 0.0
        self.item = pane.plot.getPlotItem()
        self.line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(CURSOR_COLOUR, width=1))
        self.line.hide()
        self.item.addItem(self.line, ignoreBounds=True)
        self.harmonics: list[pg.InfiniteLine] = []
        self.readout = add_readout(pane)
        pane.plot.scene().sigMouseClicked.connect(self._clicked)

    def _to_view(self, hz: float) -> float:
        return math.log10(max(hz, LOG_FLOOR_HZ)) if self.log else hz

    def sync(self, log: bool | None = None, top_hz: float | None = None) -> None:
        """Put the marks and the readout where the cursor now is."""
        if log is not None:
            self.log = bool(log)
        if top_hz is not None:
            self.top_hz = float(top_hz)
        hz = self.session.frequency_cursor
        if hz is None:
            self.line.hide()
            self._draw_harmonics([])
            self.readout.setText("")
            return
        self.line.setPos(self._to_view(hz))
        self.line.show()
        wanted = []
        if self.harmonics_shown and hz > 0:
            wanted = [k * hz for k in range(2, MAX_HARMONICS + 2) if k * hz <= self.top_hz]
        self._draw_harmonics(wanted)
        self.readout.setText(self.pane.readout_text(hz))

    def set_harmonics_shown(self, shown: bool) -> None:
        self.harmonics_shown = bool(shown)
        self.sync()

    def harmonic_frequencies(self) -> list[float]:
        """Where the harmonic markers are, in Hz: for tests and the console."""
        return [10.0 ** m.value() if self.log else m.value()
                for m in self.harmonics if m.isVisible()]

    def _draw_harmonics(self, frequencies: list[float]) -> None:
        while len(self.harmonics) < len(frequencies):
            marker = pg.InfiniteLine(angle=90, movable=False, pen=HARMONIC_PEN)
            self.item.addItem(marker, ignoreBounds=True)
            self.harmonics.append(marker)
        for i, marker in enumerate(self.harmonics):
            if i < len(frequencies):
                marker.setPos(self._to_view(frequencies[i]))
                marker.show()
            else:
                marker.hide()

    def _clicked(self, event) -> None:
        if event.button() != Qt.LeftButton or event.modifiers() & Qt.ShiftModifier:
            return
        viewbox = self.item.getViewBox()
        if not viewbox.sceneBoundingRect().contains(event.scenePos()):
            return
        self.place(viewbox.mapSceneToView(event.scenePos()).x(),
                   snap=not event.modifiers() & Qt.AltModifier)

    def place(self, x: float, snap: bool = True) -> float:
        """Set the cursor from a view x, on the nearest peak unless `snap`
        is false; returns where it went, in Hz."""
        hz = max(0.0, 10.0 ** x if self.log else x)
        values = self.pane.frequency_values() if snap else None
        if values is not None and len(values[0]):
            hz = snap_to_peak(*values, hz)
        self.session.set_frequency_cursor(hz)
        return hz

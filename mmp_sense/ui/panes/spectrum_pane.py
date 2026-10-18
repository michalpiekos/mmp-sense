"""The spectrum of the cursor window, or of the selection."""

from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer

from ...core.query import spectrum_of
from ...dsp.peaks import nearest_bin
from .band_strips import BandStrips
from .base import COLOURS, Pane
from .frequency_marks import FrequencyMarks
from .signal_pane import SelectionViewBox

__all__ = ["SpectrumPane"]

#: Long enough to look at a transient's marker after noticing it.
PEAK_HOLD_S = 2.0
#: A deeper blue than the bars, so a marker stands apart where it sits on one.
PEAK_COLOUR = "#2449c8"


class SpectrumPane(Pane):
    """An FFT that follows the cursor.

    What it transforms is the selection when there is one, and otherwise
    the analysis window around the cursor -- so moving the cursor or
    changing the window changes the chart at once, which is the whole
    point of it.

    Over each bar a short line holds the highest value the bin reached
    for `peak_hold_s` seconds of wall-clock time, then drops to the bar.
    The session's frequency bands are shaded strips; shift-drag adds one.
    A click puts the frequency cursor on the nearest peak.
    """

    kind = "spectrum"
    time_axis = False

    def __init__(self, session, signal="magnitude", parent=None) -> None:
        super().__init__(session, signal, parent)
        self.plot.setLabel("bottom", "Frequency", units="Hz")
        self.plot.setLabel("left", "Amplitude")
        self.bars = pg.BarGraphItem(
            x=np.zeros(0), height=np.zeros(0), width=1.0,
            pen=pg.mkPen(COLOURS[1], width=1), brush=pg.mkBrush(COLOURS[1]),
        )
        self.plot.addItem(self.bars)
        self.peak = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#dd8800", style=Qt.DashLine)
        )
        self.plot.addItem(self.peak, ignoreBounds=True)
        self.markers = self.plot.plot(
            pen=pg.mkPen(PEAK_COLOUR, width=2), connect="pairs"
        )
        self.bands = BandStrips(self.session, self.plot.getPlotItem())
        self._viewbox.region_dragged.connect(self.bands.dragged)
        self.bands.sync()
        self.marks = FrequencyMarks(self)
        self.clock = time.monotonic
        self._peak_hold_s = PEAK_HOLD_S
        self._frequency = np.zeros(0)
        self._width = 0.0
        self._current = np.zeros(0)
        self.held = np.zeros(0)
        self._held_at = np.zeros(0)
        # Holds expire with nothing else changing, e.g. while stopped.
        self._expiry = QTimer(self)
        self._expiry.setInterval(100)
        self._expiry.timeout.connect(self.expire_holds)
        self._expiry.start()

    @property
    def peak_hold_s(self) -> float:
        return self._peak_hold_s

    @peak_hold_s.setter
    def peak_hold_s(self, seconds: float) -> None:
        self._peak_hold_s = max(0.0, float(seconds))
        self.held = self._current.copy()
        self._held_at = np.full(len(self._current), self.clock())
        self._draw_markers()

    def make_viewbox(self):
        self._viewbox = SelectionViewBox()
        return self._viewbox

    def choices(self) -> list[str]:
        return list(self.session.signals.names)

    def frequency_values(self):
        """(bin frequencies, amplitude in dB) of what is drawn, or None."""
        if not len(self._current):
            return None
        with np.errstate(divide="ignore"):
            return self._frequency, 20.0 * np.log10(self._current)

    def readout_text(self, hz: float) -> str:
        if not len(self._current):
            return f"{hz:.5g} Hz"
        i = nearest_bin(self._frequency, hz)
        return f"{hz:.5g} Hz, bin {self._frequency[i]:.5g} Hz: {self._current[i]:.4g}"

    def set_harmonics_shown(self, shown: bool) -> None:
        self.marks.set_harmonics_shown(shown)

    def clear_bars(self) -> None:
        self.bars.setOpts(x=np.zeros(0), height=np.zeros(0))
        self.update_holds(np.zeros(0), np.zeros(0), 0.0)
        self.marks.sync()

    def update_holds(self, frequency, values, width: float) -> None:
        """Fold a new spectrum into the held maxima; new bins start afresh."""
        now = self.clock()
        values = np.asarray(values, dtype=float)
        if not np.array_equal(frequency, self._frequency):
            self._frequency = np.asarray(frequency, dtype=float).copy()
            self.held = values.copy()
            self._held_at = np.full(len(values), now)
        else:
            renew = (values >= self.held) | (now - self._held_at >= self._peak_hold_s)
            self.held = np.where(renew, values, self.held)
            self._held_at = np.where(renew, now, self._held_at)
        self._current = values.copy()
        self._width = float(width)
        self._draw_markers()

    def expire_holds(self) -> None:
        if self.collapsed or self._peak_hold_s <= 0 or not len(self.held):
            return
        expired = (self.clock() - self._held_at >= self._peak_hold_s) & (
            self.held != self._current
        )
        if not expired.any():
            return
        self.held = np.where(expired, self._current, self.held)
        self._held_at = np.where(expired, self.clock(), self._held_at)
        self._draw_markers()

    def _draw_markers(self) -> None:
        if self._peak_hold_s <= 0 or not len(self.held):
            self.markers.setData(np.zeros(0), np.zeros(0))
            return
        half = self._width / 2.0
        x = np.column_stack([self._frequency - half, self._frequency + half]).ravel()
        self.markers.setData(x, np.repeat(self.held, 2))

    def refresh(self, reasons: set[str], window=None) -> None:
        if "bands" in reasons:
            self.bands.sync()
        if "frequency" in reasons:
            self.marks.sync()
        if not reasons & {"cursor", "selection", "analysis", "data", "signals"}:
            return
        name = self.signal
        if name not in self.session.signals:
            self.clear_bars()
            return
        # The selection, else the analysis window around the cursor.
        span = self.session.selection or (
            self.session.cursor - self.session.analysis_window / 2,
            self.session.cursor + self.session.analysis_window / 2)
        spectrum = spectrum_of(self.session, name, *span)
        if not len(spectrum.frequency):
            self.clear_bars()
            self.status.setText("not enough samples")
            return
        values = spectrum.channel(0)
        self.bars.setOpts(
            x=spectrum.frequency, height=values, width=spectrum.resolution
        )
        self.update_holds(spectrum.frequency, values, spectrum.resolution)
        self.marks.sync(top_hz=float(spectrum.frequency[-1]))
        self.peak.setPos(spectrum.peak_frequency())
        t0, t1 = spectrum.source_range or (0.0, 0.0)
        self.status.setText(
            f"{t0:.3f}-{t1:.3f} s, {spectrum.resolution:.3g} Hz/bin, "
            f"peak {spectrum.peak_frequency():.4g} Hz"
        )

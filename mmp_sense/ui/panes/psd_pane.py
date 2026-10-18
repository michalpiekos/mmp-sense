"""The averaged spectrum -- power spectral density -- of the selection,
else of everything held."""

from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QCheckBox, QComboBox

from ...dsp.peaks import nearest_bin
from .band_strips import BandStrips
from .base import COLOURS, Pane
from .frequency_marks import FrequencyMarks
from .signal_pane import SelectionViewBox

__all__ = ["PsdPane", "SEGMENTS"]

#: Segment lengths offered, in samples: longer is finer in frequency,
#: shorter averages more of them. At 48 kHz, 32768 is 1.46 Hz a bin.
SEGMENTS = ("256", "512", "1024", "2048", "4096", "8192", "16384", "32768", "65536")
#: Averaging a minute is not free; twice a second is enough to watch it.
MIN_INTERVAL_S = 0.5


class PsdPane(Pane):
    """Welch's power spectral density, in units squared per Hz or in dB,
    on a linear or a logarithmic frequency axis. The session's frequency
    bands are shaded strips; shift-drag adds one. A click puts the
    frequency cursor on the nearest peak."""

    kind = "psd"
    time_axis = False

    def __init__(self, session, signal=None, parent=None) -> None:
        self.db = QCheckBox("dB")
        self.db.setChecked(True)
        self.log_frequency = QCheckBox("log f")
        self.segment = QComboBox()
        self.segment.addItems(SEGMENTS)
        self.segment.setCurrentText("1024")
        self.segment.setToolTip("Samples per segment: longer is finer, shorter averages more")
        super().__init__(session, signal or session.channels[0], parent)
        layout = self.header.layout()
        at = layout.indexOf(self.chooser) + 1
        for i, widget in enumerate((self.segment, self.db, self.log_frequency)):
            layout.insertWidget(at + i, widget)
        for box in (self.db, self.log_frequency):
            box.toggled.connect(lambda _on: self._redraw())
        self.segment.currentTextChanged.connect(lambda _t: self._redraw())
        self.plot.setLabel("bottom", "Frequency", units="Hz")
        self.curve = self.plot.plot(pen=pg.mkPen(COLOURS[1], width=1))
        self.bands = BandStrips(self.session, self.plot.getPlotItem())
        self._viewbox.region_dragged.connect(self.bands.dragged)
        self.bands.sync()
        self.frequency = np.zeros(0)
        self.density = np.zeros(0)
        self.marks = FrequencyMarks(self)
        self._last = 0.0

    def make_viewbox(self):
        self._viewbox = SelectionViewBox()
        return self._viewbox

    def choices(self) -> list[str]:
        return list(self.session.signals.names)

    def _redraw(self) -> None:
        self.refresh({"signals"})

    def frequency_values(self):
        """(bin frequencies, density in dB) of what is drawn, or None."""
        if not len(self.density):
            return None
        with np.errstate(divide="ignore"):
            return self.frequency, 10.0 * np.log10(self.density)

    def readout_text(self, hz: float) -> str:
        if not len(self.density):
            return f"{hz:.5g} Hz"
        i = nearest_bin(self.frequency, hz)
        density = self.density[i]
        if self.db.isChecked():
            with np.errstate(divide="ignore"):
                value = f"{10.0 * np.log10(density):.1f} dB"
        else:
            value = f"{density:.4g}"
        return f"{hz:.5g} Hz, bin {self.frequency[i]:.5g} Hz: {value}"

    def set_harmonics_shown(self, shown: bool) -> None:
        self.marks.set_harmonics_shown(shown)

    def refresh(self, reasons: set[str], window=None) -> None:
        if "bands" in reasons:
            self.bands.sync(self.log_frequency.isChecked())
        if "frequency" in reasons:
            self.marks.sync()
        if not reasons & {"data", "signals", "source", "selection"}:
            return
        now = time.monotonic()
        if now - self._last < MIN_INTERVAL_S and not reasons & {"signals", "selection"}:
            return
        self._last = now
        session, name = self.session, self.signal
        if name not in session.signals:
            self.curve.setData(np.zeros(0), np.zeros(0))
            self.density = np.zeros(0)
            self.marks.sync()
            return
        span = session.selection or session.span
        segment = int(self.segment.currentText())
        try:
            density = session.psd(name, *span, segment=segment)
        except Exception as exc:                              # noqa: BLE001
            self.curve.setData(np.zeros(0), np.zeros(0))
            self.density = np.zeros(0)
            self.marks.sync()
            self.status.setText(str(exc)[:120])
            return
        self.frequency, self.density = density.index.to_numpy(float), density.to_numpy(float)
        x, y = self.frequency, self.density
        units = session.signals.units_for([name])
        if self.db.isChecked():
            with np.errstate(divide="ignore"):
                y = 10.0 * np.log10(y)
            self.plot.setLabel("left", f"PSD ({units}²/Hz)" if units else "PSD", units="dB")
        else:
            self.plot.setLabel("left", f"PSD ({units}²/Hz)" if units else "PSD (1/Hz)")
        log = self.log_frequency.isChecked()
        if log:
            keep = x > 0
            x, y = x[keep], y[keep]
        self.plot.setLogMode(x=log, y=False)
        self.bands.sync(log)
        self.marks.sync(log=log, top_hz=float(self.frequency[-1]) if len(self.frequency) else 0.0)
        finite = np.isfinite(y)
        self.curve.setData(x[finite], y[finite])
        step = self.frequency[1] - self.frequency[0] if len(self.frequency) > 1 else np.nan
        self.status.setText(f"{span[0]:.3f}-{span[1]:.3f} s, {segment}-sample segments, "
                            f"{step:.3g} Hz/bin")

    def describe(self) -> dict:
        return {**super().describe(), "segment": int(self.segment.currentText()),
                "db": self.db.isChecked(), "log_frequency": self.log_frequency.isChecked()}

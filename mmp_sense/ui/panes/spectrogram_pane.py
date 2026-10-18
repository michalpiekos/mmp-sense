"""Frequency over time, drawn as an image."""

from __future__ import annotations

import time as _time

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import QComboBox

from ...core.query import spectrogram_of
from ...dsp.peaks import nearest_bin
from .base import Pane
from .frequency_marks import CURSOR_COLOUR, add_readout

__all__ = ["SpectrogramPane", "SEGMENTS", "TOPS"]

#: Samples per segment: each one's bins are rate / segment apart, so a
#: longer one tells closer frequencies apart -- engine harmonics 19 Hz
#: apart need some 8192 at 48 kHz -- and blurs time more.
SEGMENTS = ("256", "512", "1024", "2048", "4096", "8192", "16384", "32768")
#: How high the image reaches, in Hz; the colours span what is drawn.
TOPS = ("all", "100", "200", "500", "1000", "2000", "5000", "10000")

#: A spectrogram of the whole window is the most expensive thing drawn,
#: and it does not change meaningfully four times a second.
MIN_INTERVAL_S = 0.5


class SpectrogramPane(Pane):
    """The spectrogram of one signal over the whole buffer.

    A click puts the time cursor at its time and the frequency cursor at
    its frequency; the readout is the cell where the two cross.
    """

    kind = "spectrogram"

    def __init__(self, session, signal="magnitude", parent=None) -> None:
        super().__init__(session, signal, parent)
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setLabel("left", "Frequency", units="Hz")
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        self.legend = pg.ColorBarItem(
            colorMap=pg.colormap.get("viridis"), label="dB", interactive=False,
            orientation="horizontal", width=10,
        )
        # Below, not beside: a narrower view box would shift the linked time axis.
        self.legend.setImageItem(self.image, insert_in=self.plot.getPlotItem())
        self.cursor_line = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#dd8800", width=1)
        )
        self.plot.addItem(self.cursor_line, ignoreBounds=True)
        self.frequency_line = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen(CURSOR_COLOUR, width=1)
        )
        self.frequency_line.hide()
        self.plot.addItem(self.frequency_line, ignoreBounds=True)
        self.readout = add_readout(self)
        self.plot.scene().sigMouseClicked.connect(self._clicked)
        self._time = np.zeros(0)
        self._frequency = np.zeros(0)
        self._decibels = np.zeros((0, 0))
        self.region = pg.LinearRegionItem(
            brush=pg.mkBrush(80, 130, 180, 50), movable=False
        )
        self.region.setZValue(10)
        self.region.hide()
        self.plot.addItem(self.region, ignoreBounds=True)
        self.segment = QComboBox()
        self.segment.addItems(SEGMENTS)
        self.segment.setCurrentText("256")
        self.segment.setToolTip("Samples per segment: longer tells closer frequencies "
                                "apart, and blurs time")
        self.top = QComboBox()
        self.top.addItems(TOPS)
        self.top.setToolTip("Draw frequencies up to this, Hz; the colours span what is drawn")
        layout = self.header.layout()
        at = layout.indexOf(self.chooser) + 1
        for i, widget in enumerate((self.segment, self.top)):
            layout.insertWidget(at + i, widget)
        for box in (self.segment, self.top):
            box.currentTextChanged.connect(lambda _t: self._redraw())
        self._last = 0.0

    @property
    def nperseg(self) -> int:
        return int(self.segment.currentText())

    @nperseg.setter
    def nperseg(self, samples: int) -> None:
        self.segment.setCurrentText(str(int(samples)))

    @property
    def top_hz(self) -> float | None:
        text = self.top.currentText()
        return None if text == "all" else float(text)

    @top_hz.setter
    def top_hz(self, hz: float | None) -> None:
        self.top.setCurrentText("all" if hz is None else str(int(hz)))

    def _redraw(self) -> None:
        self.refresh({"signals"}, self.session.window())

    def choices(self) -> list[str]:
        return list(self.session.signals.names)

    def frequency_values(self):
        """(bin frequencies, dB) of the column at the time cursor, or None."""
        if not self._decibels.size:
            return None
        return self._frequency, self._decibels[:, nearest_bin(self._time, self.session.cursor)]

    def readout_text(self, hz: float) -> str:
        t = self.session.cursor
        if not self._decibels.size:
            return f"{hz:.5g} Hz"
        # A column stands for a whole segment, centred on its time: 1 / bin spacing long.
        reach = 1.0 / (self._frequency[1] - self._frequency[0]) if len(self._frequency) > 1 else 0.0
        if not self._time[0] - reach <= t <= self._time[-1] + reach:
            return f"{hz:.5g} Hz"
        i, j = nearest_bin(self._frequency, hz), nearest_bin(self._time, t)
        return f"{t:.3f} s, {hz:.5g} Hz: {self._decibels[i, j]:.1f} dB"

    def _sync_frequency(self) -> None:
        hz = self.session.frequency_cursor
        if hz is None:
            self.frequency_line.hide()
            self.readout.setText("")
            return
        self.frequency_line.setPos(hz)
        self.frequency_line.show()
        self.readout.setText(self.readout_text(hz))

    def _clicked(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        viewbox = self.plot.getViewBox()
        if not viewbox.sceneBoundingRect().contains(event.scenePos()):
            return
        point = viewbox.mapSceneToView(event.scenePos())
        self.session.set_cursor(float(point.x()))
        self.session.set_frequency_cursor(max(0.0, float(point.y())))

    def refresh(self, reasons: set[str], window=None) -> None:
        if reasons & {"cursor", "data"}:
            self.cursor_line.setPos(self.session.cursor)
        if reasons & {"cursor", "frequency"}:
            self._sync_frequency()
        if "selection" in reasons:
            span = self.session.selection
            if span is None:
                self.region.hide()
            else:
                self.region.setRegion(span)
                self.region.show()
        if not reasons & {"data", "signals", "source"}:
            return
        now = _time.monotonic()
        if now - self._last < MIN_INTERVAL_S and "signals" not in reasons:
            return
        self._last = now
        name = self.signal
        if window is None or window.empty or name not in self.session.signals:
            self.image.clear()
            self._decibels = np.zeros((0, 0))
            self._sync_frequency()
            return
        spectrogram = spectrogram_of(
            self.session, name, window.t_start, window.t_end, nperseg=self.nperseg
        )
        if not all(spectrogram.shape):
            self.image.clear()
            self._decibels = np.zeros((0, 0))
            self._sync_frequency()
            return
        decibels = spectrogram.decibels()
        time, frequency = spectrogram.time, spectrogram.frequency
        top = self.top_hz
        if top is not None and frequency[-1] > top:
            keep = frequency <= top
            if keep.sum() >= 2:
                decibels, frequency = decibels[keep], frequency[keep]
        self._time, self._frequency, self._decibels = time, frequency, decibels
        self._sync_frequency()
        # ImageItem draws the first axis across, so time goes first.
        self.image.setImage(decibels.T, autoLevels=False)
        # From the typical cell up: half the image is the floor, drawn
        # dark, so that what stands above it -- a line, a burst -- shows.
        low, high = float(np.median(decibels)), float(decibels.max())
        self.legend.setLevels((low, high if high > low else low + 1.0))
        self.image.setRect(QRectF(
            float(time[0]), float(frequency[0]),
            float(time[-1] - time[0]) or 1e-9,
            float(frequency[-1] - frequency[0]) or 1e-9,
        ))
        step = frequency[1] - frequency[0] if len(frequency) > 1 else float("nan")
        self.status.setText(
            f"{spectrogram.shape[0]} x {spectrogram.shape[1]}, "
            f"{self.nperseg}-sample segments, {step:.3g} Hz/bin"
        )

    def describe(self) -> dict:
        return {**super().describe(), "segment": self.nperseg, "top_hz": self.top_hz}

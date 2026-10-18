"""The session's frequency bands drawn over a frequency chart.

Each band is a shaded strip whose edges can be dragged to change it;
shift-drag on the chart adds one. On a logarithmic axis the strips sit
at log10 of their edges, as pyqtgraph puts everything else there.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg

__all__ = ["BandStrips"]

ON_BRUSH = (60, 170, 90, 55)
OFF_BRUSH = (150, 150, 150, 25)
DRAFT_BRUSH = (60, 170, 90, 30)
#: Where 0 Hz is drawn on a log axis, which has no place for it.
LOG_FLOOR_HZ = 1e-3


class BandStrips:
    """Keeps one strip per band on `plot`, in step with `session.bands`."""

    def __init__(self, session, plot: pg.PlotItem) -> None:
        self.session = session
        self.plot = plot
        self.log = False
        self.strips: dict[str, pg.LinearRegionItem] = {}
        self._drawn: tuple = ()
        self.draft = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(*DRAFT_BRUSH))
        self.draft.hide()
        plot.addItem(self.draft, ignoreBounds=True)

    def _to_view(self, hz: float) -> float:
        return math.log10(max(hz, LOG_FLOOR_HZ)) if self.log else hz

    def _from_view(self, x: float) -> float:
        hz = 10.0 ** x if self.log else x
        return 0.0 if hz <= LOG_FLOOR_HZ * 1.0001 else hz

    def sync(self, log: bool = False) -> None:
        """Redraw the strips if the bands, or the axis, changed."""
        self.log = bool(log)
        drawn = (self.log, tuple(self.session.bands))
        if drawn == self._drawn:
            return
        self._drawn = drawn
        for strip in self.strips.values():
            self.plot.removeItem(strip)
        self.strips.clear()
        for band in self.session.bands:
            strip = pg.LinearRegionItem(
                values=(self._to_view(band.low_hz), self._to_view(band.high_hz)),
                brush=pg.mkBrush(*(ON_BRUSH if band.on else OFF_BRUSH)),
            )
            strip.setToolTip(f"{band.name}: {band.low_hz:.6g}-{band.high_hz:.6g} Hz"
                             + ("" if band.on else " (not heard)"))
            strip.setZValue(-10)
            strip.sigRegionChangeFinished.connect(
                lambda s=strip, name=band.name: self._moved(name, s))
            self.plot.addItem(strip, ignoreBounds=True)
            self.strips[band.name] = strip

    def _moved(self, name: str, strip) -> None:
        try:
            band = self.session.band(name)
        except KeyError:
            return
        a, b = strip.getRegion()
        edges = []
        for x, hz in ((a, band.low_hz), (b, band.high_hz)):
            # An edge not moved keeps its value, which a log axis would round.
            edges.append(hz if np.isclose(x, self._to_view(hz)) else self._from_view(x))
        low, high = max(0.0, min(edges)), max(edges)
        try:
            self.session.edit_band(name, low_hz=low, high_hz=high)
        except ValueError:
            self._drawn = ()
            self.sync(self.log)

    def dragged(self, start: float, end: float, finished: bool) -> str | None:
        """A shift-drag in view coordinates: drawn while it lasts, a band
        when it ends. Returns the new band's name."""
        a, b = sorted((start, end))
        if not finished:
            self.draft.setRegion((a, b))
            self.draft.show()
            return None
        self.draft.hide()
        low, high = max(0.0, self._from_view(a)), self._from_view(b)
        if high <= low:
            return None
        return self.session.add_band(low, high)

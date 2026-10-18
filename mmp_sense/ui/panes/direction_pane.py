"""Where a sound comes from: an array's steered power against azimuth,
over the selection or the analysis window around the cursor."""

from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt

from .base import COLOURS, Pane

__all__ = ["DirectionPane"]

#: Steering over every azimuth is not free; twice a second is enough while
#: data arrives, and the selection or the cursor moving redraws at once.
MIN_INTERVAL_S = 0.5


class DirectionPane(Pane):
    """SRP-PHAT power of one of the session's arrays, 0..360 degrees
    counter-clockwise from +x; its peak is where the sound came from."""

    kind = "direction"
    time_axis = False

    def __init__(self, session, signal=None, parent=None) -> None:
        arrays = list(session.arrays)
        super().__init__(session, signal if signal in session.arrays else
                         (arrays[0] if arrays else ""), parent)
        self.plot.setLabel("bottom", "Azimuth", units="deg")
        self.plot.setLabel("left", "Steered power")
        self.plot.setXRange(0, 360, padding=0)
        self.curve = self.plot.plot(pen=pg.mkPen(COLOURS[1], width=1))
        self.peak = pg.InfiniteLine(angle=90, movable=False,
                                    pen=pg.mkPen("#dd8800", style=Qt.DashLine))
        self.plot.addItem(self.peak, ignoreBounds=True)
        self.azimuth = np.nan
        self._last = 0.0

    def choices(self) -> list[str]:
        return list(self.session.arrays)

    def signal_names(self) -> list[str]:
        return []

    def refresh(self, reasons: set[str], window=None) -> None:
        if not reasons & {"cursor", "selection", "analysis", "data", "signals"}:
            return
        now = time.monotonic()
        # Following the newest data moves the cursor every tick: that is data too.
        asked = reasons & {"selection", "analysis", "signals"} or (
            "cursor" in reasons and not self.session.following)
        if not asked and now - self._last < MIN_INTERVAL_S:
            return
        self._last = now
        self.update_choices()
        session = self.session
        if self.signal not in session.arrays:
            self._clear("no array: session.add_array(name, mmp_sense.Array({...}))")
            return
        try:
            span = session.selection or (session.cursor - session.analysis_window / 2,
                                         session.cursor + session.analysis_window / 2)
            power = session.direction_map(self.signal, *span)
        except Exception as exc:                              # noqa: BLE001
            self._clear(str(exc)[:120])
            return
        self.curve.setData(power.index.to_numpy(float), power.to_numpy(float))
        self.azimuth = float(power.idxmax())
        self.peak.setPos(self.azimuth)
        self.peak.setVisible(True)
        self.status.setText(f"{span[0]:.3f}-{span[1]:.3f} s, azimuth {self.azimuth:g} deg, "
                            f"power {power.max():.2f}")

    def _clear(self, why: str) -> None:
        self.curve.setData(np.zeros(0), np.zeros(0))
        self.peak.setVisible(False)
        self.azimuth = np.nan
        self.status.setText(why)

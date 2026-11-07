"""A script's result: one named (x, y) curve from `session.plot`."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from .base import COLOURS, Pane

__all__ = ["XYPane"]

NO_PLOT = "no plot: session.plot(name, x, y)"


class XYPane(Pane):
    """Draws `session.plots[name]`, with the axes labelled as the script gave them."""

    kind = "xy"
    time_axis = False

    def __init__(self, session, signal=None, parent=None) -> None:
        plots = list(session.plots)
        super().__init__(session, signal if signal in session.plots else
                         (plots[0] if plots else ""), parent)
        for side in ("bottom", "left"):
            self.plot.getAxis(side).enableAutoSIPrefix(False)
        self.curve = self.plot.plot(pen=pg.mkPen(COLOURS[1], width=1))
        self._drawn = None
        self.status.setText(NO_PLOT)

    def choices(self) -> list[str]:
        return list(self.session.plots)

    def signal_names(self) -> list[str]:
        return []

    def refresh(self, reasons: set[str], window=None) -> None:
        plot = self.session.plots.get(self.signal)
        if plot is self._drawn:
            return
        self._drawn = plot
        if plot is None:
            self.curve.setData(np.zeros(0), np.zeros(0))
            self.status.setText(NO_PLOT)
            return
        self.plot.setLabel("bottom", plot.x_label, units=plot.x_units)
        self.plot.setLabel("left", plot.y_label, units=plot.y_units)
        self.curve.setData(plot.x, plot.y, connect="finite")
        self.plot.enableAutoRange()
        self.status.setText(f"{len(plot.x)} points")

    def describe(self) -> dict:
        return {**super().describe(), "points": 0 if self._drawn is None else len(self._drawn.x)}

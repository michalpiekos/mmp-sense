"""The correlation of two signals against lag, over the cursor window or
the selection: where it peaks is how much later the second hears things."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QLabel

from ...dsp import METHODS, peak_lag
from .base import COLOURS, Pane

__all__ = ["CorrelationPane"]

#: The longest lag looked at either way, in seconds.
MAX_LAG_S = 1.0


class CorrelationPane(Pane):
    """GCC-PHAT (or plain) correlation of the chosen signal and another.

    Over the selection, else the analysis window around the cursor, as
    the spectrum is; lags up to half that stretch, and no more than
    MAX_LAG_S, either way. The peak is marked and its lag is in the
    status line: positive when the second signal is later.
    """

    kind = "correlation"
    time_axis = False

    def __init__(self, session, signal=None, parent=None) -> None:
        self.other = QComboBox()
        self.other.setMinimumWidth(160)
        self.other.setToolTip("The signal compared with the one on the left")
        self.method = QComboBox()
        self.method.addItems(METHODS)
        self.method.setToolTip("phat: sharp whatever the sound; plain: -1..1")
        names = list(session.signals.names)
        super().__init__(session, signal or (names[0] if names else ""), parent)
        layout = self.header.layout()
        at = layout.indexOf(self.chooser) + 1
        layout.insertWidget(at, QLabel("vs"))
        layout.insertWidget(at + 1, self.other)
        layout.insertWidget(at + 2, self.method)
        self.other.currentTextChanged.connect(lambda _t: self._redraw())
        self.method.currentTextChanged.connect(lambda _t: self._redraw())
        self.plot.setLabel("bottom", "Lag", units="s")
        self.plot.setLabel("left", "Correlation")
        self.curve = self.plot.plot(pen=pg.mkPen(COLOURS[1], width=1))
        self.peak = pg.InfiniteLine(angle=90, movable=False,
                                    pen=pg.mkPen("#dd8800", style=Qt.DashLine))
        self.plot.addItem(self.peak, ignoreBounds=True)
        self.plot.addItem(pg.InfiniteLine(angle=90, movable=False,
                                          pen=pg.mkPen("#888888", width=1)),
                          ignoreBounds=True)
        self.update_choices()
        self.lag = np.nan

    def choices(self) -> list[str]:
        return list(self.session.signals.names)

    def update_choices(self) -> None:
        super().update_choices()
        names = self.choices()
        current = [self.other.itemText(i) for i in range(self.other.count())]
        if names == current:
            return
        wanted = self.other.currentText() or self._partner(names)
        self.other.blockSignals(True)
        self.other.clear()
        self.other.addItems(names)
        self.other.setCurrentText(wanted if wanted in names else self._partner(names))
        self.other.blockSignals(False)

    def _partner(self, names) -> str:
        """Another stream's first signal, else the next signal along."""
        signals = self.session.signals
        if self.signal in signals:
            mine = signals.stream_of(self.signal)
            for name in names:
                if name in signals and signals.stream_of(name) != mine:
                    return name
        others = [n for n in names if n != self.signal]
        return others[0] if others else self.signal

    def set_other(self, name: str) -> None:
        self.other.setCurrentText(name)

    def _redraw(self) -> None:
        self.refresh({"signals"})

    def refresh(self, reasons: set[str], window=None) -> None:
        if not reasons & {"cursor", "selection", "analysis", "data", "signals"}:
            return
        x, y = self.signal, self.other.currentText()
        session = self.session
        if x not in session.signals or y not in session.signals:
            self._clear("pick two signals")
            return
        try:
            span = session.selection or (session.cursor - session.analysis_window / 2,
                                         session.cursor + session.analysis_window / 2)
            max_lag = min((span[1] - span[0]) / 2.0, MAX_LAG_S)
            series = session.correlation(x, y, *span, max_lag=max_lag,
                                         method=self.method.currentText())
        except Exception as exc:                              # noqa: BLE001
            self._clear(str(exc)[:120])
            return
        lags, values = series.index.to_numpy(float), series.to_numpy(float)
        self.curve.setData(lags, values)
        self.lag, peak, snr = peak_lag(lags, values)
        self.peak.setVisible(bool(np.isfinite(self.lag)))
        if np.isfinite(self.lag):
            self.peak.setPos(self.lag)
            self.status.setText(f"{span[0]:.3f}-{span[1]:.3f} s, delay "
                                f"{self.lag * 1e3:+.3f} ms, peak {peak:.3g}, snr {snr:.3g}")
        else:
            self.status.setText("no peak")

    def _clear(self, why: str) -> None:
        self.curve.setData(np.zeros(0), np.zeros(0))
        self.peak.setVisible(False)
        self.lag = np.nan
        self.status.setText(why)

    def describe(self) -> dict:
        return {**super().describe(), "other": self.other.currentText(),
                "method": self.method.currentText()}

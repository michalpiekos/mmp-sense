"""Signals against time, sharing one time axis with every other pane."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal

from ...core.query import distribution
from .base import ALL_CHANNELS, COLOURS, Pane

__all__ = ["SignalPane", "SelectionViewBox", "SeriesLegend", "DETECTION_COLOURS",
           "MAX_MARKS", "DISTRIBUTION_BINS"]

#: One colour per shown detection, apart from the curves' colours.
DETECTION_COLOURS = ["#c17d11", "#ad7fa8", "#e9b96e", "#729fcf", "#ef2929", "#8ae234"]

#: The most events of one kind a pane marks; the newest are kept.
MAX_MARKS = 200

#: How many bins the distribution beside a chart has, across its visible values.
DISTRIBUTION_BINS = 64


class SelectionViewBox(pg.ViewBox):
    """A view box where shift-drag selects an interval.

    Left-drag still pans and the wheel still zooms; selecting is the one
    thing added, because a time range is what an FFT is taken over and
    dragging it out is how one asks for it.
    """

    region_dragged = Signal(float, float, bool)

    def mouseDragEvent(self, event, axis=None):
        if event.button() == Qt.LeftButton and (event.modifiers() & Qt.ShiftModifier):
            event.accept()
            start = self.mapToView(event.buttonDownPos()).x()
            now = self.mapToView(event.pos()).x()
            self.region_dragged.emit(float(start), float(now), bool(event.isFinish()))
            return
        super().mouseDragEvent(event, axis)


class _Sample(pg.ItemSample):
    """A legend entry's line: clicks go to the pane, not straight to the curve."""

    def __init__(self, item, legend) -> None:
        super().__init__(item)
        self.legend = legend

    def mouseClickEvent(self, event) -> None:
        self.legend.entry_clicked(self.item, event)

    def mouseDoubleClickEvent(self, event) -> None:
        self.legend.entry_double_clicked(self.item, event)


class _Label(pg.LabelItem):
    """A legend entry's name, clickable as its line is."""

    def __init__(self, text, legend, series, **opts) -> None:
        super().__init__(text, **opts)
        self.legend = legend
        self.series = series            # not `item`: LabelItem's own text is that

    def mouseClickEvent(self, event) -> None:
        self.legend.entry_clicked(self.series, event)

    def mouseDoubleClickEvent(self, event) -> None:
        self.legend.entry_double_clicked(self.series, event)


class SeriesLegend(pg.LegendItem):
    """The legend as the switch: click an entry to hide or show that series,
    double-click it to show it alone, or all again."""

    def __init__(self, pane, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pane = pane
        self._last_click = None

    def addItem(self, item, name) -> None:
        sample = _Sample(item, self)
        label = _Label(name, self, item, color=self.opts["labelTextColor"],
                       justify="left", size=self.opts["labelTextSize"])
        self.items.append((sample, label))
        self._addItemToLayout(sample, label)
        self.updateSize()

    def entry_clicked(self, item, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        event.accept()
        name = item.name()
        self._last_click = name
        self.pane.set_series_visible(name, not item.isVisible())

    def entry_double_clicked(self, item, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        event.accept()
        name = item.name()
        if self._last_click == name:
            # The click that began this double-click toggled it; undo that.
            self.pane.set_series_visible(name, not item.isVisible())
        self._last_click = None
        self.pane.solo_series(name)

    def paint_entry(self, item, visible: bool) -> None:
        for sample, label in self.items:
            if sample.item is item:
                colour = self.opts["labelTextColor"] if visible else "#666666"
                label.setText(label.text, color=colour)
                sample.update()


class SignalPane(Pane):
    """One or more signals drawn against time.

    The curves are created once and given new data on every tick; a long
    window costs what a short one does because pyqtgraph is told to
    downsample by peaks and to clip to what is visible.
    """

    kind = "signal"

    def __init__(self, session, signal="channels", parent=None) -> None:
        super().__init__(session, signal, parent)
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.legend = SeriesLegend(self, offset=(-10, 10))
        self.legend.setParentItem(self.plot.getPlotItem().vb)
        self.plot.getPlotItem().legend = self.legend
        #: Series switched off in the legend, by name, for this selection.
        self.hidden_series: set[str] = set()
        self._hidden_for = signal

        self._curves: dict[str, pg.PlotDataItem] = {}
        #: Per series, its bars and outline in the distribution column.
        self._densities: dict[str, tuple[pg.BarGraphItem, pg.PlotDataItem]] = {}
        self._window = None
        self.cursor_line = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#dd8800", width=1)
        )
        self.plot.addItem(self.cursor_line, ignoreBounds=True)
        self.region = pg.LinearRegionItem(brush=pg.mkBrush(80, 130, 180, 60))
        self.region.setZValue(-10)
        self.region.hide()
        self.plot.addItem(self.region, ignoreBounds=True)

        #: Per shown detection, the lines and regions marking its events.
        self._marks: dict[str, tuple[list, list]] = {}
        self._mark_styles: dict[str, tuple] = {}
        self._detection_problem = ""

        self.region.sigRegionChanged.connect(self._region_moved)
        self._viewbox.region_dragged.connect(self._region_dragged)
        self.plot.scene().sigMouseClicked.connect(self._clicked)
        self._viewbox.sigRangeChanged.connect(lambda *_: self._draw_distribution())

    def make_viewbox(self):
        self._viewbox = SelectionViewBox()
        return self._viewbox

    def make_side(self):
        side = pg.PlotWidget()
        side.setMouseEnabled(x=False, y=False)
        side.hideButtons()
        side.setMenuEnabled(False)
        side.showGrid(x=False, y=True, alpha=0.2)
        side.getPlotItem().hideAxis("left")
        # An axis below as tall as the chart's: y is linked by pixel, so the views must match.
        side.setLabel("bottom", "density")
        side.setYLink(self.plot)
        side.getViewBox().enableAutoRange(x=True, y=False)
        side.setToolTip("How the values on screen, or in the selection, are distributed")
        return side

    def set_distribution_shown(self, shown: bool) -> None:
        super().set_distribution_shown(shown)
        self._draw_distribution()

    # -- drawing -------------------------------------------------------

    def refresh(self, reasons: set[str], window=None) -> None:
        if reasons & {"data", "signals", "source"}:
            self._draw(window)
        if reasons & {"data", "signals", "source", "detections"}:
            self._draw_detections(window)
        if reasons & {"cursor", "data"}:
            self.cursor_line.setPos(self.session.cursor)
        if reasons & {"selection"}:
            self._follow_selection()
        if reasons & {"data", "signals", "source", "selection"}:
            self._draw_distribution()

    # -- which series are drawn --------------------------------------------

    def set_series_visible(self, name: str, visible: bool) -> None:
        """Show or hide one series; hidden, it leaves the autoscale too."""
        if visible:
            self.hidden_series.discard(name)
        else:
            self.hidden_series.add(name)
        curve = self._curves.get(name)
        if curve is None:
            return
        curve.setVisible(visible)
        self.legend.paint_entry(curve, visible)
        if visible:
            self._draw(self.session.window())      # it missed the ticks while hidden
        self._draw_distribution()
        # Fit what is left, on an axis left to autoscale (a user's zoom stays).
        self.plot.getViewBox().updateAutoRange()

    def solo_series(self, name: str) -> None:
        """Show `name` alone -- or, if it already is, show them all again."""
        alone = (name not in self.hidden_series
                 and all(n in self.hidden_series for n in self._curves if n != name))
        for other in list(self._curves):
            self.set_series_visible(other, alone or other == name)

    def _draw(self, window) -> None:
        if self.signal != self._hidden_for:          # another selection: all shown
            self._hidden_for = self.signal
            self.hidden_series.clear()
        names = self.signal_names()
        for gone in [n for n in self._curves if n not in names]:
            self.plot.removeItem(self._curves.pop(gone))
        self._window = window
        if window is None or window.empty:
            for curve in self._curves.values():
                curve.setData(np.zeros(0), np.zeros(0))
            return

        samples = 0
        for i, name in enumerate(names):
            try:
                own = window.home(name)          # each stream on its own times
                values = own[name]
            except Exception as exc:                          # noqa: BLE001
                self.status.setText(f"{name}: {exc}")
                continue
            samples = max(samples, len(own))
            curve = self._curves.get(name)
            if curve is None:
                curve = self.plot.plot(
                    pen=pg.mkPen(COLOURS[i % len(COLOURS)], width=1), name=name
                )
                # Peak-preserving downsampling and clipping to the view:
                # what makes a minute of data cost what a second does.
                curve.setDownsampling(auto=True, method="peak")
                curve.setClipToView(True)
                self._curves[name] = curve
                if name in self.hidden_series:
                    curve.setVisible(False)
                    self.legend.paint_entry(curve, False)
            if name in self.hidden_series:
                continue                             # not drawn while hidden
            curve.setData(own.time, np.asarray(values, dtype=float))
        units = self.session.signals.units_for(names)
        grouped = self.signal != ALL_CHANNELS and self.session.members(self.signal) is not None
        label = self.signal if grouped else ", ".join(names)[:40]
        self.plot.setLabel("left", label, units=units or None)
        self.status.setText(f"{samples} samples")

    # -- the distribution beside it ------------------------------------

    def distribution_span(self) -> tuple[float, float]:
        """The time the distribution counts: the selection, else what is on screen."""
        span = self.session.selection
        if span is not None:
            return (min(span), max(span))
        (lo, hi), _ = self._viewbox.viewRange()
        return (lo, hi)

    def densities(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Per series drawn, the bin edges and density beside the chart."""
        if self._window is None or self._window.empty:
            return {}
        a, b = self.distribution_span()
        _, (low, high) = self._viewbox.viewRange()
        out = {}
        for name in self._curves:
            if name in self.hidden_series:
                continue
            try:
                own = self._window.home(name)
                keep = (own.time >= a) & (own.time <= b)
                values = np.asarray(own[name], dtype=float)[keep]
            except Exception:                                  # noqa: BLE001
                continue                     # _draw has said why in the status
            out[name] = distribution(values, bins=DISTRIBUTION_BINS, low=low, high=high)
        return out

    def _draw_distribution(self) -> None:
        if not self.distribution_shown or self.collapsed:
            return                           # reopened, a pane is redrawn in full
        shown = self.densities()
        for gone in [n for n in self._densities if n not in shown]:
            for item in self._densities.pop(gone):
                self.side.removeItem(item)
        for name, (edges, density) in shown.items():
            items = self._densities.get(name)
            if items is None:
                colour = self._curves[name].opts["pen"].color()
                shade = pg.mkColor(colour)
                shade.setAlpha(60)
                bars = pg.BarGraphItem(x0=[], y0=[], width=[], height=[],
                                       brush=pg.mkBrush(shade), pen=pg.mkPen(None))
                outline = pg.PlotDataItem(pen=pg.mkPen(colour, width=1), connect="finite")
                self.side.addItem(bars)
                self.side.addItem(outline)
                items = self._densities[name] = (bars, outline)
            bars, outline = items
            bars.setOpts(x0=np.zeros(len(density)), y0=edges[:-1],
                         width=density, height=np.diff(edges))
            # The step outline, turned; empty bins left out, or it would draw the zero line.
            steps = np.repeat(np.where(density > 0, density, np.nan), 2)
            outline.setData(steps, np.column_stack([edges[:-1], edges[1:]]).ravel())

    # -- detections ----------------------------------------------------

    def _draw_detections(self, window) -> None:
        """Mark every shown event node's events on this pane.

        An event is a line at its timestamp; one from a pattern -- a
        double knock -- is also a shaded span from its start to its end,
        so it reads as one thing made of the events inside it.
        """
        names = self.session.detection_names
        # Colour by place among all of them, so hiding one recolours none.
        colours = {n: DETECTION_COLOURS[i % len(DETECTION_COLOURS)]
                   for i, n in enumerate(names)}
        names = [n for n in names if self.session.drawn(n)]
        for gone in [n for n in self._marks if n not in names]:
            self._resize_marks(gone, 0, 0, None, False)
            del self._marks[gone]
            self._mark_styles.pop(gone, None)
        if window is None:
            return
        problems = []
        shown = self.session.shown
        for name in names:
            node = shown.get(name)
            colour = colours[name]
            try:
                table = self.session.evaluator.evaluate(node, window).table
            except Exception as exc:                          # noqa: BLE001
                problems.append(f"{name}: {exc}")
                table = None
            if table is None or window.empty:
                self._resize_marks(name, 0, 0, colour, False)
                continue
            table = table.iloc[-MAX_MARKS:]
            pattern = node is not None and node.spec.pattern
            stamps = table["timestamp" if "timestamp" in table else "start_time"]
            lines, regions = self._resize_marks(
                name, len(table), len(table) if pattern else 0, colour, pattern)
            for line, t in zip(lines, stamps):
                line.setPos(float(t))
            for region, (a, b) in zip(regions, zip(table["start_time"], table["end_time"])):
                region.setRegion((float(a), float(b)))
        had_problem = bool(self._detection_problem)
        self._detection_problem = "; ".join(problems)
        if problems:
            self.status.setText(self._detection_problem[:120])
        elif had_problem:
            self.status.setText(f"{len(window)} samples")

    def _resize_marks(self, name, n_lines, n_regions, colour, pattern):
        lines, regions = self._marks.setdefault(name, ([], []))
        style = (colour, pattern)
        if colour is not None and self._mark_styles.get(name, style) != style:
            for item in lines + regions:
                self.plot.removeItem(item)
            lines.clear()
            regions.clear()
        if colour is not None:
            self._mark_styles[name] = style
        while len(lines) > n_lines:
            self.plot.removeItem(lines.pop())
        while len(regions) > n_regions:
            self.plot.removeItem(regions.pop())
        while len(lines) < n_lines:
            # Alternate heights, so the labels of close events do not collide.
            height = (0.82 if pattern else 0.95) - 0.05 * (len(lines) % 2)
            line = pg.InfiniteLine(
                angle=90, movable=False,
                pen=pg.mkPen(colour, width=2 if pattern else 1,
                             style=Qt.SolidLine if pattern else Qt.DashLine),
                label=name,
                labelOpts={"position": height, "color": colour, "movable": False},
            )
            line.setZValue(5)
            self.plot.addItem(line, ignoreBounds=True)
            lines.append(line)
        while len(regions) < n_regions:
            shade = pg.mkColor(colour)
            shade.setAlpha(50)
            region = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(shade),
                                         pen=pg.mkPen(colour, width=1))
            region.setZValue(-5)
            self.plot.addItem(region, ignoreBounds=True)
            regions.append(region)
        return lines, regions

    def marks(self) -> dict[str, dict]:
        """What is marked, per detection: for tests and for looking at."""
        return {
            name: {"lines": [line.value() for line in lines],
                   "regions": [tuple(r.getRegion()) for r in regions]}
            for name, (lines, regions) in self._marks.items()
        }

    def _follow_selection(self) -> None:
        span = self.session.selection
        if span is None:
            self.region.hide()
            return
        self._updating = True
        try:
            self.region.setRegion(span)
            self.region.show()
        finally:
            self._updating = False

    # -- what the user does to it --------------------------------------

    def _clicked(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        if self.legend.sceneBoundingRect().contains(event.scenePos()):
            return                                   # a click on the legend is its own
        point = self._viewbox.mapSceneToView(event.scenePos())
        self.session.set_cursor(float(point.x()))

    def _region_dragged(self, start: float, end: float, finished: bool) -> None:
        self.session.set_selection((start, end))

    def _region_moved(self) -> None:
        if self._updating or not self.region.isVisible():
            return
        self.session.set_selection(tuple(self.region.getRegion()))

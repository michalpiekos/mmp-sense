"""The window: panes over one time axis, a cursor, and a console.

The window owns no data and computes nothing. Every tick it polls the
session, advances playback, builds one `Window` for the whole buffer and
hands it to the panes, which redraw only what changed. Everything the
user does -- clicking, dragging, pressing `l` -- is a call on the
session, so the console sees it too.

Keys:

    h / l        move the cursor by a quarter of the analysis window
    H / L        move it by a whole window
    Ctrl+h / l   move it by one sample
    n / p        jump to the next or previous peak of the focus signal
    space        play or pause the focus signal from the cursor, within the
                 selection if there is one; only its ticked bands if any
    b            add a frequency band, typed in Hz
    f            follow the newest data again
    v            select the analysis window around the cursor
    Esc          clear the selection
    [ / ]        halve or double the analysis window
    d            show or hide the distribution beside the time charts
    j / k        move the frequency cursor one bin down or up
    J / K        move it to the next peak below or above

Shift-drag on a time chart selects time; on a spectrum or PSD it adds a
frequency band, and a band's edges drag to change it.

A click on a spectrum or PSD puts the frequency cursor on the nearest
peak, Alt+click exactly where clicked; on the spectrogram a click sets
both cursors. Each frequency pane reads out its value at the cursor.

Double-click a pane's header to let it fill the window; again to restore.
"""

from __future__ import annotations

import logging
import math
import time as _time
from decimal import ROUND_DOWN, Decimal

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QPalette, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QLabel,
    QMainWindow,
    QMenu,
    QSplitter,
    QToolBar,
    QToolButton,
    QWidget,
)

from ..core.query import fade_in
from ..dsp.peaks import next_peak, step_bin
from .views import views
from .audio import make_player
from .bridge import SessionBridge
from .console import make_console
from .panes import Pane, SignalPane, SpectrogramPane, SpectrumPane, XYPane
from .panes.base import ALL_CHANNELS
from .panes.spectrum_pane import PEAK_HOLD_S
from .panes.base import QWIDGETSIZE_MAX
from .widgets import BandDock, DetectorDock, SignalDock
from .widgets.dock_title import DockTitle

__all__ = ["MainWindow"]

log = logging.getLogger("mmp_sense.ui")

#: How often the window looks at the session. Four times this would still
#: be smooth; this is what a live chart wants and a laptop can afford.
TICK_MS = 50

SPEEDS = ["0.25x", "0.5x", "1x", "2x", "4x"]
#: The Amp box: what each entry asks of the sound.
AMPS = {"amp off": None, "amp auto": "auto", "amp +12 dB": 12.0, "amp +24 dB": 24.0,
        "amp +40 dB": 40.0}

#: Everything a pane can be told changed: what one opened again needs.
ALL_REASONS = frozenset({"data", "signals", "cursor", "selection", "analysis", "detections",
                         "bands", "frequency"})


def _decimal(seconds: float) -> Decimal:
    # The shortest printed form, so 12.004517 is not 12.00451699999...
    return Decimal(repr(float(seconds)))


def format_seconds(seconds) -> str:
    """Seconds to the microsecond, truncated, as `3.250,123`: a comma after the milliseconds."""
    value = seconds if isinstance(seconds, Decimal) else _decimal(seconds)
    if not value.is_finite():
        return str(float(value))
    micros = int(value.quantize(Decimal("0.000001"), rounding=ROUND_DOWN) * 1_000_000)
    sign = "-" if micros < 0 else ""
    whole, rest = divmod(abs(micros), 1_000_000)
    millis, micro = divmod(rest, 1000)
    return f"{sign}{whole}.{millis:03d},{micro:03d}"


class MainWindow(QMainWindow):
    """One session, seen four ways at once."""

    def __init__(
        self,
        session,
        *,
        window_s: float = 30.0,
        interval_ms: int = TICK_MS,
        console: bool = True,
        listen: bool | str = False,
        peak_hold_s: float = PEAK_HOLD_S,
        player=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        #: What Play sounds through.
        self.player = player if player is not None else make_player(self)
        self.player.finished.connect(self._sound_finished)
        #: The signal being heard, or None.
        self.hearing: str | None = None
        #: How many frequency bands it is heard through; 0 for all of it.
        self.hearing_bands = 0
        #: The stretch being played: (start, end).
        self.playing_stretch: tuple[float, float] | None = None
        #: Why Play is silent, or "".
        self.silent_because = ""
        self.window_s = float(window_s)
        self.peak_hold_s = max(0.0, float(peak_hold_s))
        self.follow = True
        #: What the spectrum, the spectrogram and peak-jumping look at.
        self.focus_signal = session.channels[0]
        self.panes: list[Pane] = []
        self._ranged_from = None
        self.expanded: Pane | None = None
        self._sizes_before_expand: list[int] = []
        self._reasons: set[str] = {"data", "signals", "cursor", "selection"}
        self._busy = False
        self._tick_times: list[float] = []
        self._shutdown_console = lambda: None
        self.listener = None
        self.console = None
        self.docks: list[QDockWidget] = []

        self.setWindowTitle(f"mmp-sense -- {session.dataset}")
        self.resize(1280, 860)

        self.splitter = QSplitter(Qt.Vertical)
        self.setCentralWidget(self.splitter)

        self._build_toolbar()
        self._build_brand()
        self._build_docks(console)
        # Empty until a band is added, which opens it.
        self.set_dock_collapsed(self.bands_dock, True)
        if hasattr(self, "console_dock"):
            # Always there, out of the way until wanted.
            self.set_dock_collapsed(self.console_dock, True)
            self.console_dock.open_size = None     # not laid out yet: no size to keep
        if listen:
            self._listen(None if listen is True else listen)
        self._build_panes()
        self._build_shortcuts()

        self.bridge = SessionBridge(session, self)
        self.bridge.data_arrived.connect(lambda _p: self._mark("data"))
        self.bridge.cursor_moved.connect(lambda _t: self._mark("cursor"))
        self.bridge.frequency_moved.connect(lambda _f: self._mark("frequency"))
        self.bridge.selection_changed.connect(lambda _s: self._mark("selection"))
        self.bridge.analysis_changed.connect(self._analysis_changed)
        self.bridge.signals_changed.connect(lambda _p: self._mark("signals"))
        self.bridge.source_changed.connect(lambda _p: self._mark("source", "signals"))
        self.bridge.playback_changed.connect(self._playback_changed)
        self.bridge.intake_changed.connect(self._intake_changed)
        self.bridge.chart_requested.connect(self._chart_requested)
        self.bridge.plots_changed.connect(self._plots_changed)
        self.bridge.detections_changed.connect(lambda _p: self._mark("detections"))
        self.bridge.bands_changed.connect(lambda _p: self._mark("bands"))

        self.timer = QTimer(self)
        self.timer.setInterval(int(interval_ms))
        self.timer.timeout.connect(self.tick)
        self.timer.start()

    # -- building ------------------------------------------------------

    def _build_brand(self) -> None:
        self.brand = QLabel("MMP Systems")
        font = self.brand.font()
        font.setPointSizeF(font.pointSizeF() * 0.85)
        self.brand.setFont(font)
        self.brand.setForegroundRole(QPalette.PlaceholderText)
        self.brand.setContentsMargins(0, 0, 6, 0)
        self.statusBar().addPermanentWidget(self.brand)

    def _build_toolbar(self) -> None:
        bar = QToolBar("Controls")
        bar.setMovable(False)
        self.addToolBar(bar)

        self.start_action = QAction("Start", self)
        self.start_action.setToolTip("Take samples again, from an empty buffer")
        self.start_action.triggered.connect(self.start_intake)
        bar.addAction(self.start_action)

        self.stop_action = QAction("Stop", self)
        self.stop_action.setToolTip("Stop taking samples; what is held stays")
        self.stop_action.setEnabled(self.session.running)
        self.stop_action.triggered.connect(self.stop_intake)
        bar.addAction(self.stop_action)

        bar.addSeparator()
        self.play_action = QAction("Play", self)
        self.play_action.setToolTip(
            "Play the focus signal from the cursor, within the selection if there is one; "
            "only its ticked bands if any (space)")
        self.play_action.triggered.connect(self.toggle_play)
        bar.addAction(self.play_action)

        self.speed = QComboBox()
        self.speed.addItems(SPEEDS)
        self.speed.setCurrentText("1x")
        self.speed.setToolTip("Replay speed")
        bar.addWidget(self.speed)

        self.amp = QComboBox()
        self.amp.addItems(AMPS)
        self.amp.setCurrentText("amp auto")
        self.amp.setToolTip("How loud Play is: auto raises a weak recording to a comfortable "
                            "level, and loud moments are held down")
        self.amp.currentTextChanged.connect(self._amp_changed)
        bar.addWidget(self.amp)

        self.follow_action = QAction("Follow", self)
        self.follow_action.setCheckable(True)
        self.follow_action.setChecked(True)
        self.follow_action.setToolTip("Keep the newest data in view (f)")
        self.follow_action.toggled.connect(self._follow_toggled)
        bar.addAction(self.follow_action)

        bar.addSeparator()
        bar.addWidget(QLabel(" window "))
        self.analysis_spin = QDoubleSpinBox()
        self.analysis_spin.setRange(0.005, 30.0)
        self.analysis_spin.setDecimals(3)
        self.analysis_spin.setSingleStep(0.05)
        self.analysis_spin.setValue(self.session.analysis_window)
        self.analysis_spin.setSuffix(" s")
        self.analysis_spin.setToolTip("Analysis window: what the FFT is taken over")
        self.analysis_spin.valueChanged.connect(self._analysis_spin_changed)
        bar.addWidget(self.analysis_spin)

        bar.addWidget(QLabel(" span "))
        self.span_spin = QDoubleSpinBox()
        self.span_spin.setRange(0.1, 600.0)
        self.span_spin.setDecimals(1)
        self.span_spin.setSingleStep(5.0)
        self.span_spin.setValue(self.window_s)
        self.span_spin.setSuffix(" s")
        self.span_spin.setToolTip("How much history is on screen")
        self.span_spin.valueChanged.connect(self._span_changed)
        bar.addWidget(self.span_spin)

        bar.addWidget(QLabel(" hold "))
        self.hold_spin = QDoubleSpinBox()
        self.hold_spin.setRange(0.0, 60.0)
        self.hold_spin.setDecimals(1)
        self.hold_spin.setSingleStep(0.5)
        self.hold_spin.setValue(self.peak_hold_s)
        self.hold_spin.setSuffix(" s")
        self.hold_spin.setToolTip(
            "How long the spectrum's peak markers stay; 0 hides them"
        )
        self.hold_spin.valueChanged.connect(self.set_peak_hold)
        bar.addWidget(self.hold_spin)

        bar.addSeparator()
        previous = QAction("< peak", self)
        previous.setToolTip("Previous peak of the focus signal (p)")
        previous.triggered.connect(lambda: self.jump_peak(-1))
        bar.addAction(previous)
        following = QAction("peak >", self)
        following.setToolTip("Next peak of the focus signal (n)")
        following.triggered.connect(lambda: self.jump_peak(+1))
        bar.addAction(following)

        bar.addSeparator()
        add = QToolButton()
        add.setText("Add pane")
        add.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(add)
        menu.setToolTipsVisible(True)
        for name, view in sorted(views().items()):
            action = menu.addAction(name)
            action.setToolTip(view.description)
            action.triggered.connect(lambda _c=False, n=name: self.add_pane(n))
        add.setMenu(menu)
        bar.addWidget(add)
        tidy = QAction("Tidy", self)
        tidy.setToolTip("Make the open charts the same height (=)")
        tidy.triggered.connect(self.tidy_panes)
        bar.addAction(tidy)
        self.distribution_action = QAction("PDF", self)
        self.distribution_action.setCheckable(True)
        self.distribution_action.setToolTip(
            "Beside each chart, how its values on screen -- or in the selection -- "
            "are distributed (d)")
        self.distribution_action.toggled.connect(self._distribution_toggled)
        bar.addAction(self.distribution_action)
        self.harmonics_action = QAction("k·f", self)
        self.harmonics_action.setCheckable(True)
        self.harmonics_action.setToolTip(
            "Mark the frequency cursor's harmonics, 2f, 3f, ..., on the spectrum and PSD")
        self.harmonics_action.toggled.connect(self._harmonics_toggled)
        bar.addAction(self.harmonics_action)

    def _build_docks(self, console: bool) -> None:
        self.signal_dock = SignalDock(self.session)
        self.signal_dock.plot_requested.connect(self.plot_signal)
        self.signal_dock.drop_requested.connect(self._drop_signal)
        self.signal_dock.op_requested.connect(self.add_op)
        self.signals_dock = self._make_dock("Signals", self.signal_dock, Qt.LeftDockWidgetArea)

        self.detector_dock = DetectorDock(self.session)
        self.detectors_dock = self._make_dock(
            "Detectors", self.detector_dock, Qt.LeftDockWidgetArea)

        self.band_dock = BandDock(self.session, self.nyquist)
        self.bands_dock = self._make_dock("Bands", self.band_dock, Qt.LeftDockWidgetArea)

        if not console:
            return
        widget, shutdown = make_console(self._console_namespace())
        self._shutdown_console = shutdown
        if widget is None:
            widget = QLabel(
                "  No console: install qtconsole and ipykernel for one."
            )
        self.console = widget if widget is not None and hasattr(widget, "execute") else None
        self.console_dock = self._make_dock("Python", widget, Qt.BottomDockWidgetArea)

    # -- docks: collapsing to the title ----------------------------------

    def _make_dock(self, title: str, widget, area) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        dock.collapsed = False
        dock.open_size = None
        bar = DockTitle(dock)
        bar.collapse_clicked.connect(lambda d=dock: self.set_dock_collapsed(d, not d.collapsed))
        dock.setTitleBarWidget(bar)
        dock.topLevelChanged.connect(lambda floating, d=dock: self._dock_floated(d, floating))
        self.addDockWidget(area, dock)
        self.docks.append(dock)
        return dock

    def _dock_floated(self, dock: QDockWidget, floating: bool) -> None:
        # A floating dock is a window of its own; collapsing it means nothing.
        if floating:
            self.set_dock_collapsed(dock, False)
        dock.titleBarWidget().collapse_button.setVisible(not floating)

    def set_dock_collapsed(self, dock: QDockWidget, collapsed: bool) -> None:
        """Shrink a dock to its title bar, or open it; it keeps working.

        The left column, once every dock in it is collapsed, becomes a thin
        strip of side-on titles, so the charts get its width.
        """
        if collapsed == dock.collapsed:
            return
        if collapsed and dock.isFloating():
            return
        area = self.dockWidgetArea(dock)
        across = area in (Qt.LeftDockWidgetArea, Qt.RightDockWidgetArea)
        column = [d for d in self.docks if self.dockWidgetArea(d) == area
                  and not d.isFloating() and d.isVisible()]
        bar = dock.titleBarWidget()
        if collapsed:
            dock.open_size = dock.height() if not across or len(column) > 1 else None
            if across:
                # A collapsed dock's width says nothing of the column's.
                open_widths = [d.width() for d in column if not d.collapsed]
                if open_widths:
                    self._column_width = max(open_widths)
            dock.widget().hide()
            dock.collapsed = True
            bar.set_collapsed(True)
            dock.setMaximumHeight(bar.sizeHint().height())
            if across and column and all(d.collapsed for d in column):
                self._strip(column, True)
        else:
            stripped = across and column and all(d.collapsed for d in column)
            if stripped:
                self._strip(column, False)
            dock.collapsed = False
            bar.set_collapsed(False)
            dock.setMaximumHeight(QWIDGETSIZE_MAX)
            dock.widget().show()
            if across:
                # Qt pinned its width to the title's while it was empty; unpin.
                dock.setMinimumWidth(0)
                dock.setMaximumWidth(QWIDGETSIZE_MAX)
            if stripped:
                self.resizeDocks([dock], [getattr(self, "_column_width", 260)], Qt.Horizontal)
            if dock.open_size:
                self.resizeDocks([dock], [dock.open_size], Qt.Vertical)
            elif not across:
                self.resizeDocks([dock], [self.height() // 3], Qt.Vertical)
            if dock is getattr(self, "console_dock", None):
                bar.set_error(None)         # seen now
            if dock is getattr(self, "detectors_dock", None):
                self.detector_dock.refresh(self.session.window())

    def _strip(self, column: list, on: bool) -> None:
        """Turn a side column of collapsed docks into a thin strip, or back."""
        for d in column:
            bar = d.titleBarWidget()
            features = d.features()
            if on:
                d.setMaximumHeight(QWIDGETSIZE_MAX)
                d.setFeatures(features | QDockWidget.DockWidgetVerticalTitleBar)
                bar.set_vertical(True)
                d.setMaximumWidth(bar.sizeHint().width())
            else:
                d.setFeatures(features & ~QDockWidget.DockWidgetVerticalTitleBar)
                bar.set_vertical(False)
                d.setMaximumWidth(QWIDGETSIZE_MAX)
                d.setMaximumHeight(bar.sizeHint().height())

    def _listen(self, path=None) -> None:
        """Take code for the console on a socket; see `mmp_sense.ui.remote`."""
        if self.console is None:
            log.warning("--listen needs the console; not listening")
            return
        from .remote import ConsoleListener, ConsoleRunner

        runner = ConsoleRunner(self.console, self)
        runner.finished.connect(self._sent_finished)
        self.listener = ConsoleListener(runner, path, self).start()

    def _sent_finished(self, ok: bool, error: str) -> None:
        """A failed send turns the console's title red until it is seen:
        the console opened, or -- open already -- a send that works."""
        bar = self.console_dock.titleBarWidget()
        if not ok:
            bar.set_error(error)
        elif not self.console_dock.collapsed:
            bar.set_error(None)

    def _console_namespace(self) -> dict:
        import mmp_sense

        namespace = {
            "mmp_sense": mmp_sense,
            "np": np,
            "session": self.session,
            "pd": pd,
            "run": lambda path: mmp_sense.run_script(path, self.session),
            "show": self.session.show,
            "signal": mmp_sense.signal,
            "ui": self,
        }
        # So a script from an editor can name these: `from mmp_sense.live import ...`.
        from .. import live

        live.attach(namespace)
        return namespace

    def _build_panes(self) -> None:
        self.add_pane("signal", ALL_CHANNELS)
        self.add_pane("spectrum", self.focus_signal)
        self.add_pane("spectrogram", self.focus_signal)
        self.splitter.setSizes([3, 2, 2])

    def _build_shortcuts(self) -> None:
        def bind(keys: str, slot) -> None:
            QShortcut(QKeySequence(keys), self, activated=slot)

        bind("h", lambda: self._step(-0.25, "window"))
        bind("l", lambda: self._step(+0.25, "window"))
        bind("Shift+h", lambda: self._step(-1.0, "window"))
        bind("Shift+l", lambda: self._step(+1.0, "window"))
        bind("Ctrl+h", lambda: self._step(-1.0, "sample"))
        bind("Ctrl+l", lambda: self._step(+1.0, "sample"))
        bind("n", lambda: self.jump_peak(+1))
        bind("p", lambda: self.jump_peak(-1))
        bind("Space", self.toggle_play)
        bind("f", lambda: self.follow_action.setChecked(True))
        bind("v", self.select_analysis_window)
        bind("Escape", self.session.clear_selection)
        bind("[", lambda: self._scale_analysis(0.5))
        bind("]", lambda: self._scale_analysis(2.0))
        bind("=", self.tidy_panes)
        bind("d", self.distribution_action.toggle)
        bind("b", self.band_dock.add)
        bind("j", lambda: self.step_frequency(-1))
        bind("k", lambda: self.step_frequency(+1))
        bind("Shift+j", lambda: self.step_frequency(-1, peak=True))
        bind("Shift+k", lambda: self.step_frequency(+1, peak=True))

    # -- panes ---------------------------------------------------------

    def add_pane(self, kind: str = "signal", signal: str | None = None) -> Pane:
        self.restore_panes()
        factory = views()[kind].factory
        if signal is None:
            signal = self.focus_signal
        pane = factory(self.session, signal)
        if hasattr(pane, "peak_hold_s"):
            pane.peak_hold_s = self.peak_hold_s
        pane.closed.connect(self.remove_pane)
        pane.expand_toggled.connect(self.toggle_expanded)
        pane.collapse_toggled.connect(self._pane_collapse_toggled)
        pane.signal_changed.connect(lambda name, p=pane: self._pane_retargeted(p, name))
        pane.set_distribution_shown(self.distribution_action.isChecked())
        if hasattr(pane, "set_harmonics_shown"):
            pane.set_harmonics_shown(self.harmonics_action.isChecked())
        self.splitter.addWidget(pane)
        self.panes.append(pane)
        self._link_time_axes()
        pane.refresh({"data", "signals", "cursor", "selection", "analysis"},
                     self.session.window())
        return pane

    def remove_pane(self, pane: Pane) -> None:
        if pane not in self.panes:
            return
        self.restore_panes()
        self.panes.remove(pane)
        if pane.time_axis:
            # Unlink before it goes, and relink the rest to whichever
            # pane is first now: a link to a deleted plot is a crash.
            pane.plot.setXLink(None)
            for other in self.panes:
                if other.time_axis:
                    other.plot.setXLink(None)
        pane.setParent(None)
        pane.deleteLater()
        self._link_time_axes()

    def toggle_expanded(self, pane: Pane) -> None:
        """Let `pane` fill the window, or put every pane back if one does."""
        if self.expanded is not None:
            self.restore_panes()
            return
        if pane not in self.panes:
            return
        if pane.collapsed:
            pane.set_collapsed(False)
        self._sizes_before_expand = self.splitter.sizes()
        # Collapsed rather than hidden: the others keep their width and
        # place, which is what the pixel-mapped time-axis links rely on.
        self.splitter.setSizes([
            sum(self._sizes_before_expand) if other is pane else 0
            for other in self.panes
        ])
        self.expanded = pane

    def restore_panes(self) -> None:
        if self.expanded is None:
            return
        self.expanded = None
        self.splitter.setSizes(self._sizes_before_expand)

    def tidy_panes(self) -> None:
        """Give every open chart the same height; a collapsed one keeps its
        title bar, and an expanded one is put back first."""
        self.restore_panes()
        sizes = self.splitter.sizes()
        if not self.panes or not sum(sizes):
            return
        opened = [i for i, p in enumerate(self.panes) if not p.collapsed]
        for pane in self.panes:
            pane._open_ratio = 1.0                  # reopened, it takes an equal share
        heights = {i: p.header.sizeHint().height()
                   for i, p in enumerate(self.panes) if p.collapsed}
        room = sum(sizes) - sum(heights.values())
        if opened:
            share, extra = divmod(max(0, room), len(opened))
            heights |= {i: share + (1 if k < extra else 0) for k, i in enumerate(opened)}
        self.splitter.setSizes([heights[i] for i in range(len(self.panes))])

    def show_distribution(self, on: bool = True) -> None:
        """Show, beside every time chart, the distribution of its values on
        screen -- or in the selection -- or take it away."""
        self.distribution_action.setChecked(bool(on))

    def _distribution_toggled(self, on: bool) -> None:
        for pane in self.panes:
            pane.set_distribution_shown(on)

    def _harmonics_toggled(self, on: bool) -> None:
        for pane in self.panes:
            if hasattr(pane, "set_harmonics_shown"):
                pane.set_harmonics_shown(on)

    def _pane_collapse_toggled(self, pane: Pane) -> None:
        """Give a collapsed pane's height to the open ones, or take it back.

        A pane remembers its height relative to the open ones, not in
        pixels: the room may have grown or shrunk while it was collapsed.
        """
        if self.expanded is not None:
            self.restore_panes()
        sizes = self.splitter.sizes()
        index = self.panes.index(pane)
        header = pane.header.sizeHint().height()
        others = [i for i, p in enumerate(self.panes) if i != index and not p.collapsed]
        pool = sum(sizes[i] for i in others)
        if pane.collapsed:
            mean = pool / len(others) if others and pool else 0
            pane._open_ratio = sizes[index] / mean if mean else 1.0
            wanted = header
        else:
            ratio = getattr(pane, "_open_ratio", 1.0)
            # Its share once the others have made room: w = r (P + h - w) / n.
            room = pool + sizes[index]
            wanted = round(ratio * room / (len(others) + ratio)) if others else room
        change = wanted - sizes[index]
        if others and pool > 0:
            for i in others:
                sizes[i] = max(0, sizes[i] - round(change * sizes[i] / pool))
        sizes[index] = wanted
        self.splitter.setSizes(sizes)
        if not pane.collapsed:
            pane.refresh(ALL_REASONS, self.session.window())

    def plot_signal(self, name: str) -> None:
        """Show a signal or group, reusing a pane that already draws it."""
        if name not in self.session.signals and self.session.members(name) is None:
            return
        self.focus_signal = name
        for pane in self._signal_panes():
            if pane.signal == name:
                return
        self.add_pane("signal", name)

    def _signal_panes(self) -> list[Pane]:
        return [p for p in self.panes if not isinstance(p, XYPane)]

    def _pane_retargeted(self, pane: Pane, name: str) -> None:
        if pane.time_axis and name in self.session.signals:
            self.focus_signal = name

    def _link_time_axes(self) -> None:
        """One time axis for every pane that has one."""
        time_panes = [p for p in self.panes if p.time_axis]
        if not time_panes:
            self._ranged_from = None
            return
        first = time_panes[0]
        viewbox = first.plot.getViewBox()
        if self._ranged_from is not viewbox:
            if self._ranged_from is not None:
                try:
                    self._ranged_from.sigRangeChangedManually.disconnect(
                        self._user_ranged
                    )
                except (RuntimeError, TypeError):
                    pass          # it went with its pane
            viewbox.sigRangeChangedManually.connect(self._user_ranged)
            self._ranged_from = viewbox
        for pane in time_panes[1:]:
            pane.plot.setXLink(first.plot)

    def _user_ranged(self, *_args) -> None:
        # Panning or zooming is a decision; the window stops scrolling
        # out from under it until Follow is asked for again.
        if self.follow:
            self.follow_action.setChecked(False)

    # -- the tick ------------------------------------------------------

    def tick(self) -> None:
        if self._busy:
            # A tick that overruns must not start another; the next one
            # will catch up, and two ticks at once would only make it
            # worse.
            return
        self._busy = True
        started = _time.perf_counter()
        try:
            if self.session.poll():
                self._reasons.add("data")
            if self.session.playing:
                self.session.tick()
            self.refresh()
        except Exception as exc:                              # noqa: BLE001
            # A failing tick would otherwise fail twenty times a second.
            # Say so once and carry on; the next tick may well work.
            log.exception("tick failed")
            self.statusBar().showMessage(f"tick failed: {exc}", 5000)
        finally:
            self._busy = False
            self._tick_times.append(_time.perf_counter() - started)
            del self._tick_times[:-200]

    def refresh(self, reasons: set[str] | None = None) -> None:
        reasons = set(reasons or self._reasons)
        self._reasons.clear()
        if not reasons:
            return
        if "signals" in reasons or "source" in reasons:
            self.signal_dock.refresh()
            for pane in self.panes:
                pane.update_choices()
        # One window for the tick: a signal several panes draw is
        # computed once. Only the panes that need samples ask for it.
        window = (
            self.session.window()
            if reasons & {"data", "signals", "source", "detections"} else None
        )
        for pane in self.panes:
            if not pane.collapsed:          # redrawn in full when opened
                pane.refresh(reasons, window)
        if "bands" in reasons:
            if self.session.bands and not self.band_dock.tree.topLevelItemCount():
                self.set_dock_collapsed(self.bands_dock, False)
                if not self.bands_dock.isFloating():
                    self.resizeDocks([self.bands_dock], [self.height() // 4], Qt.Vertical)
            self.band_dock.refresh()
        if window is not None and not self.detectors_dock.collapsed:
            self.detector_dock.refresh(window)
        if "data" in reasons:
            self._apply_view()
        if "cursor" in reasons:
            self._keep_cursor_visible()
        self._update_status()

    def _mark(self, *reasons: str) -> None:
        self._reasons.update(reasons)

    def _apply_view(self) -> None:
        if not self.follow:
            return
        time_panes = [p for p in self.panes if p.time_axis]
        if not time_panes:
            return
        end = self.session.span[1]
        viewbox = time_panes[0].plot.getViewBox()
        viewbox.setXRange(end - self.window_s, end, padding=0)

    def _keep_cursor_visible(self) -> None:
        if self.follow:
            return
        time_panes = [p for p in self.panes if p.time_axis]
        if not time_panes:
            return
        viewbox = time_panes[0].plot.getViewBox()
        (lo, hi), _ = viewbox.viewRange()
        cursor = self.session.cursor
        margin = (hi - lo) * 0.05
        if lo + margin <= cursor <= hi - margin:
            return
        half = (hi - lo) / 2.0
        viewbox.setXRange(cursor - half, cursor + half, padding=0)

    def _update_status(self) -> None:
        session = self.session
        selection = session.selection
        streams = list(session.streams.values())
        if len(streams) == 1:
            parts = [f"{session.dataset}", f"{len(session.buffer)} samples",
                     f"{session.rate:.6g} Hz"]
        else:
            parts = [f"{s.name}: {len(s.buffer)} samples, {s.buffer.rate:.6g} Hz"
                     for s in streams]
        parts.append(f"cursor {format_seconds(session.cursor)} s")
        if session.frequency_cursor is not None:
            parts.append(f"frequency {session.frequency_cursor:.5g} Hz")
        if selection:
            start, end = selection
            length = (
                _decimal(end) - _decimal(start)
                if math.isfinite(start) and math.isfinite(end) else end - start
            )
            parts.append(
                f"selection {format_seconds(start)} – {format_seconds(end)} s "
                f"({format_seconds(length)} s)"
            )
        if not session.running:
            parts.append("stopped")
        if self.hearing is not None:
            heard = f"playing {self.hearing} {session.speed:g}x, {self.amp.currentText()}"
            if self.hearing_bands:
                heard += f", {self.hearing_bands} band{'s' if self.hearing_bands > 1 else ''}"
            parts.append(heard)
        elif session.playing:
            silent = f", silent: {self.silent_because}" if self.silent_because else ""
            parts.append(f"playing {session.speed:g}x{silent}")
        elif not self.follow:
            parts.append("paused view")
        if self._tick_times:
            parts.append(f"tick {1000 * np.median(self._tick_times):.1f} ms")
        self.statusBar().showMessage("   ".join(parts))

    @property
    def tick_times(self) -> list[float]:
        """Recent tick durations in seconds -- what the benchmark reads."""
        return list(self._tick_times)

    # -- what the controls do ------------------------------------------

    def _step(self, amount: float, unit: str) -> None:
        self.follow_action.setChecked(False)
        self.session.step_cursor(amount, unit=unit)

    def _scale_analysis(self, factor: float) -> None:
        self.analysis_spin.setValue(self.session.analysis_window * factor)

    def _analysis_spin_changed(self, value: float) -> None:
        if abs(value - self.session.analysis_window) > 1e-9:
            self.session.analysis_window = value

    def _analysis_changed(self, value: float) -> None:
        if abs(self.analysis_spin.value() - value) > 1e-9:
            self.analysis_spin.setValue(value)
        self._mark("analysis")

    def set_peak_hold(self, seconds: float) -> None:
        """How long every spectrum pane holds its peak markers."""
        self.peak_hold_s = max(0.0, float(seconds))
        if abs(self.hold_spin.value() - self.peak_hold_s) > 1e-9:
            self.hold_spin.setValue(self.peak_hold_s)
        for pane in self.panes:
            if hasattr(pane, "peak_hold_s"):
                pane.peak_hold_s = self.peak_hold_s

    def _span_changed(self, value: float) -> None:
        self.window_s = float(value)
        self._apply_view()

    def _follow_toggled(self, on: bool) -> None:
        self.follow = bool(on)
        if on:
            self.session.follow_live()
            self._apply_view()

    def start_intake(self) -> None:
        try:
            self.session.start_intake()
        except Exception as exc:                              # noqa: BLE001
            # A source that cannot open again -- a device that went
            # away -- must not take the window with it.
            log.exception("starting the source failed")
            self.statusBar().showMessage(f"cannot start: {exc}", 5000)

    def stop_intake(self) -> None:
        self.session.stop_intake()

    def _intake_changed(self, payload: dict) -> None:
        running = bool(payload.get("running"))
        self.stop_action.setEnabled(running)
        if running:
            # A fresh capture starts at the beginning of an empty buffer,
            # and a view left over the last one would show nothing.
            self.follow_action.setChecked(True)
        self._mark("data", "cursor", "selection")

    def toggle_play(self) -> None:
        """Play the focus signal from the cursor, or stop it."""
        if self.session.playing or self.hearing is not None:
            self.player.stop()
            self.session.pause()
            return
        session = self.session
        first, last = session.selection or session.span
        start = session.cursor if first <= session.cursor < last else first
        self.follow_action.setChecked(False)
        self.playing_stretch = (first, last)
        self._play_from(start, float(self.speed.currentText().rstrip("x")))

    def _play_from(self, start: float, speed: float) -> None:
        self.silent_because = self._start_sound(start, speed)
        self.session.play(start, speed=speed, until=self.playing_stretch[1])
        self._show_play_state()
        self._update_status()

    def _start_sound(self, start: float, speed: float) -> str:
        """Sound `start` onwards of the stretch; returns why not, or ""."""
        self.player.stop(quietly=True)
        self.hearing = None
        name = self.heard_signal()
        if name is None:
            return "no signal to hear"
        if not self.player.available:
            return self.player.problem
        first, last = self.playing_stretch
        bands = self.session.heard_bands
        limit = self.nyquist()
        if bands and limit and all(low >= limit[1] for low, _ in bands):
            return f"{name} holds nothing above {limit[1]:.6g} Hz: no ticked band is in it"
        rate = self.player.rate
        try:
            samples = self.session.audio(name, first, last, rate=rate, speed=speed,
                                         bands=bands or None, amp=AMPS[self.amp.currentText()])
        except Exception as exc:                              # noqa: BLE001
            return f"cannot play {name}: {exc}"
        own = self.session.streams[self.session.signals.stream_of(name)].buffer.t_start
        ahead = round((start - max(first, own)) / speed * rate)
        if ahead > 0:
            samples = fade_in(samples[ahead:], rate)
        if not len(samples):
            return f"{name}: nothing to play there"
        self.player.play(samples)
        self.hearing = name
        self.hearing_bands = len(bands)
        return ""

    def _amp_changed(self) -> None:
        if self.hearing is not None and self.session.playing:
            self._play_from(self.session.cursor, self.session.speed)

    def _show_play_state(self) -> None:
        playing = self.session.playing or self.hearing is not None
        self.play_action.setText("Pause" if playing else "Play")

    def _playback_changed(self, payload: dict) -> None:
        self._show_play_state()
        if self.hearing is not None and not payload.get("playing") \
                and not payload.get("finished"):
            # Paused. At its end the sound, a little behind the cursor, plays out.
            self.player.stop()
        self._mark("cursor")

    def heard_signal(self) -> str | None:
        """What Play sounds: the focus signal, or a group's first member."""
        members = self.session.members(self.focus_signal)
        name = members[0] if members else self.focus_signal
        return name if name in self.session.signals else None

    def _sound_finished(self) -> None:
        if self.hearing is None:
            return
        self.hearing = None
        if self.session.playing:
            self.session.pause()
        self._show_play_state()
        self._update_status()

    def nyquist(self) -> tuple[str, float] | None:
        """The heard signal and the highest frequency it holds."""
        name = self.heard_signal()
        if name is None:
            return None
        try:
            return name, self.session.rate_of(name) / 2.0
        except Exception:                                     # noqa: BLE001
            return None

    def select_analysis_window(self) -> None:
        half = self.session.analysis_window / 2.0
        self.session.set_selection(
            (self.session.cursor - half, self.session.cursor + half)
        )

    def jump_peak(self, direction: int) -> None:
        """Put the cursor on the next peak of the focus signal."""
        name = self.focus_signal
        if name not in self.session.signals:
            return
        times = peak_times(self.session.values(name, *self.session.span),
                           max(self.session.analysis_window, 0.02))
        if not times:
            self.statusBar().showMessage(f"no peaks in {name}", 2000)
            return
        cursor = self.session.cursor
        step = 1e-6
        if direction > 0:
            later = [t for t in times if t > cursor + step]
            target = later[0] if later else times[0]
        else:
            earlier = [t for t in times if t < cursor - step]
            target = earlier[-1] if earlier else times[-1]
        self.follow_action.setChecked(False)
        self.session.set_cursor(target)

    def frequency_values(self):
        """(bin frequencies, dB) the frequency keys step through: of the
        first open spectrum or PSD, else of a spectrogram; None if none."""
        for kinds in (("spectrum", "psd"), ("spectrogram",)):
            for pane in self.panes:
                if pane.kind in kinds and not pane.collapsed:
                    values = pane.frequency_values()
                    if values is not None and len(values[0]) > 1:
                        return values
        return None

    def step_frequency(self, direction: int, peak: bool = False) -> None:
        """Move the frequency cursor a bin, or to the next peak, up or down."""
        values = self.frequency_values()
        if values is None:
            self.statusBar().showMessage("no frequency chart to step through", 2000)
            return
        hz = self.session.frequency_cursor
        if peak:
            target = next_peak(*values, hz, direction)
            if target is None:
                self.statusBar().showMessage(
                    f"no peak {'above' if direction > 0 else 'below'}", 2000)
                return
        else:
            target = step_bin(values[0], hz, direction)
        self.session.set_frequency_cursor(target)

    def add_op(self, op_name: str):
        """Open the form for a graph op from the feature menu; returns it."""
        from .widgets.op_form import OpForm

        chosen = self.signal_dock.selected()
        signal = chosen if chosen in self.session.signals else self.focus_signal
        self.op_form = OpForm(self.session, op_name, signal, self)
        self.op_form.open()
        return self.op_form

    def _drop_signal(self, name: str) -> None:
        for pane in [p for p in self._signal_panes() if p.signal == name]:
            self.remove_pane(pane)
        try:
            self.session.hide(shown_name(self.session, name))
        except Exception as exc:                              # noqa: BLE001
            self.statusBar().showMessage(str(exc), 3000)

    def _chart_requested(self, name: str, view) -> None:
        if view:
            self.add_pane(view, name)
        else:
            self.plot_signal(name)

    def _plots_changed(self, name: str) -> None:
        exists = name in self.session.plots
        panes = [p for p in self.panes if isinstance(p, XYPane)]
        if not exists:
            for pane in [p for p in panes if p.signal == name]:
                self.remove_pane(pane)
            panes = [p for p in panes if p.signal != name]
        for pane in panes:
            pane.update_choices()
            pane.refresh({"plots"})
        if exists and not any(p.signal == name for p in panes):
            self.add_pane("xy", name)

    # -- shutting down -------------------------------------------------

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.player.stop(quietly=True)
        self.bridge.close()
        if self.listener is not None:
            self.listener.stop()
        self._shutdown_console()
        self.session.stop()
        super().closeEvent(event)


def peak_times(values: pd.Series, min_distance_s: float) -> list[float]:
    """Where a signal stands out: local maxima three standard deviations
    above its mean, at least `min_distance_s` apart."""
    from scipy.signal import find_peaks

    x = values.to_numpy(float)
    ok = np.isfinite(x)
    if ok.sum() < 3:
        return []
    time = values.index.to_numpy(float)
    rate = (len(time) - 1) / (time[-1] - time[0]) if time[-1] > time[0] else 0.0
    height = float(x[ok].mean() + 3.0 * x[ok].std())
    distance = max(1, int(min_distance_s * rate)) if rate else None
    found, _ = find_peaks(np.where(ok, x, -np.inf), height=height, distance=distance)
    return sorted(float(time[i]) for i in found)


def shown_name(session, signal: str) -> str:
    """The shown recipe a signal comes from: `hp.az` is of `hp`."""
    source = session.signals.get(signal).source if signal in session.signals else ""
    if source.startswith("graph:"):
        return source[len("graph:"):]
    raise KeyError(f"{signal} is a channel of the source and stays")


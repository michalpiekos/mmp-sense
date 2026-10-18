"""Collapsing panes and docks to their titles, and opening them again."""

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtCore import Qt                       # noqa: E402

from conftest import show_basics                    # noqa: E402
from mmp_sense.ui.main_window import MainWindow          # noqa: E402


@pytest.fixture
def window(qapp, session):
    window = MainWindow(show_basics(session), window_s=10.0, console=False)
    window.add_pane("signal", "magnitude")           # a fourth chart, a signal one
    window.splitter.setSizes([3, 2, 2, 2])
    window.resize(1200, 900)
    window.show()
    window.tick()
    qapp.processEvents()
    try:
        yield window
    finally:
        window.close()


def time_panes(window):
    return [p for p in window.panes if p.time_axis]


def count_refreshes(pane, monkeypatch):
    calls = []
    real = pane.refresh

    def counted(reasons, window=None):
        calls.append(set(reasons))
        return real(reasons, window)

    monkeypatch.setattr(pane, "refresh", counted)
    return calls


# -- chart panes -----------------------------------------------------------------


def test_a_collapsed_pane_is_its_header_and_the_others_take_the_room(window, qapp):
    before = window.splitter.sizes()
    pane = window.panes[3]
    pane.collapse_button.click()
    qapp.processEvents()
    after = window.splitter.sizes()
    assert pane.collapsed
    assert after[3] == pane.header.sizeHint().height()
    assert pane.plot.height() == 0
    assert pane.header.isVisible() and pane.chooser.currentText() == "magnitude"
    for i in (0, 1, 2):
        assert after[i] > before[i]

    pane.collapse_button.click()
    qapp.processEvents()
    assert not pane.collapsed
    opened = window.splitter.sizes()
    assert [s / sum(opened) for s in opened] == pytest.approx(
        [s / sum(before) for s in before], abs=0.02)


def test_an_opened_pane_takes_its_share_of_a_grown_room(window, qapp):
    first = window.panes[0]
    first.set_collapsed(True)
    window.panes[2].set_collapsed(True)
    sizes = window.splitter.sizes()
    grown = [sizes[0], sizes[1] * 2, sizes[2], sizes[3] * 2]
    window.splitter.setSizes(grown)       # e.g. a dock below was collapsed
    window.resize(1200, 1500)
    qapp.processEvents()
    first.set_collapsed(False)
    qapp.processEvents()
    opened = window.splitter.sizes()
    assert opened[0] == pytest.approx(opened[1], rel=0.1)
    assert opened[0] == pytest.approx(opened[3], rel=0.1)


def test_a_collapsed_pane_is_not_redrawn_until_opened(window, session, clock, monkeypatch, qapp):
    pane = window.panes[3]
    pane.set_collapsed(True)
    calls = count_refreshes(pane, monkeypatch)
    clock.advance(1.0)
    session.set_cursor(3.0)
    window.tick()
    window.tick()
    assert calls == []

    pane.set_collapsed(False)
    assert len(calls) == 1 and {"data", "cursor", "detections"} <= calls[0]
    assert pane.cursor_line.pos().x() == pytest.approx(3.0)
    window.tick()
    assert len(calls) >= 1


def test_follow_and_panning_work_with_the_first_pane_collapsed(window, clock, qapp):
    first, second = time_panes(window)[:2]
    first.set_collapsed(True)
    qapp.processEvents()
    clock.advance(2.0)
    window.tick()
    qapp.processEvents()
    (lo, hi), _ = second.plot.getViewBox().viewRange()
    assert hi == pytest.approx(window.session.buffer.t_end, abs=0.05)

    window.follow_action.setChecked(False)
    second.plot.getViewBox().setXRange(2.0, 4.0, padding=0)
    qapp.processEvents()
    for pane in time_panes(window):
        assert pane.plot.getViewBox().viewRange()[0] == pytest.approx([2.0, 4.0], abs=0.05)


def test_expanding_a_collapsed_pane_opens_it_and_others_stay_collapsed(window, qapp):
    collapsed, other = window.panes[1], window.panes[2]
    collapsed.set_collapsed(True)
    window.toggle_expanded(collapsed)
    assert not collapsed.collapsed

    collapsed.set_collapsed(True)
    window.toggle_expanded(other)
    window.toggle_expanded(other)
    qapp.processEvents()
    assert collapsed.collapsed
    assert window.splitter.sizes()[1] == collapsed.header.sizeHint().height()


def test_a_collapsed_pane_can_be_closed(window):
    pane = window.panes[1]
    pane.set_collapsed(True)
    window.remove_pane(pane)
    assert pane not in window.panes
    window.tick()


def test_the_spectrum_holds_nothing_while_collapsed(window, monkeypatch):
    spectrum = [p for p in window.panes if p.kind == "spectrum"][0]
    spectrum.set_collapsed(True)
    drawn = []
    monkeypatch.setattr(spectrum, "_draw_markers", lambda: drawn.append(1))
    spectrum.expire_holds()
    assert drawn == []


# -- docks -------------------------------------------------------------------------


def central(window, qapp):
    qapp.processEvents()
    return window.centralWidget().geometry()


def test_the_console_starts_collapsed_and_opens_to_a_third(qapp, session):
    window = MainWindow(session, window_s=10.0, console=True)
    window.resize(1200, 900)
    window.show()
    window.tick()                                  # the status bar appears
    try:
        dock = window.console_dock
        bar = dock.titleBarWidget()
        assert dock.collapsed and not dock.widget().isVisible()
        assert dock.height() <= bar.sizeHint().height() + 2
        tall = central(window, qapp).height()
        if window.console is not None:            # running all along
            window.console.kernel_manager.kernel.shell.run_cell("still = 1")
            assert window.console.kernel_manager.kernel.shell.user_ns["still"] == 1

        bar.collapse_button.click()
        qapp.processEvents()
        assert dock.widget().isVisible()
        assert dock.height() == pytest.approx(window.height() / 3, rel=0.15)
        assert central(window, qapp).height() < tall - 200

        bar.collapse_button.click()
        assert central(window, qapp).height() == pytest.approx(tall, abs=10)
    finally:
        window.close()


def test_there_is_no_option_to_leave_the_console_out(capsys):
    from mmp_sense.ui.app import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--no-console"])
    assert "unrecognized arguments: --no-console" in capsys.readouterr().err


def test_the_left_column_gives_height_then_width(window, qapp):
    signals, detectors = window.signals_dock, window.detectors_dock
    wide = central(window, qapp).width()
    detectors_tall = detectors.height()

    window.set_dock_collapsed(signals, True)
    qapp.processEvents()
    assert detectors.height() > detectors_tall + 100
    assert central(window, qapp).width() == pytest.approx(wide, abs=10)

    window.set_dock_collapsed(detectors, True)
    qapp.processEvents()
    assert signals.titleBarWidget().vertical and detectors.titleBarWidget().vertical
    assert signals.width() < 50
    assert central(window, qapp).width() > wide + 150

    window.set_dock_collapsed(detectors, False)
    qapp.processEvents()
    assert not signals.titleBarWidget().vertical
    assert central(window, qapp).width() == pytest.approx(wide, abs=10)
    assert detectors.widget().isVisible() and not signals.widget().isVisible()


def test_the_detectors_dock_waits_while_collapsed(window, monkeypatch):
    calls = []
    monkeypatch.setattr(window.detector_dock, "refresh", lambda w=None: calls.append(w))
    window.set_dock_collapsed(window.detectors_dock, True)
    window.refresh({"data"})
    assert calls == []
    window.set_dock_collapsed(window.detectors_dock, False)
    assert len(calls) == 1 and calls[0] is not None


def test_the_title_bar_floats_and_closes(window, qapp):
    dock = window.detectors_dock
    bar = dock.titleBarWidget()
    window.set_dock_collapsed(dock, True)
    bar.float_button.click()
    assert dock.isFloating() and not dock.collapsed
    assert bar.collapse_button.isHidden()
    bar.float_button.click()
    assert not dock.isFloating() and not bar.collapse_button.isHidden()
    bar.close_button.click()
    assert not dock.isVisible()


def test_tidy_gives_the_open_charts_equal_heights(window, qapp):
    window.splitter.setSizes([500, 100, 60, 200])
    qapp.processEvents()
    window.tidy_panes()
    sizes = window.splitter.sizes()
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) > 0


def test_tidy_keeps_a_collapsed_chart_collapsed_and_undoes_an_expand(window, qapp):
    collapsed = window.panes[3]
    collapsed.set_collapsed(True)
    window.toggle_expanded(window.panes[1])
    qapp.processEvents()
    window.tidy_panes()
    qapp.processEvents()
    sizes = window.splitter.sizes()
    assert window.expanded is None
    assert collapsed.collapsed and sizes[3] == collapsed.header.sizeHint().height()
    assert max(sizes[:3]) - min(sizes[:3]) <= 1
    collapsed.set_collapsed(False)                     # reopened: an equal share
    qapp.processEvents()
    sizes = window.splitter.sizes()
    assert max(sizes) - min(sizes) <= 0.1 * max(sizes)


def test_the_tidy_button_and_key(window, qapp):
    from PySide6.QtGui import QAction, QKeySequence, QShortcut

    [button] = [a for a in window.findChildren(QAction) if a.text() == "Tidy"]
    window.splitter.setSizes([500, 100, 60, 200])
    button.trigger()
    assert max(window.splitter.sizes()) - min(window.splitter.sizes()) <= 1
    window.splitter.setSizes([500, 100, 60, 200])
    [key] = [k for k in window.findChildren(QShortcut) if k.key() == QKeySequence("=")]
    key.activated.emit()                   # what pressing it in the window does
    assert max(window.splitter.sizes()) - min(window.splitter.sizes()) <= 1


@pytest.mark.parametrize("which", [0, 1, -1])
def test_a_dock_reopened_from_the_strip_is_not_held_to_its_title(window, qapp, which):
    column = [d for d in window.docks if window.dockWidgetArea(d) == Qt.LeftDockWidgetArea]
    for dock in column:
        window.set_dock_collapsed(dock, True)
        qapp.processEvents()
    dock = column[which]
    window.set_dock_collapsed(dock, False)
    qapp.processEvents()
    assert dock.maximumWidth() > dock.minimumWidth()
    assert dock.width() >= dock.widget().minimumSizeHint().width()

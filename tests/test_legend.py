"""The legend as a switch: hide, show and solo the series of a chart."""

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtCore import QPoint, Qt          # noqa: E402
from PySide6.QtTest import QTest               # noqa: E402

import mmp_sense                                     # noqa: E402
from conftest import show_basics                    # noqa: E402
from mmp_sense.ui.main_window import MainWindow      # noqa: E402


@pytest.fixture
def window(qapp, session):
    window = MainWindow(show_basics(session), window_s=10.0, console=False)
    window.resize(1200, 900)
    window.show()
    window.tick()
    qapp.processEvents()
    try:
        yield window
    finally:
        window.close()


def entry(pane, name):
    for sample, label in pane.legend.items:
        if sample.item.name() == name:
            return sample, label
    raise KeyError(name)


def at(pane, graphics_item):
    view = pane.plot
    centre = graphics_item.sceneBoundingRect().center()
    return view.mapFromScene(centre)


def click(pane, graphics_item, double=False):
    point = at(pane, graphics_item)
    if double:
        QTest.mouseClick(pane.plot.viewport(), Qt.LeftButton, pos=point)
        QTest.mouseDClick(pane.plot.viewport(), Qt.LeftButton, pos=point)
    else:
        QTest.mouseClick(pane.plot.viewport(), Qt.LeftButton, pos=point)


def visible(pane):
    return sorted(n for n, c in pane._curves.items() if c.isVisible())


def test_clicking_a_name_or_its_line_hides_and_shows_the_series(window, session, qapp):
    pane = window.panes[0]                       # channels: three curves
    cursor = session.cursor
    _sample, label = entry(pane, "accel_z")
    click(pane, label)
    assert visible(pane) == ["accel_x", "accel_y"]
    assert pane.hidden_series == {"accel_z"}
    assert "666666" in label.text or label.opts.get("color") == "#666666"
    assert session.cursor == cursor              # the legend click is not a cursor click
    sample, _label = entry(pane, "accel_z")
    click(pane, sample)
    assert visible(pane) == ["accel_x", "accel_y", "accel_z"]


def test_double_click_shows_one_alone_then_all(window, qapp):
    pane = window.panes[0]
    _sample, label = entry(pane, "accel_y")
    click(pane, label, double=True)
    assert visible(pane) == ["accel_y"]
    click(pane, label, double=True)
    assert visible(pane) == ["accel_x", "accel_y", "accel_z"]


def test_hiding_the_big_one_rescales_to_the_rest(window, qapp):
    pane = window.panes[0]                       # accel_z sits at ~9.8, the rest at 0
    vb = pane.plot.getViewBox()
    qapp.processEvents()
    _x, (low, high) = vb.viewRange()
    assert high > 8
    pane.set_series_visible("accel_z", False)
    qapp.processEvents()
    _x, (low, high) = vb.viewRange()
    assert high < 5
    pane.set_series_visible("accel_z", True)
    qapp.processEvents()
    assert vb.viewRange()[1][1] > 8


def test_a_hidden_series_rests_and_catches_up_when_shown(window, session, clock):
    pane = window.panes[0]

    def last(name):                                # curves clip to the view: compare ends
        return float(pane._curves[name].xData[-1])

    pane.set_series_visible("accel_x", False)
    before = last("accel_x")
    clock.advance(1.0)
    session.poll()
    window.refresh({"data"})
    assert last("accel_x") == before
    assert last("accel_y") > before + 0.9
    pane.set_series_visible("accel_x", True)
    assert last("accel_x") == last("accel_y")


def test_another_selection_starts_with_everything_shown(window, session):
    session.show(mmp_sense.signal("accel_x", "accel_y").highpass(cutoff_hz=5, name="hp"))
    window.tick()
    pane = window.panes[-1]
    pane.set_series_visible("hp.accel_x", False)
    window.tick()
    assert not pane._curves["hp.accel_x"].isVisible()   # kept across ticks
    pane.chooser.setCurrentText("channels")
    assert pane.hidden_series == set()
    assert visible(pane) == ["accel_x", "accel_y", "accel_z"]


def test_a_click_in_the_plot_still_moves_the_cursor(window, session, qapp):
    pane = window.panes[0]
    vb = pane.plot.getViewBox()
    (x0, x1), (y0, y1) = vb.viewRange()
    target = x0 + 0.3 * (x1 - x0)
    scene = vb.mapViewToScene(vb.mapToView(vb.mapFromView(
        __import__("pyqtgraph").Point(target, (y0 + y1) / 2))))
    QTest.mouseClick(pane.plot.viewport(), Qt.LeftButton, pos=pane.plot.mapFromScene(scene))
    assert session.cursor == pytest.approx(target, abs=0.05 * (x1 - x0))

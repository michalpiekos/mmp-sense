"""Detections on the screen: marks on the signal panes, and the dock.

Qt's offscreen platform, so this needs no display.
"""

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import mmp_sense                                          # noqa: E402
from PySide6.QtCore import Qt                       # noqa: E402
from mmp_sense.ui.main_window import MainWindow          # noqa: E402
from mmp_sense.ui.widgets.detector_dock import (  # noqa: E402
    FOUND, LATENCY, NODE, SHOW)


@pytest.fixture
def window(qapp, session):
    window = MainWindow(session, window_s=10.0, console=False)
    window.show()
    window.tick()
    try:
        yield window
    finally:
        window.close()


def knock_graph(threshold=0.25):
    hp = mmp_sense.signal().magnitude().highpass(cutoff_hz=10, name="hp")
    energy = hp.energy("80ms", name="energy")
    freq = hp.frames("80ms").dominant_hz(name="freq")
    knock = ((energy > threshold) & freq.between(60, 150)).events(
        name="knock", refractory="100ms")
    return knock.followed_by(knock, gap=("80ms", "500ms"), name="double_knock")


def signal_panes(window):
    return [p for p in window.panes if p.kind == "signal"]


def test_shown_events_are_marked_on_every_signal_pane(window, session):
    graph = knock_graph()
    session.show(graph.find("knock"))
    window.tick()
    for pane in signal_panes(window):
        marks = pane.marks()["knock"]
        assert marks["lines"] == pytest.approx([5.0], abs=0.06)
        assert marks["regions"] == []


def test_a_pattern_is_drawn_as_a_span_apart_from_its_events(window, session, source, clock):
    source.knock(at=10.25)
    source.knock(at=10.5)
    clock.advance(2.0)
    session.poll()
    graph = knock_graph()
    session.show(graph.find("knock"), graph)
    window.tick()
    pane = signal_panes(window)[0]
    knocks = pane.marks()["knock"]["lines"]
    doubles = pane.marks()["double_knock"]
    assert len(knocks) >= 2
    assert len(doubles["lines"]) == 1
    (start, end), = doubles["regions"]
    assert start == pytest.approx(10.25, abs=0.06)
    assert end > 10.5
    styles = pane._mark_styles
    assert styles["knock"] != styles["double_knock"]


def test_series_nodes_open_panes_and_the_dock_lists_everything(window, session):
    graph = knock_graph()
    session.show(graph.find("energy"), graph.find("knock"), graph)
    window.tick()
    assert any(p.signal == "energy" for p in window.panes)

    dock = window.detector_dock
    rows = {dock.tree.topLevelItem(i).text(NODE): dock.tree.topLevelItem(i)
            for i in range(dock.tree.topLevelItemCount())}
    assert list(rows) == ["energy", "knock", "double_knock"]
    assert rows["knock"].text(FOUND) == "1"
    assert rows["double_knock"].text(FOUND) == "0"
    assert rows["energy"].text(FOUND) == "-"
    assert rows["double_knock"].text(LATENCY) == "600ms"

    dock.tree.setCurrentItem(rows["double_knock"])
    assert dock.graph.toPlainText() == graph.graph()


def test_a_changed_threshold_reaches_the_screen(window, session):
    session.show(knock_graph().find("knock"))
    window.tick()
    assert len(signal_panes(window)[0].marks()["knock"]["lines"]) == 1
    session.show(knock_graph(threshold=1e9).find("knock"))
    window.tick()
    assert signal_panes(window)[0].marks()["knock"]["lines"] == []


def test_hiding_removes_the_marks(window, session):
    session.show(knock_graph().find("knock"))
    window.tick()
    session.hide("knock")
    window.tick()
    assert "knock" not in signal_panes(window)[0].marks()


def test_a_failing_detection_is_reported_not_fatal(window, session):
    session.show((mmp_sense.signal("nothing") > 1).events(name="broken"))
    window.tick()
    pane = signal_panes(window)[0]
    assert "broken" in pane.status.text()
    assert pane.marks()["broken"]["lines"] == []
    rows = [window.detector_dock.tree.topLevelItem(i).text(FOUND)
            for i in range(window.detector_dock.tree.topLevelItemCount())]
    assert rows[0].startswith("error")


def test_a_fixed_detection_clears_its_complaint(window, session):
    session.stop_intake()
    session.show((mmp_sense.signal("nothing") > 1).events(name="broken"))
    window.tick()
    session.show((mmp_sense.signal("accel_z") > 1e9).events(name="broken"))
    window.tick()
    assert "broken" not in signal_panes(window)[0].status.text()


def test_the_dock_follows_a_node_shown_again(window, session):
    events = (mmp_sense.signal("accel_z") > 1e9).events(name="e")
    session.show(events.followed_by(events, gap=(0, "1s"), name="pair"))
    window.tick()
    session.show(events.followed_by(events, gap=(0, "2s"), name="pair"))
    window.tick()
    item = window.detector_dock.tree.topLevelItem(0)
    assert item.text(LATENCY) == "2s"


# -- run from the repository's root, as its README says ------------------

import os                                           # noqa: E402
import subprocess                                   # noqa: E402
import sys                                          # noqa: E402
from pathlib import Path                            # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def from_the_root(*args):
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    return subprocess.run([sys.executable, *args], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120)


def test_the_demo_runs_from_the_repository_root():
    done = from_the_root("-m", "mmp_sense.examples.double_knock")
    assert done.returncode == 0, done.stderr
    assert "double_knock" in done.stdout


def test_the_window_and_its_console_start_from_the_repository_root():
    done = from_the_root("-c", (
        "import runpy, sys\n"
        "from PySide6.QtCore import QTimer\n"
        "from PySide6.QtWidgets import QApplication\n"
        "app = QApplication([])\n"
        "QTimer.singleShot(1500, app.quit)\n"
        "sys.argv = ['mmp-sense']\n"
        "try:\n"
        "    runpy.run_module('mmp_sense', run_name='__main__')\n"
        "except SystemExit as exc:\n"
        "    assert not exc.code, exc.code\n"
        "import mmp_sense\n"
        "assert mmp_sense.signal and mmp_sense.load and mmp_sense.Session\n"
    ))
    assert done.returncode == 0, done.stderr


def test_unticking_show_takes_one_detector_off_the_plots(window, session, source, clock):
    source.knock(at=10.25)
    source.knock(at=10.5)
    clock.advance(2.0)
    session.poll()
    graph = knock_graph()
    session.show(graph.find("knock"), graph, graph.find("energy"))
    window.tick()
    dock = window.detector_dock
    pane = signal_panes(window)[0]
    colour = pane._mark_styles["double_knock"][0]
    assert dock.tree.headerItem().text(SHOW) == "Show"
    assert dock.shown("knock") is True and dock.shown("double_knock") is True
    assert dock.shown("energy") is None      # a signal has no box

    row = dock.tree.findItems("knock", Qt.MatchExactly, NODE)[0]
    row.setCheckState(SHOW, Qt.Unchecked)
    assert not session.drawn("knock")
    window.tick()
    for each in signal_panes(window):
        assert "knock" not in each.marks()
        assert each.marks()["double_knock"]["lines"]
    assert pane._mark_styles["double_knock"][0] == colour
    assert row.text(FOUND) == "3"            # still counted

    row.setCheckState(SHOW, Qt.Checked)
    window.tick()
    assert len(pane.marks()["knock"]["lines"]) == 3


def test_the_show_box_follows_the_console(window, session):
    session.show(knock_graph().find("knock"))
    window.tick()
    session.set_drawn("knock", False)
    window.tick()
    assert window.detector_dock.shown("knock") is False
    session.set_drawn("knock", True)
    window.tick()
    assert window.detector_dock.shown("knock") is True
    session.set_drawn("knock", False)
    session.show(knock_graph().find("knock"))          # shown again: drawn again
    window.tick()
    assert window.detector_dock.shown("knock") is True

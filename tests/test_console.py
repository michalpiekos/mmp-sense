"""The console and the window on one session.

The kernel is in this process, so `session` in the console is the object
the panes are drawing -- which is the whole reason for embedding IPython
rather than writing a REPL.
"""

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("qtconsole")
pytest.importorskip("ipykernel")

from mmp_sense.ui.console import make_console                  # noqa: E402
from mmp_sense.ui.main_window import MainWindow                # noqa: E402


@pytest.fixture
def console_window(qapp, session):
    window = MainWindow(session, window_s=10.0, console=True)
    widget = window.console_dock.widget()
    if not hasattr(widget, "kernel_manager"):
        window.close()
        pytest.skip("no in-process kernel available")
    try:
        yield window, widget.kernel_manager.kernel.shell
    finally:
        window.close()


def test_the_console_namespace_is_the_running_session(console_window, session):
    _window, shell = console_window
    assert shell.user_ns["session"] is session
    assert shell.user_ns["show"] == session.show
    assert "data" not in shell.user_ns


def test_a_signal_defined_in_the_console_is_plotted_at_once(console_window, session):
    window, shell = console_window
    shell.run_cell('show(signal("accel_z").energy("20ms", name="boom"))')
    window.tick()
    assert "boom" in session.signals
    assert any(pane.signal == "boom" for pane in window.panes)
    assert "boom" in window.panes[0].choices()


def test_what_the_window_does_is_visible_in_the_console(console_window, session):
    window, shell = console_window
    session.set_cursor(5.0)
    window.tick()
    session.set_selection((4.9, 5.15))
    shell.run_cell("here = session.context.cursor\n"
                   "peak = session.spectrum_summary('accel_z')['dominant_hz']")
    assert shell.user_ns["here"] == 5.0
    assert shell.user_ns["peak"] == pytest.approx(90.0, abs=6.0)


def test_a_console_is_optional(qapp):
    widget, shutdown = make_console({"x": 1})
    assert widget is not None
    shutdown()


def settle(qapp, until, seconds=3.0):
    import time

    end = time.monotonic() + seconds
    while time.monotonic() < end and not until():
        qapp.processEvents()
        time.sleep(0.01)
    return until()


def test_the_call_tip_waits_to_be_asked(qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    def knock_detector(threshold, refractory="145ms"):
        """Three knocks, each above threshold."""

    widget, shutdown = make_console({"knock_detector": knock_detector})
    try:
        widget.show()
        widget._control.setFocus()
        settle(qapp, lambda: False, 0.5)                    # the prompt comes up
        tip = widget._call_tip_widget
        QTest.keyClicks(widget._control, "knock_detector(")
        settle(qapp, lambda: False, 0.3)
        assert "call_tip" not in widget._request_info       # typing `(` asks nothing
        assert tip.text() == ""

        QTest.keyClick(widget._control, Qt.Key_Space, Qt.ControlModifier)
        assert settle(qapp, lambda: "threshold" in tip.text())
        assert "call_tip" in widget._request_info
        assert widget.input_buffer == "knock_detector("        # the key typed nothing
    finally:
        widget._call_tip_widget.hide()
        widget.close()
        shutdown()
        widget.deleteLater()
        settle(qapp, lambda: False, 0.2)

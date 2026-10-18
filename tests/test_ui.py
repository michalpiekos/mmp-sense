"""The window: what it draws, what it follows, and what it leaves alone.

Qt's offscreen platform, so this needs no display.
"""

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import mmp_sense                                          # noqa: E402
from conftest import show_basics                    # noqa: E402
from mmp_sense.ui.main_window import MainWindow          # noqa: E402
from mmp_sense.ui.panes.base import ALL_CHANNELS         # noqa: E402


@pytest.fixture
def window(qapp, session):
    window = MainWindow(show_basics(session), window_s=10.0, console=False)
    window.show()
    window.tick()
    try:
        yield window
    finally:
        window.close()


def time_panes(window):
    return [p for p in window.panes if p.time_axis]


def test_the_default_workspace_is_the_channels_and_their_frequencies(window, session):
    assert [p.kind for p in window.panes] == ["signal", "spectrum", "spectrogram"]
    assert window.panes[0].signal == ALL_CHANNELS
    assert window.panes[1].signal == window.panes[2].signal == session.channels[0]


def test_the_brand_sits_at_the_right_of_the_status_bar(window):
    assert window.brand.text() == "MMP Systems"
    assert window.brand.isVisible()
    assert window.brand.parent() is window.statusBar()
    bar = window.statusBar()
    assert window.brand.geometry().right() > bar.width() * 0.8


def test_the_cursor_is_shared_by_every_pane(window, session):
    session.set_cursor(5.0)
    window.tick()
    for pane in window.panes:
        if hasattr(pane, "cursor_line"):
            assert pane.cursor_line.pos().x() == pytest.approx(5.0)


def test_the_spectrum_follows_the_cursor(window, session):
    session.analysis_window = 0.25
    session.set_cursor(5.0)
    window.tick()
    spectrum = [p for p in window.panes if p.kind == "spectrum"][0]
    assert "4.875-5.125 s" in spectrum.status.text()
    # The knock is at 5 s, and it is what the transform should show.
    peak = float(spectrum.status.text().split("peak ")[1].split(" Hz")[0])
    assert peak == pytest.approx(90.0, abs=6.0)

    session.set_cursor(1.0)
    window.tick()
    assert "0.875-1.125 s" in spectrum.status.text()


def test_the_spectrum_prefers_a_selection_to_the_cursor(window, session):
    session.set_selection((4.8, 5.2))
    window.tick()
    spectrum = [p for p in window.panes if p.kind == "spectrum"][0]
    assert "4.800-5.200 s" in spectrum.status.text()


def test_the_spectrum_is_one_bar_per_bin(window, session):
    session.set_selection((4.8, 5.2))
    window.tick()
    pane = [p for p in window.panes if p.kind == "spectrum"][0]
    spectrum = session.spectrum(pane.signal, 4.8, 5.2)
    opts = pane.bars.opts
    np.testing.assert_allclose(opts["x"], spectrum.index)
    np.testing.assert_allclose(opts["height"], spectrum.values)
    assert opts["width"] == pytest.approx(spectrum.index[1] - spectrum.index[0])

    session.set_selection((5.0, 5.001))
    window.tick()
    assert "not enough samples" in pane.status.text()
    assert len(pane.bars.opts["x"]) == 0


def test_the_spectrogram_has_a_decibel_scale(window, session):
    pane = [p for p in window.panes if p.kind == "spectrogram"][0]
    pane.refresh({"signals"}, session.window())
    image = pane.image.image
    assert image is not None and image.size
    assert pane.legend.axis.labelText == "dB"
    low, high = pane.legend.levels()
    assert (low, high) == pytest.approx(tuple(pane.image.getLevels()))
    assert (low, high) == pytest.approx((np.median(image), image.max()))


def test_the_spectrogram_chooses_its_segment_and_how_high_it_reaches(window, session):
    pane = [p for p in window.panes if p.kind == "spectrogram"][0]
    pane.nperseg = 512
    assert "512-sample segments" in pane.status.text()
    whole = pane.image.image.shape[1]
    assert whole == 257                        # 512 samples: 257 frequencies
    pane.top_hz = 100
    reach = pane.image.mapRectToParent(pane.image.boundingRect())
    assert 0 < pane.image.image.shape[1] < whole
    assert reach.bottom() <= 100.0
    assert pane.describe()["segment"] == 512 and pane.describe()["top_hz"] == 100.0
    pane.top_hz = None
    assert pane.top.currentText() == "all"


def test_h_and_l_move_the_cursor_by_a_quarter_window(window, session):
    session.set_cursor(5.0)
    session.analysis_window = 0.4
    window._step(+0.25, "window")
    assert session.cursor == pytest.approx(5.1)
    window._step(-0.25, "window")
    assert session.cursor == pytest.approx(5.0)
    window._step(+1.0, "sample")
    assert session.cursor == pytest.approx(5.002)


def test_n_jumps_to_the_knock(window, session):
    window.focus_signal = "energy"
    session.set_cursor(0.0)
    window.jump_peak(+1)
    assert session.cursor == pytest.approx(5.0, abs=0.05)


def test_dragging_a_region_selects_it(window, session):
    pane = time_panes(window)[0]
    pane._region_dragged(4.0, 6.0, True)
    assert session.selection == (4.0, 6.0)
    window.tick()
    for other in time_panes(window):
        assert other.region.isVisible()
        assert other.region.getRegion() == pytest.approx((4.0, 6.0))


def test_a_signal_made_in_the_console_is_plotted_without_a_restart(window, session):
    session.show(mmp_sense.signal("magnitude").energy("20ms", name="knock_energy"))
    window.tick()
    assert any(p.signal == "knock_energy" for p in window.panes)
    assert "knock_energy" in window.panes[0].choices()
    assert window.signal_dock.tree.topLevelItemCount() == len(session.signals.names)


def test_dropping_a_signal_takes_its_pane_with_it(window, session):
    session.show(mmp_sense.signal("magnitude").peak("20ms", name="spare"))
    window.plot_signal("spare")
    assert any(p.signal == "spare" for p in window.panes)
    window._drop_signal("spare")
    assert not any(p.signal == "spare" for p in window.panes)
    assert "spare" not in session.signals and "spare" not in session.shown


def test_curves_are_updated_rather_than_recreated(window, session, clock):
    pane = time_panes(window)[0]
    before = {name: id(curve) for name, curve in pane._curves.items()}
    clock.advance(1.0)
    window.tick()
    after = {name: id(curve) for name, curve in pane._curves.items()}
    assert before == after
    assert len(pane._curves["accel_x"].xData) == len(session.buffer)


def test_the_view_follows_the_newest_data_until_it_is_touched(window, session, clock):
    viewbox = time_panes(window)[0].plot.getViewBox()
    clock.advance(1.0)
    window.tick()
    (lo, hi), _ = viewbox.viewRange()
    assert hi == pytest.approx(session.buffer.t_end)
    assert lo == pytest.approx(session.buffer.t_end - 10.0)

    # Panning is a decision: the window stops scrolling underneath it.
    window._user_ranged()
    assert not window.follow
    viewbox.setXRange(2.0, 4.0, padding=0)
    clock.advance(1.0)
    window.tick()
    (lo, hi), _ = viewbox.viewRange()
    assert (lo, hi) == pytest.approx((2.0, 4.0))

    window.follow_action.setChecked(True)
    window.tick()
    (lo, hi), _ = viewbox.viewRange()
    assert hi == pytest.approx(session.buffer.t_end)


def test_a_dragged_selection_survives_the_next_tick(window, session, clock):
    session.set_selection((4.0, 4.5))
    clock.advance(0.5)
    window.tick()
    assert session.selection == (4.0, 4.5)
    assert time_panes(window)[0].region.getRegion() == pytest.approx((4.0, 4.5))


def test_a_tick_does_not_start_while_one_is_running(window, session):
    window._busy = True
    before = len(window.tick_times)
    window.tick()
    assert len(window.tick_times) == before
    window._busy = False


def test_playback_moves_the_cursor_from_the_toolbar(window, session, clock):
    session.set_cursor(2.0)
    window.toggle_play()
    assert session.playing and window.play_action.text() == "Pause"
    clock.advance(0.5)
    window.tick()
    assert session.cursor == pytest.approx(2.5, abs=1e-6)
    window.toggle_play()
    assert not session.playing


def test_a_pane_can_be_closed(window):
    pane = window.panes[-1]
    pane.closed.emit(pane)
    assert pane not in window.panes


def test_the_status_line_says_where_the_cursor_is(window, session):
    session.set_cursor(3.25)
    window.tick()
    message = window.statusBar().currentMessage()
    assert "cursor 3.250,000 s" in message
    assert "synthetic" in message


def test_the_status_line_gives_the_selection_and_its_length(window, session):
    session.set_selection((4.8, 5.2))
    window.tick()
    message = window.statusBar().currentMessage()
    assert "selection 4.800,000 – 5.200,000 s (0.400,000 s)" in message


@pytest.mark.parametrize("seconds, text", [
    (3.25, "3.250,000"),
    (12.004517, "12.004,517"),
    (0.0, "0.000,000"),
    (-0.0015, "-0.001,500"),
    (0.9999996, "0.999,999"),
    (-0.0000004, "0.000,000"),
    (1234.5, "1234.500,000"),
    (1.7e9 + 0.25, "1700000000.250,000"),
    (1e-07, "0.000,000"),
    (float("nan"), "nan"),
])
def test_seconds_are_shown_to_the_microsecond_without_rounding(seconds, text):
    from mmp_sense.ui.main_window import format_seconds

    assert format_seconds(seconds) == text


def test_closing_the_first_pane_leaves_the_rest_working(window, session, clock):
    """The time axes are linked to the first pane; it may be the one closed."""
    first = time_panes(window)[0]
    window.remove_pane(first)
    clock.advance(0.5)
    window.tick()
    remaining = time_panes(window)
    assert first not in window.panes
    window._user_ranged()
    assert not window.follow
    viewbox = remaining[0].plot.getViewBox()
    viewbox.setXRange(1.0, 2.0, padding=0)
    for pane in remaining[1:]:
        assert pane.plot.getViewBox().viewRange()[0] == pytest.approx([1.0, 2.0])


def test_a_source_that_fails_does_not_stop_the_window(window, session):
    class Broken:
        name, kind, channels, units, rate = "broken", "live", ["accel_x"], "", 100.0

        def start(self):
            return self

        def stop(self):
            pass

        def read(self):
            raise RuntimeError("the source fell over")

    session.source = Broken()
    window.tick()
    assert "the source fell over" in window.statusBar().currentMessage()
    window.tick()                    # and again, without raising


def test_stop_freezes_the_chart_and_keeps_what_is_drawn(window, session, clock):
    pane = time_panes(window)[0]
    held = len(session.buffer)

    window.stop_action.trigger()
    clock.advance(2.0)               # the simulator's clock runs on
    window.tick()

    assert not session.running
    assert len(session.buffer) == held
    assert len(pane._curves["accel_x"].xData) == held
    assert "stopped" in window.statusBar().currentMessage()
    assert not window.stop_action.isEnabled()


def test_start_empties_the_chart_and_fills_it_again(window, session, clock):
    window.stop_action.trigger()
    window.start_action.trigger()
    window.tick()

    assert session.running and window.stop_action.isEnabled()
    assert len(session.buffer) == 0
    assert "stopped" not in window.statusBar().currentMessage()

    clock.advance(1.0)
    window.tick()
    assert len(session.buffer) == 500
    assert len(time_panes(window)[0]._curves["accel_x"].xData) == 500


def test_start_brings_the_view_back_to_the_new_data(window, session, clock):
    """A view left over the last capture would show an empty chart."""
    viewbox = time_panes(window)[0].plot.getViewBox()
    window._user_ranged()
    viewbox.setXRange(2.0, 4.0, padding=0)
    assert not window.follow

    window.start_action.trigger()
    clock.advance(1.0)
    window.tick()
    assert window.follow
    (lo, hi), _ = viewbox.viewRange()
    assert hi == pytest.approx(session.buffer.t_end)


def test_the_toolbar_follows_a_stop_made_from_the_console(window, session):
    session.stop_intake()
    assert not window.stop_action.isEnabled()
    window.tick()
    assert "stopped" in window.statusBar().currentMessage()

    session.start_intake()
    assert window.stop_action.isEnabled()


def test_a_source_that_cannot_restart_is_reported_not_raised(window, session, clock):
    from mmp_sense import SyntheticAccelerometer

    class Gone(SyntheticAccelerometer):
        def restart(self):
            raise ConnectionError("the device went away")

    held = len(session.buffer)
    session.source = Gone(rate=500.0, clock=clock)
    window.start_action.trigger()
    assert "the device went away" in window.statusBar().currentMessage()
    assert len(session.buffer) == held
    window.tick()                    # and the window goes on


def double_click_header(pane, widget=None):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    QTest.mouseDClick(widget or pane.status, Qt.LeftButton)


def visible_panes(window):
    sizes = window.splitter.sizes()
    return [p for p, size in zip(window.panes, sizes) if size > 0]


def test_double_clicking_a_header_expands_and_restores_the_pane(window, qapp):
    window.resize(1000, 800)
    qapp.processEvents()
    before = window.splitter.sizes()
    spectrum = window.panes[1]

    double_click_header(spectrum)
    assert visible_panes(window) == [spectrum]
    assert window.expanded is spectrum

    double_click_header(spectrum)
    assert visible_panes(window) == window.panes
    assert window.expanded is None
    qapp.processEvents()
    after = window.splitter.sizes()
    assert np.array(after) / sum(after) == pytest.approx(
        np.array(before) / sum(before), abs=0.01
    )


@pytest.mark.parametrize("expanded", [0, 1, 2])
def test_restored_panes_are_still_linked(window, qapp, expanded):
    # Links map by pixel, and the panes' left axes differ in width.
    tolerance = 0.05
    window.follow_action.setChecked(False)
    pane = window.panes[expanded]
    double_click_header(pane)
    qapp.processEvents()
    time_panes(window)[0].plot.getViewBox().setXRange(2.0, 4.0, padding=0)
    qapp.processEvents()
    double_click_header(pane)
    qapp.processEvents()
    for each in time_panes(window):
        assert each.plot.getViewBox().viewRange()[0] == pytest.approx(
            [2.0, 4.0], abs=tolerance
        )

    time_panes(window)[-1].plot.getViewBox().setXRange(6.0, 8.0, padding=0)
    qapp.processEvents()
    for each in time_panes(window):
        assert each.plot.getViewBox().viewRange()[0] == pytest.approx(
            [6.0, 8.0], abs=tolerance
        )


def test_the_headers_controls_do_not_expand_the_pane(window):
    pane = window.panes[1]
    double_click_header(pane, pane.chooser)
    assert window.expanded is None
    assert visible_panes(window) == window.panes


def test_an_expanded_pane_that_is_not_first_still_follows(window, session, clock, qapp):
    spectrogram = window.panes[2]
    double_click_header(spectrogram)
    qapp.processEvents()
    clock.advance(2.0)
    window.tick()
    qapp.processEvents()
    (lo, hi), _ = spectrogram.plot.getViewBox().viewRange()
    assert hi == pytest.approx(session.buffer.t_end, abs=0.05)
    assert hi - lo == pytest.approx(window.window_s, rel=0.05)


def test_closing_the_expanded_pane_shows_the_rest(window):
    spectrum = window.panes[1]
    double_click_header(spectrum)
    window.remove_pane(spectrum)
    assert window.expanded is None
    assert visible_panes(window) == window.panes
    assert len(window.panes) == 2


def test_adding_a_pane_while_one_is_expanded_shows_all(window):
    double_click_header(window.panes[0])
    added = window.add_pane("signal", "magnitude")
    assert window.expanded is None
    assert visible_panes(window) == window.panes
    assert added in window.panes


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


@pytest.fixture
def spectrum_pane(window):
    pane = [p for p in window.panes if p.kind == "spectrum"][0]
    pane.clock = FakeClock()
    pane.peak_hold_s = 2.0
    return pane


def marker_heights(pane):
    _x, y = pane.markers.getData()
    return np.zeros(0) if y is None else y[::2]


def test_a_peak_marker_holds_then_drops_to_the_bar(spectrum_pane):
    pane, clock = spectrum_pane, spectrum_pane.clock
    assert pane._expiry.isActive()
    f = np.array([10.0, 20.0, 30.0])
    pane.update_holds(f, [1.0, 5.0, 2.0], 10.0)
    clock.t += 1.0
    pane.update_holds(f, [1.0, 3.0, 4.0], 10.0)
    np.testing.assert_allclose(pane.held, [1.0, 5.0, 4.0])
    np.testing.assert_allclose(marker_heights(pane), [1.0, 5.0, 4.0])
    x, _y = pane.markers.getData()
    np.testing.assert_allclose(x[:2], [5.0, 15.0])

    clock.t += 1.0          # 2 s after the 5.0, 1 s after the 4.0
    pane.expire_holds()
    np.testing.assert_allclose(pane.held, [1.0, 3.0, 4.0])
    clock.t += 1.0
    pane.expire_holds()
    np.testing.assert_allclose(marker_heights(pane), [1.0, 3.0, 4.0])


def test_a_higher_value_moves_the_marker_up_and_restarts_its_hold(spectrum_pane):
    pane, clock = spectrum_pane, spectrum_pane.clock
    f = np.array([10.0])
    pane.update_holds(f, [2.0], 10.0)
    clock.t += 1.5
    pane.update_holds(f, [3.0], 10.0)
    clock.t += 1.5
    pane.update_holds(f, [1.0], 10.0)
    np.testing.assert_allclose(pane.held, [3.0])


def test_new_bins_reset_the_markers(spectrum_pane):
    pane = spectrum_pane
    pane.update_holds(np.array([10.0, 20.0]), [9.0, 9.0], 10.0)
    pane.update_holds(np.array([5.0, 10.0, 15.0]), [1.0, 2.0, 3.0], 5.0)
    np.testing.assert_allclose(pane.held, [1.0, 2.0, 3.0])


def test_a_hold_of_zero_draws_no_markers(spectrum_pane):
    pane = spectrum_pane
    pane.update_holds(np.array([10.0]), [2.0], 10.0)
    assert len(marker_heights(pane)) == 1
    pane.peak_hold_s = 0
    assert len(marker_heights(pane)) == 0
    pane.update_holds(np.array([10.0]), [5.0], 10.0)
    assert len(marker_heights(pane)) == 0


def test_the_knock_leaves_a_marker_as_the_cursor_moves_on(window, session, spectrum_pane):
    pane, clock = spectrum_pane, spectrum_pane.clock
    session.analysis_window = 0.25
    session.set_cursor(5.0)
    window.tick()
    knock = pane.held.max()
    session.set_cursor(8.0)
    window.tick()
    bars = pane.bars.opts["height"]
    assert pane.held.max() == pytest.approx(knock)
    assert bars.max() < knock / 3
    clock.t += 2.0
    pane.expire_holds()
    np.testing.assert_allclose(pane.held, bars)


def test_the_hold_box_sets_every_spectrum_pane(window):
    added = window.add_pane("spectrum", "magnitude")
    window.hold_spin.setValue(0.5)
    for pane in window.panes:
        if pane.kind == "spectrum":
            assert pane.peak_hold_s == 0.5
    assert window.add_pane("spectrum").peak_hold_s == 0.5
    assert added.peak_hold_s == 0.5


def test_peak_hold_is_an_option():
    from mmp_sense.ui.app import parse_args

    assert parse_args([]).peak_hold == 2.0
    assert parse_args(["--peak-hold", "1.5"]).peak_hold == 1.5

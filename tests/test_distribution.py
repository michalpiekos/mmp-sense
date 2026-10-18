"""The distribution beside a chart, and session.distribution behind it."""

import numpy as np
import pandas as pd
import pytest

import mmp_sense
from mmp_sense.core.query import distribution


def test_the_density_has_area_one_over_what_it_counts():
    values = np.r_[np.linspace(-1.0, 1.0, 1001), 50.0, np.nan]
    edges, density = distribution(values, bins=20, low=-1.0, high=1.0)
    assert len(edges) == 21 and edges[0] == -1.0 and edges[-1] == 1.0
    assert np.sum(density * np.diff(edges)) == pytest.approx(1.0)


def test_no_value_in_range_is_all_zeros_and_a_constant_is_one_bin_wide():
    _, density = distribution(np.ones(10), bins=8, low=5.0, high=6.0)
    assert not density.any()
    edges, density = distribution(np.full(10, 3.0), bins=8)
    assert edges[0] < 3.0 < edges[-1]
    assert np.sum(density * np.diff(edges)) == pytest.approx(1.0)


@pytest.fixture
def ramp():
    time = np.arange(0.0, 10.0, 0.01)
    return mmp_sense.from_frame(pd.DataFrame({"a": time, "b": -time}, index=time))


def test_session_distribution_follows_the_stretch_rule(ramp):
    whole = ramp.distribution("a", bins=10)
    assert whole.index.name == "value" and whole.name == "a"
    assert whole.index[0] == pytest.approx(0.495, abs=0.01)
    ramp.set_selection((2.0, 4.0))
    selected = ramp.distribution("a", bins=10)
    assert selected.index[0] == pytest.approx(2.1, abs=0.01)
    limited = ramp.distribution("a", 0.0, 10.0, bins=4, low=0.0, high=2.0)
    assert list(limited.index) == [0.25, 0.75, 1.25, 1.75]
    assert (limited * 0.5).sum() == pytest.approx(1.0)


def test_session_distribution_is_of_one_signal(ramp):
    with pytest.raises(ValueError, match="one signal"):
        ramp.distribution(mmp_sense.signal("a", "b"))


# -- the window ----------------------------------------------------------

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from conftest import show_basics               # noqa: E402
from mmp_sense.ui.main_window import MainWindow      # noqa: E402
from mmp_sense.ui.panes.base import SIDE_WIDTH       # noqa: E402


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


def pane_of(window, kind):
    return next(p for p in window.panes if p.kind == kind)


def look_at(window, pane, x, y, qapp):
    window.follow_action.setChecked(False)
    pane.plot.getViewBox().setRange(xRange=x, yRange=y, padding=0)
    qapp.processEvents()


def test_off_by_default_and_the_button_the_key_and_ui_turn_it_on(window, qapp):
    time_panes = [p for p in window.panes if p.time_axis]
    assert not window.distribution_action.isChecked()
    assert not any(p.distribution_shown for p in window.panes)
    window.show_distribution()
    qapp.processEvents()
    assert all(p.distribution_shown for p in time_panes)
    assert pane_of(window, "spectrum").side is None
    assert all(p.side.width() == SIDE_WIDTH for p in time_panes)
    window.distribution_action.trigger()
    assert not any(p.distribution_shown for p in window.panes)
    window.add_pane("signal", "magnitude")
    assert not window.panes[-1].distribution_shown
    window.show_distribution(True)
    window.add_pane("signal", "magnitude")
    assert window.panes[-1].distribution_shown


def test_the_time_axes_stay_aligned(window, qapp):
    window.show_distribution()
    qapp.processEvents()
    signal, spectrogram = pane_of(window, "signal"), pane_of(window, "spectrogram")
    assert signal.plot.width() == spectrogram.plot.width()


def test_it_counts_what_is_on_screen(window, session, qapp):
    pane = pane_of(window, "signal")
    window.show_distribution()
    look_at(window, pane, (1.0, 4.0), (-2.0, 2.0), qapp)
    densities = pane.densities()
    assert sorted(densities) == sorted(pane.signal_names())
    name = pane.signal_names()[0]
    _, (low, high) = pane.plot.getViewBox().viewRange()
    (a, b), _ = pane.plot.getViewBox().viewRange()
    expected = session.distribution(name, a, b, bins=64, low=low, high=high)
    edges, density = densities[name]
    assert edges[0] == pytest.approx(low) and edges[-1] == pytest.approx(high)
    np.testing.assert_allclose(density, expected.to_numpy())
    bars, outline = pane._densities[name]
    np.testing.assert_allclose(bars.opts["width"], density)
    side = pane.side.getViewBox()
    assert side.viewRange()[1] == pytest.approx([low, high])
    chart = pane.plot.getViewBox()
    assert side.sceneBoundingRect().top() == pytest.approx(chart.sceneBoundingRect().top())
    assert side.height() == pytest.approx(chart.height())


def test_a_selection_is_counted_instead(window, session, qapp):
    pane = pane_of(window, "signal")
    window.show_distribution()
    look_at(window, pane, (1.0, 4.0), (-2.0, 2.0), qapp)
    name = pane.signal_names()[0]
    before = pane.densities()[name][1]
    session.set_selection((4.5, 5.5))                  # the knock
    window.refresh({"selection"})
    _, (low, high) = pane.plot.getViewBox().viewRange()
    expected = session.distribution(name, 4.5, 5.5, bins=64, low=low, high=high)
    drawn = pane._densities[name][0].opts["width"]
    np.testing.assert_allclose(drawn, expected.to_numpy())
    assert not np.allclose(drawn, before)
    session.clear_selection()
    window.refresh({"selection"})
    np.testing.assert_allclose(pane._densities[name][0].opts["width"], before)


def test_zooming_redraws_and_a_hidden_series_has_no_curve(window, qapp):
    pane = pane_of(window, "signal")
    window.show_distribution()
    look_at(window, pane, (1.0, 4.0), (-2.0, 2.0), qapp)
    name = pane.signal_names()[0]
    look_at(window, pane, (1.0, 4.0), (-0.5, 0.5), qapp)
    edges = pane._densities[name][1].getData()[1]
    assert min(edges) == pytest.approx(-0.5) and max(edges) == pytest.approx(0.5)
    pane.set_series_visible(name, False)
    assert name not in pane._densities
    pane.set_series_visible(name, True)
    assert name in pane._densities


def test_a_collapsed_chart_takes_its_column_with_it(window, qapp):
    pane = pane_of(window, "signal")
    window.show_distribution()
    pane.set_collapsed(True)
    qapp.processEvents()
    assert pane.side.height() == 0
    name = pane.signal_names()[0]
    before = pane._densities[name][0].opts["width"].copy()
    pane.plot.getViewBox().setRange(yRange=(-0.5, 0.5), padding=0)
    np.testing.assert_array_equal(pane._densities[name][0].opts["width"], before)
    pane.set_collapsed(False)
    qapp.processEvents()
    assert pane.side.height() == pane.plot.height() > 0

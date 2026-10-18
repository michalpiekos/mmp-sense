"""Graph ops in the feature menu: a form for the parameters, then show."""

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import mmp_sense                                          # noqa: E402
from mmp_sense.core.graph import register_op             # noqa: E402
from mmp_sense.core.graph.ops import menu_ops            # noqa: E402
from conftest import show_basics                    # noqa: E402
from mmp_sense.ui.main_window import MainWindow          # noqa: E402
from mmp_sense.ui.widgets.op_form import op_parameters   # noqa: E402

MENU = ["band_power", "band_share", "bandpass", "bandstop", "centroid_hz", "dominant_hz",
        "energy", "envelope", "flatness", "flux", "highpass", "line_power", "line_share",
        "lowpass", "notch", "peak", "rms",
        "rolloff_hz", "spread_hz", "zero_crossing_rate"]
ON_FRAMES = {"band_power", "band_share", "centroid_hz", "dominant_hz", "flatness", "flux",
             "line_power", "line_share", "rolloff_hz", "spread_hz"}


@pytest.fixture
def window(qapp, session):
    window = MainWindow(show_basics(session), window_s=10.0, console=False)
    window.show()
    window.tick()
    try:
        yield window
    finally:
        window.close()


def entries(window):
    box = window.signal_dock.features
    return [box.itemText(i) for i in range(box.count())]


def test_the_menu_has_the_ops(window):
    assert sorted(menu_ops()) == MENU
    assert entries(window) == [f"{name}…" for name in MENU]


def test_an_op_registered_for_the_menu_appears_when_it_opens(window):
    @register_op("double_it", menu={})
    def _double(ctx, series):
        return series.with_values(series.values * 2)

    try:
        window.signal_dock._fill_features()               # what opening it does
        assert "double_it…" in entries(window)
    finally:
        import sys
        sys.modules["mmp_sense.core.graph.ops"]._OPS.pop("double_it")


@pytest.mark.parametrize("name", MENU)
def test_each_op_adds_what_the_console_would(window, session, name):
    before = len(window.panes)
    window.focus_signal = "magnitude"
    form = window.add_op(name)
    assert form.input.currentText() == "magnitude"
    assert form.name.text() == f"{name}_magnitude"
    assert bool(form.frames) == (name in ON_FRAMES)
    assert form.try_add(), form.error.text()
    window.tick()
    assert f"{name}_magnitude" in session.shown
    assert len(window.panes) == before + 1
    source = mmp_sense.signal("magnitude")
    if name in ON_FRAMES:
        source = source.frames("200ms", "25ms")
    expected = session.evaluator.evaluate(source.apply(name, **{
        k: v for k, v in op_parameters(name) if v is not None}), session.window())
    got = session.window()[f"{name}_magnitude"]
    finite = np.isfinite(got)
    assert finite.any()
    held = mmp_sense.core.graph.series.hold(expected, session.window().time)
    assert np.allclose(got[finite], held[finite])


def test_band_share_from_the_form_peaks_at_the_knock(window, session):
    window.focus_signal = "magnitude"
    form = window.add_op("band_share")
    assert form.fields["low_hz"].text() == "80.0" and form.fields["high_hz"].text() == "100.0"
    form.frames["length"].setText("80ms")
    form.frames["hop"].setText("20ms")
    form.name.setText("share")
    assert form.try_add()
    w = session.window()
    share = w["share"]
    near = (w.time > 5.0) & (w.time < 5.1)
    assert np.nanmax(share[near]) > 0.5
    assert np.nanmin(share) >= 0 and np.nanmax(share) <= 1


@pytest.mark.parametrize("field, text, message", [
    ("high_hz", "50", "upside down"),
    ("length", "fast", "duration"),
    ("low_hz", "", "needs a value"),
])
def test_a_bad_value_is_said_and_nothing_is_added(window, session, field, text, message):
    form = window.add_op("band_power")
    (form.frames.get(field) or form.fields[field]).setText(text)
    shown = dict(session.shown)
    assert not form.try_add()
    assert message in form.error.text()
    assert form.isVisible()
    assert session.shown == shown


def test_a_name_in_use_is_made_unique(window, session):
    window.focus_signal = "magnitude"
    window.add_op("energy").try_add()
    form = window.add_op("energy")
    form.input.setCurrentText("magnitude")
    assert form.name.text() == "energy_magnitude2"


def test_dropping_a_graph_signal_hides_its_node(window, session):
    window.focus_signal = "magnitude"
    window.add_op("energy").try_add()
    window.tick()
    window._drop_signal("energy_magnitude")
    assert "energy_magnitude" not in session.shown
    assert "energy_magnitude" not in session.signals
    window.detector_dock.refresh(session.window())
    rows = [window.detector_dock.tree.topLevelItem(i).text(1)
            for i in range(window.detector_dock.tree.topLevelItemCount())]
    assert "energy_magnitude" not in rows
